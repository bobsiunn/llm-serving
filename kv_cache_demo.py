"""
KV Cache Reuse Demo

prefill: 프롬프트 전체를 한 번에 처리 → K/V 텐서를 cache에 저장
decode: 새 토큰 하나씩 생성 → cache된 K/V를 재사용, 재연산 없음

cache=True  vs  cache=False 의 시간 차이로 reuse 효과를 확인한다.
"""

import time
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL = "gpt2"
PROMPT = "The key insight about transformer models is that attention"
N_NEW_TOKENS = 10

print(f"모델 로딩: {MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
model.eval()

input_ids = tokenizer(PROMPT, return_tensors="pt").input_ids
prompt_len = input_ids.shape[1]
print(f"\n프롬프트 토큰 수: {prompt_len}")
print(f"생성할 토큰 수:   {N_NEW_TOKENS}")
print(f"모델 레이어 수:   {model.config.n_layer}")
print(f"attention heads:  {model.config.n_head}")
print(f"head dimension:   {model.config.n_embd // model.config.n_head}")


def get_cache_seq_len(kv_cache):
    """DynamicCache(transformers 5.x) 또는 tuple-of-tuples 모두 지원."""
    if hasattr(kv_cache, 'get_seq_length'):
        return kv_cache.get_seq_length()
    return kv_cache[0][0].shape[2]


def get_layer0_kv(kv_cache):
    """Layer 0의 K, V 텐서를 반환. shape: (batch, heads, seq_len, head_dim)."""
    if hasattr(kv_cache, 'layers'):
        k = kv_cache.layers[0].keys   # (batch, heads, seq_len, head_dim)
        v = kv_cache.layers[0].values
    elif hasattr(kv_cache, 'key_cache'):
        k = kv_cache.key_cache[0]
        v = kv_cache.value_cache[0]
    else:
        k, v = kv_cache[0]
    return k, v


def get_n_layers(kv_cache):
    if hasattr(kv_cache, 'layers'):
        return len(kv_cache.layers)
    if hasattr(kv_cache, 'key_cache'):
        return len(kv_cache.key_cache)
    return len(kv_cache)


# ──────────────────────────────────────────────
# 1. prefill: 프롬프트 처리 + KV cache 생성
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("[ PREFILL ]  프롬프트 전체를 한 번에 처리")
print("=" * 60)

t0 = time.perf_counter()
with torch.no_grad():
    prefill_out = model(input_ids, use_cache=True)
prefill_time = time.perf_counter() - t0

kv_cache = prefill_out.past_key_values
k0, v0 = get_layer0_kv(kv_cache)
seq_len_after_prefill = get_cache_seq_len(kv_cache)

print(f"  소요 시간:      {prefill_time*1000:.1f} ms")
print(f"  cache 레이어 수: {get_n_layers(kv_cache)}")
print(f"\n  [Layer 0] K shape: {tuple(k0.shape)}")
print(f"             → (batch, heads, seq_len, head_dim)")
print(f"  [Layer 0] V shape: {tuple(v0.shape)}")
print(f"\n  → cache seq_len={seq_len_after_prefill} == 프롬프트 토큰 수({prompt_len})  ✓")
print(f"  → 이후 decode 시 이 K/V를 재사용, 재계산 없음")


# ──────────────────────────────────────────────
# 2. decode with cache: KV reuse
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("[ DECODE — cache=True ]  KV cache 재사용")
print("=" * 60)
print(f"  step  새 토큰       cache seq_len    decode 시간")
print(f"  ----  ----------   -------------    -----------")

current_ids = input_ids[:, -1:]   # 마지막 토큰만 입력 (새 토큰 1개 기준)
past = kv_cache
token_times_cached = []

with torch.no_grad():
    for step in range(N_NEW_TOKENS):
        t0 = time.perf_counter()
        out = model(current_ids, past_key_values=past, use_cache=True)
        elapsed = time.perf_counter() - t0
        token_times_cached.append(elapsed)

        past = out.past_key_values
        next_token = out.logits[:, -1, :].argmax(-1, keepdim=True)
        current_ids = next_token

        cache_len = get_cache_seq_len(past)
        token_str = repr(tokenizer.decode(next_token[0]))
        print(f"  {step+1:>4}  {token_str:<12} {cache_len:>13}    {elapsed*1000:>8.1f} ms")

avg_cached = sum(token_times_cached) / len(token_times_cached)


# ──────────────────────────────────────────────
# 3. decode without cache: 매번 전체 재연산
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("[ DECODE — cache=False ]  매번 전체 시퀀스 재연산")
print("=" * 60)
print(f"  step  새 토큰       입력 seq_len     decode 시간")
print(f"  ----  ----------   -------------    -----------")

all_ids = input_ids
token_times_nocache = []

with torch.no_grad():
    for step in range(N_NEW_TOKENS):
        t0 = time.perf_counter()
        out = model(all_ids, use_cache=False)
        elapsed = time.perf_counter() - t0
        token_times_nocache.append(elapsed)

        next_token = out.logits[:, -1, :].argmax(-1, keepdim=True)
        all_ids = torch.cat([all_ids, next_token], dim=1)

        token_str = repr(tokenizer.decode(next_token[0]))
        seq_len = all_ids.shape[1]
        print(f"  {step+1:>4}  {token_str:<12} {seq_len:>13}    {elapsed*1000:>8.1f} ms")

avg_nocache = sum(token_times_nocache) / len(token_times_nocache)


# ──────────────────────────────────────────────
# 4. 비교 요약
# ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("[ 결과 요약 ]")
print("=" * 60)
print(f"  decode 평균 (cache=True) :  {avg_cached*1000:.1f} ms/token")
print(f"  decode 평균 (cache=False):  {avg_nocache*1000:.1f} ms/token")
print(f"  속도 향상:  {avg_nocache/avg_cached:.1f}x")
print()
print("  [ KV Cache Reuse 원리 ]")
print(f"  - prefill:  seq_len={prompt_len} 토큰의 K/V를 한 번에 계산 → cache 저장")
print(f"  - cache=True:   새 토큰 1개의 Q만 생성 → 저장된 K/V와 attention 계산")
print(f"                  attention 연산량: O(1 × cached_seq_len)  → 빠름")
print(f"  - cache=False:  매 step 전체 시퀀스를 처음부터 재계산")
print(f"                  attention 연산량: O(seq_len²)  → step마다 느려짐")
