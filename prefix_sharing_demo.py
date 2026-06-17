"""
Prefix Sharing Demo — SGLang RadixAttention의 핵심 아이디어

같은 prefix를 가진 두 요청이 있을 때:
  - 공유 없음: prefix KV를 두 번 계산
  - 공유 있음: prefix KV를 한 번만 계산 → 두 요청이 재사용

이 파일은 그 차이를 시간과 shape으로 보여준다.
"""

import time
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL = "gpt2"
SYSTEM_PREFIX = (
    "You are a helpful AI assistant. "
    "Answer questions clearly and concisely. "
    "Always be polite and professional."
)
QUERY_A = " Question: What is machine learning?"
QUERY_B = " Question: What is deep learning?"
N_ANSWER_TOKENS = 8


print(f"모델 로딩: {MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
model.eval()


def get_seq_len(kv):
    if hasattr(kv, 'get_seq_length'):
        return kv.get_seq_length()
    return kv[0][0].shape[2]


def get_layer_kv(kv, layer_idx=0):
    if hasattr(kv, 'layers'):
        return kv.layers[layer_idx].keys, kv.layers[layer_idx].values
    if hasattr(kv, 'key_cache'):
        return kv.key_cache[layer_idx], kv.value_cache[layer_idx]
    return kv[layer_idx]


def greedy_decode(model, prompt_ids, past_kv, n_tokens):
    """past_kv를 시작점으로 n_tokens 생성. 첫 입력은 prompt_ids의 마지막 토큰."""
    cur = prompt_ids[:, -1:]
    past = past_kv
    tokens = []
    with torch.no_grad():
        for _ in range(n_tokens):
            out = model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
            tokens.append(next_tok.item())
            cur = next_tok
    return tokens, past


# ── 토큰화 ──────────────────────────────────────
prefix_ids   = tokenizer(SYSTEM_PREFIX, return_tensors="pt").input_ids
query_a_ids  = tokenizer(QUERY_A, add_special_tokens=False, return_tensors="pt").input_ids
query_b_ids  = tokenizer(QUERY_B, add_special_tokens=False, return_tensors="pt").input_ids

full_a_ids = torch.cat([prefix_ids, query_a_ids], dim=1)
full_b_ids = torch.cat([prefix_ids, query_b_ids], dim=1)

prefix_len  = prefix_ids.shape[1]
query_a_len = query_a_ids.shape[1]
query_b_len = query_b_ids.shape[1]

print(f"\nPrefix 토큰 수:   {prefix_len}  ← 두 요청이 공유하는 부분")
print(f"Query A 토큰 수:  {query_a_len}  ← 요청 A만의 부분")
print(f"Query B 토큰 수:  {query_b_len}  ← 요청 B만의 부분")
print(f"\nPrefix:  {repr(SYSTEM_PREFIX[:60])}...")
print(f"Query A: {repr(QUERY_A)}")
print(f"Query B: {repr(QUERY_B)}")


# ════════════════════════════════════════════════════════
# 방법 1: 공유 없음 — prefix를 두 번 각각 계산
# ════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("[ 방법 1 ]  Prefix 공유 없음 — 두 요청 각각 처음부터 계산")
print("=" * 62)

t0 = time.perf_counter()
with torch.no_grad():
    out_a = model(full_a_ids, use_cache=True)
t_a_prefill = time.perf_counter() - t0
tokens_a_noshare, _ = greedy_decode(model, full_a_ids, out_a.past_key_values, N_ANSWER_TOKENS)

t0 = time.perf_counter()
with torch.no_grad():
    out_b = model(full_b_ids, use_cache=True)
t_b_prefill = time.perf_counter() - t0
tokens_b_noshare, _ = greedy_decode(model, full_b_ids, out_b.past_key_values, N_ANSWER_TOKENS)

total_no_share = t_a_prefill + t_b_prefill

print(f"\n  Request A prefill: {t_a_prefill*1000:>7.1f} ms  (seq_len={full_a_ids.shape[1]})")
print(f"  Request B prefill: {t_b_prefill*1000:>7.1f} ms  (seq_len={full_b_ids.shape[1]})")
print(f"  합계:              {total_no_share*1000:>7.1f} ms")
print(f"\n  → prefix {prefix_len}토큰의 KV가 2번 계산됨 (낭비)")

ans_a = tokenizer.decode(tokens_a_noshare)
ans_b = tokenizer.decode(tokens_b_noshare)
print(f"\n  A 답변 시작: {repr(ans_a)}")
print(f"  B 답변 시작: {repr(ans_b)}")


# ════════════════════════════════════════════════════════
# 방법 2: Prefix 공유 — prefix KV를 한 번만 계산
# ════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("[ 방법 2 ]  Prefix 공유 — prefix KV 한 번 계산 후 두 요청이 재사용")
print("=" * 62)

# Step 1: prefix KV 계산 (한 번만)
t0 = time.perf_counter()
with torch.no_grad():
    prefix_out = model(prefix_ids, use_cache=True)
t_prefix = time.perf_counter() - t0
prefix_kv = prefix_out.past_key_values

k0, v0 = get_layer_kv(prefix_kv)
print(f"\n  [Step 1] Prefix prefill: {t_prefix*1000:.1f} ms")
print(f"           Layer 0 K shape: {tuple(k0.shape)}")
print(f"           → seq_len={get_seq_len(prefix_kv)} 토큰의 KV가 캐시에 저장됨")

# Step 2-A: Query A만 추가로 처리 (prefix KV 재사용)
t0 = time.perf_counter()
with torch.no_grad():
    out_a2 = model(query_a_ids, past_key_values=prefix_kv, use_cache=True)
t_query_a = time.perf_counter() - t0
tokens_a_share, _ = greedy_decode(model, query_a_ids, out_a2.past_key_values, N_ANSWER_TOKENS)

k0_a, _ = get_layer_kv(out_a2.past_key_values)
print(f"\n  [Step 2A] Query A 처리:  {t_query_a*1000:.1f} ms  (query만 {query_a_len}토큰)")
print(f"           Layer 0 K shape: {tuple(k0_a.shape)}")
print(f"           → seq_len={get_seq_len(out_a2.past_key_values)}  (prefix + query A)")

# Step 2-B: Query B만 추가로 처리 (동일한 prefix_kv 재사용)
t0 = time.perf_counter()
with torch.no_grad():
    out_b2 = model(query_b_ids, past_key_values=prefix_kv, use_cache=True)
t_query_b = time.perf_counter() - t0
tokens_b_share, _ = greedy_decode(model, query_b_ids, out_b2.past_key_values, N_ANSWER_TOKENS)

k0_b, _ = get_layer_kv(out_b2.past_key_values)
print(f"\n  [Step 2B] Query B 처리:  {t_query_b*1000:.1f} ms  (query만 {query_b_len}토큰)")
print(f"           Layer 0 K shape: {tuple(k0_b.shape)}")
print(f"           → seq_len={get_seq_len(out_b2.past_key_values)}  (prefix + query B)")

total_share = t_prefix + t_query_a + t_query_b
print(f"\n  합계: {total_share*1000:.1f} ms  (prefix는 딱 한 번만)")

ans_a2 = tokenizer.decode(tokens_a_share)
ans_b2 = tokenizer.decode(tokens_b_share)
print(f"\n  A 답변 시작: {repr(ans_a2)}")
print(f"  B 답변 시작: {repr(ans_b2)}")


# ════════════════════════════════════════════════════════
# 비교 요약
# ════════════════════════════════════════════════════════
print("\n" + "=" * 62)
print("[ 비교 요약 ]")
print("=" * 62)
print(f"  공유 없음:  {total_no_share*1000:>7.1f} ms  (prefix {prefix_len}토큰 × 2번 계산)")
print(f"  공유 있음:  {total_share*1000:>7.1f} ms  (prefix {prefix_len}토큰 × 1번 계산)")
print(f"  절감:       {(total_no_share - total_share)*1000:>7.1f} ms  ({(1 - total_share/total_no_share)*100:.0f}% 감소)")
print()
print("  [ SGLang RadixAttention 아이디어 ]")
print("  - Radix Tree로 prefix를 관리 → 동일 prefix가 오면 KV 재사용")
print("  - System prompt, few-shot 예시처럼 많은 요청이 공유하는 prefix일수록 효과 극대화")
print(f"  - prefix가 길수록 (지금은 {prefix_len}토큰), 요청 수가 많을수록 절감량 증가")
