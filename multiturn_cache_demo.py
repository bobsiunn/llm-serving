"""
Multi-turn Cache Demo

대화가 쌓일수록 KV cache가 어떻게 누적되고,
이전 턴의 KV를 재계산 없이 재사용하는지 보여준다.

cache=True:  이전 턴의 KV를 계속 이어 붙임 → 이전 대화 재연산 없음
cache=False: 매 턴마다 전체 대화 히스토리를 처음부터 재계산
"""

import time
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL = "gpt2"
N_ANSWER_TOKENS = 6   # 턴당 생성할 토큰 수

TURNS = [
    "User: Hello, who are you?",
    "User: What can you help me with?",
    "User: Tell me about neural networks.",
    "User: How does attention work?",
]

print(f"모델 로딩: {MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
model.eval()


def get_seq_len(kv):
    if hasattr(kv, 'get_seq_length'):
        return kv.get_seq_length()
    return kv[0][0].shape[2]


def get_kv_memory_bytes(kv):
    """KV cache가 차지하는 메모리 크기 계산 (bytes)."""
    total = 0
    if hasattr(kv, 'layers'):
        for layer in kv.layers:
            total += layer.keys.nelement() * layer.keys.element_size()
            total += layer.values.nelement() * layer.values.element_size()
    elif hasattr(kv, 'key_cache'):
        for k, v in zip(kv.key_cache, kv.value_cache):
            total += k.nelement() * k.element_size() + v.nelement() * v.element_size()
    else:
        for k, v in kv:
            total += k.nelement() * k.element_size() + v.nelement() * v.element_size()
    return total


def greedy_generate(model, input_ids, past_kv, n_tokens):
    cur = input_ids[:, -1:]
    past = past_kv
    generated = []
    with torch.no_grad():
        for _ in range(n_tokens):
            out = model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
            generated.append(next_tok.item())
            cur = next_tok
    return generated, past


# ════════════════════════════════════════════════════════
# 방법 1: cache=True — KV 누적 재사용
# ════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("[ cache=True ]  KV cache 누적 — 이전 턴 재계산 없음")
print("=" * 65)
print(f"  {'턴':>3}  {'입력 토큰':>8}  {'cache seq_len':>14}  {'KV 메모리':>10}  {'시간':>10}")
print(f"  {'---':>3}  {'--------':>8}  {'--------------':>14}  {'----------':>10}  {'----------':>10}")

past_kv = None
history_ids = None
turn_times_cached = []

for turn_idx, user_msg in enumerate(TURNS):
    # 이번 턴 입력 토큰화
    turn_ids = tokenizer(
        "\n" + user_msg + "\nAssistant:",
        add_special_tokens=(turn_idx == 0),
        return_tensors="pt"
    ).input_ids

    # prefill: 이번 턴 입력만 처리 (이전 KV는 past_kv로 전달)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(turn_ids, past_key_values=past_kv, use_cache=True)
    t_prefill = time.perf_counter() - t0

    # decode: 답변 생성
    answer_tokens, past_kv = greedy_generate(model, turn_ids, out.past_key_values, N_ANSWER_TOKENS)
    elapsed = time.perf_counter() - t0
    turn_times_cached.append(elapsed)

    cache_len = get_seq_len(past_kv)
    kv_mem_kb = get_kv_memory_bytes(past_kv) / 1024
    answer_str = tokenizer.decode(answer_tokens)

    print(f"  {turn_idx+1:>3}  {turn_ids.shape[1]:>8}  {cache_len:>14}  {kv_mem_kb:>8.1f} KB  {elapsed*1000:>8.1f} ms")
    print(f"       입력: {repr(user_msg)}")
    print(f"       답변: {repr(answer_str)}")
    if turn_idx < len(TURNS) - 1:
        print(f"       → 다음 턴에서 위 {cache_len}토큰의 KV 재사용 (재계산 없음)")
    print()


# ════════════════════════════════════════════════════════
# 방법 2: cache=False — 매 턴 전체 히스토리 재계산
# ════════════════════════════════════════════════════════
print("=" * 65)
print("[ cache=False ]  매 턴 전체 대화 히스토리 처음부터 재계산")
print("=" * 65)
print(f"  {'턴':>3}  {'전체 seq_len':>12}  {'시간':>10}")
print(f"  {'---':>3}  {'------------':>12}  {'----------':>10}")

full_history_ids = None
turn_times_nocache = []

for turn_idx, user_msg in enumerate(TURNS):
    turn_text = ("\n" if turn_idx > 0 else "") + user_msg + "\nAssistant:"
    turn_ids = tokenizer(
        turn_text,
        add_special_tokens=(turn_idx == 0),
        return_tensors="pt"
    ).input_ids

    if full_history_ids is None:
        full_history_ids = turn_ids
    else:
        full_history_ids = torch.cat([full_history_ids, turn_ids], dim=1)

    t0 = time.perf_counter()
    # 매번 전체 히스토리를 처음부터 처리
    with torch.no_grad():
        out = model(full_history_ids, use_cache=False)
    next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
    # 답변 토큰 추가 (no cache라 매번 점점 더 긴 시퀀스 처리)
    ans_ids = full_history_ids
    for _ in range(N_ANSWER_TOKENS - 1):
        ans_ids = torch.cat([ans_ids, next_tok], dim=1)
        with torch.no_grad():
            out = model(ans_ids, use_cache=False)
        next_tok = out.logits[:, -1, :].argmax(-1, keepdim=True)
    elapsed = time.perf_counter() - t0
    turn_times_nocache.append(elapsed)

    full_history_ids = torch.cat([ans_ids, next_tok], dim=1)
    print(f"  {turn_idx+1:>3}  {full_history_ids.shape[1]:>12}  {elapsed*1000:>8.1f} ms")


# ════════════════════════════════════════════════════════
# 비교 요약
# ════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("[ 비교 요약 ]")
print("=" * 65)
print(f"  {'턴':>3}  {'cache=True':>12}  {'cache=False':>13}  {'비율':>6}")
print(f"  {'---':>3}  {'----------':>12}  {'-----------':>13}  {'------':>6}")
for i, (tc, tnc) in enumerate(zip(turn_times_cached, turn_times_nocache)):
    ratio = tnc / tc
    print(f"  {i+1:>3}  {tc*1000:>10.1f} ms  {tnc*1000:>11.1f} ms  {ratio:>5.1f}x")

total_c  = sum(turn_times_cached)
total_nc = sum(turn_times_nocache)
print(f"  {'합계':>3}  {total_c*1000:>10.1f} ms  {total_nc*1000:>11.1f} ms  {total_nc/total_c:>5.1f}x")

print()
print("  [ Multi-turn KV Cache 원리 ]")
print("  - cache=True:  새로운 턴 입력만 처리 → 이전 모든 턴 KV 재사용")
print("                 턴이 쌓여도 그 턴의 새 입력 토큰만큼만 시간 증가")
print("  - cache=False: 매 턴 전체 대화를 처음부터 재계산")
print("                 턴이 쌓일수록 처리해야 할 seq_len이 선형 증가")
print("                 → attention은 O(seq_len²)이므로 점점 느려짐")
print()
print("  [ 실제 서빙에서 ]")
print("  - vLLM, SGLang: KV cache를 PagedAttention / RadixAttention으로 관리")
print("  - 메모리 제약으로 오래된 턴의 KV를 evict하거나 offload할 수 있음")
