"""
RadixAttention Demo — SGLang RadixCache 로직을 CPU로 재현

SGLang의 RadixCache는 Radix Tree(접두사 트리)로 KV cache를 관리한다:
  - match_prefix: 새 요청의 token_ids와 가장 긴 공통 prefix를 찾아 KV 반환
  - insert: 생성 완료 후 KV를 트리에 저장
  - evict: 메모리 부족 시 LRU로 노드 제거

이 파일은 그 핵심 로직을 직접 구현하고, GPT-2 추론과 연동해
cache hit/miss 동작을 보여준다.
"""

from __future__ import annotations
import time
import logging
from array import array
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

logging.getLogger("transformers").setLevel(logging.ERROR)


# ════════════════════════════════════════════════════════════════
# RadixTree: SGLang RadixCache의 핵심 구조
# ════════════════════════════════════════════════════════════════

class RadixNode:
    def __init__(self):
        self.children: dict[int, RadixNode] = {}   # first token → child node
        self.token_ids: list[int] = []             # 이 노드가 담당하는 token 구간
        self.kv_cache = None                       # 저장된 KV (past_key_values)
        self.last_access: float = 0.0              # LRU eviction용
        self.ref_count: int = 0                    # 현재 사용 중인 요청 수


class RadixTree:
    """
    SGLang RadixCache의 핵심 동작을 재현한 Radix Tree.

    트리 구조 예시 (system_prompt + 두 개 query):
    root
    └─ [sys_tok_0 ... sys_tok_19]  ← system prompt (공유)
       ├─ [q_a_0 ... q_a_6]        ← Query A KV
       └─ [q_b_0 ... q_b_6]        ← Query B KV
    """

    def __init__(self):
        self.root = RadixNode()
        self.total_tokens_cached = 0
        self.hits = 0
        self.misses = 0

    def match_prefix(self, token_ids: list[int]) -> tuple[int, Optional[object]]:
        """
        token_ids와 가장 긴 공통 prefix를 찾아 (matched_len, kv_cache) 반환.
        matched_len == 0 이면 cache miss.
        """
        node = self.root
        matched_len = 0
        best_kv = None

        pos = 0
        while pos < len(token_ids):
            tok = token_ids[pos]
            if tok not in node.children:
                break
            child = node.children[tok]
            # 이 child 노드의 token_ids와 비교
            child_toks = child.token_ids
            match_len = 0
            for ct in child_toks:
                if pos + match_len >= len(token_ids):
                    break
                if token_ids[pos + match_len] != ct:
                    break
                match_len += 1
            pos += match_len
            matched_len += match_len
            if child.kv_cache is not None:
                best_kv = child.kv_cache
                child.last_access = time.time()
            if match_len < len(child_toks):
                break  # 부분 매칭 → 더 내려갈 수 없음
            node = child

        if matched_len > 0:
            self.hits += 1
        else:
            self.misses += 1

        return matched_len, best_kv

    def insert(self, token_ids: list[int], kv_cache) -> None:
        """
        token_ids에 해당하는 KV를 트리에 저장.
        기존 노드와 공통 prefix가 있으면 분기(split) 처리.
        """
        node = self.root
        pos = 0

        while pos < len(token_ids):
            tok = token_ids[pos]
            if tok not in node.children:
                # 새 노드 생성
                new_node = RadixNode()
                new_node.token_ids = token_ids[pos:]
                new_node.kv_cache = kv_cache
                new_node.last_access = time.time()
                node.children[tok] = new_node
                self.total_tokens_cached += len(token_ids[pos:])
                return
            child = node.children[tok]
            child_toks = child.token_ids
            # 공통 prefix 길이 계산
            common = 0
            for ct in child_toks:
                if pos + common >= len(token_ids):
                    break
                if token_ids[pos + common] != ct:
                    break
                common += 1
            if common == len(child_toks):
                # 이 노드를 완전히 통과
                pos += common
                node = child
            else:
                # 중간에서 분기 — 노드를 쪼갬 (split)
                # 기존 노드를 공통 부분과 나머지로 분리
                split_node = RadixNode()
                split_node.token_ids = child_toks[:common]
                split_node.kv_cache = None  # 중간 노드엔 KV 없음
                split_node.last_access = time.time()

                child.token_ids = child_toks[common:]
                split_node.children[child_toks[common]] = child
                node.children[tok] = split_node

                # 나머지 새 토큰으로 새 자식 생성
                new_node = RadixNode()
                new_node.token_ids = token_ids[pos + common:]
                new_node.kv_cache = kv_cache
                new_node.last_access = time.time()
                split_node.children[token_ids[pos + common]] = new_node
                self.total_tokens_cached += len(token_ids[pos + common:])
                return

        # 루프 정상 종료: 기존 노드에 KV 업데이트
        if node.kv_cache is None:
            node.kv_cache = kv_cache
            node.last_access = time.time()

    def print_tree(self, tokenizer=None):
        """트리 구조를 시각화."""
        def _print(node: RadixNode, depth: int, prefix_len: int):
            if node is self.root:
                print("  root")
            for first_tok, child in node.children.items():
                indent = "  " + "│  " * depth + "├─ "
                toks = child.token_ids
                if tokenizer:
                    decoded = repr(tokenizer.decode(toks))
                    label = f"{decoded} ({len(toks)} tokens)"
                else:
                    label = f"tokens={toks[:4]}... ({len(toks)} tokens)"
                has_kv = " [KV cached]" if child.kv_cache is not None else ""
                cum_len = prefix_len + len(toks)
                print(f"{indent}{label}{has_kv}  cumlen={cum_len}")
                _print(child, depth + 1, cum_len)
        _print(self.root, 0, 0)

    @property
    def hit_rate(self):
        total = self.hits + self.misses
        return self.hits / total if total > 0 else 0.0


# ════════════════════════════════════════════════════════════════
# GPT-2 추론 헬퍼
# ════════════════════════════════════════════════════════════════

def get_seq_len(kv):
    if hasattr(kv, 'get_seq_length'):
        return kv.get_seq_length()
    return kv[0][0].shape[2]


def run_prefill(model, token_ids: list[int], past_kv=None):
    """token_ids를 prefill. past_kv가 있으면 이어서 처리."""
    ids = torch.tensor([token_ids])
    with torch.no_grad():
        out = model(ids, past_key_values=past_kv, use_cache=True)
    return out.past_key_values


def run_decode(model, past_kv, n_tokens: int):
    """past_kv 이어서 n_tokens 생성."""
    # prefill 마지막 토큰의 logits로 첫 토큰 결정
    with torch.no_grad():
        dummy_ids = torch.tensor([[0]])  # 아무 토큰으로 시작 (past_kv가 context 담당)
        out = model(dummy_ids, past_key_values=past_kv, use_cache=True)
    cur = out.logits[:, -1, :].argmax(-1, keepdim=True)
    past = out.past_key_values
    tokens = [cur.item()]
    with torch.no_grad():
        for _ in range(n_tokens - 1):
            out = model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            cur = out.logits[:, -1, :].argmax(-1, keepdim=True)
            tokens.append(cur.item())
    return tokens, past


# ════════════════════════════════════════════════════════════════
# 메인 데모
# ════════════════════════════════════════════════════════════════

MODEL = "gpt2"
N_ANSWER_TOKENS = 5

SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Answer questions clearly and concisely. "
    "Always be polite."
)

REQUESTS = [
    "Question: What is machine learning?",
    "Question: What is deep learning?",
    "Question: What is neural network?",
    # 아래는 완전히 다른 prefix — cache miss 발생
    "Completely different topic: Tell me about the weather.",
]

print(f"모델 로딩: {MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
model.eval()

tree = RadixTree()

sys_ids = tokenizer.encode(SYSTEM_PROMPT)
print(f"\nSystem prompt 토큰 수: {len(sys_ids)}")
print(f"System prompt: {repr(SYSTEM_PROMPT[:60])}...")

print("\n" + "=" * 65)
print("[ RadixAttention 서빙 시뮬레이션 ]")
print("=" * 65)
print(f"  {'요청':>3}  {'상태':>10}  {'hit len':>8}  {'처리 토큰':>10}  {'시간':>10}  query")
print(f"  {'---':>3}  {'--------':>10}  {'-------':>8}  {'----------':>10}  {'--------':>10}  -----")

for req_idx, query in enumerate(REQUESTS):
    # 전체 입력 = system prompt + query
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    full_ids = sys_ids + query_ids

    t0 = time.perf_counter()

    # 1) RadixTree에서 prefix 매칭
    matched_len, cached_kv = tree.match_prefix(full_ids)

    if matched_len > 0 and cached_kv is not None:
        status = "HIT"
        # cache hit: matched_len 이후의 토큰만 prefill
        remaining_ids = full_ids[matched_len:]
        kv = run_prefill(model, remaining_ids, past_kv=cached_kv)
        processed_tokens = len(remaining_ids)
    else:
        status = "MISS"
        # cache miss: 전체 prefill
        kv = run_prefill(model, full_ids)
        processed_tokens = len(full_ids)

    # 2) 답변 생성
    answer_tokens, final_kv = run_decode(model, kv, N_ANSWER_TOKENS)

    elapsed = time.perf_counter() - t0

    # 3) 생성 완료 → 트리에 KV 저장 (full_ids까지의 KV)
    tree.insert(full_ids, kv)

    answer = tokenizer.decode(answer_tokens)
    query_short = repr(query[:35])
    print(f"  {req_idx+1:>3}  {status:>10}  {matched_len:>8}  {processed_tokens:>10}  {elapsed*1000:>8.1f} ms  {query_short}")

print(f"\n  총 cache hit: {tree.hits}/{tree.hits + tree.misses} 요청")
print(f"  hit rate:     {tree.hit_rate:.0%}")
print(f"  트리에 캐시된 총 토큰 수: {tree.total_tokens_cached}")


# ════════════════════════════════════════════════════════════════
# 트리 구조 시각화
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("[ Radix Tree 구조 ]")
print("=" * 65)
tree.print_tree(tokenizer)

print("\n  [ 읽는 법 ]")
print("  - 공통 prefix는 트리 상위 노드로 공유됨")
print("  - [KV cached] 표시된 노드의 KV를 다음 요청이 재사용")
print("  - 완전히 다른 prefix 요청(req 4)은 별도 분기로 저장됨")


# ════════════════════════════════════════════════════════════════
# 동일 요청 재실행 — 100% hit 확인
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("[ 동일 요청 재실행 — 모든 요청이 full cache hit ]")
print("=" * 65)
print(f"  {'요청':>3}  {'상태':>10}  {'hit len':>8}  {'처리 토큰':>10}  {'시간':>10}")
print(f"  {'---':>3}  {'--------':>10}  {'-------':>8}  {'----------':>10}  {'--------':>10}")

tree2 = RadixTree()
# 첫 번째 라운드의 KV를 그대로 재활용하기 위해 트리 복사
tree2 = tree  # 동일 트리에서 재조회

for req_idx, query in enumerate(REQUESTS):
    query_ids = tokenizer.encode(query, add_special_tokens=False)
    full_ids = sys_ids + query_ids

    t0 = time.perf_counter()
    matched_len, cached_kv = tree.match_prefix(full_ids)

    if matched_len >= len(full_ids) - 1 and cached_kv is not None:
        status = "FULL HIT"
        processed_tokens = 0
    elif matched_len > 0 and cached_kv is not None:
        status = "PART HIT"
        remaining_ids = full_ids[matched_len:]
        run_prefill(model, remaining_ids, past_kv=cached_kv)
        processed_tokens = len(remaining_ids)
    else:
        status = "MISS"
        processed_tokens = len(full_ids)

    elapsed = time.perf_counter() - t0
    print(f"  {req_idx+1:>3}  {status:>10}  {matched_len:>8}  {processed_tokens:>10}  {elapsed*1000:>8.1f} ms")
