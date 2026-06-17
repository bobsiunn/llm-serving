"""
RadixAttention (SGLang) vs PagedAttention (vLLM) 비교

두 가지 시나리오로 각각이 유리한 상황을 실측한다.

시나리오 A — RadixAttention 유리
  공유 system prompt를 가진 요청들이 들어올 때.
  RadixAttention: 토큰 단위 prefix 매칭 → 공유 prefix 전체 재사용
  PagedAttention: block 경계까지만 재사용 → partial block 낭비

시나리오 B — PagedAttention 유리 (Wide & Shallow Tree)
  공유 prefix가 없는 다양한 요청 + 제한된 메모리 budget.
  RadixAttention: 시퀀스 전체를 노드 단위로 evict → 굵게 버림
  PagedAttention: block 단위로 evict → 세밀하게 버려서 더 많은 요청 보존
"""

from __future__ import annotations
import hashlib
import time
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, DynamicCache

logging.getLogger("transformers").setLevel(logging.ERROR)

MODEL = "gpt2"
N_ANSWER_TOKENS = 5

# ── 시나리오 A 파라미터 ────────────────────────────────────────
A_BLOCK_SIZE = 16    # vLLM 기본값

A_SYSTEM_PROMPT = (
    "You are a helpful AI assistant. "
    "Answer questions clearly and concisely. "
    "Always be polite."
)
A_REQUESTS = [
    "Question: What is machine learning?",
    "Question: What is deep learning?",
    "Question: What is neural network?",
    "Completely different topic: Tell me about the weather.",
]

# ── 시나리오 B 파라미터 ────────────────────────────────────────
B_BLOCK_SIZE    = 8   # 각 요청이 1 full block + partial 구조가 되도록
B_BUDGET_BLOCKS = 2   # 메모리 budget: block 수
B_BUDGET_TOKENS = B_BUDGET_BLOCKS * B_BLOCK_SIZE

# 완전히 다른 prefix (공유 없음), 각 9~11 토큰
B_REQUESTS = [
    "Quantum mechanics is the foundation of modern physics research",
    "Renaissance paintings brought realism and perspective into art",
    "Machine learning algorithms learn patterns from labeled datasets",
    "Environmental scientists study the effects of climate on life",
]


# ════════════════════════════════════════════════════════════════
# 공통 헬퍼
# ════════════════════════════════════════════════════════════════

def run_prefill(model, token_ids: list[int], past_kv=None):
    ids = torch.tensor([token_ids])
    with torch.no_grad():
        out = model(ids, past_key_values=past_kv, use_cache=True)
    return out.past_key_values


def run_decode(model, past_kv, n_tokens: int):
    with torch.no_grad():
        out = model(torch.tensor([[0]]), past_key_values=past_kv, use_cache=True)
    cur = out.logits[:, -1, :].argmax(-1, keepdim=True)
    past = out.past_key_values
    tokens = [cur.item()]
    with torch.no_grad():
        for _ in range(n_tokens - 1):
            out = model(cur, past_key_values=past, use_cache=True)
            past = out.past_key_values
            cur = out.logits[:, -1, :].argmax(-1, keepdim=True)
            tokens.append(cur.item())
    return tokens


def slice_kv(kv, n_tokens: int):
    """past_key_values를 앞 n_tokens 분량으로 슬라이스."""
    new_cache = DynamicCache()
    if hasattr(kv, 'layers'):
        for i, layer in enumerate(kv.layers):
            new_cache.update(layer.keys[:, :, :n_tokens, :].clone(),
                             layer.values[:, :, :n_tokens, :].clone(), i)
    else:
        for i, (k, v) in enumerate(kv):
            new_cache.update(k[:, :, :n_tokens, :].clone(),
                             v[:, :, :n_tokens, :].clone(), i)
    return new_cache


def hit_status(matched: int, total: int) -> str:
    if matched == 0:           return "MISS"
    if matched >= total - 1:   return "FULL HIT"
    return "PART HIT"


# ════════════════════════════════════════════════════════════════
# RadixAttention
# ════════════════════════════════════════════════════════════════

class RadixNode:
    def __init__(self):
        self.children: dict[int, RadixNode] = {}
        self.token_ids: list[int] = []
        self.kv_cache = None
        self.last_access: float = 0.0


class RadixTree:
    def __init__(self):
        self.root = RadixNode()

    def match_prefix(self, token_ids: list[int]):
        node, matched_len, best_kv = self.root, 0, None
        pos = 0
        while pos < len(token_ids):
            tok = token_ids[pos]
            if tok not in node.children:
                break
            child = node.children[tok]
            common = 0
            for ct in child.token_ids:
                if pos + common >= len(token_ids) or token_ids[pos + common] != ct:
                    break
                common += 1
            pos += common
            matched_len += common
            if child.kv_cache is not None:
                best_kv = child.kv_cache
                child.last_access = time.time()
            if common < len(child.token_ids):
                break
            node = child
        return matched_len, best_kv

    def insert(self, token_ids: list[int], kv_cache):
        node, pos = self.root, 0
        while pos < len(token_ids):
            tok = token_ids[pos]
            if tok not in node.children:
                new = RadixNode()
                new.token_ids = token_ids[pos:]
                new.kv_cache = kv_cache
                new.last_access = time.time()
                node.children[tok] = new
                return
            child = node.children[tok]
            common = 0
            for ct in child.token_ids:
                if pos + common >= len(token_ids) or token_ids[pos + common] != ct:
                    break
                common += 1
            if common == len(child.token_ids):
                pos += common
                node = child
            else:
                split = RadixNode()
                split.token_ids = child.token_ids[:common]
                child.token_ids = child.token_ids[common:]
                split.children[child.token_ids[0]] = child
                node.children[tok] = split
                new = RadixNode()
                new.token_ids = token_ids[pos + common:]
                new.kv_cache = kv_cache
                new.last_access = time.time()
                split.children[token_ids[pos + common]] = new
                return
        node.kv_cache = node.kv_cache or kv_cache
        node.last_access = time.time()

    def print_tree(self, tokenizer=None, node=None, depth=0, prefix_len=0):
        if node is None:
            node = self.root
            print("  root")
        for child in node.children.values():
            indent = "  " + "│  " * depth + "├─ "
            label = repr(tokenizer.decode(child.token_ids)) if tokenizer else str(child.token_ids)
            kv_mark = " [KV]" if child.kv_cache is not None else ""
            cum = prefix_len + len(child.token_ids)
            print(f"{indent}{label} ({len(child.token_ids)}tok){kv_mark}  cumlen={cum}")
            self.print_tree(tokenizer, child, depth + 1, cum)


class RadixTreeLRU(RadixTree):
    """토큰 budget 초과 시 LRU leaf 노드를 통째로 evict."""

    def __init__(self, max_tokens: int):
        super().__init__()
        self.max_tokens = max_tokens
        self._cached_tokens = 0

    def _kv_leaves(self, node=None):
        if node is None:
            node = self.root
        result = []
        def _walk(n, parent, key):
            if n.kv_cache is not None and not n.children:
                result.append((n, parent, key))
            for k, c in n.children.items():
                _walk(c, n, k)
        for k, c in node.children.items():
            _walk(c, node, k)
        return result

    def _evict_lru(self):
        leaves = self._kv_leaves()
        if not leaves:
            return 0
        node, parent, key = min(leaves, key=lambda x: x[0].last_access)
        freed = len(node.token_ids)
        node.kv_cache = None
        del parent.children[key]
        self._cached_tokens -= freed
        return freed

    def insert(self, token_ids: list[int], kv_cache):
        while self._cached_tokens + len(token_ids) > self.max_tokens:
            if self._evict_lru() == 0:
                break
        super().insert(token_ids, kv_cache)
        self._cached_tokens += len(token_ids)


# ════════════════════════════════════════════════════════════════
# PagedAttention
# ════════════════════════════════════════════════════════════════

class PageBlock:
    def __init__(self, token_ids: list[int], kv_cache, block_hash: str):
        self.token_ids = token_ids
        self.kv_cache = kv_cache
        self.block_hash = block_hash
        self.last_access = time.time()


class PagedAttentionCache:
    """chain hashing 기반 block-level prefix caching. partial block은 공유 불가."""

    def __init__(self, block_size: int):
        self.block_size = block_size
        self.block_table: dict[str, PageBlock] = {}

    def _hash(self, token_ids: list[int], prev: str = "") -> str:
        return hashlib.md5((prev + str(token_ids)).encode()).hexdigest()

    def match_prefix(self, token_ids: list[int]):
        matched_len, best_kv, prev, pos = 0, None, "", 0
        while pos + self.block_size <= len(token_ids):
            bh = self._hash(token_ids[pos:pos + self.block_size], prev)
            if bh not in self.block_table:
                break
            blk = self.block_table[bh]
            blk.last_access = time.time()
            best_kv = blk.kv_cache
            matched_len += self.block_size
            prev = bh
            pos += self.block_size
        return matched_len, best_kv

    def insert(self, token_ids: list[int], full_kv):
        prev, pos = "", 0
        while pos + self.block_size <= len(token_ids):
            block_toks = token_ids[pos:pos + self.block_size]
            bh = self._hash(block_toks, prev)
            if bh not in self.block_table:
                end = pos + self.block_size
                self.block_table[bh] = PageBlock(block_toks, slice_kv(full_kv, end), bh)
            prev = bh
            pos += self.block_size

    def print_blocks(self, token_ids: list[int], tokenizer=None):
        prev, pos = "", 0
        while pos + self.block_size <= len(token_ids):
            block_toks = token_ids[pos:pos + self.block_size]
            bh = self._hash(block_toks, prev)
            end = pos + self.block_size
            label = repr(tokenizer.decode(block_toks)) if tokenizer else str(block_toks)
            cached = " [KV]" if bh in self.block_table else ""
            print(f"  Block [{pos:>2}~{end-1:>2}] {label}{cached}")
            prev = bh
            pos += self.block_size
        if pos < len(token_ids):
            partial = token_ids[pos:]
            label = repr(tokenizer.decode(partial)) if tokenizer else str(partial)
            print(f"  Block [{pos:>2}~{len(token_ids)-1:>2}] {label}  [partial — 캐시 불가]")


class PagedAttentionCacheLRU(PagedAttentionCache):
    """block budget 초과 시 LRU block을 evict."""

    def __init__(self, block_size: int, max_blocks: int):
        super().__init__(block_size)
        self.max_blocks = max_blocks

    def _evict_lru(self):
        if not self.block_table:
            return
        oldest = min(self.block_table.values(), key=lambda b: b.last_access)
        del self.block_table[oldest.block_hash]

    def insert(self, token_ids: list[int], full_kv):
        n_new = len(token_ids) // self.block_size
        while len(self.block_table) + n_new > self.max_blocks:
            self._evict_lru()
        super().insert(token_ids, full_kv)


# ════════════════════════════════════════════════════════════════
# 모델 로딩
# ════════════════════════════════════════════════════════════════

print(f"모델 로딩: {MODEL} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL)
model.eval()


# ════════════════════════════════════════════════════════════════
# 시나리오 A: RadixAttention 유리
#   공유 system prompt를 가진 요청들 + 비정렬 prefix
# ════════════════════════════════════════════════════════════════

sys_ids = tokenizer.encode(A_SYSTEM_PROMPT)
a_full_ids = [sys_ids + tokenizer.encode(q, add_special_tokens=False)
              for q in A_REQUESTS]
shared_prefix_len = len(sys_ids) + 4   # "Question: What is" 포함

print("\n" + "=" * 72)
print("[ 시나리오 A: RadixAttention 유리 — 공유 prefix, 비정렬 ]")
print("=" * 72)
print(f"  system prompt: {len(sys_ids)}토큰")
print(f"  공유 prefix:   {shared_prefix_len}토큰  (system + 'Question: What is')")
print(f"  block_size:    {A_BLOCK_SIZE}")
print(f"  RadixAttention → {shared_prefix_len}토큰 전체 재사용 가능")
print(f"  PagedAttention → {(shared_prefix_len // A_BLOCK_SIZE) * A_BLOCK_SIZE}토큰만 재사용 "
      f"({shared_prefix_len // A_BLOCK_SIZE} full blocks, "
      f"나머지 {shared_prefix_len % A_BLOCK_SIZE}토큰은 partial block 낭비)")

radix_a = RadixTree()
paged_a = PagedAttentionCache(block_size=A_BLOCK_SIZE)
a_results = []

for query, full_ids in zip(A_REQUESTS, a_full_ids):
    r_match, r_kv = radix_a.match_prefix(full_ids)
    p_match, p_kv = paged_a.match_prefix(full_ids)

    t0 = time.perf_counter()
    kv_r = run_prefill(model, full_ids[r_match:], past_kv=r_kv if r_match else None)
    run_decode(model, kv_r, N_ANSWER_TOKENS)
    t_r = time.perf_counter() - t0
    radix_a.insert(full_ids, kv_r)

    t0 = time.perf_counter()
    kv_p = run_prefill(model, full_ids[p_match:], past_kv=p_kv if p_match else None)
    run_decode(model, kv_p, N_ANSWER_TOKENS)
    t_p = time.perf_counter() - t0
    paged_a.insert(full_ids, kv_p)

    a_results.append({
        "query": query, "full_len": len(full_ids),
        "r_match": r_match, "r_compute": len(full_ids) - r_match, "r_time": t_r,
        "p_match": p_match, "p_compute": len(full_ids) - p_match, "p_time": t_p,
    })

# 요청별 결과
print(f"\n  {'req':>3}  {'query':38}  {'RadixAttention':>16}  {'PagedAttention':>16}")
print(f"  {'---':>3}  {'-----':38}  {'----------------':>16}  {'----------------':>16}")
for i, r in enumerate(a_results):
    r_info = f"{hit_status(r['r_match'], r['full_len'])} ({r['r_match']}tok)"
    p_info = f"{hit_status(r['p_match'], r['full_len'])} ({r['p_match']}tok)"
    print(f"  {i+1:>3}  {repr(r['query'][:36]):38}  {r_info:>16}  {p_info:>16}")

# 누적 비교
total_r = sum(r["r_compute"] for r in a_results)
total_p = sum(r["p_compute"] for r in a_results)
total   = sum(r["full_len"]  for r in a_results)
print(f"\n  {'':38}  {'RadixAttention':>16}  {'PagedAttention':>16}")
print(f"  {'총 계산 토큰':38}  {total_r:>16}  {total_p:>16}")
print(f"  {'token 절감률':38}  {(total-total_r)/total:>15.0%}  {(total-total_p)/total:>15.0%}")

# Radix Tree 구조
print(f"\n  [ Radix Tree ]")
radix_a.print_tree(tokenizer)

# Block Table
print(f"\n  [ Block Table (block_size={A_BLOCK_SIZE}) ]")
for query, full_ids in zip(A_REQUESTS, a_full_ids):
    print(f"  >> {repr(query[:40])}")
    paged_a.print_blocks(full_ids, tokenizer)
    print()


# ════════════════════════════════════════════════════════════════
# 시나리오 B: PagedAttention 유리
#   Wide & Shallow Tree + 제한된 메모리 budget
# ════════════════════════════════════════════════════════════════

b_ids = [tokenizer.encode(q) for q in B_REQUESTS]

print("=" * 72)
print("[ 시나리오 B: PagedAttention 유리 — Wide & Shallow Tree, Budget 제한 ]")
print("=" * 72)
print(f"  공유 prefix 없음 → Radix Tree가 루트에서 바로 분기 (wide & shallow)")
print(f"  block_size={B_BLOCK_SIZE}  /  budget={B_BUDGET_BLOCKS} blocks = {B_BUDGET_TOKENS}토큰\n")

for i, (q, ids) in enumerate(zip(B_REQUESTS, b_ids)):
    n_full = len(ids) // B_BLOCK_SIZE
    n_part = len(ids) % B_BLOCK_SIZE
    print(f"  req {i+1}: {len(ids):>2}토큰  ({n_full} full block + {n_part} partial)  {repr(q[:48])}")

seq_tok = len(b_ids[0])
print(f"\n  budget {B_BUDGET_TOKENS}토큰 기준:")
print(f"    RadixAttention → 시퀀스 전체({seq_tok}tok) 단위 eviction → {B_BUDGET_TOKENS // seq_tok}개 시퀀스 보존")
print(f"    PagedAttention → block({B_BLOCK_SIZE}tok) 단위 eviction → {B_BUDGET_BLOCKS}개 시퀀스의 첫 block 보존")

radix_b = RadixTreeLRU(max_tokens=B_BUDGET_TOKENS)
paged_b = PagedAttentionCacheLRU(block_size=B_BLOCK_SIZE, max_blocks=B_BUDGET_BLOCKS)

# 1차: cold start
print(f"\n  [ 1차 요청 — cold start ]")
print(f"  {'req':>4}  {'R cached_tokens':>16}  {'P cached_blocks':>16}  Radix Tree 상태")
print(f"  {'----':>4}  {'----------------':>16}  {'----------------':>16}  ---------------")

for i, (query, full_ids) in enumerate(zip(B_REQUESTS, b_ids)):
    kv = run_prefill(model, full_ids)
    run_decode(model, kv, N_ANSWER_TOKENS)
    radix_b.match_prefix(full_ids)
    radix_b.insert(full_ids, kv)

    kv_p = run_prefill(model, full_ids)
    run_decode(model, kv_p, N_ANSWER_TOKENS)
    paged_b.match_prefix(full_ids)
    paged_b.insert(full_ids, kv_p)

    leaves = radix_b._kv_leaves()
    leaf_info = f"leaf {len(leaves)}개 ({[len(n.token_ids) for n,_,_ in leaves]}tok)"
    print(f"  {i+1:>4}  {radix_b._cached_tokens:>16}  {len(paged_b.block_table):>16}  {leaf_info}")

print(f"\n  Radix Tree (1차 후):")
radix_b.print_tree(tokenizer)
print(f"  → width={len(radix_b.root.children)}  depth=1  (공유 prefix 없음)")

# 2차: replay
print(f"\n  [ 2차 요청 — 동일 요청 replay ]")
print(f"  {'req':>4}  {'RadixAttention':>20}  {'PagedAttention':>20}")
print(f"  {'----':>4}  {'--------------------':>20}  {'--------------------':>20}")

r_hits, r_saved, p_hits, p_saved = 0, 0, 0, 0
for i, (query, full_ids) in enumerate(zip(B_REQUESTS, b_ids)):
    r_match, _ = radix_b.match_prefix(full_ids)
    p_match, _ = paged_b.match_prefix(full_ids)
    if r_match > 0: r_hits += 1; r_saved += r_match
    if p_match > 0: p_hits += 1; p_saved += p_match
    r_s = f"HIT {r_match}tok" if r_match else "MISS"
    p_s = f"HIT {p_match}tok" if p_match else "MISS"
    print(f"  {i+1:>4}  {r_s:>20}  {p_s:>20}")

print(f"\n  {'':4}  {'RadixAttention':>20}  {'PagedAttention':>20}")
print(f"  {'HIT 요청수':4}  {r_hits:>20}  {p_hits:>20}  ← PagedAttention이 더 많은 요청 서빙")
print(f"  {'절감 토큰합':4}  {r_saved:>20}  {p_saved:>20}")
print(f"""
  이유: RadixAttention은 시퀀스 전체({seq_tok}tok)를 하나의 노드로 보존
        → budget({B_BUDGET_TOKENS}tok) / {seq_tok}tok = {B_BUDGET_TOKENS // seq_tok}개 시퀀스만 유지 가능
        PagedAttention은 {B_BLOCK_SIZE}tok block 단위로 세밀하게 보존
        → budget({B_BUDGET_BLOCKS} blocks)에 {B_BUDGET_BLOCKS}개 시퀀스의 첫 block 유지 가능""")


# ════════════════════════════════════════════════════════════════
# 최종 비교
# ════════════════════════════════════════════════════════════════

print("\n" + "=" * 72)
print("[ 최종 비교: 언제 어느 쪽이 유리한가 ]")
print("=" * 72)
print(f"""
  조건                                    RadixAttention     PagedAttention
  ─────────────────────────────────────   ──────────────     ──────────────
  공유 prefix 많음, 비정렬 (시나리오 A)   ✅ 유리             ❌ partial 낭비
  Wide & Shallow + budget 빡빡 (시나리오B) ❌ 굵은 eviction   ✅ 세밀한 eviction
  GPU 메모리 단편화 방지                   ❌ 가변 노드 크기   ✅ 고정 block 크기
  긴 system prompt / RAG context           ✅ 크게 유리        △ block 정렬 필요
  구현 복잡도                              ❌ 높음 (트리)      ✅ 낮음 (해시)
""")
