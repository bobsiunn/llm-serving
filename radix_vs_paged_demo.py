"""
RadixAttention (SGLang) vs PagedAttention (vLLM) 비교

두 가지 시나리오로 각각이 유리한 상황을 실측한다.
block_size는 두 시나리오 공통으로 8을 사용한다.

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
BLOCK_SIZE = 8   # 두 시나리오 공통

# ── 시나리오 A 파라미터 ────────────────────────────────────────
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
B_BUDGET_BLOCKS = 2
B_BUDGET_TOKENS = B_BUDGET_BLOCKS * BLOCK_SIZE   # = 16

# 공유 prefix 없음, 각 8~9 토큰 (1 full block + partial)
# budget=16tok: RadixAttention은 1개 시퀀스만 보존, PagedAttention은 2개 block 보존
B_REQUESTS = [
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
        self.eviction_log: list[list[int]] = []

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
        self.eviction_log.append(list(node.token_ids))
        node.kv_cache = None
        del parent.children[key]
        self._cached_tokens -= freed
        return freed

    def insert(self, token_ids: list[int], kv_cache):
        # 이미 트리에 있는 경로는 새 토큰으로 세지 않음
        node, pos, n_new = self.root, 0, 0
        while pos < len(token_ids):
            tok = token_ids[pos]
            if tok not in node.children:
                n_new += len(token_ids) - pos
                break
            child = node.children[tok]
            common = 0
            for ct in child.token_ids:
                if pos + common >= len(token_ids) or token_ids[pos + common] != ct:
                    break
                common += 1
            if common < len(child.token_ids):
                n_new += len(token_ids) - (pos + common)
                break
            pos += common
            node = child

        while self._cached_tokens + n_new > self.max_tokens:
            if self._evict_lru() == 0:
                break
        super().insert(token_ids, kv_cache)
        self._cached_tokens += n_new

    def pop_evictions(self) -> list[list[int]]:
        log = self.eviction_log[:]
        self.eviction_log.clear()
        return log


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
        self.eviction_log: list[list[int]] = []

    def _evict_lru(self):
        if not self.block_table:
            return
        oldest = min(self.block_table.values(), key=lambda b: b.last_access)
        self.eviction_log.append(list(oldest.token_ids))
        del self.block_table[oldest.block_hash]

    def insert(self, token_ids: list[int], full_kv):
        # 이미 캐시에 있는 block은 새 block으로 세지 않음
        prev, pos, n_new = "", 0, 0
        while pos + self.block_size <= len(token_ids):
            bh = self._hash(token_ids[pos:pos + self.block_size], prev)
            if bh not in self.block_table:
                n_new += 1
            prev = bh
            pos += self.block_size
        while len(self.block_table) + n_new > self.max_blocks:
            self._evict_lru()
        super().insert(token_ids, full_kv)

    def pop_evictions(self) -> list[list[int]]:
        log = self.eviction_log[:]
        self.eviction_log.clear()
        return log


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
print(f"[ 시나리오 A: RadixAttention 유리 — 공유 prefix, 비정렬 (block_size={BLOCK_SIZE}) ]")
print("=" * 72)
print(f"  system prompt: {len(sys_ids)}토큰")
print(f"  공유 prefix:   {shared_prefix_len}토큰  (system + 'Question: What is')")
print(f"  block_size:    {BLOCK_SIZE}")
print(f"  RadixAttention → {shared_prefix_len}토큰 전체 재사용 가능")
print(f"  PagedAttention → {(shared_prefix_len // BLOCK_SIZE) * BLOCK_SIZE}토큰만 재사용 "
      f"({shared_prefix_len // BLOCK_SIZE} full blocks, "
      f"나머지 {shared_prefix_len % BLOCK_SIZE}토큰은 partial block 낭비)")

radix_a = RadixTree()
paged_a = PagedAttentionCache(block_size=BLOCK_SIZE)
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
print(f"\n  [ Block Table (block_size={BLOCK_SIZE}) ]")
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
print(f"[ 시나리오 B: PagedAttention 유리 — Wide & Shallow Tree, Budget 제한 (block_size={BLOCK_SIZE}) ]")
print("=" * 72)
print(f"  공유 prefix 없음 → Radix Tree가 루트에서 바로 분기 (wide & shallow)")
print(f"  block_size={BLOCK_SIZE}  /  budget={B_BUDGET_BLOCKS} blocks = {B_BUDGET_TOKENS}토큰\n")

for i, (q, ids) in enumerate(zip(B_REQUESTS, b_ids)):
    n_full = len(ids) // BLOCK_SIZE
    n_part = len(ids) % BLOCK_SIZE
    print(f"  req {i+1}: {len(ids):>2}토큰  ({n_full} full block + {n_part} partial)  '{q}'")

tok1, tok2 = len(b_ids[0]), len(b_ids[1])
print(f"\n  budget {B_BUDGET_TOKENS}토큰 기준:")
print(f"    RadixAttention → 노드(시퀀스 전체) 단위 eviction")
print(f"                     req1({tok1}tok) + req2({tok2}tok) = {tok1+tok2}tok > {B_BUDGET_TOKENS}  → 동시 보존 불가, 1개만 유지")
print(f"    PagedAttention → block({BLOCK_SIZE}tok) 단위 eviction")
print(f"                     req1(1 block) + req2(1 block) = 2 blocks ≤ {B_BUDGET_BLOCKS}  → 둘 다 보존 가능")

radix_b = RadixTreeLRU(max_tokens=B_BUDGET_TOKENS)
paged_b = PagedAttentionCacheLRU(block_size=BLOCK_SIZE, max_blocks=B_BUDGET_BLOCKS)

def r_cache_state(tree: RadixTreeLRU) -> str:
    leaves = tree._kv_leaves()
    if not leaves:
        return "(empty)"
    parts = [f"'{tokenizer.decode(n.token_ids[:4])}...'({len(n.token_ids)}tok)"
             for n, _, _ in leaves]
    return f"{tree._cached_tokens}tok  [{', '.join(parts)}]"

def p_cache_state(cache: PagedAttentionCacheLRU) -> str:
    if not cache.block_table:
        return "(empty)"
    parts = [f"'{tokenizer.decode(b.token_ids[:4])}...'[{BLOCK_SIZE}tok]"
             for b in sorted(cache.block_table.values(), key=lambda b: b.last_access)]
    return f"{len(cache.block_table)} blocks  [{', '.join(parts)}]"

# ── 1차: cold-start (모두 MISS → compute → insert) ────────────
print(f"\n  [ 1차 요청 — cold start: MISS → compute → cache에 insert ]")
print(f"  (캐시가 비어있으므로 모든 요청이 MISS. 하지만 compute 후 cache에 저장됨)\n")

for i, (query, full_ids) in enumerate(zip(B_REQUESTS, b_ids)):
    n_tok = len(full_ids)
    n_blocks = n_tok // BLOCK_SIZE
    print(f"  ▶ req {i+1} (MISS): '{query[:48]}' ({n_tok}tok)")

    # RadixAttention
    r_before = radix_b._cached_tokens
    kv = run_prefill(model, full_ids)
    run_decode(model, kv, N_ANSWER_TOKENS)
    radix_b.insert(full_ids, kv)
    r_evictions = radix_b.pop_evictions()

    if r_before + n_tok > B_BUDGET_TOKENS:
        print(f"    RadixAttention:  budget 초과 ({r_before}+{n_tok}={r_before+n_tok} > {B_BUDGET_TOKENS})")
        for ev in r_evictions:
            label = tokenizer.decode(ev[:5])
            print(f"                     EVICT '{label}...' ({len(ev)}tok)  ← LRU 노드 통째로 제거")
    else:
        print(f"    RadixAttention:  budget 여유 ({r_before}+{n_tok}={r_before+n_tok} ≤ {B_BUDGET_TOKENS})")
    print(f"                     → insert req{i+1} 완료  |  cache: {r_cache_state(radix_b)}")

    # PagedAttention
    p_before = len(paged_b.block_table)
    kv_p = run_prefill(model, full_ids)
    run_decode(model, kv_p, N_ANSWER_TOKENS)
    paged_b.insert(full_ids, kv_p)
    p_evictions = paged_b.pop_evictions()

    partial = n_tok % BLOCK_SIZE
    if p_before + n_blocks > B_BUDGET_BLOCKS:
        print(f"    PagedAttention:  budget 초과 ({p_before}+{n_blocks}={p_before+n_blocks} > {B_BUDGET_BLOCKS})")
        for ev in p_evictions:
            label = tokenizer.decode(ev[:4])
            print(f"                     EVICT '{label}...'[{BLOCK_SIZE}tok] block  ← LRU block 제거")
    else:
        print(f"    PagedAttention:  budget 여유 ({p_before}+{n_blocks}={p_before+n_blocks} ≤ {B_BUDGET_BLOCKS})")
    if partial > 0:
        print(f"                     partial {partial}tok → 캐시 불가 (block 미달, 버림)")
    print(f"                     → insert req{i+1} block 완료  |  blocks: {p_cache_state(paged_b)}")
    print()

# 최종 cache 상태 요약
print(f"  ── 1차 완료 후 최종 cache 상태 ──")
print(f"  RadixAttention: {r_cache_state(radix_b)}")
print(f"  PagedAttention: {p_cache_state(paged_b)}")
print(f"")
print(f"  RadixAttention: req1({tok1}tok)+req2({tok2}tok)={tok1+tok2}tok > budget({B_BUDGET_TOKENS}tok)")
print(f"                  → req2 삽입 시 req1 evict → req2만 생존")
print(f"  PagedAttention: req1(1 block)+req2(1 block)=2 blocks = budget({B_BUDGET_BLOCKS} blocks)")
print(f"                  → req1·req2 block 둘 다 생존")
print(f"")
print(f"  ※ 1차 요청은 모두 MISS였지만 compute 후 cache에 삽입됨")
print(f"     eviction 정책 차이로 살아남은 데이터가 다름 → 2차 HIT 수 차이")

# ── 2차: 동일 요청 재실행 (현실적: MISS도 compute+insert) ──────
print(f"\n  [ 2차 요청 — 동일한 req1·req2 재도착 ]")
print(f"  (MISS 시에도 compute + insert 수행 — 실제 서빙과 동일)\n")

r_hits = p_hits = r_saved = p_saved = 0

for i, (query, full_ids) in enumerate(zip(B_REQUESTS, b_ids)):
    n_tok    = len(full_ids)
    n_blocks = n_tok // BLOCK_SIZE
    print(f"  ▶ req{i+1} (2차): '{query}' ({n_tok}tok)")

    # ── RadixAttention ──
    r_before       = radix_b._cached_tokens
    r_match, r_kv  = radix_b.match_prefix(full_ids)

    if r_match > 0:
        r_hits += 1; r_saved += r_match
        remaining = full_ids[r_match:]
        kv_r = run_prefill(model, remaining, past_kv=r_kv) if remaining else r_kv
        print(f"    RadixAttention: HIT {r_match}tok  → {len(remaining)}tok만 compute → insert")
    else:
        print(f"    RadixAttention: MISS → {n_tok}tok 전체 compute → insert")
        kv_r = run_prefill(model, full_ids)
        if r_before + n_tok > B_BUDGET_TOKENS:
            print(f"                   budget 초과 ({r_before}+{n_tok}={r_before+n_tok} > {B_BUDGET_TOKENS})")
    radix_b.insert(full_ids, kv_r)
    for ev in radix_b.pop_evictions():
        print(f"                   EVICT '{tokenizer.decode(ev[:5])}...' ({len(ev)}tok)")
    run_decode(model, kv_r, N_ANSWER_TOKENS)
    print(f"                   cache: {r_cache_state(radix_b)}")

    # ── PagedAttention ──
    p_before       = len(paged_b.block_table)
    p_match, p_kv  = paged_b.match_prefix(full_ids)

    if p_match > 0:
        p_hits += 1; p_saved += p_match
        remaining = full_ids[p_match:]
        kv_p = run_prefill(model, remaining, past_kv=p_kv) if remaining else p_kv
        print(f"    PagedAttention: HIT {p_match}tok  → {len(remaining)}tok만 compute → insert (new blocks only)")
    else:
        print(f"    PagedAttention: MISS → {n_tok}tok 전체 compute → insert")
        kv_p = run_prefill(model, full_ids)
        if p_before + n_blocks > B_BUDGET_BLOCKS:
            print(f"                   budget 초과 ({p_before}+{n_blocks}={p_before+n_blocks} > {B_BUDGET_BLOCKS})")
    paged_b.insert(full_ids, kv_p)
    for ev in paged_b.pop_evictions():
        print(f"                   EVICT '{tokenizer.decode(ev[:4])}...'[{BLOCK_SIZE}tok] block")
    if n_tok % BLOCK_SIZE:
        print(f"                   partial {n_tok % BLOCK_SIZE}tok → 버림")
    run_decode(model, kv_p, N_ANSWER_TOKENS)
    print(f"                   blocks: {p_cache_state(paged_b)}")
    print()

print(f"  {'':6}  {'RadixAttention':>18}  {'PagedAttention':>18}")
print(f"  {'HIT 수':6}  {r_hits:>18}  {p_hits:>18}  ← PagedAttention이 더 많은 요청 서빙")
print(f"  {'절감 tok':6}  {r_saved:>18}  {p_saved:>18}")


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
