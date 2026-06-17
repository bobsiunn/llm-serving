# LLM Serving 실습

KV cache 동작 원리를 GPT-2로 직접 확인하는 실습 모음입니다.

## 환경 설정

```bash
# 최초 1회만 실행
~/.local/bin/virtualenv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install transformers
```

---

## 실습 1 — KV Cache Reuse 기본 (`kv_cache_demo.py`)

**무엇을 보여주나**
- Prefill: 프롬프트 전체를 한 번에 처리하고 K/V 텐서를 cache에 저장
- Decode with cache: 새 토큰 1개의 Q만 계산 → 저장된 K/V와 attention (재계산 없음)
- Decode without cache: 매 step마다 전체 시퀀스를 처음부터 재계산

```bash
.venv/bin/python kv_cache_demo.py
```

**주요 출력**
```
[Layer 0] K shape: (1, 12, 9, 64)   # (batch, heads, seq_len, head_dim)
→ seq_len=9 == 프롬프트 토큰 수(9)

step  새 토큰   cache seq_len   decode 시간
   1  ' is'             10        87.9 ms   ← cache에 토큰 1개 추가
   2  ' focused'        11        96.2 ms
   ...

decode 평균 (cache=True) :   97.9 ms/token
decode 평균 (cache=False):  151.8 ms/token
속도 향상: 1.6x
```

---

## 실습 2 — Prefix Sharing (`prefix_sharing_demo.py`)

**무엇을 보여주나**

SGLang RadixAttention의 핵심 아이디어: 동일한 system prompt(prefix)를 공유하는 두 요청이 있을 때, prefix의 KV를 한 번만 계산하고 재사용한다.

```
[방법 1] 공유 없음   Request A: prefix + query_A 계산
                    Request B: prefix + query_B 계산  ← prefix를 두 번 계산
                    
[방법 2] 공유 있음   prefix KV 계산 (1회)
                    Request A: query_A만 추가 처리 + prefix KV 재사용
                    Request B: query_B만 추가 처리 + prefix KV 재사용
```

```bash
.venv/bin/python prefix_sharing_demo.py
```

**언제 효과가 큰가**
- Prefix가 길수록 (system prompt가 500~2000 토큰인 경우)
- 동시 요청 수가 많을수록
- 실제 SGLang은 Radix Tree로 prefix를 관리해 자동으로 재사용

---

## 실습 3 — Multi-turn Cache 누적 (`multiturn_cache_demo.py`)

**무엇을 보여주나**

대화가 쌓일수록 KV cache가 어떻게 누적되는지, 이전 턴을 재계산하지 않고 재사용하는지 확인한다.

```bash
.venv/bin/python multiturn_cache_demo.py
```

**주요 출력**
```
[ cache=True ]  KV cache 누적
턴   입력 토큰   cache seq_len   KV 메모리    시간
 1        12             18    1296 KB    922 ms
 2        13             37    2664 KB    781 ms   ← 이전 18토큰 KV 재사용
 3        12             55    3960 KB    618 ms   ← 이전 37토큰 KV 재사용
 4        11             72    5184 KB    701 ms   ← 이전 55토큰 KV 재사용

[ cache=False ]  매 턴 전체 재계산
턴   전체 seq_len   시간
 1            17    814 ms
 2            36   1492 ms   ← 점점 느려짐
 3            54   1792 ms
 4            71   2001 ms
```

---

---

## 실습 4 — RadixAttention 시뮬레이션 (`radix_attention_demo.py`)

**무엇을 보여주나**

SGLang RadixCache의 핵심 로직(Radix Tree + prefix matching)을 직접 구현하고 GPT-2 추론과 연동한다.

- `match_prefix`: 새 요청의 token_ids와 가장 긴 공통 prefix를 트리에서 탐색
- `insert`: 추론 완료 후 KV를 트리에 저장, 공통 prefix는 자동으로 노드 분기(split)
- cache hit 시 매칭된 prefix 이후 토큰만 prefill → 연산 절감

```bash
.venv/bin/pip install sglang --no-deps   # 최초 1회 (radix_cache 소스 참조용)
.venv/bin/python radix_attention_demo.py
```

**주요 출력**
```
[ RadixAttention 서빙 시뮬레이션 ]
  요청      상태   hit len   처리 토큰       시간   query
    1       MISS         0        25    1064 ms  'Question: What is machine learning?'
    2        HIT        22         3     701 ms  'Question: What is deep learning?'   ← 22토큰 재사용
    3        HIT        22         3     707 ms  'Question: What is neural network?'  ← 22토큰 재사용
    4       MISS        18        29     692 ms  'Completely different topic...'      ← system prompt만 공유

[ Radix Tree 구조 ]
  root
  └─ "You are a helpful AI..." (18 tokens)   ← system prompt 공유
     ├─ "Question: What is" (4 tokens)        ← 자동 분기된 공유 prefix
     │  ├─ " machine learning?" [KV cached]
     │  ├─ " deep learning?"    [KV cached]
     │  └─ " neural network?"   [KV cached]
     └─ "Completely different..." [KV cached]

[ 동일 요청 재실행 ]
    1   FULL HIT    25    0 토큰    0.0 ms   ← prefill 없이 즉시 반환
```

**SGLang 실제 구현과의 차이**
- 실제 SGLang: GPU KV 메모리를 페이지 단위로 관리 (PagedAttention과 유사)
- 이 데모: KV 객체를 그대로 Python 메모리에 저장 (개념 동일, 메모리 관리 생략)

---

## 실습 5 — RadixAttention vs PagedAttention 비교 (`radix_vs_paged_demo.py`)

**무엇을 보여주나**

동일한 4개 요청을 SGLang(RadixAttention)과 vLLM(PagedAttention) 방식으로 각각 처리하고 결과를 나란히 비교한다.

| | RadixAttention | PagedAttention |
|--|--|--|
| 매칭 단위 | 토큰 1개 | `BLOCK_SIZE`(=8) 토큰 |
| 22토큰 공유 prefix | 22토큰 전부 재사용 | 16토큰만 재사용 |
| partial block | 없음 (토큰 단위) | 공유 불가 |

```bash
.venv/bin/python radix_vs_paged_demo.py
```

**주요 출력**
```
                        RadixAttention  PagedAttention
총 계산 토큰                          42              56
절감 토큰 (재사용)                      62              48
token 절감률                         60%            46%

[ PagedAttention — Block Table (block_size=8) ]
  Block [ 0~ 7] 'You are a helpful AI assistant. Answer'        [KV]
  Block [ 8~15] ' questions clearly and concisely. Always be'   [KV]
  Block [16~23] ' polite.Question: What is machine learning'    [KV]  ← req마다 다름!
  Block [24~24] '?'                                             [partial — 캐시 불가]
```

**BLOCK_SIZE 조정해보기** — 파일 상단 `BLOCK_SIZE = 8`을 바꾸면 block 낭비가 얼마나 달라지는지 확인 가능

---

## 개념 요약

```
Prefill    프롬프트 토큰 전체의 K/V를 한 번에 계산 → cache 저장
           attention 연산: O(seq_len²)

Decode     새 토큰 1개의 Q만 생성 → 저장된 K/V와 attention
(cache=True)  연산량: O(1 × cached_seq_len)  → 빠름

Decode     매 step 전체 시퀀스를 처음부터 재계산
(cache=False) 연산량: O(seq_len²)  → step마다 느려짐

Prefix     동일 prefix의 KV를 여러 요청이 공유
Sharing    SGLang RadixAttention, vLLM prefix caching의 핵심
```

## 모델 정보

| 항목 | 값 |
|------|-----|
| 모델 | GPT-2 (137M) |
| 레이어 수 | 12 |
| Attention heads | 12 |
| Head dimension | 64 |
| 아키텍처 | Decoder-only |
