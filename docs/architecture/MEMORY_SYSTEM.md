# AURA v3 Memory System Architecture

> **Version**: 3.0.0  
> **Last Updated**: February 2026  
> **Status**: Production

## Overview

AURA v3 implements a **biologically-inspired multi-tier memory architecture** designed for offline-first operation. The system mimics human cognitive memory processes, featuring working memory (neural networks), episodic memory (hippocampus), semantic memory (neocortex), and long-term archive (ancestor memory).

### Design Principles

1. **Biological Plausibility** - Memory tiers mirror human cognitive architecture
2. **Offline-First** - All operations work without network connectivity
3. **Resource Efficiency** - Strict memory budgets and lazy loading
4. **Graceful Degradation** - System remains functional under resource pressure
5. **Privacy by Design** - Local storage with optional encryption

---

## Multi-Tier Memory Hierarchy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MEMORY SYSTEM ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     WORKING MEMORY (Neural)                         │   │
│  │                     Capacity: 7±2 items                             │   │
│  │                     Retention: Seconds to minutes                   │   │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐   │   │
│  │  │ Neuron  │──│ Neuron  │──│ Neuron  │──│ Neuron  │──│ Neuron  │   │   │
│  │  │ A=0.8   │  │ A=0.6   │  │ A=0.9   │  │ A=0.4   │  │ A=0.7   │   │   │
│  │  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘   │   │
│  │       │ Synapses   │           │            │            │         │   │
│  │       └────────────┴───────────┴────────────┴────────────┘         │   │
│  └─────────────────────────────────┬───────────────────────────────────┘   │
│                                    │ Consolidation (every 5 min)           │
│                                    ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   EPISODIC MEMORY (Hippocampus CA3)                 │   │
│  │                   Fast Encoding: <100ms                             │   │
│  │                   Retention: Hours to days                          │   │
│  │                                                                     │   │
│  │   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │   │
│  │   │   Episode    │    │   Episode    │    │   Episode    │         │   │
│  │   │ "User asked  │    │ "Opened app  │    │ "Error in    │         │   │
│  │   │  about X"    │    │  settings"   │    │  module Y"   │         │   │
│  │   │ emotion: 0.3 │    │ emotion: 0.1 │    │ emotion: 0.7 │         │   │
│  │   └──────────────┘    └──────────────┘    └──────────────┘         │   │
│  └─────────────────────────────────┬───────────────────────────────────┘   │
│                                    │ Slow Consolidation                    │
│                                    ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   SEMANTIC MEMORY (Neocortex)                       │   │
│  │                   Stable Knowledge Storage                          │   │
│  │                   Retention: Days to permanent                      │   │
│  │                                                                     │   │
│  │   ┌────────────┐   ┌────────────┐   ┌────────────┐                 │   │
│  │   │   FACTS    │   │  CONCEPTS  │   │PREFERENCES │                 │   │
│  │   │ "X is Y"   │   │ Categories │   │ User likes │                 │   │
│  │   │ conf: 0.9  │   │ Relations  │   │ dark mode  │                 │   │
│  │   └────────────┘   └────────────┘   └────────────┘                 │   │
│  └─────────────────────────────────┬───────────────────────────────────┘   │
│                                    │ Archive (low importance/old)          │
│                                    ▼                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   ANCESTOR MEMORY (Long-term Archive)               │   │
│  │                   Compressed Storage (zlib)                         │   │
│  │                   Optional Encryption (Fernet)                      │   │
│  │                   Retention: Permanent (with pruning)               │   │
│  │                                                                     │   │
│  │   ┌─────────────────────────────────────────────────────────────┐   │   │
│  │   │  Archived memories with summaries, tags, compression       │   │   │
│  │   │  Lazy loading - only accessed when needed                  │   │   │
│  │   └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow

### Input → Encoding → Storage → Retrieval Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            DATA FLOW PIPELINE                               │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────┐
                              │   INPUT     │
                              │  (Event/    │
                              │   Query)    │
                              └──────┬──────┘
                                     │
                                     ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                           ENCODING PHASE                                    │
│  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐          │
│  │ Text Embedding   │  │ Importance       │  │ Emotional        │          │
│  │ (Trigram 64-dim) │  │ Scoring          │  │ Valence          │          │
│  │                  │  │ (Amygdala)       │  │ Detection        │          │
│  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘          │
│           │                     │                     │                     │
│           └─────────────────────┴─────────────────────┘                     │
│                                 │                                           │
└─────────────────────────────────┼───────────────────────────────────────────┘
                                  │
                                  ▼
┌────────────────────────────────────────────────────────────────────────────┐
│                           STORAGE PHASE                                     │
│                                                                             │
│  ┌─────────────────┐    encode()     ┌─────────────────┐                   │
│  │  Working Memory │ ──────────────► │ Episodic Memory │                   │
│  │  (Neurons)      │                 │ (SQLite)        │                   │
│  └─────────────────┘                 └────────┬────────┘                   │
│                                               │                             │
│                                   consolidate_to_semantic()                 │
│                                               │                             │
│                                               ▼                             │
│                                      ┌─────────────────┐                   │
│                                      │ Semantic Memory │                   │
│                                      │ (Facts/Concepts)│                   │
│                                      └────────┬────────┘                   │
│                                               │                             │
│                                        archive()                            │
│                                               │                             │
│                                               ▼                             │
│                                      ┌─────────────────┐                   │
│                                      │ Ancestor Memory │                   │
│                                      │ (Compressed)    │                   │
│                                      └─────────────────┘                   │
└────────────────────────────────────────────────────────────────────────────┘

                                  │
                                  ▼ Query
┌────────────────────────────────────────────────────────────────────────────┐
│                          RETRIEVAL PHASE                                    │
│                    (UnifiedMemoryRetrieval - Phased Search)                │
│                                                                             │
│   Phase 1          Phase 2          Phase 3          Phase 4               │
│  ┌────────┐       ┌────────┐       ┌────────┐       ┌────────┐            │
│  │Working │ ───►  │Episodic│ ───►  │Semantic│ ───►  │Ancestor│            │
│  │Memory  │       │Memory  │       │Memory  │       │Memory  │            │
│  └────────┘       └────────┘       └────────┘       └────────┘            │
│      │                │                │                │                  │
│      └────────────────┴────────────────┴────────────────┘                  │
│                              │                                              │
│                              ▼                                              │
│                    ┌─────────────────┐                                     │
│                    │  Deduplicate &  │                                     │
│                    │  Rank by Score  │                                     │
│                    └────────┬────────┘                                     │
│                             │                                               │
│                             ▼                                               │
│                    ┌─────────────────┐                                     │
│                    │    RESULTS      │                                     │
│                    └─────────────────┘                                     │
└────────────────────────────────────────────────────────────────────────────┘
```

---

## Consolidation Pipeline

### Working → Episodic → Semantic → Archive

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       CONSOLIDATION PIPELINE                                │
│                    (MemoryCoordinator - Every 5 minutes)                    │
└─────────────────────────────────────────────────────────────────────────────┘

┌───────────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Working → Episodic                                               │
│ Trigger: Automatic (background task)                                      │
│ Interval: 300 seconds (5 minutes)                                         │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   Working Memory                         Episodic Memory                  │
│   ┌─────────────────┐                   ┌─────────────────┐              │
│   │ Active Neurons  │                   │                 │              │
│   │ (importance>0.3)│ ═══════════════►  │ episodic_traces │              │
│   │                 │   encode()        │ table           │              │
│   └─────────────────┘                   └─────────────────┘              │
│                                                                           │
│   Selection Criteria:                                                     │
│   • importance >= 0.3                                                     │
│   • Not already consolidated                                              │
│   • Has meaningful content                                                │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Episodic → Semantic                                              │
│ Trigger: Threshold-based (>100 unconsolidated traces)                     │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   Episodic Memory                        Semantic Memory                  │
│   ┌─────────────────┐                   ┌─────────────────┐              │
│   │ episodic_traces │                   │ facts           │              │
│   │ (consolidated=0)│ ═══════════════►  │ concepts        │              │
│   │                 │ extract_knowledge │ preferences     │              │
│   └─────────────────┘                   └─────────────────┘              │
│                                                                           │
│   Knowledge Extraction:                                                   │
│   • Pattern detection across episodes                                     │
│   • Fact extraction (entity relationships)                                │
│   • Concept formation (category abstraction)                              │
│   • Preference inference (behavioral patterns)                            │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Semantic → Ancestor (Archive)                                    │
│ Trigger: Age + Low Importance                                             │
├───────────────────────────────────────────────────────────────────────────┤
│                                                                           │
│   Semantic Memory                        Ancestor Memory                  │
│   ┌─────────────────┐                   ┌─────────────────┐              │
│   │ Old facts       │                   │ archived_       │              │
│   │ (age > 30 days) │ ═══════════════►  │ memories        │              │
│   │ (importance<0.3)│   archive()       │ (compressed)    │              │
│   └─────────────────┘                   └─────────────────┘              │
│                                                                           │
│   Archive Process:                                                        │
│   • Generate summary                                                      │
│   • Extract tags                                                          │
│   • Compress with zlib                                                    │
│   • Optional Fernet encryption                                            │
│   • Store with metadata                                                   │
│                                                                           │
└───────────────────────────────────────────────────────────────────────────┘
```

### Consolidation Configuration

```python
# Default consolidation settings
CONSOLIDATION_CONFIG = {
    "interval_seconds": 300,           # 5 minutes
    "working_to_episodic": {
        "min_importance": 0.3,
        "batch_size": 50
    },
    "episodic_to_semantic": {
        "trigger_threshold": 100,      # Unconsolidated traces
        "min_confidence": 0.5
    },
    "semantic_to_ancestor": {
        "age_days": 30,
        "max_importance": 0.3,
        "compression_level": 6         # zlib level
    }
}
```

---

## Importance Scoring Algorithm

### Amygdala-Inspired Scoring System

The `ImportanceScorer` implements a weighted multi-factor algorithm inspired by the amygdala's role in emotional memory processing.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      IMPORTANCE SCORING ALGORITHM                           │
│                         (Amygdala Analogue)                                 │
└─────────────────────────────────────────────────────────────────────────────┘

                         ┌─────────────────┐
                         │  Input Memory   │
                         │    Content      │
                         └────────┬────────┘
                                  │
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
           ▼                      ▼                      ▼
   ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
   │   EMOTIONAL   │     │    OUTCOME    │     │    NOVELTY    │
   │    WEIGHT     │     │    WEIGHT     │     │    WEIGHT     │
   │     30%       │     │     25%       │     │     20%       │
   └───────┬───────┘     └───────┬───────┘     └───────┬───────┘
           │                      │                      │
           │                      │                      │
           ▼                      ▼                      ▼
   ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
   │ Sentiment     │     │ Task success/ │     │ Uniqueness    │
   │ intensity,    │     │ failure,      │     │ compared to   │
   │ keywords      │     │ user feedback │     │ existing      │
   │ (error, love) │     │               │     │ memories      │
   └───────────────┘     └───────────────┘     └───────────────┘
           │                      │                      │
           │                      │                      │
           ▼                      ▼                      ▼
   ┌───────────────┐     ┌───────────────┐     ┌───────────────┐
   │  RELEVANCE    │     │  REPETITION   │     │               │
   │    WEIGHT     │     │    WEIGHT     │     │               │
   │     15%       │     │     10%       │     │               │
   └───────┬───────┘     └───────┬───────┘     │               │
           │                      │             │               │
           ▼                      ▼             │               │
   ┌───────────────┐     ┌───────────────┐     │               │
   │ Context match │     │ Frequency of  │     │               │
   │ current task  │     │ similar       │     │               │
   │               │     │ memories      │     │               │
   └───────────────┘     └───────────────┘     │               │
           │                      │             │               │
           └──────────────────────┴─────────────┴───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │    WEIGHTED SUM         │
                    │                         │
                    │ score = Σ(weight × val) │
                    │                         │
                    │ Range: 0.0 - 1.0        │
                    └─────────────────────────┘
```

### Scoring Formula

```python
def calculate_importance(memory: Memory, context: Context) -> float:
    """
    Calculate importance score using weighted factors.
    
    Formula:
    importance = (emotional × 0.30) + (outcome × 0.25) + (novelty × 0.20) 
                + (relevance × 0.15) + (repetition × 0.10)
    """
    
    WEIGHTS = {
        "emotional": 0.30,    # Emotional intensity
        "outcome": 0.25,      # Task success/failure impact
        "novelty": 0.20,      # Uniqueness vs existing memories
        "relevance": 0.15,    # Match to current context
        "repetition": 0.10    # Frequency of similar content
    }
    
    scores = {
        "emotional": analyze_emotional_intensity(memory.content),
        "outcome": evaluate_outcome_significance(memory),
        "novelty": calculate_novelty_score(memory, existing_memories),
        "relevance": compute_context_relevance(memory, context),
        "repetition": get_repetition_factor(memory)
    }
    
    return sum(WEIGHTS[k] * scores[k] for k in WEIGHTS)
```

### Emotional Keywords Detection

```python
EMOTIONAL_KEYWORDS = {
    "high_positive": ["love", "amazing", "excellent", "perfect"],
    "high_negative": ["error", "critical", "failed", "crash", "hate"],
    "medium_positive": ["good", "nice", "helpful", "thanks"],
    "medium_negative": ["problem", "issue", "wrong", "bug"]
}

# Keyword presence boosts emotional score by 0.1-0.3
```

---

## Vector Store Implementation

### Local Vector Store (Zvec-Style)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     LOCAL VECTOR STORE ARCHITECTURE                         │
│                          (Zvec-Style, Offline)                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                           EMBEDDING LAYER                                   │
│                                                                             │
│   Input Text ──► Character Trigram ──► Hash ──► 128-dim Vector             │
│                                                                             │
│   "hello" → ["hel", "ell", "llo"] → [hash1, hash2, hash3] → normalize      │
│                                                                             │
│   Properties:                                                               │
│   • Dimension: 128 (configurable)                                           │
│   • Method: Character trigrams with hashing                                 │
│   • Normalization: L2 normalized                                            │
│   • Offline: No API calls required                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           STORAGE LAYER                                     │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                        SQLite Database                              │  │
│   │                                                                     │  │
│   │   vectors table:                                                    │  │
│   │   ┌────────┬──────────┬────────────┬──────────┬────────────────┐   │  │
│   │   │   id   │  content │   vector   │ metadata │ importance     │   │  │
│   │   │  TEXT  │   TEXT   │    BLOB    │   JSON   │   REAL         │   │  │
│   │   │  PK    │          │  128×f32   │          │                │   │  │
│   │   └────────┴──────────┴────────────┴──────────┴────────────────┘   │  │
│   │                                                                     │  │
│   │   Indexes:                                                          │  │
│   │   • idx_vectors_importance (importance DESC)                        │  │
│   │   • idx_vectors_access (access_count DESC)                          │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Limits:                                                                   │
│   • Max vectors: 10,000 (configurable)                                      │
│   • Pruning: Removes lowest importance when limit reached                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SEARCH LAYER                                      │
│                                                                             │
│   Query: "user preference" ──► Embed ──► Cosine Similarity Search          │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                     Search Algorithm                                │  │
│   │                                                                     │  │
│   │   1. Embed query text → query_vector (128-dim)                      │  │
│   │   2. Load all vectors from SQLite                                   │  │
│   │   3. Compute cosine similarity: sim = dot(q, v) / (||q|| × ||v||)   │  │
│   │   4. Filter by threshold (default: 0.5)                             │  │
│   │   5. Sort by similarity descending                                  │  │
│   │   6. Return top-k results (default: 10)                             │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   Optimizations:                                                            │
│   • Product quantization for large stores (optional)                        │
│   • Batch processing for multiple queries                                   │
│   • In-memory caching for frequent access                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Vector Store API

```python
class LocalVectorStore:
    """Zvec-style local vector store for offline semantic search."""
    
    def __init__(self, db_path: str, dimension: int = 128, max_vectors: int = 10000):
        self.dimension = dimension
        self.max_vectors = max_vectors
    
    async def add(self, content: str, metadata: dict = None, importance: float = 0.5) -> str:
        """Add content with auto-generated embedding."""
        vector = self._embed(content)
        # Store in SQLite...
        return vector_id
    
    async def search(self, query: str, top_k: int = 10, threshold: float = 0.5) -> List[SearchResult]:
        """Semantic search using cosine similarity."""
        query_vector = self._embed(query)
        # Compute similarities...
        return results
    
    def _embed(self, text: str) -> np.ndarray:
        """Generate 128-dim embedding using character trigrams."""
        trigrams = [text[i:i+3] for i in range(len(text)-2)]
        vector = np.zeros(self.dimension)
        for trigram in trigrams:
            idx = hash(trigram) % self.dimension
            vector[idx] += 1
        return vector / (np.linalg.norm(vector) + 1e-8)
```

---

## Knowledge Graph Structure

### GraphPilot-Style App Topology

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       KNOWLEDGE GRAPH STRUCTURE                             │
│                     (GraphPilot-Style App Topology)                         │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                            ENTITY TYPES                                     │
│                                                                             │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │
│   │    APP      │   │   SCREEN    │   │  COMPONENT  │   │   ACTION    │    │
│   │             │   │             │   │             │   │             │    │
│   │ name        │   │ name        │   │ type        │   │ name        │    │
│   │ package     │   │ route       │   │ props       │   │ handler     │    │
│   │ version     │   │ layout      │   │ state       │   │ params      │    │
│   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘    │
│                                                                             │
│   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────────┐    │
│   │    USER     │   │   SETTING   │   │    FILE     │   │   SERVICE   │    │
│   │             │   │             │   │             │   │             │    │
│   │ preferences │   │ key         │   │ path        │   │ endpoint    │    │
│   │ history     │   │ value       │   │ type        │   │ methods     │    │
│   │ context     │   │ category    │   │ content     │   │ auth        │    │
│   └─────────────┘   └─────────────┘   └─────────────┘   └─────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          RELATIONSHIP TYPES                                 │
│                                                                             │
│   APP ─────────── contains ──────────► SCREEN                               │
│   SCREEN ──────── has_component ─────► COMPONENT                            │
│   SCREEN ──────── navigates_to ──────► SCREEN                               │
│   COMPONENT ───── triggers ──────────► ACTION                               │
│   ACTION ──────── calls ─────────────► SERVICE                              │
│   USER ────────── interacted_with ───► SCREEN                               │
│   USER ────────── configured ────────► SETTING                              │
│   SERVICE ─────── reads/writes ──────► FILE                                 │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

                         EXAMPLE GRAPH VISUALIZATION

                              ┌─────────┐
                              │  AURA   │
                              │  (APP)  │
                              └────┬────┘
                                   │ contains
                    ┌──────────────┼──────────────┐
                    │              │              │
                    ▼              ▼              ▼
              ┌──────────┐  ┌──────────┐  ┌──────────┐
              │  Home    │  │ Settings │  │  Chat    │
              │ (SCREEN) │  │ (SCREEN) │  │ (SCREEN) │
              └────┬─────┘  └────┬─────┘  └────┬─────┘
                   │             │             │
         ┌─────────┼─────┐      │        ┌────┴─────┐
         │         │     │      │        │          │
         ▼         ▼     ▼      ▼        ▼          ▼
     ┌───────┐ ┌───────┐   ┌────────┐ ┌───────┐ ┌───────┐
     │Button │ │List   │   │Toggle  │ │Input  │ │Button │
     │(COMP) │ │(COMP) │   │(COMP)  │ │(COMP) │ │(COMP) │
     └───┬───┘ └───────┘   └────┬───┘ └───────┘ └───┬───┘
         │                      │                   │
         ▼                      ▼                   ▼
     ┌────────┐           ┌──────────┐        ┌─────────┐
     │navigate│           │updatePref│        │sendMsg  │
     │(ACTION)│           │(ACTION)  │        │(ACTION) │
     └────────┘           └──────────┘        └────┬────┘
                                                   │
                                                   ▼
                                             ┌──────────┐
                                             │ ChatAPI  │
                                             │(SERVICE) │
                                             └──────────┘
```

### Knowledge Graph Schema

```python
@dataclass
class KnowledgeNode:
    """Node in the knowledge graph."""
    id: str
    type: str           # app, screen, component, action, user, setting, file, service
    name: str
    properties: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    confidence: float   # 0.0 - 1.0

@dataclass  
class KnowledgeEdge:
    """Edge connecting two nodes."""
    id: str
    source_id: str
    target_id: str
    relation: str       # contains, has_component, navigates_to, triggers, etc.
    properties: Dict[str, Any]
    weight: float       # Relationship strength
    created_at: datetime
```

---

## Temporal Knowledge Graph

### Graphiti-Style Temporal Facts

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TEMPORAL KNOWLEDGE GRAPH                                 │
│                    (Graphiti-Style Validity Intervals)                      │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         TEMPORAL FACT MODEL                                 │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Fact: "User prefers dark mode"                                    │  │
│   │                                                                     │  │
│   │   ┌───────────────────────────────────────────────────────────────┐│  │
│   │   │ Timeline                                                      ││  │
│   │   │                                                               ││  │
│   │   │ 2025-01-01          2025-06-15          NOW                   ││  │
│   │   │     │                   │                │                    ││  │
│   │   │     ▼                   ▼                ▼                    ││  │
│   │   │ ════╪═══════════════════╪════════════════╪══════► time       ││  │
│   │   │     │                   │                │                    ││  │
│   │   │     │◄─── valid_from    │◄── superseded  │                    ││  │
│   │   │     │                   │    (new fact)  │                    ││  │
│   │   │     └───────────────────┘                                     ││  │
│   │   │         valid_to                                              ││  │
│   │   │                                                               ││  │
│   │   └───────────────────────────────────────────────────────────────┘│  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                      TEMPORAL OPERATIONS                                    │
│                                                                             │
│   1. ADD FACT (with validity)                                               │
│      ┌──────────────────────────────────────────────────────────────────┐  │
│      │ add_fact("User likes Python", valid_from=NOW, valid_to=None)     │  │
│      │                                                                  │  │
│      │ Creates: Fact with open-ended validity (current truth)           │  │
│      └──────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   2. SUPERSEDE FACT                                                         │
│      ┌──────────────────────────────────────────────────────────────────┐  │
│      │ supersede_fact(old_fact_id, "User prefers Rust")                 │  │
│      │                                                                  │  │
│      │ Actions:                                                         │  │
│      │ • Set old_fact.valid_to = NOW                                    │  │
│      │ • Create new_fact.valid_from = NOW                               │  │
│      │ • Link: new_fact.supersedes = old_fact_id                        │  │
│      └──────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   3. EXPIRE FACT                                                            │
│      ┌──────────────────────────────────────────────────────────────────┐  │
│      │ expire_fact(fact_id)                                             │  │
│      │                                                                  │  │
│      │ Sets: fact.valid_to = NOW (no longer true)                       │  │
│      └──────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   4. QUERY AT TIME                                                          │
│      ┌──────────────────────────────────────────────────────────────────┐  │
│      │ get_facts_at(entity="User", timestamp=2025-03-15)                │  │
│      │                                                                  │  │
│      │ Returns: All facts where valid_from <= timestamp < valid_to      │  │
│      └──────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Temporal Schema

```python
@dataclass
class TemporalFact:
    """Fact with temporal validity."""
    id: str
    subject: str            # Entity the fact is about
    predicate: str          # Relationship/property
    object: str             # Value or target entity
    valid_from: datetime    # When fact became true
    valid_to: Optional[datetime]  # When fact stopped being true (None = current)
    supersedes: Optional[str]     # ID of fact this supersedes
    confidence: float
    source: str             # Where this fact came from
    
    def is_valid_at(self, timestamp: datetime) -> bool:
        """Check if fact was valid at given time."""
        if timestamp < self.valid_from:
            return False
        if self.valid_to and timestamp >= self.valid_to:
            return False
        return True
```

---

## SQLite Schema Diagrams

### Complete Database Schema

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SQLITE SCHEMA OVERVIEW                              │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        EPISODIC MEMORY DATABASE                             │
│                         (episodic_memory.db)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      episodic_traces                                │  │
│   ├─────────────────────────────────────────────────────────────────────┤  │
│   │ id              TEXT PRIMARY KEY                                    │  │
│   │ content         TEXT NOT NULL                                       │  │
│   │ sensory_embed   BLOB          -- 64-dim float32                     │  │
│   │ spatial_embed   BLOB          -- 64-dim float32                     │  │
│   │ temporal_embed  BLOB          -- 64-dim float32                     │  │
│   │ emotional       REAL DEFAULT 0.0                                    │  │
│   │ importance      REAL DEFAULT 0.5                                    │  │
│   │ context         TEXT          -- JSON                               │  │
│   │ consolidated    INTEGER DEFAULT 0                                   │  │
│   │ created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP                 │  │
│   │ accessed_at     TIMESTAMP                                           │  │
│   │ access_count    INTEGER DEFAULT 0                                   │  │
│   ├─────────────────────────────────────────────────────────────────────┤  │
│   │ INDEXES:                                                            │  │
│   │ • idx_episodic_importance (importance DESC)                         │  │
│   │ • idx_episodic_consolidated (consolidated)                          │  │
│   │ • idx_episodic_created (created_at DESC)                            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        SEMANTIC MEMORY DATABASE                             │
│                         (semantic_memory.db)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────────────────────┐   ┌──────────────────────────────┐      │
│   │           facts              │   │         concepts             │      │
│   ├──────────────────────────────┤   ├──────────────────────────────┤      │
│   │ id          TEXT PK          │   │ id          TEXT PK          │      │
│   │ fact        TEXT NOT NULL    │   │ name        TEXT NOT NULL    │      │
│   │ entity_type TEXT             │   │ category    TEXT             │      │
│   │ entities    TEXT (JSON)      │   │ attributes  TEXT (JSON)      │      │
│   │ relation    TEXT             │   │ related     TEXT (JSON)      │      │
│   │ confidence  REAL             │   │ confidence  REAL             │      │
│   │ evidence    INTEGER          │   │ created_at  TIMESTAMP        │      │
│   │ created_at  TIMESTAMP        │   │ updated_at  TIMESTAMP        │      │
│   │ updated_at  TIMESTAMP        │   └──────────────────────────────┘      │
│   └──────────────────────────────┘                                          │
│                                                                             │
│   ┌──────────────────────────────┐                                          │
│   │        preferences           │                                          │
│   ├──────────────────────────────┤                                          │
│   │ key         TEXT PK          │                                          │
│   │ value       TEXT NOT NULL    │                                          │
│   │ category    TEXT             │                                          │
│   │ confidence  REAL             │                                          │
│   │ occurrences INTEGER          │                                          │
│   │ created_at  TIMESTAMP        │                                          │
│   │ updated_at  TIMESTAMP        │                                          │
│   └──────────────────────────────┘                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         VECTOR STORE DATABASE                               │
│                          (vector_store.db)                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                         vectors                                     │  │
│   ├─────────────────────────────────────────────────────────────────────┤  │
│   │ id              TEXT PRIMARY KEY                                    │  │
│   │ content         TEXT NOT NULL                                       │  │
│   │ vector          BLOB NOT NULL     -- 128-dim float32 (512 bytes)    │  │
│   │ metadata        TEXT              -- JSON                           │  │
│   │ importance      REAL DEFAULT 0.5                                    │  │
│   │ access_count    INTEGER DEFAULT 0                                   │  │
│   │ created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP                 │  │
│   │ accessed_at     TIMESTAMP                                           │  │
│   ├─────────────────────────────────────────────────────────────────────┤  │
│   │ INDEXES:                                                            │  │
│   │ • idx_vectors_importance (importance DESC)                          │  │
│   │ • idx_vectors_access (access_count DESC)                            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        ANCESTOR MEMORY DATABASE                             │
│                         (ancestor_memory.db)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                     archived_memories                               │  │
│   ├─────────────────────────────────────────────────────────────────────┤  │
│   │ id                TEXT PRIMARY KEY                                  │  │
│   │ original_id       TEXT              -- Reference to source          │  │
│   │ source_type       TEXT              -- episodic/semantic/neural     │  │
│   │ content           BLOB NOT NULL     -- Compressed (zlib)            │  │
│   │ summary           TEXT              -- Human-readable summary       │  │
│   │ importance        REAL                                              │  │
│   │ tags              TEXT              -- JSON array                   │  │
│   │ compression_ratio REAL              -- Original/compressed size     │  │
│   │ encrypted         INTEGER DEFAULT 0                                 │  │
│   │ created_at        TIMESTAMP                                         │  │
│   │ archived_at       TIMESTAMP DEFAULT CURRENT_TIMESTAMP               │  │
│   │ accessed_at       TIMESTAMP                                         │  │
│   │ access_count      INTEGER DEFAULT 0                                 │  │
│   ├─────────────────────────────────────────────────────────────────────┤  │
│   │ INDEXES:                                                            │  │
│   │ • idx_archived_importance (importance DESC)                         │  │
│   │ • idx_archived_source (source_type)                                 │  │
│   │ • idx_archived_created (archived_at DESC)                           │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                    TEMPORAL KNOWLEDGE GRAPH DATABASE                        │
│                        (temporal_knowledge.db)                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌──────────────────────────────┐   ┌──────────────────────────────┐      │
│   │      temporal_entities       │   │      temporal_facts          │      │
│   ├──────────────────────────────┤   ├──────────────────────────────┤      │
│   │ id          TEXT PK          │   │ id          TEXT PK          │      │
│   │ name        TEXT NOT NULL    │   │ subject     TEXT NOT NULL    │      │
│   │ type        TEXT             │   │ predicate   TEXT NOT NULL    │      │
│   │ properties  TEXT (JSON)      │   │ object      TEXT NOT NULL    │      │
│   │ valid_from  TIMESTAMP        │   │ valid_from  TIMESTAMP        │      │
│   │ valid_to    TIMESTAMP        │   │ valid_to    TIMESTAMP        │      │
│   │ created_at  TIMESTAMP        │   │ supersedes  TEXT             │      │
│   └──────────────────────────────┘   │ confidence  REAL             │      │
│                                       │ source      TEXT             │      │
│   ┌──────────────────────────────┐   │ created_at  TIMESTAMP        │      │
│   │     temporal_relations       │   └──────────────────────────────┘      │
│   ├──────────────────────────────┤                                          │
│   │ id          TEXT PK          │                                          │
│   │ source_id   TEXT NOT NULL    │                                          │
│   │ target_id   TEXT NOT NULL    │                                          │
│   │ relation    TEXT NOT NULL    │                                          │
│   │ properties  TEXT (JSON)      │                                          │
│   │ valid_from  TIMESTAMP        │                                          │
│   │ valid_to    TIMESTAMP        │                                          │
│   │ created_at  TIMESTAMP        │                                          │
│   └──────────────────────────────┘                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Memory Budget Allocation

### Resource Management Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       MEMORY BUDGET ALLOCATION                              │
│                    (MemoryOptimizer Configuration)                          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                         BUDGET DISTRIBUTION                                 │
│                                                                             │
│   Total Memory Budget: Configurable (default guidance below)                │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                     │  │
│   │   Working Memory (Neural)     ████████░░░░░░░░░░░░  15%            │  │
│   │   • Max neurons: 1000                                               │  │
│   │   • Max clusters: 100                                               │  │
│   │   • Working set: 7±2 items                                          │  │
│   │                                                                     │  │
│   │   Episodic Memory            ████████████░░░░░░░░░  25%            │  │
│   │   • Max traces: 10,000                                              │  │
│   │   • Embedding size: 3 × 64-dim = 768 bytes/trace                    │  │
│   │   • SQLite with WAL mode                                            │  │
│   │                                                                     │  │
│   │   Semantic Memory            ████████████████░░░░░  35%            │  │
│   │   • Max facts: 50,000                                               │  │
│   │   • Max concepts: 10,000                                            │  │
│   │   • Max preferences: 5,000                                          │  │
│   │                                                                     │  │
│   │   Vector Store               ████████░░░░░░░░░░░░░  15%            │  │
│   │   • Max vectors: 10,000                                             │  │
│   │   • Vector size: 128-dim = 512 bytes/vector                         │  │
│   │   • Total: ~5MB vectors                                             │  │
│   │                                                                     │  │
│   │   Ancestor Memory (Archive)  ████░░░░░░░░░░░░░░░░░  10%            │  │
│   │   • Lazy loaded (not in RAM)                                        │  │
│   │   • Compressed storage                                              │  │
│   │   • Disk-based with LRU cache                                       │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                        PRUNING STRATEGIES                                   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ WORKING MEMORY PRUNING                                              │  │
│   │ Trigger: > 1000 neurons OR memory pressure                          │  │
│   │ Strategy:                                                           │  │
│   │   1. Calculate neuron scores: activation × importance × recency     │  │
│   │   2. Remove lowest 20% of neurons                                   │  │
│   │   3. Consolidate clusters with < 3 neurons                          │  │
│   │   4. Decay all activations by 0.95                                  │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ EPISODIC MEMORY PRUNING                                             │  │
│   │ Trigger: > 10,000 traces OR age > 30 days                           │  │
│   │ Strategy:                                                           │  │
│   │   1. Mark old traces for consolidation                              │  │
│   │   2. Extract knowledge to semantic memory                           │  │
│   │   3. Archive consolidated traces to ancestor                        │  │
│   │   4. Delete archived originals                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ VECTOR STORE PRUNING                                                │  │
│   │ Trigger: > 10,000 vectors                                           │  │
│   │ Strategy:                                                           │  │
│   │   1. Score: importance × access_count × recency                     │  │
│   │   2. Remove lowest scored vectors until < 8,000                     │  │
│   │   3. Rebuild indexes                                                │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ ANCESTOR MEMORY PRUNING                                             │  │
│   │ Trigger: Storage > threshold OR age > 1 year                        │  │
│   │ Strategy:                                                           │  │
│   │   1. Score: importance × access_count                               │  │
│   │   2. Generate summaries for removal candidates                      │  │
│   │   3. Keep summaries, delete full content                            │  │
│   │   4. Permanent delete if never accessed in 1 year                   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Memory Optimizer Configuration

```python
MEMORY_OPTIMIZER_CONFIG = {
    "thresholds": {
        "working_memory": {
            "max_neurons": 1000,
            "max_clusters": 100,
            "prune_percent": 0.20
        },
        "episodic_memory": {
            "max_traces": 10000,
            "consolidation_age_days": 7,
            "archive_age_days": 30
        },
        "semantic_memory": {
            "max_facts": 50000,
            "max_concepts": 10000,
            "max_preferences": 5000,
            "min_confidence": 0.3
        },
        "vector_store": {
            "max_vectors": 10000,
            "prune_target": 8000,
            "min_importance": 0.2
        },
        "ancestor_memory": {
            "compression_level": 6,
            "max_age_days": 365,
            "min_access_count": 1
        }
    },
    "optimization_interval_hours": 24,
    "aggressive_mode_threshold": 0.90  # Trigger at 90% capacity
}
```

---

## Neural Memory Details

### Hebbian Learning Implementation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      NEURAL MEMORY (Working Memory)                         │
│                   "Neurons that fire together wire together"                │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          NEURON STRUCTURE                                   │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                         Neuron                                      │  │
│   ├─────────────────────────────────────────────────────────────────────┤  │
│   │                                                                     │  │
│   │   id: str              Unique identifier                            │  │
│   │   content: str         Memory content                               │  │
│   │   embedding: [64]      Character trigram embedding                  │  │
│   │   activation: float    Current activation level (0-1)               │  │
│   │   importance: float    Calculated importance score                  │  │
│   │   emotional: float     Emotional valence (-1 to +1)                 │  │
│   │   created_at: datetime When neuron was created                      │  │
│   │   accessed_at: datetime Last access time                            │  │
│   │   access_count: int    Number of times accessed                     │  │
│   │                                                                     │  │
│   │   connections: Dict[str, Synapse]  Connections to other neurons     │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                          SYNAPSE STRUCTURE                                  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                         Synapse                                     │  │
│   ├─────────────────────────────────────────────────────────────────────┤  │
│   │                                                                     │  │
│   │   target_id: str       ID of connected neuron                       │  │
│   │   weight: float        Connection strength (0-1)                    │  │
│   │   created_at: datetime When connection formed                       │  │
│   │   last_fired: datetime Last co-activation                           │  │
│   │                                                                     │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                       HEBBIAN LEARNING RULE                                 │
│                                                                             │
│   When neurons A and B fire together:                                       │
│                                                                             │
│   Δw = η × A_activation × B_activation                                      │
│                                                                             │
│   Where:                                                                    │
│   • η = learning rate (default: 0.1)                                        │
│   • A_activation = activation level of neuron A                             │
│   • B_activation = activation level of neuron B                             │
│                                                                             │
│   Weight update:                                                            │
│   w_new = min(1.0, w_old + Δw)                                              │
│                                                                             │
│   Weight decay (when not co-firing):                                        │
│   w_new = w_old × decay_factor (default: 0.99)                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘

                         CLUSTER FORMATION

    Initial State              After Learning            Cluster Formed
    
    ○ A    ○ B                 ○ A ═══ ○ B              ┌─────────────┐
                                  ╲   ╱                 │  Cluster 1  │
    ○ C    ○ D                     ╲ ╱                  │ ○A ═══ ○B   │
                               ○ C ═ ○ D                │   ╲   ╱     │
    ○ E                                                 │    ○C═○D    │
                               ○ E                      └─────────────┘
                                                        
                                                        ○ E (isolated)

    Neurons form clusters when synapse weights exceed threshold (0.5)
```

---

## Unified Memory Retrieval

### Phased Search Strategy

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    UNIFIED MEMORY RETRIEVAL                                 │
│                      (Phased Search Strategy)                               │
└─────────────────────────────────────────────────────────────────────────────┘

                              Query: "How to configure settings?"
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 1: WORKING MEMORY                                               ~5ms │
│ ┌─────────────────────────────────────────────────────────────────────────┐│
│ │ Search active neurons by:                                               ││
│ │ • Embedding similarity (cosine)                                         ││
│ │ • Activation level                                                      ││
│ │ • Recency                                                               ││
│ │                                                                         ││
│ │ Result: Most recently accessed, highly activated relevant neurons       ││
│ └─────────────────────────────────────────────────────────────────────────┘│
│                        │                                                    │
│                        ▼ If insufficient results                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 2: EPISODIC MEMORY                                            ~20ms │
│ ┌─────────────────────────────────────────────────────────────────────────┐│
│ │ Pattern completion search:                                              ││
│ │ • Sensory embedding match                                               ││
│ │ • Temporal context match                                                ││
│ │ • Emotional tag filter                                                  ││
│ │                                                                         ││
│ │ Result: Recent episodes matching query pattern                          ││
│ └─────────────────────────────────────────────────────────────────────────┘│
│                        │                                                    │
│                        ▼ If insufficient results                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 3: SEMANTIC MEMORY                                            ~30ms │
│ ┌─────────────────────────────────────────────────────────────────────────┐│
│ │ Knowledge base search:                                                  ││
│ │ • Fact lookup (entity + relation)                                       ││
│ │ • Concept hierarchy traversal                                           ││
│ │ • Preference matching                                                   ││
│ │                                                                         ││
│ │ Result: Relevant facts, concepts, and preferences                       ││
│ └─────────────────────────────────────────────────────────────────────────┘│
│                        │                                                    │
│                        ▼ If insufficient results                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PHASE 4: ANCESTOR MEMORY (Lazy)                                    ~100ms │
│ ┌─────────────────────────────────────────────────────────────────────────┐│
│ │ Archive search (only if needed):                                        ││
│ │ • Tag-based filtering                                                   ││
│ │ • Summary text search                                                   ││
│ │ • Decompress on demand                                                  ││
│ │                                                                         ││
│ │ Result: Historical memories from archive                                ││
│ └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                          │
                                          ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RESULT AGGREGATION                                   │
│ ┌─────────────────────────────────────────────────────────────────────────┐│
│ │ 1. Collect results from all phases                                      ││
│ │ 2. Deduplicate by content similarity                                    ││
│ │ 3. Score: relevance × importance × recency                              ││
│ │ 4. Rank by final score                                                  ││
│ │ 5. Return top-k results                                                 ││
│ └─────────────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## API Reference

### Core Classes

```python
# Memory Coordinator - Main integration point
class MemoryCoordinator:
    async def store(self, content: str, context: dict = None) -> str
    async def retrieve(self, query: str, top_k: int = 10) -> List[MemoryResult]
    async def consolidate(self) -> ConsolidationReport
    async def get_stats(self) -> MemoryStats

# Neural Memory - Working memory
class NeuralMemory:
    async def activate(self, content: str, context: dict = None) -> Neuron
    async def recall(self, query: str, top_k: int = 5) -> List[Neuron]
    async def strengthen_connection(self, neuron_a: str, neuron_b: str)
    async def get_clusters(self) -> List[NeuronCluster]

# Episodic Memory - Recent experiences
class EpisodicMemory:
    async def encode(self, content: str, emotional: float = 0.0) -> str
    async def retrieve(self, query: str, top_k: int = 10) -> List[Episode]
    async def pattern_complete(self, partial: str) -> List[Episode]

# Semantic Memory - Knowledge storage
class SemanticMemory:
    async def store_fact(self, fact: str, entities: List[str], relation: str)
    async def store_concept(self, name: str, category: str, attributes: dict)
    async def store_preference(self, key: str, value: str, category: str)
    async def query_facts(self, entity: str = None, relation: str = None)

# Ancestor Memory - Long-term archive
class AncestorMemory:
    async def archive(self, memory_id: str, source_type: str, content: str)
    async def retrieve(self, query: str, top_k: int = 5) -> List[ArchivedMemory]
    async def prune(self, max_age_days: int = 365)
```

---

## Configuration

### Environment Variables

```bash
# Memory system configuration
AURA_MEMORY_DIR=/path/to/memory/storage
AURA_MEMORY_ENCRYPTION_KEY=your-fernet-key  # Optional

# Limits
AURA_MAX_NEURONS=1000
AURA_MAX_EPISODIC_TRACES=10000
AURA_MAX_VECTORS=10000

# Consolidation
AURA_CONSOLIDATION_INTERVAL=300  # seconds
AURA_ARCHIVE_AGE_DAYS=30
```

### Configuration File

```yaml
# config/memory.yaml
memory:
  working:
    max_neurons: 1000
    max_clusters: 100
    hebbian_learning_rate: 0.1
    activation_decay: 0.95
    
  episodic:
    max_traces: 10000
    embedding_dim: 64
    encoding_target_ms: 100
    consolidation_threshold: 100
    
  semantic:
    max_facts: 50000
    max_concepts: 10000
    max_preferences: 5000
    min_confidence: 0.3
    
  vector_store:
    dimension: 128
    max_vectors: 10000
    similarity_threshold: 0.5
    
  ancestor:
    compression_level: 6
    encryption_enabled: false
    lazy_loading: true
    max_age_days: 365
    
  consolidation:
    interval_seconds: 300
    working_to_episodic_threshold: 0.3
    episodic_to_semantic_threshold: 100
    
  optimization:
    interval_hours: 24
    aggressive_mode_threshold: 0.90
```

---

## Performance Characteristics

| Operation | Target Latency | Notes |
|-----------|---------------|-------|
| Working memory activation | < 10ms | In-memory |
| Episodic encoding | < 100ms | SQLite write |
| Semantic fact storage | < 50ms | SQLite write |
| Vector search (10K vectors) | < 100ms | Linear scan |
| Phased retrieval (all tiers) | < 200ms | Total |
| Consolidation cycle | < 5s | Background |
| Archive compression | Variable | Async |

---

## Related Documentation

- [NEURAL_ARCHITECTURE.md](./NEURAL_ARCHITECTURE.md) - Detailed neural subsystem docs
- [STORAGE_LAYER.md](./STORAGE_LAYER.md) - SQLite and persistence details
- [SKILL_SYSTEM.md](./SKILL_SYSTEM.md) - Skill memory integration
- [CONTEXT_ENGINE.md](./CONTEXT_ENGINE.md) - Context-memory interaction

---

*Document generated from AURA v3 source code analysis*
