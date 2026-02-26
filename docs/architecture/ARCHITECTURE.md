# AURA v3 Architecture Documentation

> **Autonomous Background AI Assistant for Mobile Devices**  
> Target: 8GB RAM (Termux/Android) | Local-First | Privacy-Preserving

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Component Hierarchy](#component-hierarchy)
3. [Tiered Initialization](#tiered-initialization)
4. [Memory Architecture](#memory-architecture)
5. [Agent Loop (ReAct Pattern)](#agent-loop-react-pattern)
6. [LLM Brain Integration](#llm-brain-integration)
7. [Data Flow Diagrams](#data-flow-diagrams)
8. [Module Interactions](#module-interactions)
9. [Mobile Constraints & RAM Budget](#mobile-constraints--ram-budget)
10. [Security Model](#security-model)

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           AURA v3 SYSTEM ARCHITECTURE                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        USER INTERFACE LAYER                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────┐ │   │
│  │  │ Aura Space   │  │   Termux     │  │  Floating    │  │Character │ │   │
│  │  │   Server     │  │Widget Bridge │  │   Bubbles    │  │  Sheet   │ │   │
│  │  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └────┬─────┘ │   │
│  └─────────┼─────────────────┼─────────────────┼───────────────┼───────┘   │
│            │                 │                 │               │           │
│            └─────────────────┴────────┬────────┴───────────────┘           │
│                                       ▼                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         AGENT CORE (ReAct Loop)                      │   │
│  │                                                                       │   │
│  │   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐  │   │
│  │   │   Observe   │───▶│    Think    │───▶│         Act             │  │   │
│  │   │  (Context)  │    │   (LLM)     │    │   (Tool Orchestrator)   │  │   │
│  │   └─────────────┘    └─────────────┘    └─────────────────────────┘  │   │
│  │          ▲                  │                       │                │   │
│  │          │                  ▼                       ▼                │   │
│  │   ┌──────┴──────┐    ┌─────────────┐    ┌─────────────────────────┐  │   │
│  │   │  Reflect    │◀───│Meta-Cognition│◀───│      Tool Results       │  │   │
│  │   │ (Learning)  │    │ Uncertainty │    │                         │  │   │
│  │   └─────────────┘    └─────────────┘    └─────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                       │                                     │
│            ┌──────────────────────────┼──────────────────────────┐         │
│            ▼                          ▼                          ▼         │
│  ┌─────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐ │
│  │  MEMORY SYSTEM  │    │     LLM MANAGER     │    │    TOOL REGISTRY    │ │
│  │                 │    │                     │    │                     │ │
│  │ ┌─────────────┐ │    │  ┌───────────────┐  │    │  ┌───────────────┐  │ │
│  │ │   Neural    │ │    │  │  Local LLM    │  │    │  │   Android     │  │ │
│  │ │  (Working)  │ │    │  │ (llama.cpp)   │  │    │  │    Tools      │  │ │
│  │ ├─────────────┤ │    │  ├───────────────┤  │    │  ├───────────────┤  │ │
│  │ │  Episodic   │ │    │  │  Whisper STT  │  │    │  │   File Ops    │  │ │
│  │ │  (Events)   │ │    │  ├───────────────┤  │    │  ├───────────────┤  │ │
│  │ ├─────────────┤ │    │  │   Piper TTS   │  │    │  │   Web/API     │  │ │
│  │ │  Semantic   │ │    │  ├───────────────┤  │    │  ├───────────────┤  │ │
│  │ │ (Knowledge) │ │    │  │ Cloud Fallback│  │    │  │   System      │  │ │
│  │ ├─────────────┤ │    │  │  (Optional)   │  │    │  │   Control     │  │ │
│  │ │  Ancestor   │ │    │  └───────────────┘  │    │  └───────────────┘  │ │
│  │ │ (Archive)   │ │    │                     │    │                     │ │
│  │ └─────────────┘ │    └─────────────────────┘    └─────────────────────┘ │
│  └─────────────────┘                                                       │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       BACKGROUND SERVICES                            │   │
│  │  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────────────┐ │   │
│  │  │ Proactive  │ │   Life     │ │   Social   │ │    Healthcare      │ │   │
│  │  │  Engine    │ │  Tracker   │ │   Life     │ │     Agent          │ │   │
│  │  └────────────┘ └────────────┘ └────────────┘ └────────────────────┘ │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         SECURITY LAYER                               │   │
│  │       Authentication │ Permissions │ Privacy Guard │ Encryption      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Hierarchy

```
aura-v3/src/
│
├── main.py                    # Entry point: AuraProduction class
│                              # Tiered initialization (0/1/2)
│
├── agent/                     # Core Agent System
│   ├── loop.py               # ReAct agent loop (959 lines)
│   │                         # - observe → think → act → reflect
│   │                         # - max 10 iterations per query
│   │                         # - tool execution with fallbacks
│   ├── personality_system.py # Dynamic personality adaptation
│   ├── personality.py        # Personality traits & expression
│   └── relationship_system.py# User relationship tracking
│
├── memory/                    # Multi-Tier Memory System
│   ├── memory_coordinator.py # Unifies all memory systems
│   ├── neural_memory.py      # Working memory (neurons/synapses)
│   ├── episodic_memory.py    # Event memories (hippocampus-like)
│   ├── semantic_memory.py    # Knowledge graph (neocortex-like)
│   ├── ancestor_memory.py    # Ancient patterns archive
│   ├── knowledge_graph.py    # Graph-based knowledge store
│   └── local_vector_store.py # Embeddings for similarity search
│
├── llm/                       # LLM Integration
│   ├── manager.py            # LLMManager: STT/TTS/LLM routing
│   ├── production_llm.py     # Production LLM wrapper
│   └── real_llm.py           # Direct LLM API calls
│
├── core/                      # Core Neural Systems
│   ├── neuromorphic_engine.py    # Bio-inspired processing
│   ├── neural_validated_planner.py # Memory-aware planning
│   ├── hebbian_self_correction.py  # Learn from success/failure
│   ├── neural_aware_router.py      # Context-aware model selection
│   ├── proactivity_controller.py   # Knows when NOT to act
│   ├── tool_orchestrator.py        # Deterministic tool execution
│   ├── execution_control.py        # Execution flow management
│   ├── loop_detector.py            # Prevent infinite loops
│   ├── adaptive_personality.py     # Personality adaptation
│   ├── conversation.py             # Conversation management
│   ├── user_profile.py             # User modeling
│   └── mobile_power.py             # Power management
│
├── tools/                     # Tool System
│   ├── registry.py           # Tool registration & discovery
│   ├── handlers.py           # Tool execution handlers
│   ├── base.py               # Base tool interface
│   └── android.py            # Android-specific tools
│
├── services/                  # Background Services
│   ├── proactive_engine.py       # Proactive behavior engine
│   ├── proactive_event_tracker.py# Event monitoring
│   ├── proactive_life_explorer.py# Life pattern discovery
│   ├── life_tracker.py           # Daily life tracking
│   ├── adaptive_context.py       # Context adaptation
│   ├── self_learning.py          # Continuous learning
│   ├── inner_voice.py            # Internal dialogue
│   └── intelligent_call_manager.py # Call handling
│
├── agents/                    # Specialized Agents
│   ├── coordinator.py        # Multi-agent coordination
│   ├── healthcare/           # Health monitoring agent
│   ├── social_life/          # Social interaction agent
│   └── specialized_agents/   # Domain-specific agents
│
├── mobile/                    # Mobile UI Components
│   ├── aura_space_server.py  # Web-based UI server
│   ├── termux_widget_bridge.py # Termux widget integration
│   ├── character_sheet.py    # Visual character display
│   ├── cinematic_moments.py  # Rich moment presentation
│   └── floating_bubbles.py   # Overlay notifications
│
├── context/                   # Context System
│   ├── detection.py          # Context detection
│   └── provider.py           # Context provision
│
├── security/                  # Security Layer
│   ├── auth.py               # Authentication
│   ├── permissions.py        # Permission management
│   └── privacy.py            # Privacy protection
│
├── learning/                  # Learning System
│   └── engine.py             # Learning engine
│
└── session/                   # Session Management
    └── manager.py            # Session state management
```

---

## Tiered Initialization

AURA uses a 3-tier lazy loading system to minimize startup time and RAM usage:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TIERED INITIALIZATION SYSTEM                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  TIER 0: BOOT (Immediate)                        ~30-50 MB RAM              │
│  ════════════════════════                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  [LLM Manager] ──▶ [Agent Loop] ──▶ [Memory Coordinator]            │   │
│  │        │                │                    │                       │   │
│  │        ▼                ▼                    ▼                       │   │
│  │  [Tool Registry] ◀── [Security] ◀── [Neural Memory (Minimal)]       │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼ First User Query                       │
│                                                                             │
│  TIER 1: FIRST QUERY (On-Demand)                 +50-100 MB RAM             │
│  ═══════════════════════════════                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐   │   │
│  │  │   Context     │  │  Personality  │  │   Neural Systems      │   │   │
│  │  │   Provider    │  │    System     │  │   (Planner/Router)    │   │   │
│  │  └───────────────┘  └───────────────┘  └───────────────────────┘   │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐   │   │
│  │  │   Session     │  │   Learning    │  │   Hebbian Self-       │   │   │
│  │  │   Manager     │  │    Engine     │  │   Corrector           │   │   │
│  │  └───────────────┘  └───────────────┘  └───────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼ Background (Async)                     │
│                                                                             │
│  TIER 2: BACKGROUND (Async Load)                 +100-200 MB RAM            │
│  ═══════════════════════════════                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐   │   │
│  │  │  Proactive    │  │  Healthcare   │  │    Social Life        │   │   │
│  │  │   Engine      │  │    Agent      │  │      Agent            │   │   │
│  │  └───────────────┘  └───────────────┘  └───────────────────────┘   │   │
│  │  ┌───────────────┐  ┌───────────────┐  ┌───────────────────────┐   │   │
│  │  │ Aura Space    │  │   Termux      │  │   Cinematic           │   │   │
│  │  │   Server      │  │   Widgets     │  │   Moments             │   │   │
│  │  └───────────────┘  └───────────────┘  └───────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ═══════════════════════════════════════════════════════════════════════   │
│  TOTAL FULLY LOADED:                            ~250-400 MB RAM             │
│  (Leaves 7.5+ GB for LLM inference & Android)                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Initialization Flow (main.py)

```python
class AuraProduction:
    async def initialize(self):
        # TIER 0: Boot essentials
        await self._init_tier0_boot()      # LLM, Loop, Memory, Tools, Security
        
        # Ready for queries immediately
        
    async def handle_query(self, query):
        # TIER 1: Load on first query
        if not self._tier1_loaded:
            await self._init_tier1_first_query()
        
        # Process query...
        
    async def _background_init(self):
        # TIER 2: Load in background
        await self._init_tier2_background()  # Proactive, Healthcare, Social
```

---

## Memory Architecture

AURA implements a biologically-inspired multi-tier memory system:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MEMORY ARCHITECTURE                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                        ┌─────────────────────┐                              │
│                        │  MEMORY COORDINATOR │                              │
│                        │  (Unified Interface)│                              │
│                        └──────────┬──────────┘                              │
│                                   │                                         │
│          ┌────────────────────────┼────────────────────────┐               │
│          │                        │                        │               │
│          ▼                        ▼                        ▼               │
│  ┌───────────────┐    ┌───────────────────┐    ┌───────────────────┐       │
│  │ NEURAL MEMORY │    │  EPISODIC MEMORY  │    │  SEMANTIC MEMORY  │       │
│  │   (Working)   │    │    (Events)       │    │   (Knowledge)     │       │
│  │               │    │                   │    │                   │       │
│  │  ┌─────────┐  │    │  ┌─────────────┐  │    │  ┌─────────────┐  │       │
│  │  │ Neurons │  │    │  │  Episodes   │  │    │  │  Concepts   │  │       │
│  │  │  Pool   │  │    │  │   (what,    │  │    │  │   (named    │  │       │
│  │  │ (1000)  │  │    │  │   when,     │  │    │  │   entities, │  │       │
│  │  └────┬────┘  │    │  │   where,    │  │    │  │   facts)    │  │       │
│  │       │       │    │  │   who)      │  │    │  └──────┬──────┘  │       │
│  │  ┌────▼────┐  │    │  └──────┬──────┘  │    │         │         │       │
│  │  │Synapses │  │    │         │         │    │  ┌──────▼──────┐  │       │
│  │  │(Hebbian │  │    │  ┌──────▼──────┐  │    │  │ Preferences │  │       │
│  │  │Learning)│  │    │  │   Pattern   │  │    │  │  (learned   │  │       │
│  │  └────┬────┘  │    │  │ Separation  │  │    │  │   tastes)   │  │       │
│  │       │       │    │  │ /Completion │  │    │  └──────┬──────┘  │       │
│  │  ┌────▼────┐  │    │  └─────────────┘  │    │         │         │       │
│  │  │Attention│  │    │                   │    │  ┌──────▼──────┐  │       │
│  │  │  Focus  │  │    │  Hippocampus-     │    │  │  Knowledge  │  │       │
│  │  └─────────┘  │    │  like: Fast       │    │  │    Graph    │  │       │
│  │               │    │  encoding,        │    │  │  (entities, │  │       │
│  │  Bio-inspired │    │  retrieval by     │    │  │  relations) │  │       │
│  │  working mem  │    │  similarity       │    │  └─────────────┘  │       │
│  └───────────────┘    └───────────────────┘    │                   │       │
│          │                     │               │  Neocortex-like:  │       │
│          │                     │               │  Slow learning,   │       │
│          │                     │               │  generalization   │       │
│          │                     │               └───────────────────┘       │
│          │                     │                        │                  │
│          └──────────┬──────────┴────────────────────────┘                  │
│                     │                                                       │
│                     ▼                                                       │
│          ┌───────────────────┐        ┌───────────────────┐                │
│          │  ANCESTOR MEMORY  │◀───────│  LOCAL VECTOR     │                │
│          │    (Archive)      │        │     STORE         │                │
│          │                   │        │                   │                │
│          │  Ancient patterns │        │  Embeddings for   │                │
│          │  rarely accessed  │        │  similarity       │                │
│          │  but preserved    │        │  search           │                │
│          └───────────────────┘        └───────────────────┘                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Memory Consolidation Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MEMORY CONSOLIDATION FLOW                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   New Experience                                                            │
│        │                                                                    │
│        ▼                                                                    │
│   ┌─────────────┐      Immediate activation                                │
│   │   NEURAL    │      Neurons fire, synapses strengthen                   │
│   │  (Working)  │      Attention focuses on salient info                   │
│   └──────┬──────┘      Decay: ~seconds to minutes                          │
│          │                                                                  │
│          │ Important events                                                 │
│          ▼ (emotional, novel, repeated)                                    │
│   ┌─────────────┐      Fast encoding                                       │
│   │  EPISODIC   │      Pattern separation (unique ID)                      │
│   │  (Events)   │      Context binding (time, place, people)               │
│   └──────┬──────┘      Decay: days to weeks                                │
│          │                                                                  │
│          │ Repeated retrieval                                              │
│          ▼ Sleep-like consolidation                                        │
│   ┌─────────────┐      Slow integration                                    │
│   │  SEMANTIC   │      Generalization (extract patterns)                   │
│   │ (Knowledge) │      Knowledge graph updates                             │
│   └──────┬──────┘      Decay: months to years                              │
│          │                                                                  │
│          │ Rarely accessed                                                 │
│          ▼ but valuable patterns                                           │
│   ┌─────────────┐      Archive storage                                     │
│   │  ANCESTOR   │      Compressed representations                          │
│   │  (Archive)  │      Eternal retention                                   │
│   └─────────────┘                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Neural Memory Details (neural_memory.py)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       NEURAL MEMORY INTERNALS                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                         NEURON POOL                                  │  │
│   │                                                                      │  │
│   │    ○───○───○───○───○───○───○───○───○───○     ○ = Neuron             │  │
│   │    │ \ │ / │ \ │ / │ \ │ / │ \ │ / │         │ = Synapse            │  │
│   │    ○───○───○───○───○───○───○───○───○───○                            │  │
│   │    │ / │ \ │ / │ \ │ / │ \ │ / │ \ │                                │  │
│   │    ○───○───○───○───○───○───○───○───○───○     1000 neurons           │  │
│   │                                              Sparse activation       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   NEURON STRUCTURE:                                                         │
│   ┌─────────────────────────────────────┐                                  │
│   │  Neuron {                           │                                  │
│   │    id: str                          │                                  │
│   │    content: str       # What it represents                             │
│   │    embedding: [float] # Vector representation                          │
│   │    activation: float  # Current activation (0-1)                       │
│   │    last_activated: datetime                                            │
│   │    activation_count: int                                               │
│   │  }                                  │                                  │
│   └─────────────────────────────────────┘                                  │
│                                                                             │
│   SYNAPSE STRUCTURE:                                                        │
│   ┌─────────────────────────────────────┐                                  │
│   │  Synapse {                          │                                  │
│   │    source_id: str                   │                                  │
│   │    target_id: str                   │                                  │
│   │    weight: float      # Strength (0-1)                                 │
│   │    last_strengthened: datetime                                         │
│   │  }                                  │                                  │
│   └─────────────────────────────────────┘                                  │
│                                                                             │
│   HEBBIAN LEARNING:                                                         │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                      │  │
│   │   "Neurons that fire together, wire together"                       │  │
│   │                                                                      │  │
│   │   When neurons A and B activate simultaneously:                     │  │
│   │     synapse(A→B).weight += learning_rate × A.activation × B.activation │
│   │                                                                      │  │
│   │   Over time: frequently co-activated neurons form strong bonds      │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ATTENTION MECHANISM:                                                      │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                                                                      │  │
│   │   Input Query ──▶ Embedding ──▶ Similarity Search ──▶ Top-K Neurons │  │
│   │                                                                      │  │
│   │   Activated neurons spread activation to connected neurons          │  │
│   │   (spreading activation with decay)                                 │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Agent Loop (ReAct Pattern)

The core of AURA is a ReAct (Reasoning + Acting) agent loop:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ReAct AGENT LOOP                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                         User Query                                          │
│                             │                                               │
│                             ▼                                               │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      OBSERVE PHASE                                   │  │
│   │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌────────────┐  │  │
│   │  │   Query     │  │   Memory    │  │   Context   │  │    User    │  │  │
│   │  │  Analysis   │  │  Retrieval  │  │  Detection  │  │  Profile   │  │  │
│   │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └─────┬──────┘  │  │
│   │         └────────────────┴────────────────┴───────────────┘         │  │
│   └─────────────────────────────────┬───────────────────────────────────┘  │
│                                     │                                       │
│                                     ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                       THINK PHASE (LLM)                              │  │
│   │                                                                      │  │
│   │   System Prompt:                                                     │  │
│   │   ┌───────────────────────────────────────────────────────────────┐ │  │
│   │   │ You are AURA, an autonomous AI assistant.                     │ │  │
│   │   │ Available tools: {tool_schemas}                               │ │  │
│   │   │ User context: {context}                                       │ │  │
│   │   │ Relevant memories: {memories}                                 │ │  │
│   │   │                                                               │ │  │
│   │   │ Respond with ONE of:                                          │ │  │
│   │   │ 1. {"thought": "...", "tool": "...", "args": {...}}          │ │  │
│   │   │ 2. {"thought": "...", "response": "..."}                     │ │  │
│   │   └───────────────────────────────────────────────────────────────┘ │  │
│   │                                                                      │  │
│   │   + Meta-cognition (uncertainty detection)                          │  │
│   │   + Self-reflection on previous steps                               │  │
│   └─────────────────────────────────┬───────────────────────────────────┘  │
│                                     │                                       │
│                        ┌────────────┴────────────┐                         │
│                        │                         │                         │
│                        ▼                         ▼                         │
│              ┌─────────────────┐       ┌─────────────────┐                 │
│              │   Tool Call     │       │ Direct Response │                 │
│              │   Requested     │       │    (Done)       │                 │
│              └────────┬────────┘       └─────────────────┘                 │
│                       │                                                     │
│                       ▼                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                        ACT PHASE                                     │  │
│   │                                                                      │  │
│   │   ┌───────────────────────────────────────────────────────────────┐ │  │
│   │   │                   TOOL ORCHESTRATOR                           │ │  │
│   │   │                                                               │ │  │
│   │   │   1. Security Check ──▶ Permission verification               │ │  │
│   │   │   2. Validation ──▶ Argument type/range checking              │ │  │
│   │   │   3. Execution ──▶ Deterministic tool call                    │ │  │
│   │   │   4. Result Capture ──▶ Success/failure + output              │ │  │
│   │   └───────────────────────────────────────────────────────────────┘ │  │
│   │                                                                      │  │
│   │   Fallback Strategy:                                                │  │
│   │   ┌───────────────────────────────────────────────────────────────┐ │  │
│   │   │  Try primary tool ──▶ On failure ──▶ Try alternative tool    │ │  │
│   │   │                                  ──▶ On failure ──▶ Report   │ │  │
│   │   └───────────────────────────────────────────────────────────────┘ │  │
│   └─────────────────────────────────┬───────────────────────────────────┘  │
│                                     │                                       │
│                                     ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      REFLECT PHASE                                   │  │
│   │                                                                      │  │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐ │  │
│   │   │   Store     │  │  Hebbian    │  │     Update Iteration        │ │  │
│   │   │  Result in  │  │   Self-     │  │     Counter                 │ │  │
│   │   │   Memory    │  │ Correction  │  │     (max: 10)               │ │  │
│   │   └─────────────┘  └─────────────┘  └─────────────────────────────┘ │  │
│   │                                                                      │  │
│   │   If iteration < 10 and more work needed:                           │  │
│   │       └──▶ Return to THINK PHASE                                    │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Agent Loop Implementation (loop.py)

```python
class AgentLoop:
    MAX_ITERATIONS = 10
    
    async def run(self, user_query: str) -> str:
        context = await self._observe(user_query)
        
        for iteration in range(self.MAX_ITERATIONS):
            # THINK: Ask LLM what to do
            llm_response = await self._think(context, iteration)
            
            if llm_response.has_final_response:
                return llm_response.response
            
            if llm_response.has_tool_call:
                # ACT: Execute the tool
                result = await self._act(llm_response.tool_call)
                
                # REFLECT: Learn from result
                await self._reflect(result, llm_response.thought)
                
                # Add result to context for next iteration
                context.add_tool_result(result)
        
        return "I've reached my iteration limit. Here's what I found..."
```

---

## LLM Brain Integration

AURA's LLM Manager handles multiple AI models for different tasks:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LLM MANAGER ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                            ┌─────────────────┐                              │
│                            │   LLM MANAGER   │                              │
│                            │                 │                              │
│                            │  Model Router   │                              │
│                            │  Token Counter  │                              │
│                            │  Rate Limiter   │                              │
│                            └────────┬────────┘                              │
│                                     │                                       │
│          ┌──────────────────────────┼──────────────────────────┐           │
│          │                          │                          │           │
│          ▼                          ▼                          ▼           │
│  ┌───────────────┐        ┌─────────────────┐        ┌───────────────────┐ │
│  │  LOCAL LLM    │        │   SPEECH (STT)  │        │   SPEECH (TTS)    │ │
│  │               │        │                 │        │                   │ │
│  │  llama.cpp    │        │  Whisper.cpp    │        │   Piper TTS       │ │
│  │  server       │        │  (local)        │        │   (local)         │ │
│  │               │        │                 │        │                   │ │
│  │  Supported:   │        │  Models:        │        │  Voices:          │ │
│  │  - Llama 3    │        │  - tiny         │        │  - en_US          │ │
│  │  - Mistral    │        │  - base         │        │  - custom         │ │
│  │  - Phi-3      │        │  - small        │        │                   │ │
│  │  - Qwen       │        │                 │        │                   │ │
│  │               │        │  Optimized for  │        │  ~50ms latency    │ │
│  │  Q4/Q5/Q8     │        │  mobile ARM     │        │  on mobile        │ │
│  │  quantization │        │                 │        │                   │ │
│  └───────────────┘        └─────────────────┘        └───────────────────┘ │
│          │                                                                  │
│          │ Fallback (if local unavailable)                                 │
│          ▼                                                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                       CLOUD FALLBACK (Optional)                      │   │
│  │                                                                      │   │
│  │   ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│  │   │   OpenAI     │  │   Anthropic  │  │   Google Gemini          │  │   │
│  │   │   GPT-4      │  │   Claude     │  │   (via API)              │  │   │
│  │   └──────────────┘  └──────────────┘  └──────────────────────────┘  │   │
│  │                                                                      │   │
│  │   Only used when:                                                   │   │
│  │   - Local model unavailable                                         │   │
│  │   - Complex task requiring larger model                             │   │
│  │   - User explicitly requests cloud                                  │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  NEURAL-AWARE ROUTER:                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                      │   │
│  │   Query ──▶ Complexity Analysis ──▶ Context Size ──▶ Model Selection │   │
│  │                                                                      │   │
│  │   Simple query + small context  ──▶ Small local model (fast)        │   │
│  │   Complex query + large context ──▶ Large local model (accurate)    │   │
│  │   Very complex + critical       ──▶ Cloud fallback (if enabled)     │   │
│  │                                                                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tool Schema Format (JSON)

Tools are exposed to the LLM as JSON schemas:

```json
{
  "tools": [
    {
      "name": "send_message",
      "description": "Send a message via SMS, WhatsApp, or Telegram",
      "parameters": {
        "type": "object",
        "properties": {
          "recipient": {"type": "string", "description": "Contact name or number"},
          "message": {"type": "string", "description": "Message content"},
          "platform": {"type": "string", "enum": ["sms", "whatsapp", "telegram"]}
        },
        "required": ["recipient", "message"]
      }
    },
    {
      "name": "set_reminder",
      "description": "Set a reminder for a specific time",
      "parameters": {
        "type": "object",
        "properties": {
          "title": {"type": "string"},
          "datetime": {"type": "string", "format": "date-time"},
          "priority": {"type": "string", "enum": ["low", "medium", "high"]}
        },
        "required": ["title", "datetime"]
      }
    }
  ]
}
```

---

## Data Flow Diagrams

### Query Processing Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        QUERY PROCESSING FLOW                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   USER INPUT                                                                │
│       │                                                                     │
│       │  "Remind me to call mom tomorrow at 3pm"                           │
│       ▼                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      INPUT PROCESSING                                │  │
│   │                                                                      │  │
│   │   ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │  │
│   │   │   Voice?    │───▶│  Whisper    │───▶│   Text Query            │ │  │
│   │   │   (Audio)   │ Y  │    STT      │    │                         │ │  │
│   │   └──────┬──────┘    └─────────────┘    │   "Remind me to call    │ │  │
│   │          │ N                            │    mom tomorrow at 3pm" │ │  │
│   │          └──────────────────────────────┴─────────────────────────┘ │  │
│   └─────────────────────────────────┬───────────────────────────────────┘  │
│                                     │                                       │
│                                     ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                     CONTEXT ENRICHMENT                               │  │
│   │                                                                      │  │
│   │   Query ──┬──▶ Memory Retrieval ──▶ "User's mom is Sarah"          │  │
│   │           │                                                          │  │
│   │           ├──▶ User Profile ──▶ "User prefers SMS for family"       │  │
│   │           │                                                          │  │
│   │           ├──▶ Context Detection ──▶ "Currently at work"            │  │
│   │           │                                                          │  │
│   │           └──▶ Time Context ──▶ "Tomorrow = 2024-01-16"             │  │
│   └─────────────────────────────────┬───────────────────────────────────┘  │
│                                     │                                       │
│                                     ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                        LLM REASONING                                 │  │
│   │                                                                      │  │
│   │   Enriched Query + Tool Schemas ──▶ LLM                             │  │
│   │                                                                      │  │
│   │   LLM Output:                                                       │  │
│   │   {                                                                  │  │
│   │     "thought": "User wants a reminder to call their mom Sarah...",  │  │
│   │     "tool": "set_reminder",                                         │  │
│   │     "args": {                                                        │  │
│   │       "title": "Call mom (Sarah)",                                  │  │
│   │       "datetime": "2024-01-16T15:00:00",                            │  │
│   │       "priority": "medium"                                          │  │
│   │     }                                                                │  │
│   │   }                                                                  │  │
│   └─────────────────────────────────┬───────────────────────────────────┘  │
│                                     │                                       │
│                                     ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      TOOL EXECUTION                                  │  │
│   │                                                                      │  │
│   │   Tool Orchestrator:                                                │  │
│   │   1. Security check: ✓ set_reminder allowed                        │  │
│   │   2. Validation: ✓ datetime valid, title present                   │  │
│   │   3. Execute: Create reminder in Android                           │  │
│   │   4. Result: {"success": true, "reminder_id": "rem_123"}           │  │
│   └─────────────────────────────────┬───────────────────────────────────┘  │
│                                     │                                       │
│                                     ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                     RESPONSE GENERATION                              │  │
│   │                                                                      │  │
│   │   LLM (with tool result) ──▶ "I've set a reminder to call your     │  │
│   │                               mom Sarah tomorrow at 3:00 PM."       │  │
│   │                                                                      │  │
│   │   ┌─────────────┐    ┌─────────────┐                               │  │
│   │   │   Voice?    │───▶│  Piper TTS  │───▶ Audio Response            │  │
│   │   │   Output    │ Y  │             │                                │  │
│   │   └──────┬──────┘    └─────────────┘                               │  │
│   │          │ N                                                        │  │
│   │          └──────────────────────────▶ Text Response                │  │
│   └─────────────────────────────────┬───────────────────────────────────┘  │
│                                     │                                       │
│                                     ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      LEARNING & MEMORY                               │  │
│   │                                                                      │  │
│   │   Store in Episodic Memory:                                         │  │
│   │   - Event: reminder set for calling mom                            │  │
│   │   - Time: now                                                       │  │
│   │   - Context: at work                                                │  │
│   │   - Outcome: success                                                │  │
│   │                                                                      │  │
│   │   Hebbian Learning:                                                 │  │
│   │   - Strengthen: "mom" ←→ "Sarah" synapse                           │  │
│   │   - Strengthen: "call mom" ←→ "set_reminder" association           │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proactive Behavior Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       PROACTIVE BEHAVIOR FLOW                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   BACKGROUND MONITORING (Tier 2)                                            │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                    PROACTIVE ENGINE                                  │  │
│   │                                                                      │  │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────┐ │  │
│   │   │   Event     │  │    Life     │  │      Calendar               │ │  │
│   │   │  Tracker    │  │  Tracker    │  │      Monitor                │ │  │
│   │   └──────┬──────┘  └──────┬──────┘  └────────────┬────────────────┘ │  │
│   │          │                │                      │                  │  │
│   │          └────────────────┴──────────────────────┘                  │  │
│   │                           │                                          │  │
│   │                           ▼                                          │  │
│   │                  ┌─────────────────┐                                │  │
│   │                  │ Event Detected  │                                │  │
│   │                  │                 │                                │  │
│   │                  │ • Meeting in 1h │                                │  │
│   │                  │ • Low battery   │                                │  │
│   │                  │ • Traffic alert │                                │  │
│   │                  └────────┬────────┘                                │  │
│   └───────────────────────────┼──────────────────────────────────────────┘  │
│                               │                                             │
│                               ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                  PROACTIVITY CONTROLLER                              │  │
│   │                                                                      │  │
│   │   "Should I notify the user?"                                       │  │
│   │                                                                      │  │
│   │   Factors:                                                          │  │
│   │   ┌─────────────────────────────────────────────────────────────┐   │  │
│   │   │ • Event importance:     High (meeting) ✓                    │   │  │
│   │   │ • User context:         Not busy ✓                          │   │  │
│   │   │ • User preferences:     Wants meeting reminders ✓           │   │  │
│   │   │ • Recent notifications: None in last hour ✓                 │   │  │
│   │   │ • Time of day:          Working hours ✓                     │   │  │
│   │   └─────────────────────────────────────────────────────────────┘   │  │
│   │                                                                      │  │
│   │   Decision: NOTIFY                                                  │  │
│   │                                                                      │  │
│   │   ┌─────────────────────────────────────────────────────────────┐   │  │
│   │   │ Conditions to NOT notify:                                   │   │  │
│   │   │ • User in meeting (DND mode)                                │   │  │
│   │   │ • Night time (sleep hours)                                  │   │  │
│   │   │ • Too many recent notifications                             │   │  │
│   │   │ • Low-importance event                                      │   │  │
│   │   │ • User disabled this notification type                      │   │  │
│   │   └─────────────────────────────────────────────────────────────┘   │  │
│   └─────────────────────────────┬───────────────────────────────────────┘  │
│                                 │                                           │
│                                 ▼                                           │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                     NOTIFICATION DELIVERY                            │  │
│   │                                                                      │  │
│   │   ┌────────────────┐  ┌────────────────┐  ┌──────────────────────┐  │  │
│   │   │ Floating       │  │    Termux      │  │   Cinematic          │  │  │
│   │   │   Bubble       │  │   Widget       │  │    Moment            │  │  │
│   │   │ (quick glance) │  │ (persistent)   │  │ (important events)   │  │  │
│   │   └────────────────┘  └────────────────┘  └──────────────────────┘  │  │
│   │                                                                      │  │
│   │   "You have a meeting with the design team in 1 hour.               │  │
│   │    Would you like me to prepare any notes?"                         │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Interactions

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        MODULE INTERACTION MAP                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│                              main.py                                        │
│                          (AuraProduction)                                   │
│                                │                                            │
│            ┌───────────────────┼───────────────────┐                       │
│            │                   │                   │                       │
│            ▼                   ▼                   ▼                       │
│      ┌──────────┐       ┌──────────┐       ┌──────────────┐               │
│      │   LLM    │◀─────▶│  Agent   │◀─────▶│    Memory    │               │
│      │ Manager  │       │   Loop   │       │ Coordinator  │               │
│      └────┬─────┘       └────┬─────┘       └──────┬───────┘               │
│           │                  │                    │                        │
│           │                  │         ┌─────────┼─────────┐              │
│           │                  │         │         │         │              │
│           │                  │         ▼         ▼         ▼              │
│           │                  │    ┌────────┐ ┌────────┐ ┌────────┐        │
│           │                  │    │ Neural │ │Episodic│ │Semantic│        │
│           │                  │    │ Memory │ │ Memory │ │ Memory │        │
│           │                  │    └────────┘ └────────┘ └────────┘        │
│           │                  │                                             │
│           │                  │                                             │
│           │    ┌─────────────┴─────────────┐                              │
│           │    │                           │                              │
│           │    ▼                           ▼                              │
│           │  ┌──────────────┐      ┌──────────────────┐                   │
│           │  │     Tool     │      │    Neural        │                   │
│           │  │ Orchestrator │      │    Systems       │                   │
│           │  └──────┬───────┘      └────────┬─────────┘                   │
│           │         │                       │                             │
│           │         │              ┌────────┼────────┐                    │
│           │         │              │        │        │                    │
│           │         ▼              ▼        ▼        ▼                    │
│           │  ┌──────────────┐  ┌────────┐ ┌────────┐ ┌────────┐          │
│           │  │     Tool     │  │Planner │ │ Router │ │Hebbian │          │
│           │  │   Registry   │  │        │ │        │ │Correct │          │
│           │  └──────┬───────┘  └────────┘ └────────┘ └────────┘          │
│           │         │                                                     │
│           │         │                                                     │
│           │  ┌──────┴────────────────┐                                   │
│           │  │                       │                                   │
│           │  ▼                       ▼                                   │
│           │  ┌──────────────┐  ┌──────────────┐                          │
│           │  │   Android    │  │    Base      │                          │
│           │  │    Tools     │  │    Tools     │                          │
│           │  └──────────────┘  └──────────────┘                          │
│           │                                                               │
│           │                                                               │
│           ▼                                                               │
│      ┌─────────────────────────────────────────────────────────────────┐ │
│      │                    BACKGROUND SERVICES                          │ │
│      │                                                                  │ │
│      │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐  │ │
│      │  │ Proactive  │  │Healthcare  │  │ Social     │  │   Life   │  │ │
│      │  │  Engine    │  │  Agent     │  │  Life      │  │ Tracker  │  │ │
│      │  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └────┬─────┘  │ │
│      │        │               │               │              │        │ │
│      │        └───────────────┴───────────────┴──────────────┘        │ │
│      │                               │                                 │ │
│      │                               ▼                                 │ │
│      │                    ┌─────────────────────┐                     │ │
│      │                    │   Proactivity       │                     │ │
│      │                    │    Controller       │                     │ │
│      │                    │  (when to notify)   │                     │ │
│      │                    └─────────────────────┘                     │ │
│      └─────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│                                                                          │
│      ┌─────────────────────────────────────────────────────────────────┐ │
│      │                        MOBILE UI                                │ │
│      │                                                                  │ │
│      │  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌──────────┐  │ │
│      │  │Aura Space  │  │  Termux    │  │ Floating   │  │Character │  │ │
│      │  │  Server    │  │  Widget    │  │  Bubbles   │  │  Sheet   │  │ │
│      │  └────────────┘  └────────────┘  └────────────┘  └──────────┘  │ │
│      └─────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│                                                                          │
│      ┌─────────────────────────────────────────────────────────────────┐ │
│      │                       SECURITY LAYER                            │ │
│      │                                                                  │ │
│      │  All modules must pass through security for sensitive ops       │ │
│      │  ┌────────────┐  ┌────────────┐  ┌────────────────────────────┐ │ │
│      │  │   Auth     │  │Permissions │  │     Privacy Guard          │ │ │
│      │  └────────────┘  └────────────┘  └────────────────────────────┘ │ │
│      └─────────────────────────────────────────────────────────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Interaction Patterns

| From | To | Purpose |
|------|-----|---------|
| `main.py` | `agent/loop.py` | Dispatches user queries |
| `agent/loop.py` | `llm/manager.py` | Gets LLM completions |
| `agent/loop.py` | `memory/memory_coordinator.py` | Retrieves/stores memories |
| `agent/loop.py` | `core/tool_orchestrator.py` | Executes tools |
| `core/tool_orchestrator.py` | `tools/registry.py` | Looks up tool handlers |
| `core/tool_orchestrator.py` | `security/` | Validates permissions |
| `services/proactive_engine.py` | `core/proactivity_controller.py` | Decides when to notify |
| `memory/memory_coordinator.py` | `memory/neural_memory.py` | Working memory ops |
| `memory/memory_coordinator.py` | `memory/episodic_memory.py` | Event storage |
| `memory/memory_coordinator.py` | `memory/semantic_memory.py` | Knowledge queries |

---

## Mobile Constraints & RAM Budget

### Target Device: 8GB RAM Android (Termux)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          8GB RAM BUDGET ALLOCATION                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ████████████████████████████████████████████████████████████████████████  │
│   │                         8 GB TOTAL                                    │ │
│   ████████████████████████████████████████████████████████████████████████  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ ANDROID OS + SYSTEM APPS                           ~2.5 - 3.0 GB    │  │
│   │ ██████████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░  │  │
│   │ • Android framework                                                 │  │
│   │ • Background apps                                                   │  │
│   │ • System services                                                   │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ TERMUX + PYTHON RUNTIME                            ~200 - 400 MB    │  │
│   │ ████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │  │
│   │ • Termux environment                                                │  │
│   │ • Python interpreter                                                │  │
│   │ • Libraries (numpy, etc.)                                          │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ LOCAL LLM (llama.cpp)                              ~2.0 - 4.0 GB    │  │
│   │ █████████████████████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░  │  │
│   │ • Q4 quantized 7B model: ~4 GB                                     │  │
│   │ • Q4 quantized 3B model: ~2 GB                                     │  │
│   │ • Inference context: ~500 MB                                       │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ AURA APPLICATION                                   ~250 - 500 MB    │  │
│   │ ██████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │  │
│   │                                                                     │  │
│   │   Tier 0 (Boot):           ~30-50 MB                               │  │
│   │   ├─ LLM Manager:          ~10 MB                                  │  │
│   │   ├─ Agent Loop:           ~5 MB                                   │  │
│   │   ├─ Memory (minimal):     ~10 MB                                  │  │
│   │   ├─ Tools:                ~3 MB                                   │  │
│   │   └─ Security:             ~2 MB                                   │  │
│   │                                                                     │  │
│   │   Tier 1 (First Query):    ~50-100 MB                              │  │
│   │   ├─ Context Provider:     ~20 MB                                  │  │
│   │   ├─ Personality:          ~15 MB                                  │  │
│   │   ├─ Neural Systems:       ~30 MB                                  │  │
│   │   ├─ Session:              ~10 MB                                  │  │
│   │   └─ Learning:             ~25 MB                                  │  │
│   │                                                                     │  │
│   │   Tier 2 (Background):     ~100-200 MB                             │  │
│   │   ├─ Proactive Engine:     ~30 MB                                  │  │
│   │   ├─ Healthcare Agent:     ~40 MB                                  │  │
│   │   ├─ Social Life:          ~30 MB                                  │  │
│   │   ├─ Mobile UI:            ~50 MB                                  │  │
│   │   └─ Other Services:       ~50 MB                                  │  │
│   │                                                                     │  │
│   │   Memory Systems:          ~50-150 MB (scales with data)           │  │
│   │   ├─ Neural Memory:        ~20-50 MB (1000 neurons)                │  │
│   │   ├─ Episodic Memory:      ~10-30 MB                               │  │
│   │   ├─ Semantic Memory:      ~15-50 MB                               │  │
│   │   └─ Vector Store:         ~5-20 MB                                │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ WHISPER STT (if active)                            ~100 - 500 MB    │  │
│   │ ████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │  │
│   │ • tiny model: ~100 MB                                              │  │
│   │ • base model: ~200 MB                                              │  │
│   │ • small model: ~500 MB                                             │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ PIPER TTS (if active)                              ~50 - 100 MB     │  │
│   │ ██░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │  │
│   │ • Voice model + runtime                                            │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │ BUFFER / HEADROOM                                  ~500 MB - 1 GB   │  │
│   │ ████████████░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░  │  │
│   │ • OS memory pressure handling                                      │  │
│   │ • Burst allocation during inference                                │  │
│   │ • Background app switching                                         │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   MEMORY OPTIMIZATION STRATEGIES:                                           │
│                                                                             │
│   1. Lazy Loading: Only load modules when needed (tiered init)             │
│   2. Unload Unused: STT/TTS unloaded when not in use                       │
│   3. Model Quantization: Q4/Q5 quantization for LLM                        │
│   4. Memory Pooling: Reuse allocations for neural memory                   │
│   5. Disk Offload: Swap inactive memories to disk                          │
│   6. Batch Processing: Process in chunks to limit peak usage               │
│   7. Connection Pooling: Reuse HTTP/DB connections                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Recommended Model Configurations

| RAM Available | LLM Model | STT Model | Notes |
|---------------|-----------|-----------|-------|
| 4 GB free | Phi-3 Mini Q4 (2GB) | Whisper tiny | Minimal viable |
| 5 GB free | Llama 3 8B Q4 (4GB) | Whisper tiny | Balanced |
| 6 GB free | Llama 3 8B Q5 (5GB) | Whisper base | Recommended |
| 7+ GB free | Llama 3 8B Q8 (7GB) | Whisper small | Best quality |

---

## Security Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           SECURITY ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌─────────────────────────────────────────────────────────────────────┐  │
│   │                      SECURITY LAYERS                                 │  │
│   │                                                                      │  │
│   │   ┌─────────────────────────────────────────────────────────────┐   │  │
│   │   │ Layer 1: AUTHENTICATION                                     │   │  │
│   │   │                                                             │   │  │
│   │   │ • Biometric (fingerprint, face)                            │   │  │
│   │   │ • PIN/Password fallback                                    │   │  │
│   │   │ • Session tokens with expiry                               │   │  │
│   │   └─────────────────────────────────────────────────────────────┘   │  │
│   │                              │                                       │  │
│   │                              ▼                                       │  │
│   │   ┌─────────────────────────────────────────────────────────────┐   │  │
│   │   │ Layer 2: PERMISSIONS                                        │   │  │
│   │   │                                                             │   │  │
│   │   │ • Tool-level permissions (send_message, access_files)      │   │  │
│   │   │ • Data-level permissions (contacts, calendar, location)    │   │  │
│   │   │ • Action-level permissions (delete, modify, share)         │   │  │
│   │   │                                                             │   │  │
│   │   │ Permission Matrix:                                          │   │  │
│   │   │ ┌─────────────┬──────┬───────┬────────┬─────────┐          │   │  │
│   │   │ │ Tool        │ Read │ Write │ Delete │ Network │          │   │  │
│   │   │ ├─────────────┼──────┼───────┼────────┼─────────┤          │   │  │
│   │   │ │ send_msg    │  ✓   │   ✓   │   ✗    │    ✓    │          │   │  │
│   │   │ │ read_file   │  ✓   │   ✗   │   ✗    │    ✗    │          │   │  │
│   │   │ │ web_search  │  ✗   │   ✗   │   ✗    │    ✓    │          │   │  │
│   │   │ │ delete_file │  ✓   │   ✗   │   ✓    │    ✗    │          │   │  │
│   │   │ └─────────────┴──────┴───────┴────────┴─────────┘          │   │  │
│   │   └─────────────────────────────────────────────────────────────┘   │  │
│   │                              │                                       │  │
│   │                              ▼                                       │  │
│   │   ┌─────────────────────────────────────────────────────────────┐   │  │
│   │   │ Layer 3: PRIVACY GUARD                                      │   │  │
│   │   │                                                             │   │  │
│   │   │ • PII detection and redaction                              │   │  │
│   │   │ • Sensitive data classification                            │   │  │
│   │   │ • Data retention policies                                  │   │  │
│   │   │ • Export/sharing controls                                  │   │  │
│   │   │                                                             │   │  │
│   │   │ Protected Data Types:                                       │   │  │
│   │   │ • Passwords, API keys, tokens                              │   │  │
│   │   │ • Financial information                                     │   │  │
│   │   │ • Health data                                               │   │  │
│   │   │ • Location history                                          │   │  │
│   │   │ • Private conversations                                     │   │  │
│   │   └─────────────────────────────────────────────────────────────┘   │  │
│   │                              │                                       │  │
│   │                              ▼                                       │  │
│   │   ┌─────────────────────────────────────────────────────────────┐   │  │
│   │   │ Layer 4: ENCRYPTION                                         │   │  │
│   │   │                                                             │   │  │
│   │   │ • At-rest: AES-256 for stored memories                     │   │  │
│   │   │ • In-transit: TLS 1.3 for all network calls                │   │  │
│   │   │ • Key management: Android Keystore                         │   │  │
│   │   └─────────────────────────────────────────────────────────────┘   │  │
│   │                                                                      │  │
│   └─────────────────────────────────────────────────────────────────────┘  │
│                                                                             │
│   TOOL EXECUTION SECURITY FLOW:                                             │
│                                                                             │
│   Tool Request ──▶ Auth Check ──▶ Permission Check ──▶ Privacy Scan        │
│                         │              │                    │               │
│                         ▼              ▼                    ▼               │
│                      [DENY]        [DENY]              [REDACT]             │
│                         │              │                    │               │
│                         └──────────────┴────────────────────┘               │
│                                        │                                    │
│                                        ▼                                    │
│                              ┌─────────────────┐                           │
│                              │    EXECUTE      │                           │
│                              │   (if passed)   │                           │
│                              └─────────────────┘                           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Quick Reference

### File Locations

| Component | File | Lines |
|-----------|------|-------|
| Entry Point | `src/main.py` | 1229 |
| Agent Loop | `src/agent/loop.py` | 959 |
| Memory Coordinator | `src/memory/memory_coordinator.py` | 192 |
| Neural Memory | `src/memory/neural_memory.py` | 728 |
| Episodic Memory | `src/memory/episodic_memory.py` | 454 |
| Semantic Memory | `src/memory/semantic_memory.py` | 541 |
| LLM Manager | `src/llm/manager.py` | 568 |

### Key Classes

| Class | Purpose |
|-------|---------|
| `AuraProduction` | Main orchestrator, tiered initialization |
| `AgentLoop` | ReAct reasoning loop |
| `MemoryCoordinator` | Unified memory interface |
| `NeuralMemory` | Bio-inspired working memory |
| `EpisodicMemory` | Event storage and retrieval |
| `SemanticMemory` | Knowledge graph and concepts |
| `LLMManager` | LLM/STT/TTS routing |
| `ToolOrchestrator` | Deterministic tool execution |
| `ProactivityController` | Notification decisions |

### Design Principles

1. **Local-First**: All core functionality works offline
2. **Privacy-Preserving**: User data never leaves device (unless explicitly shared)
3. **Memory-Efficient**: Tiered loading, lazy initialization, aggressive cleanup
4. **Bio-Inspired**: Memory systems modeled on human cognition
5. **Autonomous**: Proactive assistance without being intrusive

---

*Generated for AURA v3 | Target: 8GB RAM Mobile (Termux/Android)*
