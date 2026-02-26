# AURA v3 Phase 2: Refinement & Resurrection Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix all P0 bugs to make AURA bootable, optimize performance for 8GB mobile, consolidate duplicate systems, and make unrealized visions real with proper algorithms — all WITHOUT reducing AURA's power or changing its architecture.

**Architecture:** Preserve existing ReAct loop, memory hierarchy, personality system, proactive services, and all rich capabilities. Every change is surgical — fix what's broken, optimize what's slow, unify what's duplicated, and make real what was placeholder.

**Tech Stack:** Python 3.8+, SQLite, llama.cpp, asyncio, Termux (Android)

**Creator's Directive:** AURA is life-changing AND practical. It reads messages, manages routines, adapts to ANY user (CEO or teacher). Rich default capabilities show people what a living smart entity can do. NEVER reduce power anywhere. Every fix must make AURA MORE capable, not less.

---

## Execution Strategy

5 waves, executed sequentially. Each wave is independently testable and committable.
Work smart and slowly with full dedication — no rushing.

| Wave | Name | Goal | Estimated Tasks |
|------|------|------|-----------------|
| 1 | Make AURA Alive | Fix P0 boot/crash bugs so AURA can start | 10 |
| 2 | Stop the Bleeding | Fix memory/performance P0s | 10 |
| 3 | Unify | Consolidate 6 duplicate systems (keep ALL capabilities) | 6 |
| 4 | Make Visions Real | Redesign 7 unrealized concepts with real algorithms | 7 |
| 5 | Harden | Tests, installation, docs | 5 |

**Constraint: No power reduction.** Every change ADDS capability or FIXES broken capability. Nothing is removed unless it's dead code that was never reachable.

---

## Wave 1: Make AURA Alive (P0 Boot/Crash Bugs)

These bugs prevent AURA from starting or crash core systems on first use. Must be fixed first.

### Task 1.1: Fix main.py import crashes (get_agent, initialize_relationship_system)

**Problem:** `main.py:200` imports `get_agent` from `src.agent` — doesn't exist. `main.py:282` imports `initialize_relationship_system` — doesn't exist. AURA crashes on boot.

**Files:**
- Modify: `src/agent/__init__.py` (add missing exports)
- Modify: `src/agent/loop.py` (add `get_agent` factory function)
- Modify: `src/agent/relationship_system.py` (verify `initialize_relationship_system` exists, add if missing)
- Test: Manual boot test `python -c "from src.agent import get_agent, initialize_relationship_system"`

**Approach:**
1. Read `src/agent/loop.py` and `src/agent/relationship_system.py` to understand what `get_agent` and `initialize_relationship_system` should return
2. Create proper factory functions that match what `main.py` expects
3. Export them from `__init__.py`
4. Verify imports work

**Key rule:** `get_agent` should return a configured `ReActAgent` or `AgentFactory`. `initialize_relationship_system` should return the relationship system instance. Study how `main.py` uses the return values to determine exact signatures.

### Task 1.2: Fix main.py _termux_bridge initialization order

**Problem:** `main.py:558` passes `self._termux_bridge` to `IntelligentCallManager`, but `_termux_bridge` is created in Phase 6 (`_init_addons`) while `IntelligentCallManager` is created in Phase 5.5 (`_init_proactive_services`). `_termux_bridge` is always None.

**Files:**
- Modify: `src/main.py` — move `_termux_bridge` initialization before `_init_proactive_services`, OR lazy-inject it after addons init

**Approach:**
1. Find where `_termux_bridge` is created in `_init_addons()`
2. Either move that single initialization earlier (before Phase 5.5), or add a post-init step that injects `_termux_bridge` into the call manager after Phase 6
3. Verify `IntelligentCallManager` has a setter for `termux_bridge` or accepts late binding

### Task 1.3: Fix error_recovery.py typo crash

**Problem:** `src/utils/error_recovery.py:219` uses `attemptions` instead of `attempts`. NameError crashes the entire error recovery system — meaning ALL error recovery throughout AURA is non-functional.

**Files:**
- Modify: `src/utils/error_recovery.py:219` — change `attemptions` to `attempts`

**Step:** Single character fix. `self._calculate_backoff(attemptions)` → `self._calculate_backoff(attempts)`

### Task 1.4: Fix security.py b64decode→b64encode bug

**Problem:** `src/security/security.py:194` uses `base64.b64decode(salt)` when storing the salt, but salt is raw bytes that should be ENCODED to base64 for JSON storage. This means no user can ever set a password — authentication setup crashes.

**Files:**
- Modify: `src/security/security.py:194` — change `b64decode` to `b64encode`

**Step:** `"salt": base64.b64decode(salt).decode()` → `"salt": base64.b64encode(salt).decode()`

### Task 1.5: Fix task_engine.py forward reference crash

**Problem:** `src/tasks/task_engine.py:1015` references `AgentResponse` and `AgentState` which aren't imported — NameError breaks agent integration with the task engine.

**Files:**
- Modify: `src/tasks/task_engine.py` — add proper imports or use string annotations
- Read: `src/agent/loop.py` to confirm exact class names

**Approach:**
1. Find the exact line where `AgentResponse`/`AgentState` are referenced
2. Add `from src.agent.loop import AgentResponse, AgentState` at top, or use `TYPE_CHECKING` guard
3. Verify no circular imports

### Task 1.6: Fix security_layers.py lock() method overwrite

**Problem:** In `src/core/security_layers.py`, the `lock()` method assigns `self._is_locked = True`, but `_is_locked` is also a method name elsewhere — the boolean overwrites the method, breaking auth checks.

**Files:**
- Modify: `src/core/security_layers.py` — rename the boolean to `self._locked_state` or similar, keeping the method `_is_locked()` intact

**Approach:**
1. Read the class to find both the method `_is_locked()` and the property assignment
2. Rename the boolean flag to avoid collision (e.g., `self._lock_engaged`)
3. Update all references to the boolean
4. Verify the method is preserved

### Task 1.7: Fix session/manager.py — sessions can't reload

**Problem:** `list_keys()` returns hashes but `load_session()` expects IDs — sessions written to disk can never be loaded back. Also hardcoded salt `b"AURA_SESSION_SALT_v1"`.

**Files:**
- Modify: `src/session/manager.py` — fix `list_keys()` to return usable identifiers, fix `load_session()` to accept what `list_keys()` returns

**Approach:**
1. Read both methods to understand the key/hash mismatch
2. Decide: either store a mapping of ID→hash, or use the ID directly as filename
3. Fix the salt to be configurable (but keep default for backward compatibility)

### Task 1.8: Fix context_provider.py non-existent attribute

**Problem:** `src/context/context_provider.py:426` references `ctx.is_in_meeting` which doesn't exist on `FullContext` dataclass — throws AttributeError at runtime.

**Files:**
- Modify: `src/context/context_provider.py:426` — either add `is_in_meeting` to FullContext dataclass, or use `getattr(ctx, 'is_in_meeting', False)`

### Task 1.9: Fix onboarding_service.py truncated file

**Problem:** `src/services/onboarding_service.py` is truncated/corrupt — service is broken.

**Files:**
- Modify: `src/services/onboarding_service.py` — read what exists, determine what's missing, complete the file

### Task 1.10: Remove hardcoded Telegram bot token from version control

**Problem:** **CRITICAL SECURITY** — `scripts/production_setup.py:513` and `scripts/auto_setup.py:64` contain a hardcoded Telegram bot token `8504361506:AAGMlzkRS3GN0m_kbINviqHlu8Pb4X0DkeY`. This must be revoked and replaced with environment variable references.

**Files:**
- Modify: `scripts/production_setup.py:513` — replace token with `os.environ.get('TELEGRAM_BOT_TOKEN', '')`
- Modify: `scripts/auto_setup.py:64` — same
- Modify: `config/security.yaml` — replace hardcoded token with placeholder

**IMPORTANT:** The actual token must be revoked via Telegram BotFather. Add a note in the commit message.

---

## Wave 2: Stop the Bleeding (Memory & Performance P0s)

These issues cause memory explosions, slow responses, and CPU waste on mobile.

### Task 2.1: Fix agent loop unbounded history

**Problem:** `src/agent/loop.py` — `messages` list in `process()` grows unbounded during ReAct iterations. On complex tasks with many tool calls, this can consume significant memory and make LLM calls progressively slower.

**Files:**
- Modify: `src/agent/loop.py` — add `max_history` parameter, trim oldest messages when exceeded (keeping system prompt always)

**Approach:** Add sliding window: keep system prompt + last N messages. Default N=20 (configurable). This preserves AURA's conversational ability while bounding memory.

### Task 2.2: Fix local_vector_store.py — lazy load vectors

**Problem:** `src/memory/local_vector_store.py` loads ALL vectors into memory on startup. With many memories, this explodes RAM.

**Files:**
- Modify: `src/memory/local_vector_store.py` — implement lazy loading: load index metadata on startup, load actual vectors on-demand with LRU cache

**Approach:**
1. On startup, only load vector count and index metadata
2. On query, load relevant vectors from SQLite using LIMIT
3. Add LRU cache (maxsize configurable, default 1000 vectors) for hot vectors
4. Keep the API identical — callers don't change

### Task 2.3: Fix episodic_memory.py — lazy load embeddings

**Problem:** `src/memory/episodic_memory.py` eagerly loads up to 500 embeddings on initialization.

**Files:**
- Modify: `src/memory/episodic_memory.py` — defer embedding loading to first query, use pagination

**Approach:** Same pattern as 2.2 — load on demand, cache hot embeddings, bound cache size.

### Task 2.4: Fix knowledge_graph.py — O(edges) neighbor lookup

**Problem:** `get_neighbors()` scans all edges linearly. With a growing knowledge graph, this gets progressively slower.

**Files:**
- Modify: `src/memory/knowledge_graph.py` — add SQLite index on edge source/target columns, use indexed queries

**Approach:**
1. Add `CREATE INDEX IF NOT EXISTS idx_edges_source ON edges(source_id)` and same for `target_id`
2. Rewrite `get_neighbors()` to use `WHERE source_id = ? OR target_id = ?` with the index
3. This is a pure optimization — no API change, no capability reduction

### Task 2.5: Fix memory_retrieval.py — batch DB queries

**Problem:** 5+ sequential SQLite round trips per memory query. Each round trip has Python↔SQLite overhead.

**Files:**
- Modify: `src/memory/memory_retrieval.py` — batch queries into 1-2 round trips using UNION or multi-statement

### Task 2.6: Fix coordinator.py — polling → asyncio.Event

**Problem:** `src/agents/coordinator.py` uses polling-based waiting with `asyncio.sleep()` loops. Wastes CPU and adds latency.

**Files:**
- Modify: `src/agents/coordinator.py` — replace polling loops with `asyncio.Event.wait()` and `asyncio.Event.set()`

### Task 2.7: Fix tool_first_planner.py — cache schema context

**Problem:** `src/core/tool_first_planner.py` rebuilds the entire tool schema context for every LLM call. Schema doesn't change between calls.

**Files:**
- Modify: `src/core/tool_first_planner.py` — cache the schema string, invalidate only when tools change

### Task 2.8: Fix background_manager.py — non-blocking psutil

**Problem:** `src/services/background_manager.py` makes blocking `psutil` calls that block the asyncio event loop.

**Files:**
- Modify: `src/services/background_manager.py` — wrap psutil calls in `asyncio.get_event_loop().run_in_executor(None, ...)`

### Task 2.9: Add SQLite connection pooling

**Problem:** 9 SQLite databases across memory subsystems. Each opens/closes connections per operation. Connection setup has overhead.

**Files:**
- Create: `src/utils/db_pool.py` — simple SQLite connection pool (reuse connections per thread)
- Modify: Memory modules to use pooled connections

**Approach:** Create a lightweight connection pool that keeps 1-2 connections per database file open. Use `threading.local()` for thread-safe access. Memory modules call `pool.get_connection(db_path)` instead of `sqlite3.connect()`.

### Task 2.10: Implement lazy loading in main.py

**Problem:** `main.py` eagerly loads ALL 40+ components in `initialize()`. ~100-300 MB at startup before first query. Claimed "lazy loading" doesn't exist.

**Files:**
- Modify: `src/main.py` — implement tiered initialization:
  - **Tier 0 (boot):** LLM, agent loop, basic memory — ~30-50 MB
  - **Tier 1 (first query):** Context, personality, user profile — loaded on first process() call
  - **Tier 2 (on demand):** Proactive services, voice, mobile features — loaded when first used

**Key rule:** ALL features remain available. Nothing is removed. They're just loaded when needed instead of all at boot. AURA is just as powerful — it just starts faster.

---

## Wave 3: Unify (Consolidate Duplicate Systems)

6 duplicate/parallel systems that add confusion and maintenance burden. Consolidation keeps ALL capabilities from both sides.

### Task 3.1: Unify LLM systems (production_llm.py + real_llm.py)

**Problem:** Two complete LLM systems exist side by side, not connected. `production_llm.py` (1,559 lines) is the better one — real RAM calculations, thermal monitoring, model recommendations, streaming. `real_llm.py` (504 lines) is simpler but has some backends `production_llm.py` doesn't.

**Files:**
- Modify: `src/llm/production_llm.py` — ensure it covers all backends from `real_llm.py`
- Modify: `src/llm/manager.py` — point to `production_llm.py` as canonical
- Modify: `src/llm/real_llm.py` — deprecate, redirect to production_llm
- Modify: All files importing from `real_llm` to use the canonical system

**Key rule:** Keep ALL backends (LlamaCpp, Gpt4All, Ollama, Transformers, Mock). `production_llm.py` becomes the single source of truth.

### Task 3.2: Unify tool execution (android.py + handlers.py)

**Problem:** `src/tools/android.py` and `src/tools/handlers.py` are two separate tool execution paths doing the same things differently, not connected to each other.

**Files:**
- Analyze both files to identify unique capabilities in each
- Merge into single tool execution system (keep `handlers.py` as primary, merge android-specific handlers into it)
- Fix registry-handler disconnect (handler fields are all `None`)

### Task 3.3: Unify context systems

**Problem:** Two overlapping context systems that don't reference each other.

**Files:**
- Modify: `src/context/` — merge into single event-driven context system
- Convert polling (10-second intervals) to event-driven where possible

### Task 3.4: Unify voice pipelines

**Problem:** `pipeline.py` (legacy, Telegram) and `real_time_pipeline.py` (production) — two full pipelines maintained simultaneously.

**Files:**
- Modify: `src/voice/` — make `real_time_pipeline.py` the primary, integrate Telegram-specific features from `pipeline.py`
- Remove `stt_updated.py` (incomplete, cuts off mid-function) — keep `stt.py`

### Task 3.5: Unify JSONPlan classes

**Problem:** `src/core/tool_orchestrator.py` and `src/core/neural_validated_planner.py` both define `JSONPlan` classes independently.

**Files:**
- Create single `JSONPlan` definition (in whichever file is more appropriate)
- Import from canonical location in the other file

### Task 3.6: Fix life_tracker.py broken serialization

**Problem:** `src/services/life_tracker.py` has broken serialization — data loss on save/load.

**Files:**
- Modify: `src/services/life_tracker.py` — fix serialization to properly save/load all tracked data

---

## Wave 4: Make Visions Real (Redesign Unrealized Concepts)

These are BRILLIANT concepts that were designed with genuine vision but implemented with placeholder code (random.choice, pass, if-else keyword matching). The goal is to make them REAL with proper algorithms while keeping the concepts intact.

**Creator's directive:** "The neuromorphic engine's 5 sub-agents aren't complexity theater to delete — they're an unrealized vision that needs to be MADE REAL with proper algorithms, not if-else/random.choice()."

### Task 4.1: Neuromorphic Engine — Real Sub-Agent Processing

**Problem:** `src/core/neuromorphic_engine.py` (689 lines) — MultiAgentOrchestrator has 5 sub-agents (ContextAnalyzer, PatternRecognizer, CreativityEngine, MemoryConsolidator, ProactivePlanner) but ALL tick() methods are empty `pass` statements. The concept is brilliant (parallel sub-agents like F.R.I.D.A.Y.), but nothing actually runs.

**Files:**
- Modify: `src/core/neuromorphic_engine.py`

**Approach (first-principles):**
Each sub-agent should have a REAL lightweight algorithm:
1. **ContextAnalyzer**: On tick, analyze recent conversation/context for shifts — use sliding window + change detection (not LLM, just statistical)
2. **PatternRecognizer**: On tick, scan recent interactions for recurring patterns — use frequency counting with time decay
3. **CreativityEngine**: On tick, generate connection ideas between recent topics — use co-occurrence matrix of memory topics
4. **MemoryConsolidator**: On tick, check if working memory needs consolidation to long-term — use importance scoring threshold
5. **ProactivePlanner**: On tick, check time-based triggers AND learned user patterns — replace hardcoded time checks with adaptive scheduling

**Key constraint:** Each tick must complete in <50ms on mobile. No LLM calls in tick. Pure algorithmic.

### Task 4.2: Adaptive Personality — Real Personality-Driven Generation

**Problem:** `src/core/adaptive_personality.py` (541 lines) — Claims "real-time generation based on personality" but IS scripted `random.choice()` from hardcoded lists like `["Hey there!", "Hi!", "Hello!"]`.

**Files:**
- Modify: `src/core/adaptive_personality.py`

**Approach (first-principles):**
1. Define personality as a numerical vector (Big Five dimensions: openness, conscientiousness, extraversion, agreeableness, neuroticism)
2. Each personality dimension influences response style through weights:
   - High openness → more creative/exploratory language
   - High conscientiousness → more structured/organized responses
   - High extraversion → more enthusiastic/social tone
3. Replace `random.choice()` with personality-weighted selection from response templates
4. Add personality evolution — vector shifts based on user interactions over time
5. Store personality vector in SQLite (persistence — currently in-memory only)

### Task 4.3: Conversation Learning — Real Learning from Conversations

**Problem:** `src/core/conversation.py` (417 lines) — `learn_from_conversation()` just logs and does nothing. Prepends random filler phrases.

**Files:**
- Modify: `src/core/conversation.py`

**Approach:**
1. `learn_from_conversation()` should extract: user preferences, topic interests, communication style
2. Store learned facts in memory system (episodic memory with high importance)
3. Replace random filler phrases with context-appropriate intros based on time-of-day + relationship stage + conversation topic

### Task 4.4: AGI Module — Real Algorithmic Intelligence

**Problem:** `src/agi/__init__.py` (482 lines) — Emotional detection is keyword matching, predictions are time-of-day heuristics, goals are list-appending. `LongTermPlanner.learn_habit` is literally `pass`.

**Files:**
- Modify: `src/agi/__init__.py`

**Approach:**
1. **Emotional detection**: Replace keyword lists with valence/arousal scoring using a lightweight emotion lexicon (ANEW-style word list, ~1000 words). Score = average valence of detected emotion words
2. **Predictions**: Replace time-of-day heuristics with Markov chain on observed user behavior sequences. After N observations, predict likely next action given current state
3. **Goal management**: Add priority scoring, progress tracking, deadline awareness
4. **learn_habit**: Implement frequency+time tracking — if action X happens at time Y more than 3 times, flag as habit

### Task 4.5: Capability Gap — Real Strategy Execution

**Problem:** `src/addons/capability_gap.py` (495 lines) — Conceptually brilliant (AURA knowing what it CAN'T do) but all strategies return hardcoded success and `_check_prerequisites` always returns True.

**Files:**
- Modify: `src/addons/capability_gap.py`

**Approach:**
1. `_check_prerequisites` should actually check: is the required tool available? Is the permission granted? Is the required service running?
2. Strategies should execute real logic — if a capability is missing, try alternatives, report honestly, learn what works
3. Track gap history — if the same gap is hit repeatedly, proactively suggest solutions

### Task 4.6: User Profile — Add Persistence

**Problem:** `src/core/user_profile.py` (567 lines) — Big Five psychological profiling works in-memory but has NO persistence. All learning is lost on restart.

**Files:**
- Modify: `src/core/user_profile.py` — add SQLite persistence for profile data
- Create table `user_profiles` with columns for each Big Five dimension + metadata

### Task 4.7: Hebbian Self-Correction — Connect to Working Neural Memory

**Problem:** `src/core/hebbian_self_correction.py` (353 lines) — Depends entirely on neural_memory injection. Without a working neural memory, every method returns None.

**Files:**
- Modify: `src/core/hebbian_self_correction.py` — add graceful standalone mode that works even without neural memory, using a local activation map
- Ensure proper connection when neural memory IS available

---

## Wave 5: Harden (Tests, Installation, Docs)

### Task 5.1: Fix broken async tests

**Problem:** 29 test methods in `test_ui_feelings_meter.py` and `test_ui_inner_voice.py` use deprecated `@asyncio.coroutine`/`yield from` — they pass vacuously (never actually run the test logic).

**Files:**
- Modify: `tests/test_ui_feelings_meter.py` — convert to `async def` + `await`
- Modify: `tests/test_ui_inner_voice.py` — same

### Task 5.2: Add tests for Wave 1 bug fixes

**Files:**
- Modify/Create tests for each P0 fix from Wave 1
- Target: every fixed bug has a regression test

### Task 5.3: Consolidate installation scripts

**Problem:** 8 overlapping scripts with different Python version requirements, different dependencies, different repo names.

**Files:**
- Create: `scripts/install.sh` (unified) — single entry point that detects environment (Termux vs desktop vs Docker) and runs appropriate setup
- Update: `requirements.txt` — single source of truth for dependencies
- Deprecate: old scripts (don't delete — just add deprecation notice at top pointing to unified script)

### Task 5.4: Fix config inconsistencies

**Files:**
- Modify: `config.example.yaml` — align with `config/config.yaml` structure
- Modify: `config/security.yaml` — remove hardcoded token, add documentation comments

### Task 5.5: Update compliance docs to reflect reality

**Files:**
- Modify: `compliance/COMPLIANCE.md` — mark Telegram/WhatsApp as requiring internet (honest documentation)
- Modify: `compliance/PRIVACY.md` — note that 100% offline applies to core AURA, channels may require connectivity
- Remove false claims about SOC 2 progress

---

## Verification Checkpoints

After each wave, run:

```bash
# Wave 1: Boot test
python -c "import asyncio; from src.main import AuraProduction; a = AuraProduction(); asyncio.run(a.initialize())"

# Wave 2: Memory test
python -c "from src.memory import get_neural_memory; m = get_neural_memory(); print('Memory OK')"

# Wave 3: Import unity test
python -c "from src.llm import get_llm_manager; from src.tools.registry import ToolRegistry; print('Unified OK')"

# Wave 4: Feature test
python -c "from src.core.neuromorphic_engine import NeuromorphicEngine; print('Visions OK')"

# Wave 5: Full test suite
python -m pytest tests/ -v
```

---

## Post-Execution

After all waves complete:
1. Run full test suite
2. Verify AURA boots on desktop (simulated mobile constraints)
3. Commit with proper message
4. Push to GitHub
5. Update TECHNICAL_AUDIT_REPORT.md with Phase 2 results
