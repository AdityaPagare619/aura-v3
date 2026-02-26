"""
AURA v3 - NEUROMORPHIC PROCESSING ENGINE
========================================

AURA's Core Processing Architecture
====================================

UNIQUE DESIGN PRINCIPLES (Not copied from OpenClaw):
----------------------------------------------------
1. EVENT-DRIVEN (SNN-inspired): Only processes when triggered
2. SPARSE ACTIVATION: Like brain neurons, not all at once
3. HARDWARE-AWARE: Thermal, RAM, battery optimization at core level
4. PARALLEL SUB-AGENTS: Like F.R.I.D.A.Y.'s sub-robots
5. EMOTIONAL CONTEXT: Like JARVIS/FRIDAY conversation style
6. PROACTIVE THINKING: Self-initiates actions, not just reactive

Mathematical Foundation:
- Event-driven processing: f(t) = Σ spike_i(t) * weight_i
- Sparse attention: Only activate top-k neurons (k << n)
- Thermal budget: E_total <= E_max * (1 - T_current/T_max)
- Memory hierarchy: L1(cache) < L2(RAM) < L3(storage)
"""

import asyncio
import logging
import time
import math
import hashlib
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import deque, defaultdict, Counter
import threading

logger = logging.getLogger(__name__)


# ============================================================================
# SUB-AGENT DATA STRUCTURES
# ============================================================================


@dataclass
class ContextShift:
    """Detected shift in conversation/context"""

    shift_type: str  # topic, emotion, intent, style
    from_context: str
    to_context: str
    confidence: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class RecognizedPattern:
    """A recognized recurring pattern"""

    pattern_type: str
    pattern_signature: str
    frequency: int
    confidence: float
    last_seen: datetime
    decay_factor: float = 1.0


@dataclass
class CreativeConnection:
    """A creative association between topics"""

    topic_a: str
    topic_b: str
    connection_type: str  # causal, temporal, semantic, analogical
    strength: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ConsolidationCandidate:
    """Working memory item that may need consolidation"""

    item_id: str
    content: str
    importance_score: float
    recency_score: float
    access_count: int
    should_consolidate: bool


@dataclass
class ProactiveSuggestion:
    """A proactive suggestion from the planner"""

    suggestion_type: str
    message: str
    urgency: float  # 0-1
    trigger: str
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# HARDWARE-AWARE PROCESSING
# ============================================================================


class ThermalState(Enum):
    """Device thermal state for adaptive processing"""

    COLD = "cold"  # < 30°C - full power available
    NORMAL = "normal"  # 30-40°C - normal operation
    WARM = "warm"  # 40-45°C - reduce computation
    HOT = "hot"  # 45-50°C - minimal processing
    CRITICAL = "critical"  # > 50°C - only essential


class ResourceBudget:
    """
    Hardware resource budget manager

    Mathematical model:
    - CPU budget: B_cpu = B_max * f(T_thermal)
    - Memory budget: B_mem = min(RAM_available * 0.3, B_max)
    - Energy budget: E_per_token = E_base * complexity_factor

    Where f(T) is thermal scaling factor
    """

    def __init__(self):
        self.max_cpu_percent = 70  # Conservative for mobile
        self.max_memory_mb = 512  # Leave headroom
        self.thermal_state = ThermalState.NORMAL

        # Adaptive scaling
        self._thermal_scaling = {
            ThermalState.COLD: 1.0,
            ThermalState.NORMAL: 0.9,
            ThermalState.WARM: 0.6,
            ThermalState.HOT: 0.3,
            ThermalState.CRITICAL: 0.1,
        }

    def get_thermal_factor(self) -> float:
        """Get processing scale factor based on thermal state"""
        return self._thermal_scaling.get(self.thermal_state, 0.5)

    def get_cpu_budget(self) -> float:
        """Get available CPU budget in percent"""
        return self.max_cpu_percent * self.get_thermal_factor()

    def get_memory_budget(self) -> int:
        """Get available memory budget in MB"""
        base = self.max_memory_mb
        thermal_factor = self.get_thermal_factor()
        return int(base * thermal_factor)

    def update_thermal(self, temperature: float):
        """Update thermal state from temperature reading"""
        if temperature < 30:
            self.thermal_state = ThermalState.COLD
        elif temperature < 40:
            self.thermal_state = ThermalState.NORMAL
        elif temperature < 45:
            self.thermal_state = ThermalState.WARM
        elif temperature < 50:
            self.thermal_state = ThermalState.HOT
        else:
            self.thermal_state = ThermalState.CRITICAL


# ============================================================================
# EVENT-DRIVEN PROCESSING (SNN-inspired)
# ============================================================================


@dataclass
class NeuralEvent:
    """
    A spike/event in the processing network

    Inspired by biological neurons:
    - Only fires when input exceeds threshold
    - Carries payload (like neurotransmitter)
    - Has temporal component
    """

    event_type: str  # What kind of event
    payload: Any  # Data being processed
    timestamp: datetime
    priority: int = 0  # Higher = more urgent
    source: str = ""  # Origin of event
    context: Dict = field(default_factory=dict)


class EventDrivenProcessor:
    """
    Event-driven processing engine

    Unlike OpenClaw's continuous loop, this processes only when events occur.

    Mathematical model:
    output = Σ (input_i * weight_i) for active inputs where input_i > threshold

    Benefits:
    - Energy efficiency (only process when needed)
    - Lower latency (no polling)
    - Natural parallelization (independent events)
    """

    def __init__(self, resource_budget: ResourceBudget):
        self.resource_budget = resource_budget
        self._event_queue: asyncio.PriorityQueue = None
        self._handlers: Dict[str, List[Callable]] = {}
        self._active = False

        # Statistics
        self.events_processed = 0
        self.events_dropped = 0

        # Threshold for processing (sparse activation)
        self.activation_threshold = 0.5

    async def start(self):
        """Start the event processor"""
        self._event_queue = asyncio.PriorityQueue()
        self._active = True

        # Start event processing loop
        asyncio.create_task(self._process_events())

    async def stop(self):
        """Stop the event processor"""
        self._active = False

    def emit(self, event: NeuralEvent):
        """
        Emit an event into the processing network

        This is the main entry point for all AURA processing
        """
        if not self._active:
            return

        # Check resource budget
        if not self._check_resources(event):
            self.events_dropped += 1
            logger.warning(
                f"Event dropped due to resource constraints: {event.event_type}"
            )
            return

        # Priority inversion: higher priority events first
        priority = -event.priority  # Negative for max-heap behavior
        self._event_queue.put_nowait((priority, event))

    def _check_resources(self, event: NeuralEvent) -> bool:
        """Check if resources available for processing"""
        thermal_factor = self.resource_budget.get_thermal_factor()

        # Drop non-essential events when hot
        if thermal_factor < 0.3 and event.priority < 5:
            return False

        return True

    async def _process_events(self):
        """Main event processing loop"""
        while self._active:
            try:
                # Wait for event with timeout
                priority, event = await asyncio.wait_for(
                    self._event_queue.get(), timeout=1.0
                )

                # Process the event
                await self._handle_event(event)
                self.events_processed += 1

            except asyncio.TimeoutError:
                # No events, continue waiting
                continue
            except Exception as e:
                logger.error(f"Event processing error: {e}")

    async def _handle_event(self, event: NeuralEvent):
        """Handle a single event"""
        handlers = self._handlers.get(event.event_type, [])

        for handler in handlers:
            try:
                await handler(event)
            except Exception as e:
                logger.error(f"Handler error for {event.event_type}: {e}")

    def register_handler(self, event_type: str, handler: Callable):
        """Register a handler for an event type"""
        if event_type not in self._handlers:
            self._handlers[event_type] = []
        self._handlers[event_type].append(handler)


# ============================================================================
# MULTI-AGENT PARALLEL EXECUTION (F.R.I.D.A.Y. Sub-robots)
# ============================================================================


@dataclass
class SubAgent:
    """
    A sub-agent (like F.R.I.D.A.Y.'s sub-robots)

    Each sub-agent specializes in one domain:
    - ATTENTION_AGENT: Watches for user attention needs
    - MEMORY_AGENT: Manages memory consolidation
    - ACTION_AGENT: Executes planned actions
    - MONITOR_AGENT: Monitors background tasks
    - COMMUNICATION_AGENT: Handles messaging
    """

    agent_id: str
    name: str
    specialty: str
    is_active: bool = True
    current_task: Optional[str] = None
    last_active: datetime = field(default_factory=datetime.now)


class MultiAgentOrchestrator:
    """
    Multi-agent parallel execution system

    Unlike OpenClaw's single agent loop, AURA runs multiple specialized
    sub-agents in parallel, like F.R.I.D.A.Y. controlling multiple processes.

    Mathematical model:
    Total_throughput = Σ(Agent_i_throughput) for all active agents
    Resource_per_agent = Total_resources / Active_agents

    Key insight: Not all agents need to run at full capacity always.
    """

    def __init__(self, resource_budget: ResourceBudget):
        self.resource_budget = resource_budget
        self._agents: Dict[str, SubAgent] = {}
        self._task_queue: asyncio.Queue = asyncio.Queue()
        self._running = False

        # Register default sub-agents
        self._register_default_agents()

        # =====================================================================
        # SUB-AGENT STATE: Shared data structures for real processing
        # =====================================================================

        # Context Analyzer state - sliding window change detection
        self._context_window: deque = deque(maxlen=20)  # Recent context tokens
        self._context_embeddings: deque = deque(
            maxlen=20
        )  # Simple hash-based "embeddings"
        self._last_context_hash: int = 0
        self._context_shift_threshold: float = 0.4  # Jaccard distance threshold
        self._detected_shifts: deque = deque(maxlen=10)

        # Pattern Recognizer state - frequency counting with time decay
        self._pattern_counts: Counter = Counter()
        self._pattern_timestamps: Dict[str, datetime] = {}
        self._pattern_decay_rate: float = 0.95  # Decay per tick
        self._recognized_patterns: deque = deque(maxlen=20)
        self._min_pattern_frequency: int = 3

        # Creativity Engine state - co-occurrence matrix
        self._topic_cooccurrence: Dict[str, Counter] = defaultdict(Counter)
        self._recent_topics: deque = deque(maxlen=30)
        self._creative_connections: deque = deque(maxlen=15)

        # Memory Consolidator state - importance scoring
        self._working_memory_items: Dict[
            str, Dict
        ] = {}  # id -> {content, importance, access_count, last_access}
        self._consolidation_threshold: float = 0.7
        self._consolidation_candidates: deque = deque(maxlen=10)
        self._last_consolidation: datetime = datetime.now()

        # Proactive Planner state - adaptive scheduling
        self._user_activity_patterns: Dict[int, Counter] = defaultdict(
            Counter
        )  # hour -> action counts
        self._learned_triggers: Dict[
            str, Dict
        ] = {}  # trigger_id -> {hour, day, action, frequency}
        self._pending_suggestions: deque = deque(maxlen=5)
        self._last_suggestion_time: datetime = datetime.now() - timedelta(hours=1)

        # Performance tracking
        self._tick_timings: Dict[str, deque] = {
            "attention": deque(maxlen=100),
            "memory": deque(maxlen=100),
            "action": deque(maxlen=100),
            "monitor": deque(maxlen=100),
            "communication": deque(maxlen=100),
        }

    def _register_default_agents(self):
        """Register F.R.I.D.A.Y.-style sub-agents"""
        default_agents = [
            SubAgent(
                agent_id="attention",
                name="Context Analyzer",
                specialty="CONTEXT_ANALYSIS",
            ),
            SubAgent(
                agent_id="memory",
                name="Memory Consolidator",
                specialty="MEMORY_CONSOLIDATION",
            ),
            SubAgent(
                agent_id="action",
                name="Pattern Recognizer",
                specialty="PATTERN_RECOGNITION",
            ),
            SubAgent(
                agent_id="monitor",
                name="Creativity Engine",
                specialty="CREATIVE_CONNECTIONS",
            ),
            SubAgent(
                agent_id="communication",
                name="Proactive Planner",
                specialty="PROACTIVE_PLANNING",
            ),
        ]

        for agent in default_agents:
            self._agents[agent.agent_id] = agent

    async def start(self):
        """Start all sub-agents"""
        self._running = True

        # Start each agent's processing loop
        for agent_id, agent in self._agents.items():
            if agent.is_active:
                asyncio.create_task(self._run_agent(agent))

    async def stop(self):
        """Stop all sub-agents"""
        self._running = False

    async def _run_agent(self, agent: SubAgent):
        """Run a single sub-agent's processing loop"""
        while self._running:
            try:
                start_time = time.perf_counter()

                # Agent-specific processing
                if agent.agent_id == "attention":
                    await self._attention_agent_tick(agent)
                elif agent.agent_id == "memory":
                    await self._memory_agent_tick(agent)
                elif agent.agent_id == "action":
                    await self._action_agent_tick(agent)
                elif agent.agent_id == "monitor":
                    await self._monitor_agent_tick(agent)
                elif agent.agent_id == "communication":
                    await self._communication_agent_tick(agent)

                # Record timing for performance monitoring
                elapsed_ms = (time.perf_counter() - start_time) * 1000
                self._tick_timings[agent.agent_id].append(elapsed_ms)

                # Warn if tick exceeds 50ms target
                if elapsed_ms > 50:
                    logger.warning(
                        f"Agent {agent.agent_id} tick took {elapsed_ms:.1f}ms (>50ms target)"
                    )

                agent.last_active = datetime.now()

                # Adaptive sleep based on thermal state
                sleep_time = self._calculate_agent_sleep(agent)
                await asyncio.sleep(sleep_time)

            except Exception as e:
                logger.error(f"Agent {agent.agent_id} error: {e}")

    def _calculate_agent_sleep(self, agent: SubAgent) -> float:
        """Calculate agent tick rate based on resources"""
        thermal_factor = self.resource_budget.get_thermal_factor()

        # Base tick rates (seconds)
        base_ticks = {
            "attention": 0.5,  # Fast - context analysis
            "memory": 30,  # Slow - periodic consolidation
            "action": 2,  # Medium - pattern recognition
            "monitor": 5,  # Medium - creativity engine
            "communication": 10,  # Slow - proactive planning
        }

        base = base_ticks.get(agent.agent_id, 1.0)
        return base * (1.0 / max(thermal_factor, 0.1))

    # =========================================================================
    # CONTEXT ANALYZER (attention agent)
    # Sliding window + change detection for context shifts
    # =========================================================================

    async def _attention_agent_tick(self, agent: SubAgent):
        """
        Context Analyzer: Detect shifts in conversation/context

        Algorithm:
        1. Extract tokens from recent context window
        2. Compute simple hash-based fingerprint
        3. Compare with previous fingerprint using Jaccard distance
        4. Signal context shift if distance exceeds threshold

        Complexity: O(n) where n = window size (20 items max)
        Target: <10ms per tick
        """
        try:
            # No context to analyze
            if len(self._context_window) < 2:
                return

            # Step 1: Extract current context tokens (bag of words)
            current_tokens = set()
            for item in list(self._context_window)[-5:]:  # Last 5 items
                if isinstance(item, str):
                    tokens = self._tokenize_simple(item)
                    current_tokens.update(tokens)

            # Step 2: Compute fingerprint (hash of sorted tokens)
            current_hash = hash(frozenset(current_tokens))

            # Step 3: Compare with previous context using Jaccard distance
            if self._last_context_hash != 0 and len(current_tokens) > 0:
                # Get previous tokens from embeddings (simplified)
                prev_tokens = set()
                for emb in list(self._context_embeddings)[-5:-1]:  # Exclude current
                    if isinstance(emb, set):
                        prev_tokens.update(emb)

                if prev_tokens:
                    # Jaccard distance = 1 - (intersection / union)
                    intersection = len(current_tokens & prev_tokens)
                    union = len(current_tokens | prev_tokens)
                    jaccard_sim = intersection / union if union > 0 else 1.0
                    jaccard_dist = 1.0 - jaccard_sim

                    # Step 4: Signal shift if distance exceeds threshold
                    if jaccard_dist > self._context_shift_threshold:
                        shift = ContextShift(
                            shift_type=self._classify_shift(
                                prev_tokens, current_tokens
                            ),
                            from_context=self._summarize_tokens(prev_tokens),
                            to_context=self._summarize_tokens(current_tokens),
                            confidence=min(jaccard_dist, 1.0),
                        )
                        self._detected_shifts.append(shift)
                        logger.debug(
                            f"Context shift detected: {shift.shift_type} (conf={shift.confidence:.2f})"
                        )

            # Update state
            self._context_embeddings.append(current_tokens)
            self._last_context_hash = current_hash

        except Exception as e:
            logger.error(f"Context analyzer error: {e}")

    def _tokenize_simple(self, text: str) -> Set[str]:
        """Simple tokenization - split and lowercase, filter short tokens"""
        if not text:
            return set()
        # Remove punctuation and split
        cleaned = "".join(
            c if c.isalnum() or c.isspace() else " " for c in text.lower()
        )
        tokens = {t for t in cleaned.split() if len(t) > 2}
        return tokens

    def _classify_shift(self, prev_tokens: Set[str], curr_tokens: Set[str]) -> str:
        """Classify the type of context shift"""
        # Emotion indicators
        emotion_words = {
            "happy",
            "sad",
            "angry",
            "frustrated",
            "excited",
            "worried",
            "anxious",
            "calm",
        }
        if curr_tokens & emotion_words:
            return "emotion"

        # Topic shift - mostly new tokens
        new_tokens = curr_tokens - prev_tokens
        if len(new_tokens) > len(curr_tokens) * 0.7:
            return "topic"

        # Intent shift - action words
        intent_words = {"want", "need", "should", "must", "will", "going", "planning"}
        if curr_tokens & intent_words:
            return "intent"

        return "general"

    def _summarize_tokens(self, tokens: Set[str]) -> str:
        """Create summary from token set"""
        sorted_tokens = sorted(list(tokens))[:5]  # Top 5 alphabetically
        return " ".join(sorted_tokens)

    def add_context(self, content: str):
        """Add content to context window (called externally)"""
        self._context_window.append(content)

    def get_context_shifts(self) -> List[ContextShift]:
        """Get recent context shifts"""
        return list(self._detected_shifts)

    # =========================================================================
    # PATTERN RECOGNIZER (action agent)
    # Frequency counting with time decay
    # =========================================================================

    async def _action_agent_tick(self, agent: SubAgent):
        """
        Pattern Recognizer: Identify recurring patterns in interactions

        Algorithm:
        1. Apply time decay to existing pattern counts
        2. Scan recent context for patterns (n-grams, action sequences)
        3. Update frequency counts
        4. Return patterns exceeding threshold with confidence

        Complexity: O(n*m) where n = recent items, m = pattern length
        Target: <15ms per tick
        """
        try:
            # Step 1: Apply time decay to all pattern counts
            now = datetime.now()
            patterns_to_remove = []

            for pattern_key, count in list(self._pattern_counts.items()):
                last_seen = self._pattern_timestamps.get(pattern_key)
                if last_seen:
                    # Decay based on time since last seen
                    hours_elapsed = (now - last_seen).total_seconds() / 3600
                    decay = self._pattern_decay_rate**hours_elapsed
                    new_count = count * decay

                    if new_count < 0.5:  # Too decayed, remove
                        patterns_to_remove.append(pattern_key)
                    else:
                        self._pattern_counts[pattern_key] = new_count

            for key in patterns_to_remove:
                del self._pattern_counts[key]
                if key in self._pattern_timestamps:
                    del self._pattern_timestamps[key]

            # Step 2: Extract patterns from recent context
            recent_items = list(self._context_window)[-10:]  # Last 10 items

            # Extract bigram patterns
            for i in range(len(recent_items) - 1):
                if isinstance(recent_items[i], str) and isinstance(
                    recent_items[i + 1], str
                ):
                    # Create pattern signature from token overlap
                    tokens_a = self._tokenize_simple(recent_items[i])
                    tokens_b = self._tokenize_simple(recent_items[i + 1])

                    if tokens_a and tokens_b:
                        # Pattern = sorted common themes
                        pattern_key = self._create_pattern_key(tokens_a, tokens_b)
                        if pattern_key:
                            self._pattern_counts[pattern_key] += 1
                            self._pattern_timestamps[pattern_key] = now

            # Step 3: Identify patterns exceeding threshold
            for pattern_key, count in self._pattern_counts.items():
                if count >= self._min_pattern_frequency:
                    # Calculate confidence based on count and consistency
                    confidence = min(count / 10.0, 1.0)

                    pattern = RecognizedPattern(
                        pattern_type="behavioral",
                        pattern_signature=pattern_key,
                        frequency=int(count),
                        confidence=confidence,
                        last_seen=self._pattern_timestamps.get(pattern_key, now),
                        decay_factor=self._pattern_decay_rate,
                    )

                    # Avoid duplicates
                    existing_sigs = {
                        p.pattern_signature for p in self._recognized_patterns
                    }
                    if pattern_key not in existing_sigs:
                        self._recognized_patterns.append(pattern)
                        logger.debug(
                            f"Pattern recognized: {pattern_key} (freq={count:.1f})"
                        )

        except Exception as e:
            logger.error(f"Pattern recognizer error: {e}")

    def _create_pattern_key(
        self, tokens_a: Set[str], tokens_b: Set[str]
    ) -> Optional[str]:
        """Create pattern key from two token sets"""
        # Find common themes
        common = tokens_a & tokens_b
        if common:
            return "_".join(sorted(list(common))[:3])

        # Or create sequential pattern
        combined = sorted(list(tokens_a)[:2] + list(tokens_b)[:2])
        if len(combined) >= 2:
            return "->".join(combined[:2])

        return None

    def add_pattern_observation(self, pattern_type: str, signature: str):
        """Add pattern observation externally"""
        self._pattern_counts[signature] += 1
        self._pattern_timestamps[signature] = datetime.now()

    def get_recognized_patterns(self) -> List[RecognizedPattern]:
        """Get recently recognized patterns"""
        return list(self._recognized_patterns)

    # =========================================================================
    # CREATIVITY ENGINE (monitor agent)
    # Co-occurrence matrix for topic associations
    # =========================================================================

    async def _monitor_agent_tick(self, agent: SubAgent):
        """
        Creativity Engine: Generate connections between topics

        Algorithm:
        1. Update co-occurrence matrix from recent topics
        2. Find high co-occurrence pairs
        3. Generate creative connection types (causal, temporal, semantic)
        4. Return associations with strength scores

        Complexity: O(n^2) where n = recent topics (30 max)
        Target: <20ms per tick
        """
        try:
            # Step 1: Extract topics from recent context
            recent_items = list(self._context_window)[-10:]

            for item in recent_items:
                if isinstance(item, str):
                    topics = self._extract_topics(item)
                    for topic in topics:
                        if topic not in self._recent_topics:
                            self._recent_topics.append(topic)

            # Step 2: Update co-occurrence matrix
            topics_list = list(self._recent_topics)
            window_size = 5  # Topics within 5 positions co-occur

            for i, topic_a in enumerate(topics_list):
                for j in range(
                    max(0, i - window_size), min(len(topics_list), i + window_size + 1)
                ):
                    if i != j:
                        topic_b = topics_list[j]
                        if topic_a != topic_b:
                            self._topic_cooccurrence[topic_a][topic_b] += 1

            # Step 3: Find strong co-occurrences and generate connections
            for topic_a, cooccur_counts in list(self._topic_cooccurrence.items())[
                :20
            ]:  # Limit iterations
                for topic_b, count in cooccur_counts.most_common(3):  # Top 3 per topic
                    if count >= 2:  # Minimum co-occurrence threshold
                        # Calculate strength (normalized)
                        max_count = (
                            max(cooccur_counts.values()) if cooccur_counts else 1
                        )
                        strength = count / max_count

                        if strength >= 0.5:
                            connection = CreativeConnection(
                                topic_a=topic_a,
                                topic_b=topic_b,
                                connection_type=self._infer_connection_type(
                                    topic_a, topic_b
                                ),
                                strength=strength,
                            )

                            # Avoid duplicates
                            existing = {
                                (c.topic_a, c.topic_b)
                                for c in self._creative_connections
                            }
                            if (topic_a, topic_b) not in existing and (
                                topic_b,
                                topic_a,
                            ) not in existing:
                                self._creative_connections.append(connection)
                                logger.debug(
                                    f"Creative connection: {topic_a} <-> {topic_b} ({connection.connection_type})"
                                )

        except Exception as e:
            logger.error(f"Creativity engine error: {e}")

    def _extract_topics(self, text: str) -> List[str]:
        """Extract topic keywords from text"""
        if not text:
            return []

        # Simple topic extraction - nouns and important words
        tokens = self._tokenize_simple(text)

        # Filter to likely topics (longer words, not common)
        common_words = {
            "the",
            "and",
            "but",
            "for",
            "with",
            "this",
            "that",
            "have",
            "has",
            "are",
            "was",
            "were",
        }
        topics = [t for t in tokens if len(t) > 3 and t not in common_words]

        return topics[:5]  # Max 5 topics per item

    def _infer_connection_type(self, topic_a: str, topic_b: str) -> str:
        """Infer the type of connection between topics"""
        # Temporal indicators
        temporal_words = {"before", "after", "then", "when", "while", "during"}
        if topic_a in temporal_words or topic_b in temporal_words:
            return "temporal"

        # Causal indicators
        causal_words = {"because", "cause", "result", "effect", "lead", "make"}
        if topic_a in causal_words or topic_b in causal_words:
            return "causal"

        # Check for semantic similarity (same prefix)
        if topic_a[:3] == topic_b[:3]:
            return "semantic"

        return "analogical"

    def add_topic(self, topic: str):
        """Add topic externally"""
        self._recent_topics.append(topic)

    def get_creative_connections(self) -> List[CreativeConnection]:
        """Get creative connections"""
        return list(self._creative_connections)

    # =========================================================================
    # MEMORY CONSOLIDATOR (memory agent)
    # Importance scoring with threshold-based consolidation
    # =========================================================================

    async def _memory_agent_tick(self, agent: SubAgent):
        """
        Memory Consolidator: Check if working memory needs consolidation

        Algorithm:
        1. Calculate importance score for each working memory item
           Score = (recency * 0.3) + (access_frequency * 0.3) + (base_importance * 0.4)
        2. Identify items exceeding consolidation threshold
        3. Trigger consolidation for high-importance items
        4. Clean up low-importance old items

        Complexity: O(n) where n = working memory items
        Target: <15ms per tick
        """
        try:
            now = datetime.now()
            consolidation_needed = []
            items_to_remove = []

            # Step 1 & 2: Score and identify consolidation candidates
            for item_id, item_data in list(self._working_memory_items.items()):
                # Calculate recency score (exponential decay)
                last_access = item_data.get("last_access", now)
                hours_since_access = (now - last_access).total_seconds() / 3600
                recency_score = math.exp(
                    -hours_since_access / 24
                )  # Decay over 24 hours

                # Calculate access frequency score
                access_count = item_data.get("access_count", 1)
                freq_score = min(access_count / 10.0, 1.0)  # Normalize to 0-1

                # Base importance from item
                base_importance = item_data.get("importance", 0.5)

                # Combined score
                combined_score = (
                    (recency_score * 0.3) + (freq_score * 0.3) + (base_importance * 0.4)
                )

                # Check for consolidation
                if combined_score >= self._consolidation_threshold:
                    candidate = ConsolidationCandidate(
                        item_id=item_id,
                        content=item_data.get("content", ""),
                        importance_score=combined_score,
                        recency_score=recency_score,
                        access_count=access_count,
                        should_consolidate=True,
                    )
                    consolidation_needed.append(candidate)
                elif recency_score < 0.1 and freq_score < 0.2:
                    # Low importance + old = remove
                    items_to_remove.append(item_id)

            # Step 3: Trigger consolidation (mark candidates)
            for candidate in consolidation_needed:
                self._consolidation_candidates.append(candidate)
                logger.debug(
                    f"Consolidation candidate: {candidate.item_id} (score={candidate.importance_score:.2f})"
                )

            # Step 4: Clean up old low-importance items
            for item_id in items_to_remove:
                if item_id in self._working_memory_items:
                    del self._working_memory_items[item_id]

            self._last_consolidation = now

        except Exception as e:
            logger.error(f"Memory consolidator error: {e}")

    def add_working_memory(self, item_id: str, content: str, importance: float = 0.5):
        """Add item to working memory (called externally)"""
        now = datetime.now()
        if item_id in self._working_memory_items:
            # Update existing
            self._working_memory_items[item_id]["access_count"] += 1
            self._working_memory_items[item_id]["last_access"] = now
        else:
            # Add new
            self._working_memory_items[item_id] = {
                "content": content,
                "importance": importance,
                "access_count": 1,
                "created_at": now,
                "last_access": now,
            }

    def get_consolidation_candidates(self) -> List[ConsolidationCandidate]:
        """Get items needing consolidation"""
        return list(self._consolidation_candidates)

    def clear_consolidation_candidates(self):
        """Clear after consolidation is complete"""
        self._consolidation_candidates.clear()

    # =========================================================================
    # PROACTIVE PLANNER (communication agent)
    # Adaptive scheduling based on learned user patterns
    # =========================================================================

    async def _communication_agent_tick(self, agent: SubAgent):
        """
        Proactive Planner: Generate proactive suggestions based on patterns

        Algorithm:
        1. Check learned time-based patterns (what user does at this hour)
        2. Check adaptive triggers (learned from past behavior)
        3. Avoid suggestion fatigue (cooldown between suggestions)
        4. Return proactive suggestions when appropriate

        Complexity: O(p) where p = number of learned patterns
        Target: <10ms per tick
        """
        try:
            now = datetime.now()
            current_hour = now.hour
            current_day = now.strftime("%A")

            # Check cooldown (no suggestions within 30 minutes)
            time_since_last = (now - self._last_suggestion_time).total_seconds()
            if time_since_last < 1800:  # 30 minutes cooldown
                return

            # Step 1: Check time-based patterns
            hour_patterns = self._user_activity_patterns.get(current_hour, Counter())

            if hour_patterns:
                # Find most common action at this hour
                most_common = hour_patterns.most_common(1)
                if most_common:
                    action, frequency = most_common[0]
                    if frequency >= 3:  # Seen at least 3 times at this hour
                        # Calculate urgency based on frequency
                        urgency = min(frequency / 10.0, 0.9)

                        suggestion = ProactiveSuggestion(
                            suggestion_type="time_pattern",
                            message=f"Based on your usual routine, you often {action} around this time.",
                            urgency=urgency,
                            trigger=f"hour_{current_hour}",
                        )
                        self._pending_suggestions.append(suggestion)
                        self._last_suggestion_time = now
                        logger.debug(
                            f"Proactive suggestion: {action} (freq={frequency})"
                        )

            # Step 2: Check adaptive triggers
            for trigger_id, trigger_data in list(self._learned_triggers.items()):
                trigger_hour = trigger_data.get("hour")
                trigger_day = trigger_data.get("day")
                trigger_action = trigger_data.get("action")
                trigger_freq = trigger_data.get("frequency", 0)

                # Check if trigger matches current time
                hour_match = trigger_hour is None or trigger_hour == current_hour
                day_match = trigger_day is None or trigger_day == current_day

                if hour_match and day_match and trigger_freq >= 2:
                    urgency = min(trigger_freq / 5.0, 0.8)

                    suggestion = ProactiveSuggestion(
                        suggestion_type="learned_trigger",
                        message=f"Time for: {trigger_action}",
                        urgency=urgency,
                        trigger=trigger_id,
                    )

                    # Avoid duplicate suggestions
                    existing_triggers = {s.trigger for s in self._pending_suggestions}
                    if trigger_id not in existing_triggers:
                        self._pending_suggestions.append(suggestion)
                        self._last_suggestion_time = now

        except Exception as e:
            logger.error(f"Proactive planner error: {e}")

    def learn_user_activity(
        self, action: str, hour: Optional[int] = None, day: Optional[str] = None
    ):
        """Learn user activity pattern (called externally)"""
        if hour is None:
            hour = datetime.now().hour
        if day is None:
            day = datetime.now().strftime("%A")

        # Update hour-based patterns
        self._user_activity_patterns[hour][action] += 1

        # Create/update trigger
        trigger_id = f"{day}_{hour}_{action}"[:32]
        if trigger_id in self._learned_triggers:
            self._learned_triggers[trigger_id]["frequency"] += 1
        else:
            self._learned_triggers[trigger_id] = {
                "hour": hour,
                "day": day,
                "action": action,
                "frequency": 1,
            }

    def get_proactive_suggestions(self) -> List[ProactiveSuggestion]:
        """Get pending proactive suggestions"""
        suggestions = list(self._pending_suggestions)
        self._pending_suggestions.clear()
        return suggestions

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def assign_task(self, agent_id: str, task: str):
        """Assign a task to a specific agent"""
        if agent_id in self._agents:
            self._agents[agent_id].current_task = task

    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        return {
            agent_id: {
                "name": agent.name,
                "specialty": agent.specialty,
                "is_active": agent.is_active,
                "current_task": agent.current_task,
                "last_active": agent.last_active.isoformat(),
                "avg_tick_ms": self._get_avg_tick_time(agent_id),
            }
            for agent_id, agent in self._agents.items()
        }

    def _get_avg_tick_time(self, agent_id: str) -> float:
        """Get average tick time for an agent"""
        timings = self._tick_timings.get(agent_id, deque())
        if not timings:
            return 0.0
        return sum(timings) / len(timings)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get detailed performance statistics"""
        return {
            "context_shifts_detected": len(self._detected_shifts),
            "patterns_recognized": len(self._recognized_patterns),
            "creative_connections": len(self._creative_connections),
            "consolidation_candidates": len(self._consolidation_candidates),
            "pending_suggestions": len(self._pending_suggestions),
            "working_memory_items": len(self._working_memory_items),
            "tick_timings": {
                agent_id: {
                    "avg_ms": self._get_avg_tick_time(agent_id),
                    "max_ms": max(timings) if timings else 0,
                    "samples": len(timings),
                }
                for agent_id, timings in self._tick_timings.items()
            },
        }


# ============================================================================
# PROACTIVE ACTION PLANNER
# ============================================================================


@dataclass
class ProactiveAction:
    """
    A proactive action that AURA decides to take

    Unlike reactive responses, proactive actions are AURA-initiated.
    """

    action_id: str
    action_type: str  # What kind of action
    reason: str  # Why AURA decided this
    urgency: int  # 1-10, higher = more urgent
    prerequisites: List[str]  # What must happen first
    expected_outcome: str  # What AURA expects to happen
    scheduled_time: Optional[datetime] = None
    is_background: bool = False  # Can run in background?
    status: str = "pending"  # pending, running, completed, failed


class ProactivePlanner:
    """
    Proactive action planning system

    This is what makes AURA "think" vs just "respond".
    Like JARVIS who anticipates Tony's needs.

    Decision model:
    - Evaluate current context
    - Identify potential needs
    - Weigh action vs inaction
    - Plan execution

    Adaptive Scheduling:
    - Learns user's actual wake/sleep patterns from activity
    - No hardcoded times - adapts to user lifestyle
    - Tracks action success rates for optimization
    """

    def __init__(self, orchestrator: MultiAgentOrchestrator):
        self.orchestrator = orchestrator
        self._actions: Dict[str, ProactiveAction] = {}
        self._action_history: deque = deque(maxlen=100)

        # Adaptive scheduling state
        self._activity_hours: Counter = Counter()  # Track which hours user is active
        self._morning_hour: int = 7  # Default, will adapt
        self._evening_hour: int = 20  # Default, will adapt
        self._last_schedule_update: Optional[datetime] = None
        self._action_success_rates: Dict[str, List[bool]] = defaultdict(list)

        # Opportunity detection weights
        self._opportunity_weights = {
            "time_based": 0.3,  # Time of day patterns
            "user_state": 0.3,  # Inferred user state
            "pattern_match": 0.2,  # Previous successful actions
            "context_change": 0.2,  # Context shift detection
        }

    async def evaluate_opportunities(self) -> List[ProactiveAction]:
        """
        Evaluate if there are proactive actions to take

        This is called periodically to "think" about what to do
        """
        opportunities = []

        # Check time-based opportunities
        opportunities.extend(await self._check_time_opportunities())

        # Check user state opportunities
        opportunities.extend(await self._check_user_state_opportunities())

        # Check pattern-based opportunities
        opportunities.extend(await self._check_pattern_opportunities())

        # Score and filter opportunities
        scored = []
        for opp in opportunities:
            score = self._score_opportunity(opp)
            if score > 0.5:  # Threshold
                scored.append((score, opp))

        # Sort by score and return top opportunities
        scored.sort(key=lambda x: x[0], reverse=True)
        return [opp for _, opp in scored[:3]]

    def _score_opportunity(self, action: ProactiveAction) -> float:
        """Score an opportunity based on multiple factors"""
        # Base score from urgency
        score = action.urgency / 10.0

        # Time decay - don't repeat too soon
        if action.action_type in self._action_history:
            recency = 0.5  # Reduce score for recent actions
        else:
            recency = 1.0

        return score * recency

    async def _check_time_opportunities(self) -> List[ProactiveAction]:
        """
        Check time-based opportunities using adaptive scheduling.

        Instead of hardcoded hours, uses learned user activity patterns
        to determine optimal times for morning/evening routines.
        """
        now = datetime.now()
        opportunities = []

        # Update adaptive schedule periodically (once per day)
        if (
            self._last_schedule_update is None
            or (now - self._last_schedule_update).days >= 1
        ):
            self._update_adaptive_schedule()
            self._last_schedule_update = now

        # Morning - check schedule (adaptive hour)
        if now.hour == self._morning_hour:
            # Check success rate - skip if user usually ignores
            success_rate = self._get_action_success_rate("check_schedule")
            if success_rate > 0.3:  # Only suggest if >30% engagement
                opportunities.append(
                    ProactiveAction(
                        action_id=f"morning_{now.date()}",
                        action_type="check_schedule",
                        reason=f"Morning time ({self._morning_hour}:00) - check your day ahead",
                        urgency=int(6 * success_rate + 4),  # Scale urgency by success
                        prerequisites=[],
                        expected_outcome="Present today's schedule",
                        is_background=True,
                    )
                )

        # Evening - summarize day (adaptive hour)
        if now.hour == self._evening_hour:
            success_rate = self._get_action_success_rate("daily_summary")
            if success_rate > 0.3:
                opportunities.append(
                    ProactiveAction(
                        action_id=f"evening_{now.date()}",
                        action_type="daily_summary",
                        reason=f"Evening time ({self._evening_hour}:00) - summarize the day",
                        urgency=int(5 * success_rate + 4),
                        prerequisites=[],
                        expected_outcome="Present day summary",
                    )
                )

        # Also check for peak activity hours - good time for reminders
        peak_hours = self._get_peak_activity_hours(top_n=3)
        if now.hour in peak_hours and now.minute < 10:
            opportunities.extend(await self._check_pattern_opportunities())

        return opportunities

    def _update_adaptive_schedule(self) -> None:
        """
        Update morning/evening hours based on learned activity patterns.

        Morning = first hour with significant activity (after 4 AM)
        Evening = last hour with significant activity (before midnight)
        """
        if not self._activity_hours:
            return  # Keep defaults if no data

        # Find morning hour: first active hour after 4 AM
        total_activity = sum(self._activity_hours.values())
        if total_activity < 10:
            return  # Not enough data

        # Calculate activity threshold (10% of average)
        avg_activity = total_activity / 24
        threshold = avg_activity * 0.1

        # Find first significant activity (morning)
        for hour in range(4, 12):
            if self._activity_hours[hour] >= threshold:
                self._morning_hour = hour
                break

        # Find last significant activity (evening)
        for hour in range(23, 17, -1):
            if self._activity_hours[hour] >= threshold:
                self._evening_hour = hour
                break

    def _get_action_success_rate(self, action_type: str) -> float:
        """Get success rate for an action type (0.0-1.0)."""
        history = self._action_success_rates.get(action_type, [])
        if not history:
            return 0.7  # Default assumption
        return sum(history) / len(history)

    def _get_peak_activity_hours(self, top_n: int = 3) -> List[int]:
        """Get the top N hours with most user activity."""
        if not self._activity_hours:
            return [9, 14, 19]  # Defaults
        return [hour for hour, _ in self._activity_hours.most_common(top_n)]

    def record_user_activity(self, hour: Optional[int] = None) -> None:
        """Record user activity for adaptive scheduling."""
        if hour is None:
            hour = datetime.now().hour
        self._activity_hours[hour] += 1

    def record_action_result(self, action_type: str, success: bool) -> None:
        """Record whether user engaged with a proactive action."""
        history = self._action_success_rates[action_type]
        history.append(success)
        # Keep last 20 results
        if len(history) > 20:
            self._action_success_rates[action_type] = history[-20:]

    async def _check_user_state_opportunities(self) -> List[ProactiveAction]:
        """
        Check opportunities based on inferred user state.

        Uses orchestrator's context analysis to detect:
        - Topic focus changes (might need related info)
        - Repeated queries (might need deeper help)
        - Time gaps (might need re-engagement)
        """
        opportunities = []

        # Get context shifts from orchestrator
        context_shifts = self.orchestrator.get_context_shifts()

        for shift in context_shifts[-3:]:  # Recent shifts only
            if shift.magnitude > 0.5:  # Significant shift
                opportunities.append(
                    ProactiveAction(
                        action_id=f"context_shift_{hash(shift.new_context) % 10000}",
                        action_type="context_assistance",
                        reason=f"Noticed topic change to: {shift.new_context[:50]}",
                        urgency=int(5 + shift.magnitude * 3),
                        prerequisites=[],
                        expected_outcome="Offer relevant context or resources",
                        is_background=True,
                    )
                )

        # Check for recognized patterns that might need action
        patterns = self.orchestrator.get_recognized_patterns()
        for pattern in patterns[-5:]:
            if pattern.frequency >= 3 and pattern.confidence > 0.6:
                opportunities.append(
                    ProactiveAction(
                        action_id=f"pattern_{pattern.pattern_id}",
                        action_type="pattern_shortcut",
                        reason=f"You often do: {pattern.description[:50]}",
                        urgency=5,
                        prerequisites=[],
                        expected_outcome="Offer to automate or shortcut this pattern",
                        is_background=True,
                    )
                )

        return opportunities

    async def _check_pattern_opportunities(self) -> List[ProactiveAction]:
        """
        Check opportunities based on learned patterns from orchestrator.

        Uses creative connections and recognized patterns to suggest
        proactive actions the user might benefit from.
        """
        opportunities = []

        # Get creative connections (topic associations)
        connections = self.orchestrator.get_creative_connections()

        for conn in connections[-3:]:
            if conn.strength > 0.5:
                opportunities.append(
                    ProactiveAction(
                        action_id=f"connection_{conn.source_topic}_{conn.target_topic}",
                        action_type="topic_connection",
                        reason=f"'{conn.source_topic}' often relates to '{conn.target_topic}'",
                        urgency=4,
                        prerequisites=[],
                        expected_outcome="Suggest exploring related topic",
                        is_background=True,
                    )
                )

        # Get proactive suggestions from orchestrator
        suggestions = self.orchestrator.get_proactive_suggestions()

        for suggestion in suggestions:
            if suggestion.confidence > 0.5:
                opportunities.append(
                    ProactiveAction(
                        action_id=f"suggestion_{suggestion.action_type}_{int(suggestion.confidence * 100)}",
                        action_type=suggestion.action_type,
                        reason=suggestion.reason,
                        urgency=int(suggestion.confidence * 8),
                        prerequisites=[],
                        expected_outcome="Execute suggested action",
                        is_background=True,
                    )
                )

        return opportunities

    async def execute_action(self, action: ProactiveAction):
        """Execute a proactive action"""
        # Check prerequisites
        for prereq in action.prerequisites:
            if prereq not in self._actions:
                logger.warning(f"Prerequisite missing: {prereq}")
                return

        action.status = "running"

        # Execute based on type
        if action.action_type == "check_schedule":
            await self._execute_schedule_check(action)
        elif action.action_type == "daily_summary":
            await self._execute_daily_summary(action)

        action.status = "completed"
        self._action_history.append(action.action_type)
        self._actions[action.action_id] = action

    async def _execute_schedule_check(self, action: ProactiveAction):
        """Check and present schedule"""
        # Would integrate with calendar
        logger.info("Checking schedule...")

    async def _execute_daily_summary(self, action: ProactiveAction):
        """Generate daily summary"""
        # Would integrate with memory
        logger.info("Generating daily summary...")


# ============================================================================
# MAIN NEUROMORPHIC ENGINE
# ============================================================================


class NeuromorphicEngine:
    """
    AURA's Core Processing Engine

    This is the heart of AURA - combining:
    1. Event-driven processing (SNN-inspired)
    2. Hardware-aware resource management
    3. Multi-agent parallel execution
    4. Proactive planning

    Unlike OpenClaw's single-threaded loop, this is a truly parallel,
    hardware-optimized architecture.
    """

    def __init__(self):
        # Hardware resource management
        self.resource_budget = ResourceBudget()

        # Event-driven processing
        self.event_processor = EventDrivenProcessor(self.resource_budget)

        # Multi-agent orchestration
        self.orchestrator = MultiAgentOrchestrator(self.resource_budget)

        # Proactive planning
        self.proactive_planner = ProactivePlanner(self.orchestrator)

        # State
        self._running = False

    async def start(self):
        """Start the neuromorphic engine"""
        logger.info("Starting AURA Neuromorphic Engine...")

        # Start components
        await self.event_processor.start()
        await self.orchestrator.start()

        # Register event handlers
        self._register_handlers()

        self._running = True
        logger.info("AURA Neuromorphic Engine running")

    async def stop(self):
        """Stop the neuromorphic engine"""
        self._running = False
        await self.event_processor.stop()
        await self.orchestrator.stop()

    def _register_handlers(self):
        """Register event handlers"""
        self.event_processor.register_handler("user_message", self._handle_user_message)
        self.event_processor.register_handler(
            "scheduled_task", self._handle_scheduled_task
        )
        self.event_processor.register_handler(
            "context_change", self._handle_context_change
        )

    async def _handle_user_message(self, event: NeuralEvent):
        """Handle incoming user message"""
        logger.info(f"Processing user message: {event.payload[:50]}...")

    async def _handle_scheduled_task(self, event: NeuralEvent):
        """Handle scheduled task trigger"""
        pass

    async def _handle_context_change(self, event: NeuralEvent):
        """Handle context change"""
        pass

    async def emit_user_message(self, message: str):
        """Emit a user message event"""
        event = NeuralEvent(
            event_type="user_message",
            payload=message,
            timestamp=datetime.now(),
            priority=8,  # High priority
            source="user",
        )
        self.event_processor.emit(event)

    def get_status(self) -> Dict[str, Any]:
        """Get engine status"""
        return {
            "running": self._running,
            "thermal_state": self.resource_budget.thermal_state.value,
            "cpu_budget": self.resource_budget.get_cpu_budget(),
            "memory_budget": self.resource_budget.get_memory_budget(),
            "events_processed": self.event_processor.events_processed,
            "agents": self.orchestrator.get_agent_status(),
        }


# Global instance
_engine: Optional[NeuromorphicEngine] = None


def get_neuromorphic_engine() -> NeuromorphicEngine:
    """Get or create neuromorphic engine"""
    global _engine
    if _engine is None:
        _engine = NeuromorphicEngine()
    return _engine
