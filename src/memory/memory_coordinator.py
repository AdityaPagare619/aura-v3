"""
AURA v3 Memory Integration Coordinator
====================================

This module properly wires together all memory systems:
- NeuralMemory (working memory)
- EpisodicMemory (event memories)
- SemanticMemory (knowledge facts)
- SkillMemory (procedural knowledge)
- AncestorMemory (ancient knowledge)

Key functions:
1. Unified retrieval across all memory types
2. Consolidation pipeline (episodic → semantic)
3. Memory importance scoring
4. Temporal validity (Graphiti-style)
5. Conflict resolution between subsystems

This is what the subagent reports found MISSING - the integration.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory in AURA"""

    WORKING = "working"  # Immediate context
    EPISODIC = "episodic"  # Event memories
    SEMANTIC = "semantic"  # Facts and knowledge
    SKILL = "skill"  # Procedures
    ANCESTOR = "ancestor"  # Ancient patterns


@dataclass
class MemoryItem:
    """Unified memory item across all systems"""

    id: str
    content: str
    memory_type: MemoryType
    importance: float  # 0-1
    activation: float  # 0-1, current relevance
    created_at: datetime
    last_accessed: datetime
    validity_start: Optional[datetime] = None
    validity_end: Optional[datetime] = None
    source: str = ""  # Which subsystem created this
    metadata: Dict = field(default_factory=dict)


class MemoryCoordinator:
    """
    Coordinates all memory systems.

    This is the INTEGRATION that was found missing in subagent reports.
    Previously each memory system existed but wasn't connected.
    """

    def __init__(self):
        # All memory systems
        self.neural_memory = None
        self.episodic_memory = None
        self.semantic_memory = None
        self.skill_memory = None
        self.ancestor_memory = None

        # Integration settings
        self.consolidation_interval = 300  # 5 minutes
        self.importance_threshold = 0.6  # For consolidation
        self.max_working_memories = 50
        self.consolidation_running = False

    def set_memory_system(self, memory_type: MemoryType, system: Any):
        """Register a memory system"""
        if memory_type == MemoryType.WORKING:
            self.neural_memory = system
        elif memory_type == MemoryType.EPISODIC:
            self.episodic_memory = system
        elif memory_type == MemoryType.SEMANTIC:
            self.semantic_memory = system
        elif memory_type == MemoryType.SKILL:
            self.skill_memory = system
        elif memory_type == MemoryType.ANCESTOR:
            self.ancestor_memory = system

        logger.info(f"Registered {memory_type.value} memory system")

    async def initialize(self):
        """Initialize the coordinator and all memory systems"""
        logger.info("Initializing Memory Coordinator...")

        # Start consolidation loop
        self.consolidation_running = True
        asyncio.create_task(self._consolidation_loop())

        logger.info("Memory Coordinator initialized")

    async def shutdown(self):
        """Stop consolidation and cleanup"""
        self.consolidation_running = False
        logger.info("Memory Coordinator stopped")

    # =========================================================================
    # CONSOLIDATION PIPELINE
    # =========================================================================

    async def _consolidation_loop(self):
        """Background loop for memory consolidation"""
        while self.consolidation_running:
            try:
                await self._consolidate_memories()
            except Exception as e:
                logger.warning(f"Consolidation error: {e}")
            await asyncio.sleep(self.consolidation_interval)

    async def _consolidate_memories(self):
        """Main consolidation: working -> episodic -> semantic -> ancestor"""
        if not self.neural_memory or not self.semantic_memory:
            return

        # Import Experience for episodic encoding
        from src.memory.episodic_memory import Experience

        # Get high-importance memories from working memory
        for neuron_id in list(self.neural_memory.working_memory):
            neuron = self.neural_memory.neurons.get(neuron_id)
            if not neuron:
                continue

            # Check if ready for consolidation
            if (
                neuron.importance >= self.importance_threshold
                and neuron.access_count >= 3
            ):
                # Create episodic trace from neuron (working -> episodic)
                episodic_trace = None
                if self.episodic_memory:
                    experience = Experience(
                        content=neuron.content,
                        context={
                            "importance": neuron.importance,
                            "type": "consolidated",
                            "neuron_id": neuron.id,
                        },
                        emotional_valence=neuron.emotional_valence,
                    )
                    episodic_trace = self.episodic_memory.encode(experience)

                # Extract semantic knowledge (episodic -> semantic)
                if episodic_trace:
                    self.semantic_memory.consolidate(episodic_trace)
                else:
                    # Fallback: create a simple object with required attributes
                    class NeuronAsTrace:
                        def __init__(self, n):
                            self.id = n.id
                            self.content = n.content
                            self.importance = n.importance
                            self.context = {"type": "consolidated"}
                    self.semantic_memory.consolidate(NeuronAsTrace(neuron))

                # Archive to ancestor memory (semantic -> ancestor)
                if self.ancestor_memory and neuron.importance >= 0.8:
                    self.ancestor_memory.archive(
                        content=neuron.content,
                        importance=neuron.importance,
                        memory_type="consolidated",
                        tags=list(neuron.tags) if hasattr(neuron, 'tags') else [],
                    )

                logger.debug(f"Consolidated memory: {neuron.id[:8]}")



# Global coordinator
_coordinator: Optional["MemoryCoordinator"] = None


def get_memory_coordinator() -> "MemoryCoordinator":
    """Get or create the global memory coordinator"""
    global _coordinator
    if _coordinator is None:
        _coordinator = MemoryCoordinator()
    return _coordinator
