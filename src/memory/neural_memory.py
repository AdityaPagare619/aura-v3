"""
AURA v3 Neural Memory System
Biologically inspired memory architecture
Mimics brain neurons, synapses, and memory consolidation
"""

import asyncio
import logging
import numpy as np
import os
import json
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import deque
import hashlib
import math

logger = logging.getLogger(__name__)

try:
    from src.core.security_layers import get_security_layers

    SECURITY_AVAILABLE = True
except ImportError:
    SECURITY_AVAILABLE = False


class MemoryType(Enum):
    """Types of memory (like brain regions)"""

    SENSORY = "sensory"  # Immediate perception
    WORKING = "working"  # Short-term, like RAM
    EPISODIC = "episodic"  # Specific events
    SEMANTIC = "semantic"  # Facts and knowledge
    PROCEDURAL = "procedural"  # Skills and habits
    EMOTIONAL = "emotional"  # Feelings and reactions


class MemoryStrength(Enum):
    """How strong a memory connection is"""

    WEAK = 0.2
    MEDIUM = 0.5
    STRONG = 0.8
    VERY_STRONG = 0.95


@dataclass
class Neuron:
    """
    A single memory neuron

    Like a brain neuron:
    - Has activation level
    - Connects to other neurons (synapses)
    - Strengthens with use (Hebbian learning)
    - Forgets over time (decay)
    """

    id: str
    memory_type: MemoryType

    # Content
    content: str
    embedding: List[float] = field(default_factory=list)

    # Activation
    activation: float = 0.0
    threshold: float = 0.5

    # Connections (synapses)
    connections: Dict[str, float] = field(default_factory=dict)  # neuron_id -> strength

    # Temporal
    created_at: datetime = field(default_factory=datetime.now)
    last_activated: datetime = field(default_factory=datetime.now)
    access_count: int = 0

    # Emotional valence (-1 to 1)
    emotional_valence: float = 0.0

    # Importance (0-1)
    importance: float = 0.5

    # Tags for retrieval
    tags: Set[str] = field(default_factory=set)


@dataclass
class Synapse:
    """Connection between two neurons"""

    source_id: str
    target_id: str
    strength: float  # 0-1
    last_fired: datetime = field(default_factory=datetime.now)
    firing_count: int = 0


@dataclass
class MemoryCluster:
    """
    A cluster of related memories (like a memory trace in brain)
    """

    id: str
    neurons: List[str]  # Neuron IDs
    activation_level: float = 0.0
    created_at: datetime = field(default_factory=datetime.now)
    last_activated: datetime = field(default_factory=datetime.now)
    cluster_type: str = "general"  # event, person, topic, etc.
    emotional_tag: str = ""


class NeuralMemory:
    """
    Neural-inspired Memory System

    Inspired by:
    - Hebbian learning ("neurons that fire together, wire together")
    - Memory consolidation (短->long term)
    - Attention mechanisms
    - Forgetting curves

    Unlike traditional databases, this learns and adapts!
    """

    def __init__(self):
        # Neurons storage
        self.neurons: Dict[str, Neuron] = {}

        # Synapses (connections)
        self.synapses: Dict[str, Synapse] = {}

        # Memory clusters
        self.clusters: Dict[str, MemoryCluster] = {}

        # Working memory (like RAM)
        self.working_memory: deque = deque(maxlen=7)  # Miller's 7±2

        # Attention window (what's currently in focus)
        self.attention: List[str] = []  # Neuron IDs

        # Settings
        self.decay_rate = 0.01  # Forgetting
        self.consolidation_threshold = 0.7
        self.learning_rate = 0.1

        # Stats
        self.total_activations = 0

    # =========================================================================
    # NEURON MANAGEMENT
    # =========================================================================

    def create_neuron(
        self,
        content: str,
        memory_type: MemoryType,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        tags: Set[str] = None,
    ) -> Neuron:
        """Create a new memory neuron"""

        # Generate embedding (simplified - would use actual embeddings)
        embedding = self._generate_embedding(content)

        neuron = Neuron(
            id=str(hashlib.md5(content.encode()).hexdigest()[:12]),
            memory_type=memory_type,
            content=content,
            embedding=embedding,
            importance=importance,
            emotional_valence=emotional_valence,
            tags=tags or set(),
        )

        self.neurons[neuron.id] = neuron

        # Add to attention
        self.attention.append(neuron.id)

        # Update working memory
        self.working_memory.append(neuron.id)

        logger.debug(f"Created neuron: {neuron.id[:8]} - {content[:30]}...")

        return neuron

    def _generate_embedding(self, content: str) -> List[float]:
        """
        Generate semantic embedding using TF-IDF-like character n-grams.
        This is a REAL offline embedding - not random!

        Uses character trigrams to capture:
        - Word patterns
        - Subword similarity
        - Character-level semantics

        Dimensions: 64 (fixed for mobile efficiency)
        """
        # Normalize content
        content_lower = content.lower().strip()

        # Generate character trigrams (n=3)
        trigrams = {}
        for i in range(len(content_lower) - 2):
            trigram = content_lower[i : i + 3]
            trigrams[trigram] = trigrams.get(trigram, 0) + 1

        # Create fixed-size vector using hash trick (deterministic, not random!)
        embedding = [0.0] * 64

        # Map trigrams to fixed positions using deterministic hash
        for trigram, count in trigrams.items():
            # Deterministic position based on trigram characters
            pos = (
                (ord(trigram[0]) * 256 + ord(trigram[1])) * 256 + ord(trigram[2])
            ) % 64
            embedding[pos] = count

        # Normalize to unit vector
        magnitude = sum(x**2 for x in embedding) ** 0.5
        if magnitude > 0:
            embedding = [x / magnitude for x in embedding]

        return embedding

    def get_neuron(self, neuron_id: str) -> Optional[Neuron]:
        """Get a neuron by ID"""
        return self.neurons.get(neuron_id)

    # =========================================================================
    # NEURAL ACTIVATION (FIRING)
    # =========================================================================

    def activate(self, neuron_id: str, strength: float = 1.0) -> float:
        """Activate a neuron - like firing in the brain"""

        neuron = self.neurons.get(neuron_id)
        if not neuron:
            return 0.0

        # Update activation
        neuron.activation = min(neuron.activation + strength, 1.0)
        neuron.last_activated = datetime.now()
        neuron.access_count += 1
        self.total_activations += 1

        # Fire to connected neurons (propagation)
        self._propagate_activation(neuron, strength)

        return neuron.activation

    def _propagate_activation(self, source_neuron: Neuron, strength: float):
        """Propagate activation to connected neurons"""

        for target_id, connection_strength in source_neuron.connections.items():
            target = self.neurons.get(target_id)
            if target:
                # Activation decays with distance
                propagated = strength * connection_strength * 0.8
                target.activation = min(target.activation + propagated, 1.0)

    def decay_all(self):
        """Apply decay to all neurons (like forgetting)"""

        for neuron in self.neurons.values():
            # Time-based decay
            time_since = (datetime.now() - neuron.last_activated).total_seconds()
            decay = self.decay_rate * (time_since / 3600)  # Per hour

            # Importance affects decay (important memories decay slower)
            importance_factor = 1.0 - (neuron.importance * 0.5)
            decay *= importance_factor

            neuron.activation = max(neuron.activation - decay, 0.0)

    # =========================================================================
    # LEARNING (HEBBIAN)
    # =========================================================================

    def learn(
        self,
        content: str,
        memory_type: MemoryType,
        related_to: List[str] = None,
        importance: float = 0.5,
        emotional_valence: float = 0.0,
        tags: Set[str] = None,
    ) -> Neuron:
        """
        Learn something new - create neuron and strengthen connections

        Hebbian learning: "Neurons that fire together, wire together"
        """

        # Create neuron
        neuron = self.create_neuron(
            content=content,
            memory_type=memory_type,
            importance=importance,
            emotional_valence=emotional_valence,
            tags=tags,
        )

        # Connect to related neurons (Hebbian learning)
        if related_to:
            for related_id in related_to:
                self._strengthen_connection(related_id, neuron.id)

        # Check for cluster formation
        self._maybe_form_cluster(neuron)

        return neuron

    def _strengthen_connection(self, source_id: str, target_id: str):
        """Strengthen connection between two neurons"""

        source = self.neurons.get(source_id)
        target = self.neurons.get(target_id)

        if not source or not target:
            return

        # Hebbian: strengthen if both active
        if source.activation > 0.5 and target.activation > 0.5:
            current_strength = source.connections.get(target_id, 0.3)
            new_strength = min(current_strength + self.learning_rate, 1.0)
            source.connections[target_id] = new_strength

            # Also strengthen reverse
            target.connections[source_id] = new_strength

    def _maybe_form_cluster(self, neuron: Neuron):
        """Form memory cluster if enough connections"""

        # Find strongly connected neurons
        strong_connections = [
            nid for nid, strength in neuron.connections.items() if strength > 0.6
        ]

        if len(strong_connections) >= 2:
            # Check if already in a cluster
            existing_cluster = self._find_cluster(neuron.id)

            if not existing_cluster:
                # Form new cluster
                cluster = MemoryCluster(
                    id=f"cluster_{len(self.clusters)}",
                    neurons=[neuron.id] + strong_connections[:4],
                    cluster_type=neuron.memory_type.value,
                )
                self.clusters[cluster.id] = cluster
                logger.debug(f"Formed cluster: {cluster.id}")

    def _find_cluster(self, neuron_id: str) -> Optional[MemoryCluster]:
        """Find cluster containing neuron"""
        for cluster in self.clusters.values():
            if neuron_id in cluster.neurons:
                return cluster
        return None

    # =========================================================================
    # RECALL (RETRIEVAL)
    # =========================================================================

    async def recall(
        self, query: str, memory_types: List[MemoryType] = None, limit: int = 5
    ) -> List[Neuron]:
        """
        Recall memories similar to query

        Like brain retrieval:
        - Uses embeddings for similarity
        - Considers recency
        - Factors in importance
        """

        # Generate query embedding
        query_embedding = self._generate_embedding(query)

        # Score all neurons
        scored_neurons = []

        for neuron in self.neurons.values():
            # Filter by type
            if memory_types and neuron.memory_type not in memory_types:
                continue

            # Calculate similarity (cosine)
            similarity = self._cosine_similarity(query_embedding, neuron.embedding)

            # Factor in recency
            recency = self._recency_factor(neuron)

            # Factor in importance
            importance = neuron.importance

            # Factor in activation
            activation = neuron.activation

            # Combined score
            score = (
                similarity * 0.4 + recency * 0.2 + importance * 0.2 + activation * 0.2
            )

            scored_neurons.append((neuron, score))

        # Sort by score
        scored_neurons.sort(key=lambda x: x[1], reverse=True)

        # Activate top results
        results = []
        for neuron, score in scored_neurons[:limit]:
            self.activate(neuron.id, strength=score)
            results.append(neuron)

        return results

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity"""
        if not a or not b:
            return 0.0

        dot = sum(x * y for x, y in zip(a, b))
        mag_a = math.sqrt(sum(x * x for x in a))
        mag_b = math.sqrt(sum(x * x for x in b))

        if mag_a == 0 or mag_b == 0:
            return 0.0

        return dot / (mag_a * mag_b)

    def _recency_factor(self, neuron: Neuron) -> float:
        """Calculate recency factor (exponential decay)"""
        hours_ago = (datetime.now() - neuron.last_activated).total_seconds() / 3600
        return math.exp(-0.1 * hours_ago)  # Half-life ~7 hours

    # =========================================================================
    # ATTENTION
    # =========================================================================

    def get_attention(self) -> List[Neuron]:
        """Get currently attended memories"""
        return [self.neurons[nid] for nid in self.attention[-5:] if nid in self.neurons]

    def shift_attention(self, neuron_id: str):
        """Shift attention to new neuron"""
        if neuron_id in self.attention:
            self.attention.remove(neuron_id)
        self.attention.append(neuron_id)

        # Keep attention window small
        if len(self.attention) > 7:
            self.attention = self.attention[-7:]

    # =========================================================================
    # CONSOLIDATION (Short-term -> Long-term)
    # =========================================================================

    async def consolidate(self):
        """Consolidate working memory to long-term"""

        for neuron_id in self.working_memory:
            neuron = self.neurons.get(neuron_id)
            if not neuron:
                continue

            # If accessed enough, strengthen importance
            if neuron.access_count > 3:
                neuron.importance = min(neuron.importance + 0.1, 1.0)
                logger.debug(f"Consolidated: {neuron.id[:8]}")

        # Apply decay to non-consolidated
        self.decay_all()

    # =========================================================================
    # FORGETTING
    # =========================================================================

    def forget_weak(self):
        """Forget weak/unimportant memories"""

        to_remove = []

        for neuron_id, neuron in self.neurons.items():
            # If not accessed for long and low importance
            hours_since = (
                datetime.now() - neuron.last_activated
            ).total_seconds() / 3600

            if hours_since > 24 and neuron.importance < 0.3 and neuron.activation < 0.1:
                to_remove.append(neuron_id)

        for neuron_id in to_remove:
            del self.neurons[neuron_id]

        logger.info(f"Forgot {len(to_remove)} weak memories")

    # =========================================================================
    # ENCRYPTION (SECURE STORAGE)
    # =========================================================================

    def __init__(self, storage_path: str = "data/memory"):
        self.storage_path = storage_path
        self._security = None
        self._encryption_enabled = False

        if SECURITY_AVAILABLE:
            try:
                self._security = get_security_layers()
                self._encryption_enabled = self._security.is_encryption_enabled()
            except Exception as e:
                logger.warning(f"Security layers not available: {e}")

        os.makedirs(storage_path, exist_ok=True)

    def _serialize_neuron(self, neuron: Neuron) -> Dict:
        """Serialize neuron to dict"""
        return {
            "id": neuron.id,
            "memory_type": neuron.memory_type.value,
            "content": neuron.content,
            "embedding": neuron.embedding,
            "activation": neuron.activation,
            "threshold": neuron.threshold,
            "connections": neuron.connections,
            "created_at": neuron.created_at.isoformat(),
            "last_activated": neuron.last_activated.isoformat(),
            "access_count": neuron.access_count,
            "emotional_valence": neuron.emotional_valence,
            "importance": neuron.importance,
            "tags": list(neuron.tags),
        }

    def _deserialize_neuron(self, data: Dict) -> Neuron:
        """Deserialize neuron from dict"""
        return Neuron(
            id=data["id"],
            memory_type=MemoryType(data["memory_type"]),
            content=data["content"],
            embedding=data.get("embedding", []),
            activation=data.get("activation", 0.0),
            threshold=data.get("threshold", 0.5),
            connections=data.get("connections", {}),
            created_at=datetime.fromisoformat(data["created_at"]),
            last_activated=datetime.fromisoformat(data["last_activated"]),
            access_count=data.get("access_count", 0),
            emotional_valence=data.get("emotional_valence", 0.0),
            importance=data.get("importance", 0.5),
            tags=set(data.get("tags", [])),
        )

    def save(self, filepath: Optional[str] = None) -> bool:
        """Save memory to file with optional encryption"""
        if filepath is None:
            filepath = os.path.join(self.storage_path, "neural_memory.json")

        data = {
            "neurons": {
                nid: self._serialize_neuron(n) for nid, n in self.neurons.items()
            },
            "clusters": {
                cid: {
                    "id": c.id,
                    "neurons": c.neurons,
                    "activation_level": c.activation_level,
                    "created_at": c.created_at.isoformat(),
                    "last_activated": c.last_activated.isoformat(),
                    "cluster_type": c.cluster_type,
                    "emotional_tag": c.emotional_tag,
                }
                for cid, c in self.clusters.items()
            },
            "synapses": [
                {
                    "source_id": s.source_id,
                    "target_id": s.target_id,
                    "strength": s.strength,
                    "last_fired": s.last_fired.isoformat(),
                    "firing_count": s.firing_count,
                }
                for s in self.synapses.values()
            ],
            "working_memory": list(self.working_memory),
            "attention": self.attention,
            "saved_at": datetime.now().isoformat(),
        }

        json_data = json.dumps(data, indent=2)

        if self._encryption_enabled and self._security:
            encrypted = self._security.encryption.encrypt(json_data)
            if encrypted:
                json_data = encrypted
                filepath = filepath.replace(".json", ".enc")

        try:
            with open(filepath, "w") as f:
                f.write(json_data)
            logger.info(
                f"Memory saved to {filepath} (encrypted: {self._encryption_enabled})"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save memory: {e}")
            return False

    def load(self, filepath: Optional[str] = None) -> bool:
        """Load memory from file with decryption"""
        if filepath is None:
            enc_path = os.path.join(self.storage_path, "neural_memory.enc")
            json_path = os.path.join(self.storage_path, "neural_memory.json")

            if os.path.exists(enc_path):
                filepath = enc_path
            elif os.path.exists(json_path):
                filepath = json_path
            else:
                logger.info("No saved memory found")
                return False

        try:
            with open(filepath, "r") as f:
                json_data = f.read()

            is_encrypted = filepath.endswith(".enc")

            if is_encrypted and self._encryption_enabled and self._security:
                decrypted = self._security.encryption.decrypt(json_data)
                if decrypted:
                    json_data = decrypted
                else:
                    logger.error("Failed to decrypt memory")
                    return False

            data = json.loads(json_data)

            self.neurons = {
                nid: self._deserialize_neuron(ndata)
                for nid, ndata in data.get("neurons", {}).items()
            }

            self.clusters = {
                cid: MemoryCluster(
                    id=cdata["id"],
                    neurons=cdata["neurons"],
                    activation_level=cdata.get("activation_level", 0.0),
                    created_at=datetime.fromisoformat(cdata["created_at"]),
                    last_activated=datetime.fromisoformat(cdata["last_activated"]),
                    cluster_type=cdata.get("cluster_type", "general"),
                    emotional_tag=cdata.get("emotional_tag", ""),
                )
                for cid, cdata in data.get("clusters", {}).items()
            }

            self.synapses = {
                f"{s['source_id']}_{s['target_id']}": Synapse(
                    source_id=s["source_id"],
                    target_id=s["target_id"],
                    strength=s["strength"],
                    last_fired=datetime.fromisoformat(s["last_fired"]),
                    firing_count=s.get("firing_count", 0),
                )
                for s in data.get("synapses", [])
            }

            self.working_memory = deque(data.get("working_memory", []), maxlen=7)
            self.attention = data.get("attention", [])

            logger.info(f"Memory loaded from {filepath} (encrypted: {is_encrypted})")
            return True
        except Exception as e:
            logger.error(f"Failed to load memory: {e}")
            return False

    # =========================================================================
    # STATUS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get memory statistics"""

        type_counts = {}
        for neuron in self.neurons.values():
            mt = neuron.memory_type.value
            type_counts[mt] = type_counts.get(mt, 0) + 1

        return {
            "total_neurons": len(self.neurons),
            "total_synapses": len(self.synapses),
            "total_clusters": len(self.clusters),
            "by_type": type_counts,
            "working_memory_size": len(self.working_memory),
            "attention_size": len(self.attention),
            "total_activations": self.total_activations,
            "encryption_enabled": self._encryption_enabled,
        }


# ==============================================================================
# INSTANCE
# ==============================================================================

_neural_memory: Optional[NeuralMemory] = None


def get_neural_memory() -> NeuralMemory:
    """Get or create neural memory instance"""
    global _neural_memory
    if _neural_memory is None:
        _neural_memory = NeuralMemory()
    return _neural_memory


__all__ = [
    "NeuralMemory",
    "Neuron",
    "Synapse",
    "MemoryCluster",
    "MemoryType",
    "MemoryStrength",
    "get_neural_memory",
]
