"""
AURA v3 Hebbian Self-Correction
Integrates with existing neural memory learning mechanism

Instead of external Generator-Verifier-Reviser (GVR) loop,
we leverage AURA's existing Hebbian learning for self-correction.

When an action:
- SUCCEEDS: Strengthen synapses (neurons that fire together, wire together)
- FAILS: Weaken synapses + try alternative paths via activation spread

This is BIOLOGICALLY inspired and makes AURA truly learn from mistakes!

STANDALONE MODE:
- Works completely without neural_memory injection
- Uses LocalActivationMap for lightweight Hebbian learning
- Persists learned connections to SQLite
"""

import asyncio
import logging
import sqlite3
import json
import os
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class ActionOutcome(Enum):
    """Outcome of an executed action"""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


@dataclass
class HebbianCorrection:
    """Record of correction applied"""

    action_sequence: List[str]
    outcome: ActionOutcome
    synapse_changes: Dict[str, float]  # neuron_id -> strength change
    timestamp: datetime
    context: Dict[str, Any]


class LocalActivationMap:
    """
    Lightweight Hebbian learning without neural memory dependency.

    Implements "neurons that fire together, wire together" principle:
    - Strengthens connections between co-activated concepts
    - Applies time decay to all connections
    - Returns strongest associated concepts

    Mobile-first: designed for low memory footprint.
    """

    def __init__(self, storage_path: Optional[str] = None):
        # connections[concept_a][concept_b] = strength
        self.connections: Dict[str, Dict[str, float]] = {}

        # Track concept metadata
        self.concept_metadata: Dict[str, Dict[str, Any]] = {}

        # Default storage path
        self.storage_path = storage_path or "data/learning/corrections"
        self.db_path = os.path.join(self.storage_path, "hebbian_connections.db")

        # Settings
        self.max_strength = 1.0
        self.min_strength = 0.01
        self.default_strength = 0.1

        # Initialize storage
        self._init_storage()

        # Load existing data
        self._load_from_db()

    def _init_storage(self):
        """Initialize SQLite storage"""
        os.makedirs(self.storage_path, exist_ok=True)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Connections table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS connections (
                concept_a TEXT NOT NULL,
                concept_b TEXT NOT NULL,
                strength REAL NOT NULL DEFAULT 0.1,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                co_activation_count INTEGER DEFAULT 1,
                PRIMARY KEY (concept_a, concept_b)
            )
        """)

        # Concept metadata table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS concept_metadata (
                concept TEXT PRIMARY KEY,
                importance REAL DEFAULT 0.5,
                emotional_valence REAL DEFAULT 0.0,
                access_count INTEGER DEFAULT 0,
                success_count INTEGER DEFAULT 0,
                failure_count INTEGER DEFAULT 0,
                last_accessed TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # Corrections history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS corrections_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                action_sequence TEXT NOT NULL,
                outcome TEXT NOT NULL,
                synapse_changes TEXT,
                context TEXT,
                created_at TEXT NOT NULL
            )
        """)

        # Index for faster lookups
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_connections_concept_a 
            ON connections(concept_a)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_connections_strength 
            ON connections(strength DESC)
        """)

        conn.commit()
        conn.close()
        logger.debug(f"Initialized Hebbian storage at {self.db_path}")

    def _load_from_db(self):
        """Load connections and metadata from SQLite"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Load connections
            cursor.execute("SELECT concept_a, concept_b, strength FROM connections")
            for row in cursor.fetchall():
                concept_a, concept_b, strength = row
                if concept_a not in self.connections:
                    self.connections[concept_a] = {}
                self.connections[concept_a][concept_b] = strength

            # Load metadata
            cursor.execute("""
                SELECT concept, importance, emotional_valence, access_count, 
                       success_count, failure_count 
                FROM concept_metadata
            """)
            for row in cursor.fetchall():
                concept = row[0]
                self.concept_metadata[concept] = {
                    "importance": row[1],
                    "emotional_valence": row[2],
                    "access_count": row[3],
                    "success_count": row[4],
                    "failure_count": row[5],
                }

            conn.close()
            logger.debug(f"Loaded {len(self.connections)} concepts from DB")
        except Exception as e:
            logger.warning(f"Failed to load from DB: {e}")

    def strengthen(self, concept_a: str, concept_b: str, amount: float = 0.1) -> float:
        """
        Strengthen connection when concepts co-activate.
        Hebbian: "neurons that fire together, wire together"

        Returns new connection strength.
        """
        # Normalize concepts
        concept_a = concept_a.lower().strip()
        concept_b = concept_b.lower().strip()

        if concept_a == concept_b:
            return 0.0  # No self-connections

        # Initialize if needed
        if concept_a not in self.connections:
            self.connections[concept_a] = {}
        if concept_b not in self.connections:
            self.connections[concept_b] = {}

        # Get current strength (bidirectional)
        current_a_b = self.connections[concept_a].get(concept_b, self.default_strength)
        current_b_a = self.connections[concept_b].get(concept_a, self.default_strength)

        # Strengthen both directions
        new_strength = min(current_a_b + amount, self.max_strength)
        self.connections[concept_a][concept_b] = new_strength
        self.connections[concept_b][concept_a] = new_strength

        # Persist
        self._save_connection(concept_a, concept_b, new_strength)

        logger.debug(
            f"Strengthened {concept_a}<->{concept_b}: {current_a_b:.3f} -> {new_strength:.3f}"
        )
        return new_strength

    def weaken(self, concept_a: str, concept_b: str, amount: float = 0.15) -> float:
        """
        Weaken connection (anti-Hebbian for failed associations).

        Returns new connection strength.
        """
        concept_a = concept_a.lower().strip()
        concept_b = concept_b.lower().strip()

        if (
            concept_a not in self.connections
            or concept_b not in self.connections[concept_a]
        ):
            return 0.0

        current = self.connections[concept_a].get(concept_b, 0.0)
        new_strength = max(current - amount, self.min_strength)

        self.connections[concept_a][concept_b] = new_strength
        if concept_b in self.connections:
            self.connections[concept_b][concept_a] = new_strength

        # Persist
        self._save_connection(concept_a, concept_b, new_strength)

        logger.debug(
            f"Weakened {concept_a}<->{concept_b}: {current:.3f} -> {new_strength:.3f}"
        )
        return new_strength

    def decay(self, rate: float = 0.01):
        """
        Apply time decay to all connections.
        Simulates forgetting of unused associations.
        """
        to_remove = []

        for concept_a, targets in self.connections.items():
            for concept_b, strength in targets.items():
                new_strength = max(strength - rate, 0.0)

                if new_strength <= self.min_strength:
                    to_remove.append((concept_a, concept_b))
                else:
                    self.connections[concept_a][concept_b] = new_strength

        # Remove very weak connections
        for concept_a, concept_b in to_remove:
            if concept_a in self.connections:
                self.connections[concept_a].pop(concept_b, None)
            self._remove_connection(concept_a, concept_b)

        logger.debug(
            f"Applied decay (rate={rate}), removed {len(to_remove)} weak connections"
        )

    def get_associated(self, concept: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Get strongest associated concepts.

        Returns list of (concept, strength) tuples sorted by strength descending.
        """
        concept = concept.lower().strip()

        if concept not in self.connections:
            return []

        associations = self.connections[concept]
        sorted_assocs = sorted(associations.items(), key=lambda x: x[1], reverse=True)

        return sorted_assocs[:top_k]

    def get_all_associated(
        self, concept: str, min_strength: float = 0.1
    ) -> List[Tuple[str, float]]:
        """Get all associations above minimum strength threshold."""
        concept = concept.lower().strip()

        if concept not in self.connections:
            return []

        return [
            (c, s) for c, s in self.connections[concept].items() if s >= min_strength
        ]

    def co_activate(self, concepts: List[str], amount: float = 0.05):
        """
        Strengthen connections between all concepts that co-activate.
        Used when multiple concepts appear together in context.
        """
        concepts = [c.lower().strip() for c in concepts if c.strip()]

        for i, concept_a in enumerate(concepts):
            for concept_b in concepts[i + 1 :]:
                self.strengthen(concept_a, concept_b, amount)

    def record_outcome(self, concept: str, success: bool):
        """Record success/failure for a concept."""
        concept = concept.lower().strip()

        if concept not in self.concept_metadata:
            self.concept_metadata[concept] = {
                "importance": 0.5,
                "emotional_valence": 0.0,
                "access_count": 0,
                "success_count": 0,
                "failure_count": 0,
            }

        self.concept_metadata[concept]["access_count"] += 1

        if success:
            self.concept_metadata[concept]["success_count"] += 1
            # Positive emotional valence
            current = self.concept_metadata[concept]["emotional_valence"]
            self.concept_metadata[concept]["emotional_valence"] = min(
                current + 0.1, 1.0
            )
        else:
            self.concept_metadata[concept]["failure_count"] += 1
            # Negative emotional valence
            current = self.concept_metadata[concept]["emotional_valence"]
            self.concept_metadata[concept]["emotional_valence"] = max(
                current - 0.15, -1.0
            )

        self._save_concept_metadata(concept)

    def get_concept_score(self, concept: str) -> float:
        """
        Get a combined score for a concept based on success rate and importance.
        """
        concept = concept.lower().strip()

        if concept not in self.concept_metadata:
            return 0.5  # Neutral default

        meta = self.concept_metadata[concept]
        total = meta["success_count"] + meta["failure_count"]

        if total == 0:
            return 0.5

        success_rate = meta["success_count"] / total
        importance = meta["importance"]
        valence = (meta["emotional_valence"] + 1) / 2  # Normalize -1..1 to 0..1

        # Combined score
        return success_rate * 0.5 + importance * 0.3 + valence * 0.2

    def _save_connection(self, concept_a: str, concept_b: str, strength: float):
        """Persist connection to SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            now = datetime.now().isoformat()

            cursor.execute(
                """
                INSERT INTO connections (concept_a, concept_b, strength, created_at, updated_at, co_activation_count)
                VALUES (?, ?, ?, ?, ?, 1)
                ON CONFLICT(concept_a, concept_b) DO UPDATE SET
                    strength = ?,
                    updated_at = ?,
                    co_activation_count = co_activation_count + 1
            """,
                (concept_a, concept_b, strength, now, now, strength, now),
            )

            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to save connection: {e}")

    def _remove_connection(self, concept_a: str, concept_b: str):
        """Remove connection from SQLite."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                DELETE FROM connections 
                WHERE (concept_a = ? AND concept_b = ?) 
                   OR (concept_a = ? AND concept_b = ?)
            """,
                (concept_a, concept_b, concept_b, concept_a),
            )

            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to remove connection: {e}")

    def _save_concept_metadata(self, concept: str):
        """Persist concept metadata to SQLite."""
        try:
            meta = self.concept_metadata.get(concept, {})
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            now = datetime.now().isoformat()

            cursor.execute(
                """
                INSERT INTO concept_metadata 
                    (concept, importance, emotional_valence, access_count, success_count, failure_count, last_accessed, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(concept) DO UPDATE SET
                    importance = ?,
                    emotional_valence = ?,
                    access_count = ?,
                    success_count = ?,
                    failure_count = ?,
                    last_accessed = ?
            """,
                (
                    concept,
                    meta.get("importance", 0.5),
                    meta.get("emotional_valence", 0.0),
                    meta.get("access_count", 0),
                    meta.get("success_count", 0),
                    meta.get("failure_count", 0),
                    now,
                    now,
                    meta.get("importance", 0.5),
                    meta.get("emotional_valence", 0.0),
                    meta.get("access_count", 0),
                    meta.get("success_count", 0),
                    meta.get("failure_count", 0),
                    now,
                ),
            )

            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to save concept metadata: {e}")

    def save_correction_history(self, correction: HebbianCorrection):
        """Save correction to history."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO corrections_history 
                    (action_sequence, outcome, synapse_changes, context, created_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    json.dumps(correction.action_sequence),
                    correction.outcome.value,
                    json.dumps(correction.synapse_changes),
                    json.dumps(correction.context),
                    correction.timestamp.isoformat(),
                ),
            )

            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Failed to save correction history: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Get activation map statistics."""
        total_connections = sum(len(targets) for targets in self.connections.values())

        return {
            "total_concepts": len(self.connections),
            "total_connections": total_connections // 2,  # Bidirectional
            "total_metadata_entries": len(self.concept_metadata),
            "storage_path": self.db_path,
        }


class HebbianSelfCorrector:
    """
    Self-correction using Hebbian learning.

    STANDALONE MODE:
    - Works without neural_memory using LocalActivationMap
    - Lightweight, mobile-first implementation
    - Persists to SQLite

    INTEGRATED MODE:
    - When neural_memory IS available, use it
    - Seamless switching between modes

    Core Hebbian principles:
    - When action succeeds → strengthen relevant synapses
    - When action fails → weaken synapses, try alternative paths

    This is how REAL brains learn!
    """

    def __init__(self, neural_memory=None, storage_path: Optional[str] = None):
        self.neural_memory = neural_memory

        # Always create LocalActivationMap for standalone mode
        self.local_activation_map = LocalActivationMap(storage_path)

        # Learning settings
        self.success_strengthen_factor = 0.2  # How much to strengthen on success
        self.failure_weaken_factor = 0.3  # How much to weaken on failure
        self.spread_activation_decay = 0.5  # For finding alternative paths

        # History for analysis (in-memory, also persisted via LocalActivationMap)
        self.correction_history: List[HebbianCorrection] = []

        # Mode tracking
        self._using_neural_memory = neural_memory is not None

        logger.info(
            f"HebbianSelfCorrector initialized (neural_memory: {self._using_neural_memory})"
        )

    @property
    def is_standalone(self) -> bool:
        """Check if running in standalone mode (no neural memory)."""
        return self.neural_memory is None

    def set_neural_memory(self, neural_memory):
        """Dynamically inject neural memory for integrated mode."""
        self.neural_memory = neural_memory
        self._using_neural_memory = neural_memory is not None
        logger.info(f"Neural memory {'connected' if neural_memory else 'disconnected'}")

    async def record_outcome(
        self, action: str, params: Dict, outcome: ActionOutcome, context: Dict
    ) -> Optional[List[str]]:
        """
        Record action outcome and apply Hebbian learning.

        Returns list of alternative actions to try if failed.

        Works in both standalone and integrated modes.
        """
        # Create action signature
        action_signature = (
            f"{action}:{','.join(f'{k}={v}' for k, v in sorted(params.items()))}"
        )

        # Extract context concepts for Hebbian co-activation
        context_concepts = self._extract_context_concepts(action, params, context)

        synapse_changes = {}

        if outcome == ActionOutcome.SUCCESS:
            # STRENGTHEN synapses for successful actions
            if self.neural_memory:
                related_neurons = await self.neural_memory.recall(
                    query=action, memory_types=["procedural", "episodic"], limit=10
                )
                await self._strengthen_synapses(related_neurons, action_signature)
                synapse_changes = {
                    n.id: self.success_strengthen_factor for n in related_neurons
                }

            # Also strengthen in local activation map (always)
            self._local_strengthen(action, context_concepts, success=True)

            logger.info(
                f"Hebbian learning: Strengthened connections for successful action '{action}'"
            )
            return None  # No alternatives needed

        elif outcome == ActionOutcome.FAILURE:
            # WEAKEN synapses for failed actions
            if self.neural_memory:
                related_neurons = await self.neural_memory.recall(
                    query=action, memory_types=["procedural", "episodic"], limit=10
                )
                await self._weaken_synapses(related_neurons, action_signature)
                synapse_changes = {
                    n.id: -self.failure_weaken_factor for n in related_neurons
                }

            # Weaken in local activation map
            self._local_weaken(action, context_concepts)

            # Find alternative paths
            alternatives = await self._find_alternative_paths(action, params, context)

            logger.info(
                f"Hebbian learning: Weakened connections, found {len(alternatives)} alternatives"
            )

            # Record for history
            correction = HebbianCorrection(
                action_sequence=[action_signature],
                outcome=outcome,
                synapse_changes=synapse_changes,
                timestamp=datetime.now(),
                context=context,
            )
            self.correction_history.append(correction)
            self.local_activation_map.save_correction_history(correction)

            return alternatives

        elif outcome == ActionOutcome.PARTIAL:
            # Partial strengthening
            factor = self.success_strengthen_factor / 2

            if self.neural_memory:
                related_neurons = await self.neural_memory.recall(
                    query=action, memory_types=["procedural", "episodic"], limit=10
                )
                await self._strengthen_synapses(
                    related_neurons, action_signature, factor
                )

            # Partial strengthen in local map
            self._local_strengthen(
                action, context_concepts, success=True, factor=factor
            )

            # Still try alternatives
            alternatives = await self._find_alternative_paths(action, params, context)
            return alternatives

        return None

    def _extract_context_concepts(
        self, action: str, params: Dict, context: Dict
    ) -> List[str]:
        """Extract concepts from action, params, and context for Hebbian learning."""
        concepts = [action]

        # Add param values as concepts
        for key, value in params.items():
            if isinstance(value, str):
                concepts.append(value)

        # Add context values
        for key, value in context.items():
            if isinstance(value, str):
                concepts.append(value)
            elif key == "current_activity":
                concepts.append(str(value))

        return concepts

    def _local_strengthen(
        self,
        action: str,
        context_concepts: List[str],
        success: bool = True,
        factor: float = None,
    ):
        """Strengthen connections in local activation map."""
        factor = factor or self.success_strengthen_factor

        # Co-activate all concepts
        self.local_activation_map.co_activate(
            [action] + context_concepts, amount=factor
        )

        # Record outcome
        self.local_activation_map.record_outcome(action, success=success)

    def _local_weaken(self, action: str, context_concepts: List[str]):
        """Weaken connections in local activation map."""
        # Weaken action connections to context
        for concept in context_concepts:
            self.local_activation_map.weaken(
                action, concept, self.failure_weaken_factor
            )

        # Record failure
        self.local_activation_map.record_outcome(action, success=False)

    async def _strengthen_synapses(
        self, neurons: List, action_signature: str, factor: float = None
    ):
        """Strengthen connections (Hebbian: fire together, wire together)"""
        if not neurons:
            return

        factor = factor or self.success_strengthen_factor

        for neuron in neurons:
            # Increase importance
            neuron.importance = min(neuron.importance + factor, 1.0)

            # Strengthen connections to context neurons
            for conn_id in neuron.connections:
                current_strength = neuron.connections[conn_id]
                neuron.connections[conn_id] = min(
                    current_strength + factor * current_strength, 1.0
                )

            # Add positive emotional valence
            neuron.emotional_valence = min(neuron.emotional_valence + factor * 0.2, 1.0)

    async def _weaken_synapses(self, neurons: List, action_signature: str):
        """Weaken connections for failed actions"""
        if not neurons:
            return

        factor = self.failure_weaken_factor

        for neuron in neurons:
            # Decrease importance
            neuron.importance = max(neuron.importance - factor, 0.1)

            # Weaken connections
            for conn_id in neuron.connections:
                current_strength = neuron.connections[conn_id]
                neuron.connections[conn_id] = max(current_strength - factor, 0.1)

            # Add negative emotional valence
            neuron.emotional_valence = max(
                neuron.emotional_valence - factor * 0.3, -1.0
            )

    async def _find_alternative_paths(
        self, failed_action: str, params: Dict, context: Dict
    ) -> List[str]:
        """
        Find alternative action paths using neural activation spread
        OR local activation map associations.

        Unique approach:
        - Find actions that are CONNECTED in memory
        - But NOT the failed action
        """
        alternatives = []

        # Try local activation map first (always available)
        local_alternatives = self.local_activation_map.get_associated(
            failed_action, top_k=10
        )
        for concept, strength in local_alternatives:
            if concept != failed_action and strength > 0.3:
                # Only add if it looks like an action (simple heuristic)
                if not concept.startswith("_") and len(concept) > 2:
                    alternatives.append(concept)

        # Try neural memory if available
        if self.neural_memory:
            try:
                # Get neurons for the failed action
                failed_neurons = await self.neural_memory.recall(
                    query=failed_action, memory_types=["procedural"], limit=5
                )

                if failed_neurons:
                    # Find connected neurons (alternatives)
                    for neuron in failed_neurons:
                        # Check connected neurons
                        for conn_id, strength in neuron.connections.items():
                            if strength > 0.3:  # Strong connection
                                conn_neuron = self.neural_memory.get_neuron(conn_id)
                                if conn_neuron:
                                    # Extract action name
                                    if conn_neuron.memory_type.value == "procedural":
                                        action_name = conn_neuron.content.split(":")[0]
                                        if (
                                            action_name != failed_action
                                            and action_name not in alternatives
                                        ):
                                            alternatives.append(action_name)

                # Also try neurons that were activated around same time
                if context.get("current_activity"):
                    context_neurons = await self.neural_memory.recall(
                        query=context["current_activity"],
                        memory_types=["episodic"],
                        limit=5,
                    )

                    for neuron in context_neurons:
                        if neuron.connections and neuron not in failed_neurons:
                            for conn_id, strength in neuron.connections.items():
                                if strength > 0.5:  # Strong connection
                                    conn = self.neural_memory.get_neuron(conn_id)
                                    if conn and conn.memory_type.value == "procedural":
                                        action = conn.content.split(":")[0]
                                        if (
                                            action != failed_action
                                            and action not in alternatives
                                        ):
                                            alternatives.append(action)

            except Exception as e:
                logger.warning(f"Error finding alternatives from neural memory: {e}")

        return alternatives[:5]  # Limit to 5 alternatives

    async def get_action_recommendation(
        self, desired_action: str, context: Dict
    ) -> Dict[str, Any]:
        """
        Get action recommendation considering learned history.

        Returns:
        - recommended_action: Best action based on history
        - confidence: How confident we are
        - reason: Why this action
        - alternatives: Other options

        Works in both standalone and integrated modes.
        """
        # Check local activation map first (always available)
        local_score = self.local_activation_map.get_concept_score(desired_action)
        local_alternatives = self.local_activation_map.get_associated(
            desired_action, top_k=3
        )

        # If neural memory available, also check there
        if self.neural_memory:
            neurons = await self.neural_memory.recall(
                query=desired_action, memory_types=["procedural"], limit=5
            )

            if neurons:
                # Find most successful related action
                best_neuron = None
                best_score = -1

                for neuron in neurons:
                    # Score = importance * (1 + emotional_valence)
                    # Higher importance + positive emotion = better
                    score = neuron.importance * (1 + neuron.emotional_valence)

                    if score > best_score:
                        best_score = score
                        best_neuron = neuron

                if best_neuron and best_score > 0.3:
                    recommended = best_neuron.content.split(":")[0]

                    return {
                        "recommended_action": recommended,
                        "confidence": min(best_score, 1.0),
                        "reason": f"Based on past {best_neuron.access_count} uses with {best_neuron.importance:.0%} importance",
                        "alternatives": [
                            n.content.split(":")[0]
                            for n in neurons
                            if n.id != best_neuron.id
                        ][:3],
                        "mode": "integrated",
                    }

        # Fall back to local activation map
        if local_score > 0.6:
            return {
                "recommended_action": desired_action,
                "confidence": local_score,
                "reason": f"Good success rate in local history (score: {local_score:.2f})",
                "alternatives": [c for c, s in local_alternatives],
                "mode": "standalone",
            }
        elif local_alternatives and local_alternatives[0][1] > 0.5:
            best_alt = local_alternatives[0][0]
            return {
                "recommended_action": best_alt,
                "confidence": local_alternatives[0][1],
                "reason": f"Strongly associated alternative (strength: {local_alternatives[0][1]:.2f})",
                "alternatives": [desired_action]
                + [c for c, s in local_alternatives[1:]],
                "mode": "standalone",
            }

        return {
            "recommended_action": desired_action,
            "confidence": 0.5,
            "reason": "No strong learning history available",
            "alternatives": [c for c, s in local_alternatives],
            "mode": "standalone" if not self.neural_memory else "integrated",
        }

    def record_user_correction(
        self, incorrect_action: str, correct_action: str, context: Dict = None
    ):
        """
        Track corrections made by user.
        Strengthen correct paths, weaken incorrect ones.

        This is the key self-correction feature!
        """
        context = context or {}
        context_concepts = list(context.values()) if context else []

        # Weaken incorrect path
        self.local_activation_map.weaken(incorrect_action, correct_action, 0.2)
        for concept in context_concepts:
            if isinstance(concept, str):
                self.local_activation_map.weaken(incorrect_action, concept, 0.1)
        self.local_activation_map.record_outcome(incorrect_action, success=False)

        # Strengthen correct path
        self.local_activation_map.strengthen(
            correct_action, correct_action, 0.0
        )  # No-op but creates entry
        for concept in context_concepts:
            if isinstance(concept, str):
                self.local_activation_map.strengthen(correct_action, concept, 0.15)
        self.local_activation_map.record_outcome(correct_action, success=True)

        # Create correction record
        correction = HebbianCorrection(
            action_sequence=[incorrect_action, correct_action],
            outcome=ActionOutcome.SUCCESS,  # Correction was made
            synapse_changes={
                incorrect_action: -0.2,
                correct_action: 0.15,
            },
            timestamp=datetime.now(),
            context={"user_corrected": True, "original_context": context},
        )
        self.correction_history.append(correction)
        self.local_activation_map.save_correction_history(correction)

        logger.info(
            f"User correction recorded: '{incorrect_action}' -> '{correct_action}'"
        )

    def apply_decay(self, rate: float = 0.01):
        """Apply time decay to all learned connections."""
        self.local_activation_map.decay(rate)
        logger.debug(f"Applied decay (rate={rate})")

    def get_correction_stats(self) -> Dict:
        """Get statistics about corrections made"""
        local_stats = self.local_activation_map.get_stats()

        if not self.correction_history:
            return {
                "total_corrections": 0,
                "success_rate": 0.0,
                "most_common_failures": [],
                "mode": "standalone" if not self.neural_memory else "integrated",
                **local_stats,
            }

        failures = [
            c for c in self.correction_history if c.outcome == ActionOutcome.FAILURE
        ]

        # Count failure types
        failure_counts = {}
        for correction in failures:
            for action in correction.action_sequence:
                action_name = action.split(":")[0]
                failure_counts[action_name] = failure_counts.get(action_name, 0) + 1

        return {
            "total_corrections": len(self.correction_history),
            "failures": len(failures),
            "success_rate": 1 - (len(failures) / len(self.correction_history))
            if self.correction_history
            else 0.0,
            "most_common_failures": sorted(
                failure_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
            "mode": "standalone" if not self.neural_memory else "integrated",
            **local_stats,
        }


# Factory function
def create_hebbian_corrector(
    neural_memory=None, storage_path: Optional[str] = None
) -> HebbianSelfCorrector:
    """Create Hebbian self-corrector (works standalone or with neural_memory)"""
    return HebbianSelfCorrector(neural_memory=neural_memory, storage_path=storage_path)


# Singleton for global access
_hebbian_corrector: Optional[HebbianSelfCorrector] = None


def get_hebbian_corrector(
    neural_memory=None, storage_path: Optional[str] = None
) -> HebbianSelfCorrector:
    """Get or create global Hebbian corrector instance"""
    global _hebbian_corrector
    if _hebbian_corrector is None:
        _hebbian_corrector = create_hebbian_corrector(neural_memory, storage_path)
    elif neural_memory is not None and _hebbian_corrector.neural_memory is None:
        # Inject neural memory if provided and not already set
        _hebbian_corrector.set_neural_memory(neural_memory)
    return _hebbian_corrector
