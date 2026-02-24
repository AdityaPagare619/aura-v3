"""
AURA v3 Hebbian Self-Correction
Integrates with existing neural memory learning mechanism

Instead of external Generator-Verifier-Reviser (GVR) loop,
we leverage AURA's existing Hebbian learning for self-correction.

When an action:
- SUCCEEDS: Strengthen synapses (neurons that fire together, wire together)
- FAILS: Weaken synapses + try alternative paths via activation spread

This is BIOLOGICALLY inspired and makes AURA truly learn from mistakes!
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

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


class HebbianSelfCorrector:
    """
    Self-correction using Hebbian learning.

    Instead of separate GVR loop, this uses AURA's neural memory:
    - When action succeeds → strengthen relevant synapses
    - When action fails → weaken synapses, try alternative paths

    This is how REAL brains learn!
    """

    def __init__(self, neural_memory=None):
        self.neural_memory = neural_memory

        # Learning settings
        self.success_strengthen_factor = 0.2  # How much to strengthen on success
        self.failure_weaken_factor = 0.3  # How much to weaken on failure
        self.spread_activation_decay = 0.5  # For finding alternative paths

        # History for analysis
        self.correction_history: List[HebbianCorrection] = []

    async def record_outcome(
        self, action: str, params: Dict, outcome: ActionOutcome, context: Dict
    ) -> Optional[List[str]]:
        """
        Record action outcome and apply Hebbian learning.

        Returns list of alternative actions to try if failed.
        """

        if not self.neural_memory:
            logger.warning("No neural memory - cannot learn from outcome")
            return None

        # Create action signature
        action_signature = (
            f"{action}:{','.join(f'{k}={v}' for k, v in sorted(params.items()))}"
        )

        # Find related neurons
        related_neurons = await self.neural_memory.recall(
            query=action, memory_types=["procedural", "episodic"], limit=10
        )

        synapse_changes = {}

        if outcome == ActionOutcome.SUCCESS:
            # STRENGTHEN synapses for successful actions
            await self._strengthen_synapses(related_neurons, action_signature)
            synapse_changes = {
                n.id: self.success_strengthen_factor for n in related_neurons
            }

            logger.info(
                f"Hebbian learning: Strengthened {len(related_neurons)} neurons for successful action"
            )

            return None  # No alternatives needed

        elif outcome == ActionOutcome.FAILURE:
            # WEAKEN synapses for failed actions
            await self._weaken_synapses(related_neurons, action_signature)
            synapse_changes = {
                n.id: -self.failure_weaken_factor for n in related_neurons
            }

            # Find alternative paths via activation spread
            alternatives = await self._find_alternative_paths(action, params, context)

            logger.info(
                f"Hebbian learning: Weakened {len(related_neurons)} neurons, found {len(alternatives)} alternatives"
            )

            # Record for history
            self.correction_history.append(
                HebbianCorrection(
                    action_sequence=[action_signature],
                    outcome=outcome,
                    synapse_changes=synapse_changes,
                    timestamp=datetime.now(),
                    context=context,
                )
            )

            return alternatives

        elif outcome == ActionOutcome.PARTIAL:
            # Partial strengthening
            factor = self.success_strengthen_factor / 2
            await self._strengthen_synapses(related_neurons, action_signature, factor)

            # Still try alternatives
            alternatives = await self._find_alternative_paths(action, params, context)

            return alternatives

        return None

    async def _strengthen_synapses(
        self, neurons: List, action_signature: str, factor: float = None
    ):
        """Strengthen connections (Hebbian: fire together, wire together)"""

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
        Find alternative action paths using neural activation spread.

        This is AURA's unique approach:
        - Instead of generic alternatives
        - Find actions that are CONNECTED in neural memory
        - But NOT the failed action
        """

        if not self.neural_memory:
            return []

        try:
            # Get neurons for the failed action
            failed_neurons = await self.neural_memory.recall(
                query=failed_action, memory_types=["procedural"], limit=5
            )

            if not failed_neurons:
                return []

            # Find connected neurons (alternatives)
            alternative_actions = []
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
                                    and action_name not in alternative_actions
                                ):
                                    alternative_actions.append(action_name)

            # Also try neurons that were activated around same time
            # but didn't fail (context neurons)
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
                                        and action not in alternative_actions
                                    ):
                                        alternative_actions.append(action)

            return alternative_actions[:5]  # Limit to 5 alternatives

        except Exception as e:
            logger.warning(f"Error finding alternatives: {e}")
            return []

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
        """

        if not self.neural_memory:
            return {
                "recommended_action": desired_action,
                "confidence": 0.5,
                "reason": "No learning history available",
                "alternatives": [],
            }

        # Check history for similar desired action
        neurons = await self.neural_memory.recall(
            query=desired_action, memory_types=["procedural"], limit=5
        )

        if not neurons:
            return {
                "recommended_action": desired_action,
                "confidence": 0.4,
                "reason": "No prior experience with this action",
                "alternatives": [],
            }

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
                    n.content.split(":")[0] for n in neurons if n.id != best_neuron.id
                ][:3],
            }

        return {
            "recommended_action": desired_action,
            "confidence": 0.5,
            "reason": "Using default action",
            "alternatives": [],
        }

    def get_correction_stats(self) -> Dict:
        """Get statistics about corrections made"""

        if not self.correction_history:
            return {
                "total_corrections": 0,
                "success_rate": 0.0,
                "most_common_failures": [],
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
            "success_rate": 1 - (len(failures) / len(self.correction_history)),
            "most_common_failures": sorted(
                failure_counts.items(), key=lambda x: x[1], reverse=True
            )[:5],
        }


# Factory function
def create_hebbian_corrector(neural_memory=None) -> HebbianSelfCorrector:
    """Create Hebbian self-corrector"""
    return HebbianSelfCorrector(neural_memory)
