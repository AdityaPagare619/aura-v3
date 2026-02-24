"""
AURA v3 Neural-Aware Model Router
Dynamic model selection based on neural state + task complexity

Instead of just task complexity, this considers:
- User's current neural attention state
- Emotional valence (urgency)
- Importance of current context
- Working memory capacity

This makes model selection CONTEXT-AWARE, not just complexity-aware.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model complexity tiers"""

    ROUTER = "router"  # Small, fast (LFM2.5-1.2B)
    REASONER = "reasoner"  # Medium (Phi-4-mini)
    EXPERT = "expert"  # Large (DeepSeek-R1-7B)


@dataclass
class ModelConfig:
    """Configuration for a model tier"""

    name: str
    tier: ModelTier
    memory_mb: int
    capabilities: List[str]
    always_loaded: bool = False
    preload_trigger: str = ""


@dataclass
class NeuralDecision:
    """Decision from neural-aware router"""

    selected_tier: ModelTier
    confidence: float
    reasoning: str
    factors: Dict[str, float]
    alternatives: List[ModelTier] = field(default_factory=list)


class NeuralAwareModelRouter:
    """
    Router that considers neural state, not just task complexity.

    This is AURA's unique approach:
    - Task complexity (from research)
    - PLUS: User's neural attention state
    - PLUS: Emotional urgency
    - PLUS: Context importance

    Example:
    - Simple query BUT user stressed → use reasoner (better quality)
    - Complex task BUT user idle → use expert freely
    - User in "flow" → keep same model for consistency
    """

    def __init__(
        self,
        llm_manager=None,
        neural_memory=None,
        config: Dict[str, ModelConfig] = None,
    ):
        self.llm_manager = llm_manager
        self.neural_memory = neural_memory

        # Default model configuration
        self.model_config = config or self._default_config()

        # State tracking
        self.current_tier: ModelTier = ModelTier.ROUTER
        self.tier_switch_count = 0

        # Settings
        self.max_memory_mb = 4096  # Default mobile limit
        self.switch_cooldown_seconds = 5

    def _default_config(self) -> Dict[str, ModelConfig]:
        """Default model configuration"""
        return {
            "router": ModelConfig(
                name="LiquidAI/LFM2.5-1.2B-Thinking",
                tier=ModelTier.ROUTER,
                memory_mb=900,
                capabilities=["planning", "simple_reasoning", "routing"],
                always_loaded=True,
            ),
            "reasoner": ModelConfig(
                name="microsoft/phi-4-mini-flash-reasoning",
                tier=ModelTier.REASONER,
                memory_mb=2400,
                capabilities=["complex_reasoning", "math", "code"],
                preload_trigger="complex",
            ),
            "expert": ModelConfig(
                name="deepseek/DeepSeek-R1-Distill-Qwen-7B",
                tier=ModelTier.EXPERT,
                memory_mb=4800,
                capabilities=["advanced_reasoning", "debugging", "analysis"],
                preload_trigger="advanced",
            ),
        }

    async def select_model(
        self, task_description: str, user_context: Dict[str, Any]
    ) -> NeuralDecision:
        """
        Select best model considering BOTH task complexity AND neural state.

        Returns decision with reasoning.
        """

        # Step 1: Analyze task complexity
        task_complexity = self._analyze_task_complexity(task_description)

        # Step 2: Get neural state
        neural_state = await self._get_neural_state(user_context)

        # Step 3: Calculate combined score for each tier
        tier_scores = {}

        for tier_name, config in self.model_config.items():
            score = 0.0
            factors = {}

            # Factor 1: Task complexity match (40%)
            complexity_match = self._calculate_complexity_match(
                task_complexity, config.capabilities
            )
            score += complexity_match * 0.4
            factors["task_complexity"] = complexity_match

            # Factor 2: Neural importance (25%)
            importance_factor = neural_state.get("context_importance", 0.5)
            # Higher importance = prefer more capable model
            score += importance_factor * 0.25
            factors["neural_importance"] = importance_factor

            # Factor 3: Emotional urgency (20%)
            urgency = abs(neural_state.get("emotional_valence", 0))
            emotional_boost = urgency * 0.2
            score += emotional_boost
            factors["emotional_urgency"] = emotional_boost

            # Factor 4: Working memory availability (15%)
            memory_factor = neural_state.get("working_memory_available", 0.7)
            score += memory_factor * 0.15
            factors["working_memory"] = memory_factor

            tier_scores[tier_name] = {
                "score": score,
                "factors": factors,
                "config": config,
            }

        # Step 4: Select best tier
        best_tier = max(tier_scores.items(), key=lambda x: x[1]["score"])
        best_name = best_tier[0]
        best_score = best_tier[1]

        # Step 5: Consider switching cost
        current_config = self.model_config.get(self.current_tier.value)

        # If current model is close in score, stay with it (reduce switching)
        if current_config:
            current_score = tier_scores.get(self.current_tier.value, {}).get("score", 0)
            score_diff = best_score["score"] - current_score

            if score_diff < 0.15:  # Less than 15% difference
                logger.info(
                    f"Staying with current tier {self.current_tier.value} (score diff: {score_diff:.2f})"
                )
                return NeuralDecision(
                    selected_tier=self.current_tier,
                    confidence=current_score,
                    reasoning="Staying with current model - minimal improvement from switching",
                    factors=tier_scores.get(self.current_tier.value, {}).get(
                        "factors", {}
                    ),
                    alternatives=[ModelTier(best_name)],
                )

        # Step 6: Generate reasoning
        reasoning = self._generate_reasoning(
            best_name, best_score["factors"], task_complexity, neural_state
        )

        # Get alternatives (other tiers with decent scores)
        alternatives = [
            ModelTier(name)
            for name, data in tier_scores.items()
            if data["score"] > 0.3 and name != best_name
        ]

        return NeuralDecision(
            selected_tier=ModelTier(best_name),
            confidence=best_score["score"],
            reasoning=reasoning,
            factors=best_score["factors"],
            alternatives=alternatives,
        )

    def _analyze_task_complexity(self, task_description: str) -> float:
        """Analyze task complexity (0-1 scale)"""

        # Keywords indicating complexity
        complex_keywords = [
            "analyze",
            "compare",
            "debug",
            "optimize",
            "design",
            "create",
            "implement",
            "solve",
            "reason",
            "explain why",
            "math",
            "code",
            "algorithm",
            "research",
        ]

        simple_keywords = [
            "what",
            "when",
            "where",
            "who",
            "remind",
            "tell",
            "set",
            "turn",
            "open",
            "send",
            "show",
        ]

        task_lower = task_description.lower()

        complex_count = sum(1 for kw in complex_keywords if kw in task_lower)
        simple_count = sum(1 for kw in simple_keywords if kw in task_lower)

        total = complex_count + simple_count
        if total == 0:
            return 0.3  # Default to low-medium

        # Score: more complex keywords = higher complexity
        complexity = complex_count / max(complex_count + simple_count, 1)

        # Boost if task is long
        if len(task_description) > 100:
            complexity = min(complexity + 0.2, 1.0)

        return complexity

    async def _get_neural_state(self, user_context: Dict) -> Dict[str, float]:
        """Get user's current neural state"""

        state = {
            "context_importance": 0.5,
            "emotional_valence": 0.0,
            "attention_focus": 0.5,
            "working_memory_available": 0.7,
            "current_activity": "idle",
        }

        # From explicit context if provided
        if user_context:
            state["context_importance"] = user_context.get("importance", 0.5)
            state["emotional_valence"] = user_context.get("emotional_valence", 0.0)
            state["current_activity"] = user_context.get("activity", "idle")

        # From neural memory if available
        if self.neural_memory:
            try:
                # Check working memory capacity
                if hasattr(self.neural_memory, "working_memory"):
                    wm = self.neural_memory.working_memory
                    capacity = wm.maxlen or 7
                    used = len(wm)
                    state["working_memory_available"] = 1 - (used / max(capacity, 1))

                # Check attention focus
                if hasattr(self.neural_memory, "attention"):
                    state["attention_focus"] = min(
                        len(self.neural_memory.attention) / 5, 1.0
                    )

                # Get importance from high-importance neurons
                if hasattr(self.neural_memory, "neurons"):
                    high_importance = [
                        n
                        for n in self.neural_memory.neurons.values()
                        if n.importance > 0.7
                    ]
                    if high_importance:
                        avg_importance = sum(
                            n.importance for n in high_importance
                        ) / len(high_importance)
                        state["context_importance"] = max(
                            state["context_importance"], avg_importance
                        )

            except Exception as e:
                logger.warning(f"Error getting neural state: {e}")

        return state

    def _calculate_complexity_match(
        self, task_complexity: float, capabilities: List[str]
    ) -> float:
        """Calculate how well capabilities match task complexity"""

        if task_complexity < 0.3:
            # Simple task - router is fine
            if "routing" in capabilities or "planning" in capabilities:
                return 0.9
            return 0.5

        elif task_complexity < 0.6:
            # Medium task - reasoner needed
            if "complex_reasoning" in capabilities:
                return 0.9
            return 0.6

        else:
            # Complex task - expert needed
            if "advanced_reasoning" in capabilities:
                return 0.9
            return 0.5

    def _generate_reasoning(
        self,
        selected_tier: str,
        factors: Dict[str, float],
        task_complexity: float,
        neural_state: Dict[str, float],
    ) -> str:
        """Generate human-readable reasoning"""

        reasons = []

        # Task factor
        if task_complexity < 0.3:
            reasons.append("simple task")
        elif task_complexity < 0.6:
            reasons.append("moderate complexity")
        else:
            reasons.append("complex task")

        # Neural factors
        if neural_state.get("emotional_valence", 0) > 0.3:
            reasons.append("user seems positive/engaged")
        elif neural_state.get("emotional_valence", 0) < -0.3:
            reasons.append("user may need careful handling")

        if neural_state.get("context_importance", 0) > 0.7:
            reasons.append("high importance context")

        if neural_state.get("working_memory_available", 1) < 0.3:
            reasons.append("limited working memory")

        return f"Selected {selected_tier}: {', '.join(reasons)}"

    async def preload_if_needed(self, predicted_next_tier: ModelTier):
        """Preload next likely model if memory allows"""

        config = self.model_config.get(predicted_next_tier.value)
        if not config or config.always_loaded:
            return

        # Check if we have memory
        available_memory = self._get_available_memory()

        if available_memory >= config.memory_mb * 1.2:  # 20% buffer
            logger.info(f"Preloading {predicted_next_tier.value} model")
            # In real implementation, would trigger async load
            # await self.llm_manager.load_model(config.name)

    def _get_available_memory(self) -> int:
        """Get available memory in MB"""
        # Would use psutil in real implementation
        # For now, return configured max
        return self.max_memory_mb

    def update_memory_limit(self, limit_mb: int):
        """Update memory limit (e.g., from power manager)"""
        self.max_memory_mb = limit_mb
        logger.info(f"Updated memory limit to {limit_mb}MB")


# Factory function
def create_neural_router(
    llm_manager=None, neural_memory=None, config: Dict[str, ModelConfig] = None
) -> NeuralAwareModelRouter:
    """Create neural-aware model router"""
    return NeuralAwareModelRouter(llm_manager, neural_memory, config)
