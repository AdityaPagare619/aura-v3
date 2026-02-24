"""
AURA v3 Neural-Validated Planner
Tool-First planning with neural pattern validation

This is AURA's unique approach: Instead of generic external verifiers,
we use AURA's neural memory to validate plans against USER's learned patterns.

This makes planning PERSONALIZED - it knows what the user typically does,
what's important to them, and their emotional relationship with actions.
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class ValidationResult(Enum):
    """Result of neural validation"""

    APPROVED = "approved"
    REJECTED = "rejected"
    NEEDS_REVISION = "needs_revision"
    UNCERTAIN = "uncertain"


@dataclass
class NeuralValidationResult:
    """Result of validating against neural patterns"""

    result: ValidationResult
    confidence: float  # 0-1
    pattern_match_score: float  # How well it matches learned patterns
    importance_score: float  # Based on importance of similar past actions
    emotional_alignment: float  # -1 to 1, matches user's emotional context
    reasons: List[str] = field(default_factory=list)
    suggested_revisions: List[str] = field(default_factory=list)


@dataclass
class JSONPlan:
    """Structured plan from LLM"""

    reasoning: str
    action: str
    params: Dict[str, Any]
    confidence: float
    alternatives: List[Dict] = field(default_factory=list)


class NeuralPatternValidator:
    """
    Validates plans against AURA's neural memory patterns.

    Instead of generic rule checking, this:
    - Checks if plan matches user's learned behavior patterns
    - Considers importance of similar past actions
    - Evaluates emotional alignment with user's current state
    """

    def __init__(self, neural_memory=None):
        self.neural_memory = neural_memory
        self.min_confidence_threshold = 0.5

    async def validate(
        self, plan: JSONPlan, user_context: Dict
    ) -> NeuralValidationResult:
        """
        Validate a plan against neural memory patterns.

        This is what makes AURA unique - personalized validation!
        """
        if not self.neural_memory:
            # No neural memory - fall back to basic validation
            return await self._basic_validation(plan, user_context)

        reasons = []
        pattern_scores = []
        importance_scores = []
        emotional_scores = []

        # 1. Check pattern matching
        pattern_result = await self._check_pattern_match(plan, user_context)
        pattern_scores.append(pattern_result["score"])
        if pattern_result["reason"]:
            reasons.append(pattern_result["reason"])

        # 2. Check importance
        importance_result = await self._check_importance(plan, user_context)
        importance_scores.append(importance_result["score"])
        if importance_result["reason"]:
            reasons.append(importance_result["reason"])

        # 3. Check emotional alignment
        emotional_result = await self._check_emotional_alignment(plan, user_context)
        emotional_scores.append(emotional_result["score"])
        if emotional_result["reason"]:
            reasons.append(emotional_result["reason"])

        # 4. Calculate overall confidence
        # Weighted combination - pattern match matters most
        overall_confidence = (
            pattern_scores[0] * 0.4
            + sum(importance_scores) / max(len(importance_scores), 1) * 0.3
            + (emotional_scores[0] + 1) / 2 * 0.3  # Normalize -1:1 to 0:1
        )

        # 5. Determine result
        if overall_confidence >= self.min_confidence_threshold:
            result = ValidationResult.APPROVED
        elif overall_confidence >= 0.3:
            result = ValidationResult.NEEDS_REVISION
        else:
            result = ValidationResult.REJECTED

        # 6. Generate suggestions if needs revision
        suggestions = []
        if result == ValidationResult.NEEDS_REVISION:
            suggestions = self._generate_suggestions(
                plan, pattern_scores, importance_scores, emotional_scores
            )

        return NeuralValidationResult(
            result=result,
            confidence=overall_confidence,
            pattern_match_score=sum(pattern_scores) / max(len(pattern_scores), 1),
            importance_score=sum(importance_scores) / max(len(importance_scores), 1),
            emotional_alignment=sum(emotional_scores) / max(len(emotional_scores), 1),
            reasons=reasons,
            suggested_revisions=suggestions,
        )

    async def _check_pattern_match(self, plan: JSONPlan, user_context: Dict) -> Dict:
        """Check if action matches user's learned patterns"""

        # Get similar past actions from neural memory
        if not self.neural_memory:
            return {"score": 0.5, "reason": "No neural memory available"}

        try:
            # Search for similar action in neural memory
            query = f"action: {plan.action} context: {user_context.get('current_activity', 'unknown')}"
            neurons = await self.neural_memory.recall(
                query=query, memory_types=["procedural", "episodic"], limit=5
            )

            if not neurons:
                return {
                    "score": 0.3,  # Low - no pattern
                    "reason": "No learned pattern for this action",
                }

            # Calculate pattern strength from synapse connections
            total_strength = 0
            for neuron in neurons:
                if plan.action.lower() in neuron.content.lower():
                    # Check connection strength to current context
                    connections = neuron.connections
                    if connections:
                        avg_strength = sum(connections.values()) / len(connections)
                        total_strength += avg_strength * neuron.importance

            pattern_score = min(total_strength / max(len(neurons), 1), 1.0)

            return {
                "score": pattern_score,
                "reason": f"Matched {len(neurons)} similar experiences",
            }

        except Exception as e:
            logger.warning(f"Pattern matching error: {e}")
            return {"score": 0.4, "reason": "Could not check patterns"}

    async def _check_importance(self, plan: JSONPlan, user_context: Dict) -> Dict:
        """Check importance based on similar past actions"""

        if not self.neural_memory:
            return {"score": 0.5, "reason": "No neural memory"}

        try:
            # Get memories about this action type
            neurons = await self.neural_memory.recall(
                query=plan.action, memory_types=["episodic", "semantic"], limit=3
            )

            if not neurons:
                return {"score": 0.4, "reason": "No importance data available"}

            # Average importance from similar actions
            avg_importance = sum(n.importance for n in neurons) / len(neurons)

            # Consider user context importance
            context_importance = user_context.get("importance", 0.5)

            combined_importance = avg_importance * 0.7 + context_importance * 0.3

            return {
                "score": combined_importance,
                "reason": f"Historical importance: {avg_importance:.2f}",
            }

        except Exception as e:
            return {"score": 0.4, "reason": "Could not check importance"}

    async def _check_emotional_alignment(
        self, plan: JSONPlan, user_context: Dict
    ) -> Dict:
        """Check emotional alignment with user's current state"""

        if not self.neural_memory:
            return {"score": 0.0, "reason": "No neural memory"}

        try:
            # Get user's current emotional state
            user_emotional_valence = user_context.get("emotional_valence", 0.0)

            # Get emotional valence of similar past actions
            neurons = await self.neural_memory.recall(
                query=plan.action, memory_types=["emotional", "episodic"], limit=3
            )

            if not neurons:
                return {
                    "score": 0.0,  # Neutral - no emotional data
                    "reason": "No emotional history for this action",
                }

            # Check alignment
            avg_past_valence = sum(n.emotional_valence for n in neurons) / len(neurons)

            # Calculate alignment (-1 = opposite, 1 = same)
            emotional_alignment = -abs(avg_past_valence - user_emotional_valence) + 1

            reason = (
                "Positive alignment"
                if emotional_alignment > 0.5
                else "Negative alignment"
            )

            return {"score": emotional_alignment, "reason": reason}

        except Exception as e:
            return {"score": 0.0, "reason": "Could not check emotions"}

    def _generate_suggestions(
        self,
        plan: JSONPlan,
        pattern_scores: List[float],
        importance_scores: List[float],
        emotional_scores: List[float],
    ) -> List[str]:
        """Generate revision suggestions based on what failed"""

        suggestions = []

        if sum(pattern_scores) / len(pattern_scores) < 0.3:
            suggestions.append(
                "This doesn't match your typical behavior. Consider a different approach."
            )

        if sum(importance_scores) / len(importance_scores) < 0.3:
            suggestions.append(
                "This action doesn't seem important based on your history."
            )

        if sum(emotional_scores) / len(emotional_scores) < 0:
            suggestions.append(
                "This might not align with how you're feeling right now."
            )

        return suggestions

    async def _basic_validation(
        self, plan: JSONPlan, user_context: Dict
    ) -> NeuralValidationResult:
        """Fallback basic validation without neural memory"""

        # Simple heuristic validation
        has_action = bool(plan.action)
        has_params = isinstance(plan.params, dict)
        reasonable_confidence = plan.confidence >= 0.3

        if has_action and has_params and reasonable_confidence:
            return NeuralValidationResult(
                result=ValidationResult.APPROVED,
                confidence=0.6,
                pattern_match_score=0.5,
                importance_score=0.5,
                emotional_alignment=0.0,
                reasons=["Basic validation passed"],
            )

        return NeuralValidationResult(
            result=ValidationResult.REJECTED,
            confidence=0.2,
            pattern_match_score=0.0,
            importance_score=0.0,
            emotional_alignment=0.0,
            reasons=["Basic validation failed"],
        )


class NeuralValidatedPlanner:
    """
    Main planner that combines Tool-First generation with Neural validation.

    This is AURA's unique approach:
    1. Generate JSON plan (Tool-First) - reduces hallucination
    2. Validate against neural patterns - personalized
    3. Revise if needed - learns from mistakes
    """

    def __init__(self, llm_manager=None, neural_memory=None):
        self.llm_manager = llm_manager
        self.neural_memory = neural_memory
        self.validator = NeuralPatternValidator(neural_memory)

        # Settings
        self.max_revision_attempts = 2
        self.min_approval_confidence = 0.6

    async def create_plan(
        self, user_intent: str, available_tools: List[Dict], user_context: Dict
    ) -> Tuple[JSONPlan, NeuralValidationResult]:
        """
        Create and validate a plan.

        Returns both the plan and validation result.
        """

        # Step 1: Generate JSON plan using LLM
        plan = await self._generate_json_plan(user_intent, available_tools)

        # Step 2: Validate against neural memory
        validation = await self.validator.validate(plan, user_context)

        # Step 3: If needs revision, try to fix
        attempts = 0
        while (
            validation.result == ValidationResult.NEEDS_REVISION
            and attempts < self.max_revision_attempts
        ):
            plan = await self._revise_plan(plan, validation, available_tools)
            validation = await self.validator.validate(plan, user_context)
            attempts += 1

        # Step 4: If still not approved, use best effort
        if validation.result == ValidationResult.REJECTED:
            logger.warning(f"Plan rejected, using best effort: {validation.reasons}")

        return plan, validation

    async def _generate_json_plan(
        self, user_intent: str, available_tools: List[Dict]
    ) -> JSONPlan:
        """Generate JSON plan from LLM (Tool-First approach)"""

        if not self.llm_manager:
            # Fallback without LLM
            return JSONPlan(
                reasoning="No LLM available",
                action="respond_directly",
                params={"message": user_intent},
                confidence=0.3,
            )

        # Build tool schema for prompt
        tool_schema = self._build_tool_schema(available_tools)

        prompt = f"""You are AURA, a personal AI assistant.

Available tools:
{tool_schema}

User request: {user_intent}

Generate a JSON plan. Respond ONLY with valid JSON:
{{
    "reasoning": "Brief explanation",
    "action": "tool_name or respond_directly",
    "params": {{"param": "value"}},
    "confidence": 0.0-1.0,
    "alternatives": [{{"action": "...", "params": {{}}}}]
}}

If no tool fits, use "respond_directly" with a message."""

        try:
            response = await self.llm_manager.chat(
                message=prompt,
                system_prompt="You are AURA. Generate valid JSON plans only.",
                conversation_history=[],
            )

            # Parse JSON from response
            plan_data = self._parse_json_response(response.text)

            return JSONPlan(
                reasoning=plan_data.get("reasoning", ""),
                action=plan_data.get("action", "respond_directly"),
                params=plan_data.get("params", {}),
                confidence=plan_data.get("confidence", 0.5),
                alternatives=plan_data.get("alternatives", []),
            )

        except Exception as e:
            logger.error(f"Plan generation error: {e}")
            return JSONPlan(
                reasoning="Generation failed",
                action="respond_directly",
                params={"message": f"I had trouble planning: {user_intent}"},
                confidence=0.2,
            )

    async def _revise_plan(
        self,
        plan: JSONPlan,
        validation: NeuralValidationResult,
        available_tools: List[Dict],
    ) -> JSONPlan:
        """Revise plan based on validation feedback"""

        revision_prompt = f"""Revise this plan based on feedback.

Original plan: action={plan.action}, params={plan.params}
Reasoning: {plan.reasoning}

Feedback:
{chr(10).join(validation.suggested_revisions)}

Available tools:
{self._build_tool_schema(available_tools)}

Generate revised JSON plan:
{{
    "reasoning": "Explanation of changes",
    "action": "tool_name",
    "params": {{"param": "value"}},
    "confidence": 0.0-1.0
}}"""

        try:
            response = await self.llm_manager.chat(
                message=revision_prompt,
                system_prompt="Revise the plan based on feedback. Output valid JSON.",
                conversation_history=[],
            )

            plan_data = self._parse_json_response(response.text)

            return JSONPlan(
                reasoning=plan_data.get("reasoning", plan.reasoning),
                action=plan_data.get("action", plan.action),
                params=plan_data.get("params", plan.params),
                confidence=plan_data.get("confidence", plan.confidence),
                alternatives=[],
            )

        except Exception as e:
            logger.error(f"Plan revision error: {e}")
            # Return original plan if revision fails
            return plan

    def _build_tool_schema(self, tools: List[Dict]) -> str:
        """Build tool schema for prompt"""
        schema_lines = []
        for tool in tools:
            name = tool.get("name", "unknown")
            desc = tool.get("description", "")
            params = tool.get("parameters", {})
            schema_lines.append(f"- {name}: {desc}")
            if params:
                schema_lines.append(f"  Parameters: {', '.join(params.keys())}")
        return "\n".join(schema_lines)

    def _parse_json_response(self, response: str) -> Dict:
        """Parse JSON from LLM response"""
        import re

        # Try direct JSON parse
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in response
        try:
            match = re.search(r"\{.*\}", response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass

        # Fallback
        return {
            "reasoning": "Parse failed",
            "action": "respond_directly",
            "params": {"message": response},
            "confidence": 0.2,
        }


# Factory function
def create_neural_planner(
    llm_manager=None, neural_memory=None
) -> NeuralValidatedPlanner:
    """Create neural-validated planner"""
    return NeuralValidatedPlanner(llm_manager, neural_memory)
