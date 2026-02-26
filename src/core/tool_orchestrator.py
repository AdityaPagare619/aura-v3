"""
AURA v3 Tool Execution Orchestrator
Implements "Tool-First, Not Token-First" architecture

Per research: "Rather than forcing a SLM to simulate entire workflows in raw text,
the SLM should be strictly constrained to generating structured JSON plans"

This orchestrator:
1. Validates JSON plans from LLM
2. Executes tools deterministically
3. Handles errors and rollbacks
4. Returns concise results to LLM
"""

import asyncio
import json
import logging
import warnings
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

# Import unified JSONPlan and ToolAction from neural_validated_planner
from src.core.neural_validated_planner import (
    JSONPlan,
    ToolAction,
)

logger = logging.getLogger(__name__)


class PlanStatus(Enum):
    """Status of a plan execution"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"
    VALIDATION_FAILED = "validation_failed"


class ActionRisk(Enum):
    """Risk level of an action"""

    SAFE = "safe"  # No side effects
    LOW = "low"  # Minor side effects
    MEDIUM = "medium"  # Significant changes
    HIGH = "high"  # Irreversible changes
    DANGEROUS = "dangerous"  # Could cause harm


# DEPRECATED: These are kept for backward compatibility
# Use imports from src.core.neural_validated_planner instead
def _deprecated_tool_action():
    warnings.warn(
        "ToolAction from tool_orchestrator is deprecated. "
        "Import from src.core.neural_validated_planner instead.",
        DeprecationWarning,
        stacklevel=3,
    )


def _deprecated_json_plan():
    warnings.warn(
        "JSONPlan from tool_orchestrator is deprecated. "
        "Import from src.core.neural_validated_planner instead.",
        DeprecationWarning,
        stacklevel=3,
    )


@dataclass
class ExecutionResult:
    """Result of executing an action"""

    action: ToolAction
    status: PlanStatus
    result: Any = None
    error: Optional[str] = None
    duration_ms: int = 0
    risk_level: ActionRisk = ActionRisk.SAFE


class PlanValidator:
    """
    Validates JSON plans from LLM before execution.
    Prevents hallucinations by enforcing structure.
    """

    def __init__(self):
        self.required_fields = ["actions"]
        self.valid_actions = self._load_valid_actions()

    def _load_valid_actions(self) -> Dict[str, Dict]:
        """Load valid actions and their schemas"""
        return {
            "open_app": {
                "params": ["app_name"],
                "risk": ActionRisk.LOW,
                "requires_confirmation": False,
            },
            "close_app": {
                "params": ["app_name"],
                "risk": ActionRisk.LOW,
                "requires_confirmation": False,
            },
            "tap_screen": {
                "params": ["x", "y"],
                "risk": ActionRisk.MEDIUM,
                "requires_confirmation": False,
            },
            "swipe_screen": {
                "params": ["direction"],
                "risk": ActionRisk.MEDIUM,
                "requires_confirmation": False,
            },
            "type_text": {
                "params": ["text"],
                "risk": ActionRisk.LOW,
                "requires_confirmation": False,
            },
            "send_message": {
                "params": ["contact", "message"],
                "risk": ActionRisk.HIGH,
                "requires_confirmation": True,
            },
            "make_call": {
                "params": ["contact"],
                "risk": ActionRisk.HIGH,
                "requires_confirmation": True,
            },
            "get_notifications": {
                "params": [],
                "risk": ActionRisk.SAFE,
                "requires_confirmation": False,
            },
            "take_screenshot": {
                "params": [],
                "risk": ActionRisk.SAFE,
                "requires_confirmation": False,
            },
            "get_current_app": {
                "params": [],
                "risk": ActionRisk.SAFE,
                "requires_confirmation": False,
            },
        }

    def validate(self, plan_json: Dict) -> Tuple[bool, Optional[str]]:
        """
        Validate a plan from LLM.
        Returns: (is_valid, error_message)
        """
        # Check required fields
        for field in self.required_fields:
            if field not in plan_json:
                return False, f"Missing required field: {field}"

        # Validate actions
        actions = plan_json.get("actions", [])
        if not actions:
            return False, "Plan has no actions"

        if len(actions) > 10:
            return False, "Plan has too many actions (max 10)"

        # Validate each action
        for i, action in enumerate(actions):
            if not isinstance(action, dict):
                return False, f"Action {i} is not a dictionary"

            if "action" not in action:
                return False, f"Action {i} missing 'action' field"

            action_name = action["action"]
            if action_name not in self.valid_actions:
                return False, f"Unknown action: {action_name}"

            # Validate params
            valid_schema = self.valid_actions[action_name]
            required_params = valid_schema.get("params", [])
            for param in required_params:
                if param not in action.get("params", {}):
                    return (
                        False,
                        f"Action {action_name} missing required param: {param}",
                    )

        return True, None

    def get_risk_level(self, action: str) -> ActionRisk:
        """Get risk level for an action"""
        return self.valid_actions.get(action, {}).get("risk", ActionRisk.MEDIUM)

    def requires_confirmation(self, action: str) -> bool:
        """Check if action requires user confirmation"""
        return self.valid_actions.get(action, {}).get("requires_confirmation", False)


class ToolOrchestrator:
    """
    Deterministic tool execution orchestrator.

    Takes JSON plans from LLM and executes them safely.
    This is the key to hallucination-free operation.
    """

    def __init__(self, tool_registry=None):
        self.tool_registry = tool_registry
        self.validator = PlanValidator()
        self.execution_history: List[Dict] = []
        self.tool_handlers: Dict[str, Callable] = {}

    def register_handler(self, tool_name: str, handler: Callable):
        """Register a tool handler"""
        self.tool_handlers[tool_name] = handler
        logger.info(f"Registered handler for tool: {tool_name}")

    async def execute_plan(
        self, plan: JSONPlan, confirm_callback: Optional[Callable] = None
    ) -> List[ExecutionResult]:
        """
        Execute a validated JSON plan.

        Args:
            plan: The JSON plan to execute
            confirm_callback: Optional callback for user confirmation

        Returns:
            List of execution results
        """
        results = []

        # Check if any action requires confirmation
        needs_confirmation = False
        for action in plan.actions:
            if self.validator.requires_confirmation(action.action):
                needs_confirmation = True
                break

        if needs_confirmation and confirm_callback:
            # Request confirmation
            confirmed = await confirm_callback(plan)
            if not confirmed:
                return [
                    ExecutionResult(
                        action=ToolAction(action="confirmation", params={}),
                        status=PlanStatus.FAILED,
                        error="User rejected confirmation",
                    )
                ]

        # Execute each action in sequence
        for action in plan.actions:
            result = await self._execute_single_action(action)
            results.append(result)

            # Stop on failure if not recoverable
            if result.status == PlanStatus.FAILED:
                if not self._is_recoverable(action):
                    logger.warning(f"Action {action.action} failed, stopping plan")
                    break

        # Store execution history
        self.execution_history.append(
            {
                "plan": plan.to_dict(),
                "results": [r.__dict__ for r in results],
                "timestamp": datetime.now().isoformat(),
            }
        )

        return results

    async def _execute_single_action(self, action: ToolAction) -> ExecutionResult:
        """Execute a single action"""
        start_time = datetime.now()

        try:
            # Get handler
            handler = self.tool_handlers.get(action.action)
            if not handler:
                return ExecutionResult(
                    action=action,
                    status=PlanStatus.FAILED,
                    error=f"No handler for action: {action.action}",
                )

            # Execute with timeout
            result = await asyncio.wait_for(
                handler(**action.params),
                timeout=30.0,  # 30 second timeout per action
            )

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            return ExecutionResult(
                action=action,
                status=PlanStatus.COMPLETED,
                result=result,
                duration_ms=duration_ms,
                risk_level=self.validator.get_risk_level(action.action),
            )

        except asyncio.TimeoutError:
            return ExecutionResult(
                action=action,
                status=PlanStatus.FAILED,
                error="Action timed out",
                duration_ms=30000,
                risk_level=self.validator.get_risk_level(action.action),
            )

        except Exception as e:
            return ExecutionResult(
                action=action,
                status=PlanStatus.FAILED,
                error=str(e),
                duration_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                risk_level=self.validator.get_risk_level(action.action),
            )

    def _is_recoverable(self, action: ToolAction) -> bool:
        """Check if failure is recoverable"""
        # Read-only actions are recoverable
        safe_actions = ["get_notifications", "get_current_app", "take_screenshot"]
        return action.action in safe_actions

    def parse_llm_response(self, llm_text: str) -> Optional[JSONPlan]:
        """
        Parse LLM response into JSON plan.
        Tries to extract JSON from raw text.
        """
        # Try direct JSON parse first
        try:
            data = json.loads(llm_text)
            is_valid, error = self.validator.validate(data)
            if is_valid:
                return self._dict_to_plan(data)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in text
        import re

        json_pattern = r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}"
        matches = re.findall(json_pattern, llm_text, re.DOTALL)

        for match in matches:
            try:
                data = json.loads(match)
                is_valid, error = self.validator.validate(data)
                if is_valid:
                    return self._dict_to_plan(data)
            except:
                continue

        # Failed to parse
        logger.warning("Could not parse LLM response as JSON plan")
        return None

    def _dict_to_plan(self, data: Dict) -> JSONPlan:
        """Convert dict to JSONPlan"""
        actions = []
        for a in data.get("actions", []):
            actions.append(
                ToolAction(
                    action=a.get("action", ""),
                    params=a.get("params", {}),
                    tool_name=a.get("tool", ""),
                )
            )

        return JSONPlan(
            reasoning=data.get("reasoning", ""),
            actions=actions,
            confidence=data.get("confidence", 0.5),
            alternatives=data.get("alternatives", []),
            requires_confirmation=data.get("requires_confirmation", False),
        )

    def get_execution_stats(self) -> Dict:
        """Get execution statistics"""
        if not self.execution_history:
            return {"total_plans": 0}

        total = len(self.execution_history)
        successful = sum(
            1
            for h in self.execution_history
            if all(r["status"] == "completed" for r in h["results"])
        )

        return {
            "total_plans": total,
            "successful": successful,
            "success_rate": successful / total if total > 0 else 0,
        }


# Global instance
_orchestrator: Optional[ToolOrchestrator] = None


def get_orchestrator() -> ToolOrchestrator:
    """Get or create the global orchestrator"""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = ToolOrchestrator()
    return _orchestrator
