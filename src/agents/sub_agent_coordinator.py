"""
AURA v3 Sub-Agent Coordinator
============================

Coordinates all sub-agents (healthcare, social, tasks) and routes
requests to the appropriate agent based on context.

This enables:
- Unified interface for all sub-agents
- Automatic routing based on user intent
- Shared memory and learning across agents
- Conflict prevention between agents
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class SubAgentType(Enum):
    """Types of sub-agents in AURA"""

    HEALTHCARE = "healthcare"
    SOCIAL = "social"
    TASKS = "tasks"
    CREATIVE = "creative"
    RESEARCH = "research"


@dataclass
class AgentRequest:
    """Request to a sub-agent"""

    request_type: str
    user_input: str
    context: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5  # 1-10
    requires_confirmation: bool = False


@dataclass
class AgentResponse:
    """Response from a sub-agent"""

    agent_type: SubAgentType
    message: str
    actions_taken: List[Dict] = field(default_factory=list)
    confidence: float = 1.0
    requires_user_action: bool = False


class SubAgentCoordinator:
    """
    Coordinates all sub-agents in AURA.

    Acts as the central router that:
    1. Receives requests from the main agent loop
    2. Routes to appropriate sub-agent
    3. Combines responses
    4. Handles conflicts between agents
    """

    def __init__(self, neural_memory=None):
        self.neural_memory = neural_memory

        # Sub-agent references
        self._healthcare_agent = None
        self._social_agent = None
        self._task_agent = None

        # Agent state
        self._active_agents: Dict[SubAgentType, bool] = {
            SubAgentType.HEALTHCARE: False,
            SubAgentType.SOCIAL: False,
            SubAgentType.TASKS: False,
            SubAgentType.CREATIVE: False,
            SubAgentType.RESEARCH: False,
        }

        # Recent activity for conflict detection
        self._recent_actions: List[Dict] = []

    def register_agent(self, agent_type: SubAgentType, agent: Any):
        """Register a sub-agent"""
        if agent_type == SubAgentType.HEALTHCARE:
            self._healthcare_agent = agent
        elif agent_type == SubAgentType.SOCIAL:
            self._social_agent = agent
        elif agent_type == SubAgentType.TASKS:
            self._task_agent = agent

        self._active_agents[agent_type] = True
        logger.info(f"Registered sub-agent: {agent_type.value}")

    async def process_request(self, request: AgentRequest) -> List[AgentResponse]:
        """Process a request through appropriate sub-agents"""
        responses = []

        # Determine which agents should handle this request
        target_agents = self._determine_targets(request)

        # Process through each target agent
        for agent_type in target_agents:
            try:
                response = await self._call_agent(agent_type, request)
                if response:
                    responses.append(response)
            except Exception as e:
                logger.error(f"Error calling {agent_type.value}: {e}")

        return responses

    def _determine_targets(self, request: AgentRequest) -> List[SubAgentType]:
        """Determine which agents should handle the request"""
        targets = []
        user_input = request.user_input.lower()

        # Healthcare keywords
        health_keywords = [
            "health",
            "fitness",
            "workout",
            "exercise",
            "sleep",
            "diet",
            "food",
            "weight",
            "heart",
            "mood",
            "energy",
            "sick",
        ]
        if any(kw in user_input for kw in health_keywords):
            targets.append(SubAgentType.HEALTHCARE)

        # Social keywords
        social_keywords = [
            "social",
            "friend",
            "message",
            "call",
            "contact",
            "relationship",
            "party",
            "event",
            "reminder",
            "birthday",
        ]
        if any(kw in user_input for kw in social_keywords):
            targets.append(SubAgentType.SOCIAL)

        # Task keywords
        task_keywords = [
            "task",
            "todo",
            "remind",
            "schedule",
            "calendar",
            "meeting",
            "deadline",
            "plan",
            "organize",
        ]
        if any(kw in user_input for kw in task_keywords):
            targets.append(SubAgentType.TASKS)

        # If no specific target, check context
        if not targets:
            if request.context.get("health_focus"):
                targets.append(SubAgentType.HEALTHCARE)
            elif request.context.get("social_focus"):
                targets.append(SubAgentType.SOCIAL)
            elif request.context.get("task_focus"):
                targets.append(SubAgentType.TASKS)

        return targets

    async def _call_agent(
        self, agent_type: SubAgentType, request: AgentRequest
    ) -> Optional[AgentResponse]:
        """Call a specific sub-agent"""
        if agent_type == SubAgentType.HEALTHCARE:
            return await self._handle_healthcare(request)
        elif agent_type == SubAgentType.SOCIAL:
            return await self._handle_social(request)
        elif agent_type == SubAgentType.TASKS:
            return await self._handle_tasks(request)

        return None

    async def _handle_healthcare(self, request: AgentRequest) -> AgentResponse:
        """Handle healthcare-related requests"""
        if not self._healthcare_agent:
            return AgentResponse(
                agent_type=SubAgentType.HEALTHCARE, message="", confidence=0.0
            )

        try:
            # Process through healthcare agent
            result = await self._healthcare_agent.process(request.user_input)

            return AgentResponse(
                agent_type=SubAgentType.HEALTHCARE,
                message=result.get("message", ""),
                actions_taken=result.get("actions", []),
                confidence=result.get("confidence", 0.8),
            )
        except Exception as e:
            logger.error(f"Healthcare agent error: {e}")
            return AgentResponse(
                agent_type=SubAgentType.HEALTHCARE,
                message="I had trouble processing that health request.",
                confidence=0.3,
            )

    async def _handle_social(self, request: AgentRequest) -> AgentResponse:
        """Handle social-related requests"""
        if not self._social_agent:
            return AgentResponse(
                agent_type=SubAgentType.SOCIAL, message="", confidence=0.0
            )

        try:
            # Process through social agent
            result = await self._social_agent.process(request.user_input)

            return AgentResponse(
                agent_type=SubAgentType.SOCIAL,
                message=result.get("message", ""),
                actions_taken=result.get("actions", []),
                confidence=result.get("confidence", 0.8),
            )
        except Exception as e:
            logger.error(f"Social agent error: {e}")
            return AgentResponse(
                agent_type=SubAgentType.SOCIAL,
                message="I had trouble processing that social request.",
                confidence=0.3,
            )

    async def _handle_tasks(self, request: AgentRequest) -> AgentResponse:
        """Handle task-related requests"""
        # For now, return a placeholder - task handling is in services
        return AgentResponse(agent_type=SubAgentType.TASKS, message="", confidence=0.0)

    def check_conflicts(self, proposed_action: Dict) -> bool:
        """Check if proposed action conflicts with recent actions"""
        action_type = proposed_action.get("type")

        for recent in self._recent_actions[-5:]:
            if recent.get("type") == action_type:
                # Check time proximity
                recent_time = datetime.fromisoformat(
                    recent.get("timestamp", "2020-01-01")
                )
                if (datetime.now() - recent_time).total_seconds() < 300:  # 5 minutes
                    return True

        return False

    def record_action(self, action: Dict):
        """Record an action for conflict detection"""
        action["timestamp"] = datetime.now().isoformat()
        self._recent_actions.append(action)

        # Keep only recent actions
        if len(self._recent_actions) > 50:
            self._recent_actions = self._recent_actions[-50:]


# Global instance
_coordinator: Optional[SubAgentCoordinator] = None


def get_sub_agent_coordinator() -> SubAgentCoordinator:
    """Get or create the sub-agent coordinator"""
    global _coordinator
    if _coordinator is None:
        _coordinator = SubAgentCoordinator()
    return _coordinator
