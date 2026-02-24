"""
AURA v3 Agent Loop - THE BRAIN
LLM-powered reasoning loop inspired by ReAct and neural attention
This is where ALL reasoning happens - the LLM is the brain!
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Any, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import deque

from src.llm.manager import LLMManager, get_llm_manager
from src.memory import NeuralMemory, get_neural_memory, MemoryType

logger = logging.getLogger(__name__)


class AgentState(Enum):
    """States of the agent"""

    IDLE = "idle"
    THINKING = "thinking"  # Reasoning - LLM processing
    ACTING = "acting"  # Executing tools
    OBSERVING = "observing"  # Gathering context
    WAITING_APPROVAL = "waiting_approval"
    WAITING_USER = "waiting_user"
    COMPLETED = "completed"
    ERROR = "error"


class ReasoningStep(Enum):
    """Steps in ReAct reasoning"""

    OBSERVE = "observe"
    THINK = "think"
    ACT = "act"
    REFLECT = "reflect"
    FINALIZE = "finalize"


@dataclass
class Thought:
    """A single thought/reasoning step"""

    step: ReasoningStep
    content: str
    timestamp: datetime
    tool_used: Optional[str] = None
    tool_result: Optional[str] = None
    confidence: float = 1.0


@dataclass
class AgentContext:
    """Complete context for agent reasoning"""

    # User input
    user_message: str
    conversation_id: str

    # Temporal context
    time_of_day: str
    day_of_week: str

    # User state
    user_busyness: float = 0.0
    interruption_willingness: float = 0.0

    # Relevant memories
    relevant_memories: List[Dict] = field(default_factory=list)
    recent_conversations: List[Dict] = field(default_factory=list)

    # Active patterns
    active_patterns: List[str] = field(default_factory=list)

    # Pending tasks
    pending_tasks: List[str] = field(default_factory=list)

    # User profile
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    communication_style: str = "balanced"


@dataclass
class AgentResponse:
    """Response from the agent"""

    message: str
    state: AgentState
    thoughts: List[Thought] = field(default_factory=list)
    actions_taken: List[Dict] = field(default_factory=list)
    context_updates: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    should_notify: bool = False
    notification_message: Optional[str] = None


class AuraAgentLoop:
    """
    AURA's BRAIN - LLM-powered reasoning loop

    This is the core of AURA - where ALL reasoning happens.
    Inspired by:
    - ReAct (Reasoning + Acting)
    - Neural attention mechanisms
    - Human thought processes

    The LLM does ALL reasoning - algorithms are just the nervous system!
    """

    def __init__(self, llm_manager: LLMManager = None):
        self.llm = llm_manager or get_llm_manager()

        # Reasoning state
        self.state = AgentState.IDLE
        self.conversation_id = str(uuid.uuid4())[:8]
        self.thought_history: deque = deque(maxlen=100)

        # Context
        self.context: Optional[AgentContext] = None

        # Tools (will be registered)
        self.tools: Dict[str, Callable] = {}
        self.tool_schemas: List[Dict] = []

        # Memory system - CONNECTED to neural memory
        self.neural_memory = get_neural_memory()

        # NEURAL SYSTEMS (AURA-Native - wired in main.py)
        self.neural_planner = None
        self.hebbian_corrector = None
        self.model_router = None

        # TOOL ORCHESTRATOR (wired in main.py)
        self.tool_orchestrator = None

        # Settings
        self.max_thought_steps = 10
        self.thinking_timeout = 60.0

        # Callbacks
        self.on_think: Optional[Callable] = None
        self.on_act: Optional[Callable] = None
        self.on_complete: Optional[Callable] = None

    # =========================================================================
    # TOOL REGISTRATION
    # =========================================================================

    def register_tool(
        self, name: str, handler: Callable, description: str, parameters: Dict[str, Any]
    ):
        """Register a tool for the agent to use"""
        self.tools[name] = handler
        self.tool_schemas.append(
            {"name": name, "description": description, "parameters": parameters}
        )
        logger.info(f"Registered tool: {name}")

    def get_tool_schemas(self) -> str:
        """Get tool schemas as JSON for LLM"""
        return json.dumps(self.tool_schemas, indent=2)

    # =========================================================================
    # NEURAL SYSTEMS SETTERS (wired from main.py)
    # =========================================================================

    def set_neural_systems(self, planner=None, hebbian=None, router=None):
        """Set neural system references (called from main.py)"""
        self.neural_planner = planner
        self.hebbian_corrector = hebbian
        self.model_router = router
        logger.info("Neural systems wired to agent")

    def set_tool_orchestrator(self, orchestrator):
        """Set tool orchestrator for deterministic execution"""
        self.tool_orchestrator = orchestrator
        logger.info("Tool orchestrator wired to agent")

    async def _select_model(self, user_context: Dict) -> str:
        """Use neural-aware router to select best model"""
        if not self.model_router:
            return "default"

        try:
            result = await self.model_router.select_model(
                task_description=user_context.get("task", "general"),
                user_context=user_context,
            )
            return result.selected_tier.value
        except Exception as e:
            logger.warning(f"Model routing failed: {e}")
            return "default"

    async def _validate_plan(self, user_intent: str, user_context: Dict):
        """Use neural-validated planner for planning"""
        if not self.neural_planner:
            return None

        try:
            plan, validation = await self.neural_planner.create_plan(
                user_intent=user_intent,
                available_tools=self.tool_schemas,
                user_context=user_context,
            )
            return {"plan": plan, "validation": validation}
        except Exception as e:
            logger.warning(f"Neural planning failed: {e}")
            return None

    async def _record_outcome(
        self, action: str, params: Dict, success: bool, context: Dict
    ):
        """Use Hebbian self-corrector to learn from outcomes"""
        if not self.hebbian_corrector:
            return

        try:
            from src.core.hebbian_self_correction import ActionOutcome

            outcome = ActionOutcome.SUCCESS if success else ActionOutcome.FAILURE
            await self.hebbian_corrector.record_outcome(
                action=action, params=params, outcome=outcome, context=context
            )
        except Exception as e:
            logger.warning(f"Hebbian learning failed: {e}")

    # =========================================================================
    # CONTEXT GATHERING (OBSERVE)
    # =========================================================================

    async def _observe(self, user_message: str) -> AgentContext:
        """Gather context - observe the situation"""
        self.state = AgentState.OBSERVING

        # Get temporal context
        now = datetime.now()
        time_of_day = self._get_time_period(now)
        day_of_week = now.strftime("%A")

        # Get user state from adaptive context - use REAL values, not hardcoded!
        user_busyness, interruption_willingness = await self._get_user_state()

        # Get relevant memories
        relevant_memories = await self._retrieve_memories(user_message)

        # Get recent conversations
        recent_conversations = await self._get_recent_conversations()

        # Get active patterns
        active_patterns = await self._get_active_patterns()

        # Get pending tasks
        pending_tasks = await self._get_pending_tasks()

        # Get user preferences
        user_preferences = await self._get_user_preferences()

        context = AgentContext(
            user_message=user_message,
            conversation_id=self.conversation_id,
            time_of_day=time_of_day,
            day_of_week=day_of_week,
            user_busyness=user_busyness,
            interruption_willingness=interruption_willingness,
            relevant_memories=relevant_memories,
            recent_conversations=recent_conversations,
            active_patterns=active_patterns,
            pending_tasks=pending_tasks,
            user_preferences=user_preferences,
        )

        self.context = context

        # Log observation
        thought = Thought(
            step=ReasoningStep.OBSERVE,
            content=f"Observing: User said '{user_message[:50]}...' at {time_of_day}",
            timestamp=datetime.now(),
        )
        self.thought_history.append(thought)

        return context

    def _get_time_period(self, dt: datetime) -> str:
        """Get time period"""
        hour = dt.hour
        if 5 <= hour < 12:
            return "morning"
        elif 12 <= hour < 17:
            return "afternoon"
        elif 17 <= hour < 21:
            return "evening"
        else:
            return "night"

    async def _retrieve_memories(self, query: str) -> List[Dict]:
        """Retrieve relevant memories using neural memory"""
        try:
            neurons = await self.neural_memory.recall(
                query=query,
                memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC],
                limit=5,
            )
            return [
                {
                    "content": n.content,
                    "type": n.memory_type.value,
                    "importance": n.importance,
                }
                for n in neurons
            ]
        except Exception as e:
            logger.error(f"Memory retrieval error: {e}")
            return []

    async def _get_recent_conversations(self) -> List[Dict]:
        """Get recent conversation history"""
        try:
            from src.session import get_session_manager

            session_mgr = get_session_manager()
            session = session_mgr.get_session()
            if session:
                return [
                    {"role": "user", "content": turn.user_input}
                    for turn in session.turns[-5:]
                ]
        except Exception as e:
            logger.warning(f"Could not get recent conversations: {e}")
        return []

    async def _get_active_patterns(self) -> List[str]:
        """Get active user patterns from adaptive context"""
        try:
            from src.services.adaptive_context import AdaptiveContextEngine

            ctx_engine = AdaptiveContextEngine()
            patterns = await ctx_engine.get_all_patterns()
            return [p.get("pattern_key", "") for p in patterns[:5]] if patterns else []
        except Exception as e:
            logger.warning(f"Could not get active patterns: {e}")
        return []

    async def _get_pending_tasks(self) -> List[str]:
        """Get pending tasks from task context"""
        try:
            from src.services.task_context import TaskContextPreservation

            task_ctx = TaskContextPreservation()
            tasks = await task_ctx.get_active_tasks()
            return [t.task_id for t in tasks[:5]] if tasks else []
        except Exception as e:
            logger.warning(f"Could not get pending tasks: {e}")
        return []

    async def _get_pending_tasks(self) -> List[str]:
        """Get pending tasks from task context"""
        try:
            from src.services.task_context import TaskContextPreservation

            task_ctx = TaskContextPreservation()
            tasks = await task_ctx.get_pending_tasks()
            return tasks if tasks else []
        except Exception as e:
            logger.warning(f"Could not get pending tasks: {e}")
        return []

    async def _get_user_preferences(self) -> Dict[str, Any]:
        """Get user preferences from profile/learning"""
        try:
            from src.core.user_profile import get_user_profiler

            profiler = get_user_profiler("default")
            if profiler:
                return profiler.get_context()
        except Exception as e:
            logger.warning(f"Could not get user preferences: {e}")
        return {}

    async def _get_user_state(self) -> tuple:
        """
        Get real user state from context provider - NOT hardcoded!
        Returns: (user_busyness, interruption_willingness)
        """
        try:
            from src.context import get_context_provider

            ctx_provider = get_context_provider()

            # Get device context for activity-based busyness estimation
            device_ctx = await ctx_provider.get_device_context()

            # Estimate busyness based on device state
            # If screen is on and user is active, they're likely busy
            user_busyness = 0.5  # default
            if device_ctx:
                screen_on = device_ctx.get("screen_on", False)
                if screen_on:
                    # Check what app is running to estimate busyness
                    # For now, default to moderate busyness when screen is on
                    user_busyness = 0.6
                else:
                    user_busyness = 0.3  # Screen off = less busy

                # Check battery level - low battery = user might be busy/away
                battery = device_ctx.get("battery_level", 100)
                if battery < 20:
                    user_busyness = min(user_busyness + 0.1, 1.0)

            # Get user's learned interruption willingness from profile
            try:
                from src.core.user_profile import get_user_profiler

                profiler = get_user_profiler("default")
                if profiler:
                    profile = profiler.get_context()
                    # Try to get interruption willingness from learned profile
                    interruption_willingness = profile.get(
                        "interruption_tolerance", 0.5
                    )
                else:
                    interruption_willingness = 0.5
            except:
                interruption_willingness = 0.5

            return (user_busyness, interruption_willingness)

        except Exception as e:
            logger.warning(f"Could not get user state from context: {e}")
            return (0.5, 0.5)  # Safe defaults

    async def _store_interaction(self, context: AgentContext, response_message: str):
        """
        Store the interaction in neural memory for learning.
        This is the CRITICAL feedback loop that enables AURA to learn from interactions.
        """
        try:
            # Determine importance based on context
            importance = 0.5

            # Higher importance if user mentioned something personal
            if any(
                word in context.user_message.lower()
                for word in ["remember", "don't forget", "important", "always"]
            ):
                importance = 0.8

            # Higher importance if there were tool actions
            has_actions = any(t.tool_used for t in self.thought_history if t.tool_used)
            if has_actions:
                importance = 0.7

            # Extract related neuron IDs from retrieval for Hebbian learning
            related_ids = [
                m.id for m in context.relevant_memories[:5] if hasattr(m, "id")
            ]

            # Store the interaction as an episodic memory
            await self.neural_memory.learn(
                content=f"User: {context.user_message[:100]} | AURA: {response_message[:100]}",
                memory_type=MemoryType.EPISODIC,
                related_to=related_ids if related_ids else None,
                importance=importance,
                emotional_valence=0.0,  # Could analyze sentiment
                tags={context.time_of_day, context.day_of_week},
            )

            logger.debug(
                f"Stored interaction in neural memory (importance: {importance})"
            )

        except Exception as e:
            logger.warning(f"Failed to store interaction in memory: {e}")

    # =========================================================================
    # REASONING (THINK) - THE CORE BRAIN
    # =========================================================================

    async def _think(self, context: AgentContext) -> Thought:
        """Think - let LLM reason about what to do"""
        self.state = AgentState.THINKING

        # Build prompt for LLM
        prompt = self._build_reasoning_prompt(context)

        # Let LLM think
        try:
            response = await self.llm.chat(
                message=prompt,
                conversation_history=self._get_conversation_history(),
                system_prompt=self._get_system_prompt(),
            )

            thought_content = response.text

        except Exception as e:
            logger.error(f"LLM thinking error: {e}")
            thought_content = f"I need to analyze this: {context.user_message}"

        thought = Thought(
            step=ReasoningStep.THINK, content=thought_content, timestamp=datetime.now()
        )
        self.thought_history.append(thought)

        return thought

    def _build_reasoning_prompt(self, context: AgentContext) -> str:
        """Build prompt for reasoning"""
        prompt = f"""You are AURA, a personal AI assistant. Think step by step about how to help the user.

Current situation:
- User said: "{context.user_message}"
- Time: {context.time_of_day} on {context.day_of_week}
- User's current busyness: {context.user_busyness:.0%}
- User's willingness to be interrupted: {context.interruption_willingness:.0%}

"""

        # Add relevant memories
        if context.relevant_memories:
            prompt += f"Relevant memories:\n"
            for mem in context.relevant_memories[:3]:
                prompt += f"- {mem.get('content', '')[:100]}\n"
            prompt += "\n"

        # Add active patterns
        if context.active_patterns:
            prompt += f"Active patterns detected: {', '.join(context.active_patterns[:3])}\n\n"

        # Add pending tasks
        if context.pending_tasks:
            prompt += f"Pending tasks: {', '.join(context.pending_tasks[:3])}\n\n"

        # Add tools available
        if self.tool_schemas:
            prompt += f"Available tools:\n{self.get_tool_schemas()}\n\n"

        prompt += """Think about what to do next. Consider:
1. What the user is asking for
2. What context you have about the user
3. What tools/actions might help
4. Whether you should act proactively

IMPORTANT - How to use tools:
If you need to use a tool, respond with:
<tool_call>tool_name|param1=value1|param2=value2</tool_call>

For example:
- To open an app: <tool_call>open_app|app_name=whatsapp</tool_call>
- To send a message: <tool_call>send_message|recipient=John|message=Hello</tool_call>
- To check notifications: <tool_call>get_notifications</tool_call>

If no tool is needed, just respond naturally.

Respond with your reasoning and then decide on your next action."""

        return prompt

    def _get_system_prompt(self) -> str:
        """Get system prompt for AURA"""
        return """You are AURA, an intelligent personal assistant.

Your characteristics:
- Proactive: Don't just wait for commands, anticipate needs
- Adaptive: Learn from interactions and adapt to user preferences
- Privacy-first: All processing is local, never share private data
- Helpful: Focus on making the user's life easier

IMPORTANT - When you need to use a tool:
Use this format in your response: <tool_call>tool_name|param1=value1|param2=value2</tool_call>

Available tools include:
- open_app: Open an application
- close_app: Close an application
- send_message: Send a message via any app
- make_call: Make a phone call
- get_notifications: Get recent notifications
- take_screenshot: Take a screenshot
- get_current_app: Get current app info

Example: If user asks "Open WhatsApp", respond:
I'll open WhatsApp for you.
<tool_call>open_app|app_name=whatsapp</tool_call>

Communication style:
- Be conversational but concise
- Use context to personalize responses
- When uncertain, ask clarifying questions
- Don't overwhelm with information"""

    def _get_conversation_history(self) -> List[Dict]:
        """Get conversation history for context"""
        history = []
        for thought in list(self.thought_history)[-10:]:
            if thought.step == ReasoningStep.THINK:
                history.append({"role": "assistant", "content": thought.content})
            elif thought.step == ReasoningStep.ACT and thought.tool_result:
                history.append(
                    {"role": "user", "content": f"Tool result: {thought.tool_result}"}
                )
        return history

    # =========================================================================
    # ACTION (ACT)
    # =========================================================================

    async def _act(self, thought: Thought, context: AgentContext) -> Thought:
        """Act - execute tools if needed"""
        self.state = AgentState.ACTING

        # Parse the thought content for tool calls
        # Format: <tool_call>tool_name|param1=value1|param2=value2</tool_call>
        tool_result = None

        import re

        tool_match = re.search(
            r"<tool_call>(.+?)</tool_call>", thought.content, re.IGNORECASE
        )

        if tool_match:
            tool_spec = tool_match.group(1)
            parts = tool_spec.split("|")

            if parts:
                tool_name = parts[0].strip()
                params = {}

                for param in parts[1:]:
                    if "=" in param:
                        key, value = param.split("=", 1)
                        params[key.strip()] = value.strip()

                # Execute tool
                if tool_name in self.tools:
                    try:
                        logger.info(
                            f"Executing tool: {tool_name} with params: {params}"
                        )
                        result = await self.tools[tool_name](**params)
                        tool_result = str(result)

                        # Update history
                        thought.tool_used = tool_name
                        thought.tool_result = tool_result[:500]  # Truncate for history

                        logger.info(f"Tool {tool_name} executed successfully")
                    except Exception as e:
                        tool_result = f"Error: {str(e)}"
                        logger.error(f"Tool execution error: {e}")
                else:
                    tool_result = f"Tool '{tool_name}' not found"
        else:
            # No tool call detected - this is a direct response
            tool_result = None

        action = Thought(
            step=ReasoningStep.ACT,
            content=thought.content,
            timestamp=datetime.now(),
            tool_used=tool_match.group(1).split("|")[0] if tool_match else None,
            tool_result=tool_result,
        )

        self.thought_history.append(action)

        return action

    # =========================================================================
    # REFLECTION
    # =========================================================================

    async def _reflect(self, context: AgentContext) -> Thought:
        """Reflect on the response and ensure quality"""

        # Get the last action result
        last_tool_result = None
        for t in reversed(self.thought_history):
            if t.tool_result:
                last_tool_result = t.tool_result
                break

        # Simple quality checks
        quality_notes = []

        # Check 1: Was a tool used?
        if last_tool_result and not last_tool_result.startswith("Error"):
            quality_notes.append("Tool executed successfully")
        elif last_tool_result and last_tool_result.startswith("Error"):
            quality_notes.append(f"Tool error: {last_tool_result[:100]}")

        # Check 2: Response length
        if context.user_message and len(context.user_message) > 100:
            quality_notes.append("Complex query handled")

        # Check 3: Context awareness
        if context.time_of_day:
            quality_notes.append(f"Time context: {context.time_of_day}")

        thought = Thought(
            step=ReasoningStep.REFLECT,
            content=f"Reflection: {'; '.join(quality_notes) if quality_notes else 'Standard response'}",
            timestamp=datetime.now(),
        )

        self.thought_history.append(thought)

        return thought

    # =========================================================================
    # MAIN LOOP
    # =========================================================================

    async def process(self, user_message: str) -> AgentResponse:
        """
        Main processing loop - THE BRAIN IN ACTION

        This is where the magic happens:
        OBSERVE → THINK → PLAN → ACT → LEARN → RESPOND
        Now with REAL neural integration!
        """
        logger.info(f"Agent processing: {user_message[:50]}...")

        # Step 1: Observe - gather context
        context = await self._observe(user_message)

        # Step 2: Think - let LLM reason
        thought = await self._think(context)

        # Step 3: Validate plan using neural-validated planner (NEW!)
        plan_result = None
        if self.neural_planner and thought.content:
            plan_result = await self._validate_plan(
                user_message, {"thought": thought.content, "context": context.__dict__}
            )
            if plan_result:
                logger.info(
                    f"Neural plan validated: {plan_result.get('validation', {}).get('result', 'unknown')}"
                )

        # Step 4: Act - execute if needed
        action = await self._act(thought, context)

        # Step 5: Record outcome for Hebbian learning (NEW!)
        if action.tool_used:
            await self._record_outcome(
                action=action.tool_used,
                params={},  # Could track actual params
                success=action.tool_result is not None
                and not action.tool_result.startswith("Error"),
                context={"user_message": user_message, "result": action.tool_result},
            )

        # Step 6: Reflect - ensure quality
        reflection = await self._reflect(context)

        # Step 7: Finalize - generate response
        self.state = AgentState.COMPLETED

        # Generate final response
        response_message = await self._generate_response(context)

        # Create response object
        response = AgentResponse(
            message=response_message,
            state=self.state,
            thoughts=list(self.thought_history),
            actions_taken=[{"thought": t.content} for t in self.thought_history],
            confidence=0.9,
        )

        # CRITICAL FIX: Store interaction in neural memory for learning
        await self._store_interaction(context, response_message)

        return response

    async def _generate_response(self, context: AgentContext) -> str:
        """Generate final response using LLM"""

        # Build context summary
        context_summary = f"""
User: {context.user_message}
Time: {context.time_of_day}, {context.day_of_week}
Patterns: {", ".join(context.active_patterns[:3]) if context.active_patterns else "None detected"}
"""

        prompt = f"""Based on the context:
{context_summary}

Generate a helpful, concise response to the user. Consider:
- Their current state ({context.user_busyness:.0%} busy)
- Their communication style preference
- Whether proactive suggestions would be helpful

Respond naturally as AURA would."""

        try:
            response = await self.llm.chat(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Response generation error: {e}")
            return f"I received: {context.user_message}"

    # =========================================================================
    # CONTINUOUS MODE (for proactive actions)
    # =========================================================================

    async def run_continuous(self):
        """Run agent in continuous mode for proactive actions"""
        logger.info("Starting continuous agent mode")

        while True:
            try:
                # Check for proactive opportunities
                should_act, action = await self._check_proactive_opportunities()

                if should_act:
                    logger.info(f"Proactive action: {action}")
                    # Execute proactive action

                await asyncio.sleep(60)  # Check every minute

            except Exception as e:
                logger.error(f"Continuous mode error: {e}")
                await asyncio.sleep(60)

    async def _check_proactive_opportunities(self) -> tuple:
        """Check if there's something proactive to do"""
        # This would check:
        # - Upcoming events
        # - Detected interests
        # - Pending preparations
        # - User patterns

        return False, ""

    # =========================================================================
    # STATE MANAGEMENT
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get agent status"""
        return {
            "state": self.state.value,
            "conversation_id": self.conversation_id,
            "thought_steps": len(self.thought_history),
            "tools_registered": len(self.tools),
            "context_available": self.context is not None,
        }

    async def reset(self):
        """Reset agent state for new conversation"""
        self.state = AgentState.IDLE
        self.conversation_id = str(uuid.uuid4())[:8]
        self.thought_history.clear()
        self.context = None


# ==============================================================================
# FACTORY
# ==============================================================================

_agent_instance: Optional[AuraAgentLoop] = None


def get_agent() -> AuraAgentLoop:
    """Get or create agent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = AuraAgentLoop()
    return _agent_instance


__all__ = [
    "AuraAgentLoop",
    "AgentState",
    "ReasoningStep",
    "Thought",
    "AgentContext",
    "AgentResponse",
    "get_agent",
]
