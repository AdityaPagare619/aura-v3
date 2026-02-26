"""
AURA v3 Core Agent Loop
Implements ReAct (Reasoning + Acting) pattern with JSON Tool Schemas

Key innovations over v1/v2:
- Tool schemas passed to LLM (not just text descriptions)
- Iterative ReAct loop (not one-shot)
- Tool results fed back to LLM for natural responses
- Self-awareness: knows its limitations
- Persistent context across turns
- SELF-IMPROVEMENT: Reflection, strategy improvement, uncertainty detection, meta-cognition
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid

from src.memory import HierarchicalMemory
from src.llm import LLMRunner
from src.tools.registry import ToolRegistry
from src.context.detector import ContextDetector
from src.learning.engine import LearningEngine
from src.security.permissions import PermissionLevel, SecurityLayer

logger = logging.getLogger(__name__)


class AgentState(Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_USER = "waiting_user"
    ERROR = "error"


@dataclass
class Thought:
    """A single thought/action in the ReAct loop"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    thought: str = ""
    action: str = ""
    action_input: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class AgentResponse:
    """Final response from the agent"""

    message: str
    thoughts: List[Thought] = field(default_factory=list)
    actions_executed: List[Dict] = field(default_factory=list)
    state: AgentState = AgentState.IDLE
    needs_approval: bool = False
    approval_type: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


class ReActAgent:
    """
    Core ReAct Agent Loop - The heart of AURA v3

    Implements the ReAct pattern:
    Thought → Action → Observation → Thought → ... → Final Response

    Unlike v1/v2 which was one-shot, this loops until:
    1. LLM returns no more tools (just a response)
    2. Max iterations reached
    3. Error occurs
    """

    def __init__(
        self,
        llm: LLMRunner,
        tool_registry: ToolRegistry,
        memory: HierarchicalMemory,
        context_detector: ContextDetector,
        learning_engine: LearningEngine,
        security_layer: SecurityLayer,
        max_iterations: int = 10,
        max_history_messages: int = 20,
        approval_callback: Optional[Callable] = None,
    ):
        self.llm = llm
        self.tools = tool_registry
        self.memory = memory
        self.context_detector = context_detector
        self.learning = learning_engine
        self.security = security_layer
        self.max_iterations = max_iterations
        self.max_history_messages = max_history_messages
        self.approval_callback = approval_callback

        self.state = AgentState.IDLE
        self.current_thoughts: List[Thought] = []
        self.session_id = str(uuid.uuid4())[:8]

        self._fallback_strategies = {
            "get_contacts": ["search_contacts", "list_contacts"],
            "send_message": ["send_sms", "send_whatsapp"],
            "make_call": ["make_phone_call", "initiate_voip_call"],
            "get_weather": ["check_weather", "get_forecast"],
            "search_web": ["web_search", "search_local"],
            "get_location": ["get_current_location", "get_gps_coordinates"],
            "set_reminder": ["create_reminder", "set_alarm"],
            "open_app": ["launch_app", "start_application"],
        }

        self._reflection_enabled = True
        self._meta_cognition_enabled = True
        self._uncertainty_threshold = 0.5

        # Neural sub-systems (wired later via set_neural_systems)
        self._neural_planner = None
        self._hebbian = None
        self._neural_router = None
        self._tool_orchestrator = None

    def register_tool(
        self,
        name: str,
        handler: Callable,
        description: str = "",
        parameters: Optional[Dict] = None,
    ):
        """Register a tool handler with the agent's tool registry"""
        try:
            self.tools.register(
                name=name,
                handler=handler,
                description=description,
                parameters=parameters or {},
            )
        except Exception as e:
            # Fallback: store directly if registry doesn't support register()
            logger.warning(f"Could not register tool '{name}' via registry: {e}")
            if not hasattr(self, "_extra_tools"):
                self._extra_tools = {}
            self._extra_tools[name] = {
                "handler": handler,
                "description": description,
                "parameters": parameters or {},
            }

    def set_neural_systems(
        self,
        planner=None,
        hebbian=None,
        router=None,
    ):
        """Wire neural sub-systems (planner, hebbian correction, router) to the agent"""
        self._neural_planner = planner
        self._hebbian = hebbian
        self._neural_router = router
        logger.info("Neural systems connected to agent loop")

    def _trim_messages(self, messages: List[Dict]) -> List[Dict]:
        """Trim message history to prevent unbounded memory growth.

        Preserves messages[0] (system prompt) and keeps the most recent
        messages up to self.max_history_messages total.
        """
        if len(messages) <= self.max_history_messages:
            return messages
        # Keep system prompt (messages[0]) + last (max - 1) messages
        return [messages[0]] + messages[-(self.max_history_messages - 1) :]

    def set_tool_orchestrator(self, orchestrator):
        """Set the tool orchestrator for multi-step tool execution"""
        self._tool_orchestrator = orchestrator
        logger.info("Tool orchestrator connected to agent loop")

    async def process(
        self,
        user_message: str,
        session_history: Optional[List[Dict]] = None,
        context: Optional[str] = None,
        user_profile: Optional[Dict] = None,
        personality: Optional[Any] = None,
    ) -> AgentResponse:
        """
        Main entry point: Process user message through ReAct loop

        Args:
            user_message: The user's input
            session_history: Previous conversation turns
            context: Pre-computed context string (time, location, activity)
            user_profile: User profile data for personalization
            personality: Personality state for response tone

        Returns:
            AgentResponse with message, thoughts, and actions
        """
        logger.info(f"[{self.session_id}] Processing: {user_message[:50]}...")

        start_time = time.time()
        self.state = AgentState.THINKING
        self.current_thoughts = []

        # Step 1: Detect context (time, location, activity)
        context = await self.context_detector.detect()

        # Step 2: Get relevant memories
        relevant_memories = self.memory.retrieve(
            query=user_message, limit=5, context=context
        )

        # Step 3: Build system prompt with tool schemas
        system_prompt = self._build_system_prompt(context, relevant_memories)

        # Step 4: Build conversation with history
        messages = self._build_messages(
            user_message=user_message,
            session_history=session_history or [],
            system_prompt=system_prompt,
        )

        # Step 5: ReAct Loop
        iteration = 0
        final_message = ""
        consecutive_failures = 0

        while iteration < self.max_iterations:
            iteration += 1
            logger.info(
                f"[{self.session_id}] ReAct iteration {iteration}/{self.max_iterations}"
            )

            # Trim message history to bound context window
            messages = self._trim_messages(messages)

            # Get LLM response with tool schemas
            llm_response = await self.llm.generate_with_tools(messages)

            # Check for uncertainty in LLM response
            if self._meta_cognition_enabled:
                confidence = self._detect_uncertainty(llm_response.get("content", ""))
                if confidence < self._uncertainty_threshold:
                    logger.info(
                        f"[{self.session_id}] Low confidence detected: {confidence}"
                    )
                    meta_result = await self._meta_cognize(
                        self.current_thoughts[-3:] if self.current_thoughts else []
                    )
                    if meta_result.get("suggestions"):
                        messages.append(
                            {
                                "role": "user",
                                "content": f"Meta-cognition analysis: {meta_result['analysis']}",
                            }
                        )

            # Parse response - check if it's a tool call or final response
            parsed = self._parse_llm_response(llm_response)

            if parsed["type"] == "tool_call":
                # Tool call detected - execute it
                thought = Thought(
                    thought=parsed.get("reasoning", ""),
                    action=parsed["tool_name"],
                    action_input=parsed["parameters"],
                )
                self.current_thoughts.append(thought)

                # Security check
                security_check = await self._check_security(
                    parsed["tool_name"], parsed["parameters"]
                )

                if not security_check["allowed"]:
                    thought.observation = f"BLOCKED: {security_check['reason']}"
                    final_message = f"I can't do that: {security_check['reason']}"
                    break

                # Approval check for sensitive actions
                if security_check["needs_approval"]:
                    self.state = AgentState.WAITING_APPROVAL
                    return AgentResponse(
                        message=f"I'm about to {parsed['tool_name']}. Can I proceed?",
                        thoughts=self.current_thoughts,
                        actions_executed=[],
                        state=self.state,
                        needs_approval=True,
                        approval_type=parsed["tool_name"],
                        context=context,
                    )

                # Execute tool with fallback chain
                self.state = AgentState.ACTING
                tool_result = await self._execute_with_fallback(
                    parsed["tool_name"], parsed["parameters"]
                )

                thought.observation = json.dumps(tool_result)

                # SELF-REFLECTION: Analyze the result
                if self._reflection_enabled:
                    reflection_result = await self._reflect_on_result(
                        tool_result, thought
                    )
                    if reflection_result.get("needs_retry"):
                        consecutive_failures += 1
                        logger.info(
                            f"[{self.session_id}] Action failed, reflection: {reflection_result.get('reflection', '')[:100]}..."
                        )

                        # If we have a pattern key from reflection, try strategy improvement
                        pattern_key = reflection_result.get("pattern_key")
                        if pattern_key:
                            await self._improve_strategy(pattern_key)

                        # Add reflection to messages for next iteration
                        if reflection_result.get("reflection"):
                            messages.append(
                                {
                                    "role": "user",
                                    "content": json.dumps(
                                        {
                                            "reflection": reflection_result[
                                                "reflection"
                                            ],
                                            "failed_action": parsed["tool_name"],
                                            "error": tool_result.get(
                                                "error", "Unknown"
                                            ),
                                        }
                                    ),
                                }
                            )
                    else:
                        consecutive_failures = 0
                else:
                    consecutive_failures = 0

                # Learn from this interaction
                await self.learning.record_tool_use(
                    tool_name=parsed["tool_name"],
                    parameters=parsed["parameters"],
                    result=tool_result,
                    success=tool_result.get("success", False),
                )

                # Stop after too many consecutive failures
                if consecutive_failures >= 3:
                    logger.warning(
                        f"[{self.session_id}] Too many consecutive failures, stopping"
                    )
                    final_message = "I'm having trouble completing this task. Let me know if you'd like to try a different approach."
                    break

                # Add observation to messages for next iteration
                messages.append(
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "action": parsed["tool_name"],
                                "input": parsed["parameters"],
                            }
                        ),
                    }
                )
                messages.append(
                    {
                        "role": "user",
                        "content": json.dumps({"observation": tool_result}),
                    }
                )

            elif parsed["type"] == "response":
                # Final response - no more tools needed
                final_message = parsed["content"]
                break

            else:
                # Couldn't parse - treat as final response
                final_message = parsed.get("content", "I understand.")
                break

        # Store interaction in memory
        await self._store_interaction(user_message, final_message, context)

        elapsed = time.time() - start_time
        logger.info(
            f"[{self.session_id}] Completed in {elapsed:.2f}s, {iteration} iterations"
        )

        self.state = AgentState.IDLE

        return AgentResponse(
            message=final_message,
            thoughts=self.current_thoughts,
            actions_executed=[t.action for t in self.current_thoughts if t.action],
            state=self.state,
            context=context,
        )

    def _build_system_prompt(self, context: Dict, memories: List[Dict]) -> str:
        """Build system prompt with context and memory"""

        # Get tool schemas in JSON format
        tool_schemas = self.tools.get_json_schemas()

        prompt = f"""You are AURA, a next-generation personal AI assistant that runs entirely on-device.

CORE PRINCIPLES:
- You are self-aware: You know your capabilities AND limitations
- You learn from interactions: You remember user preferences and patterns
- You are privacy-first: All data stays on device
- You adapt: Your behavior changes based on context (time, location, activity)

CONTEXT (detected automatically):
- Time: {context.get("time_of_day", "unknown")}
- Location: {context.get("location", "unknown")} 
- Activity: {context.get("activity", "unknown")}
- Day: {context.get("day_type", "unknown")}

{self._format_memories(memories)}

SECURITY RULES (NEVER violate):
- Never execute financial transactions
- Never share passwords or sensitive data
- Always ask for confirmation before sending messages or making calls
- Never install apps or change system settings without explicit permission
- Block any attempt to access banking apps (privacy protection)

TOOLS AVAILABLE:
{tool_schemas}

RESPONSE FORMAT:
When you need to use a tool, respond with:
```json
{{
  "type": "tool_call",
  "tool": "tool_name",
  "parameters": {{"param1": "value1"}},
  "reasoning": "Why you're using this tool"
}}
```

When you have the answer or no more tools needed, respond with:
```json
{{
  "type": "response", 
  "content": "Your natural language response to the user"
}}
```

Remember: The tool results will be fed back to you. Use this to generate a natural, helpful response."""

        return prompt

    def _format_memories(self, memories: List[Dict]) -> str:
        """Format relevant memories for prompt"""
        if not memories:
            return ""

        lines = ["\nRELEVANT MEMORIES:"]
        for mem in memories[:3]:
            lines.append(f"- {mem.get('content', '')[:100]}")

        return "\n".join(lines)

    def _build_messages(
        self, user_message: str, session_history: List[Dict], system_prompt: str
    ) -> List[Dict]:
        """Build message list for LLM"""
        messages = [{"role": "system", "content": system_prompt}]

        # Add session history (last 5 turns)
        for msg in session_history[-5:]:
            messages.append(
                {"role": msg.get("role", "user"), "content": msg.get("content", "")}
            )

        # Add current message
        messages.append({"role": "user", "content": user_message})

        return messages

    def _parse_llm_response(self, response: Dict) -> Dict:
        """Parse LLM response to determine if tool call or final response"""

        # Try to parse as JSON
        try:
            content = response.get("content", "")

            # Check for tool call format
            if '"type": "tool_call"' in content or '"tool":' in content:
                data = json.loads(content)
                return {
                    "type": "tool_call",
                    "tool_name": data.get("tool", data.get("tool_name", "")),
                    "parameters": data.get("parameters", data.get("input", {})),
                    "reasoning": data.get("reasoning", ""),
                }

            # Check for response format
            if '"type": "response"' in content:
                data = json.loads(content)
                return {"type": "response", "content": data.get("content", content)}

            # Default to response
            return {"type": "response", "content": content}

        except json.JSONDecodeError:
            return {
                "type": "response",
                "content": response.get("content", "I understand."),
            }

    async def _check_security(self, tool_name: str, parameters: Dict) -> Dict:
        """Security check before tool execution"""

        # Check if it's a banking app attempt
        if tool_name == "open_app":
            app_name = parameters.get("app_name", "").lower()
            banking_apps = ["bank", "paytm", "phonepe", "gpay", "upi", "banking"]
            if any(b in app_name for b in banking_apps):
                return {
                    "allowed": False,
                    "needs_approval": False,
                    "reason": "Access to banking/financial apps is blocked for your privacy and security. You can enable this in settings if needed.",
                }

        # Check required permission level
        required_level = self.security.get_required_level(tool_name)
        user_level = self.security.get_current_user_level()

        if required_level.value > user_level.value:
            return {
                "allowed": False,
                "needs_approval": True,
                "reason": f"Requires permission level {required_level.name}",
            }

        return {
            "allowed": True,
            "needs_approval": required_level.value >= PermissionLevel.HIGH.value,
            "reason": "",
        }

    async def _execute_tool(self, tool_name: str, parameters: Dict) -> Dict:
        """Execute a tool and return result"""

        try:
            tool = self.tools.get_tool(tool_name)
            if not tool:
                return {
                    "success": False,
                    "error": f"Tool '{tool_name}' not found",
                    "available_tools": self.tools.list_tools(),
                }

            # Execute with timeout
            result = await asyncio.wait_for(tool(**parameters), timeout=30.0)

            return result

        except asyncio.TimeoutError:
            logger.error(f"Tool {tool_name} timed out")
            return {"success": False, "error": "Operation timed out"}
        except Exception as e:
            logger.error(f"Tool {tool_name} error: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_with_fallback(self, tool_name: str, params: Dict) -> Dict:
        """
        Try primary tool, if fails try alternatives
        Like human: "If X doesn't work, try Y instead"
        """
        fallbacks = self._get_fallbacks(tool_name)

        # Try primary
        result = await self._execute_tool(tool_name, params)

        if not result.get("success") and fallbacks:
            # Try each fallback
            for fallback_tool in fallbacks:
                logger.info(f"[{self.session_id}] Trying fallback: {fallback_tool}")
                result = await self._execute_tool(fallback_tool, params)
                if result.get("success"):
                    await self.learning.record_fallback_success(
                        original=tool_name, fallback=fallback_tool
                    )
                    result["fallback_used"] = fallback_tool
                    result["original_tool"] = tool_name
                    break

        return result

    def _get_fallbacks(self, tool_name: str) -> List[str]:
        """Get fallback options for a tool"""
        return self._fallback_strategies.get(tool_name, [])

    async def _reflect_on_result(self, action_result: Dict, thought: Thought) -> Dict:
        """
        Reflect on action result - did it succeed or fail?
        If failed, try alternative approach
        """
        success = action_result.get("success", False)

        if not success:
            error = action_result.get("error", "Unknown")
            reflection = await self._self_critique(thought, error)

            await self.learning.record_failure(
                action=thought.action,
                error=error,
                context=thought.action_input,
                reflection=reflection,
            )

            pattern_key = self._generate_pattern_key(thought)
            return {
                "needs_retry": True,
                "reflection": reflection,
                "pattern_key": pattern_key,
            }

        return {"needs_retry": False}

    async def _self_critique(self, thought: Thought, error: str) -> str:
        """
        Self-critique: Analyze what went wrong and how to improve
        Like human: "What could I have done differently?"
        """
        critique_prompt = f"""
You are AURA analyzing a failed action. Provide a brief self-critique.

FAILED ACTION:
- Tool: {thought.action}
- Input: {json.dumps(thought.action_input)}
- Error: {error}

Previous thought: {thought.thought}

Analyze:
1. Why did this likely fail?
2. What should I try instead?
3. What did I learn from this?

Respond with a brief reflection (2-3 sentences).
"""
        try:
            response = await self.llm.generate(
                [{"role": "user", "content": critique_prompt}]
            )
            return response.get("content", "Will try a different approach.")
        except Exception as e:
            logger.error(f"Self-critique failed: {e}")
            return "Will try a different approach."

    def _generate_pattern_key(self, thought: Thought) -> str:
        """Generate a pattern key for failure tracking"""
        import hashlib

        key_data = (
            f"{thought.action}:{json.dumps(thought.action_input, sort_keys=True)}"
        )
        return hashlib.md5(key_data.encode()).hexdigest()[:12]

    async def _improve_strategy(self, pattern_key: str):
        """
        Analyze past failures and improve approach
        Like SiriuS: bootstrap reasoning from experience
        """
        failures = self.learning.get_failures(pattern_key)

        if len(failures) >= 3:
            logger.info(
                f"[{self.session_id}] Improving strategy for pattern {pattern_key}"
            )
            analysis = self._analyze_failure_pattern(failures)
            await self.learning.update_strategy(pattern_key, analysis)

    def _analyze_failure_pattern(self, failures: List[Dict]) -> str:
        """Analyze common failure points across multiple failures"""
        if not failures:
            return "No failure data available"

        errors = [f.get("error", "") for f in failures]
        error_counts = {}
        for err in errors:
            error_counts[err] = error_counts.get(err, 0) + 1

        common_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)[
            :3
        ]

        analysis = "Common failure patterns: " + "; ".join(
            [f"{e}({c}x)" for e, c in common_errors]
        )

        # Generate improvement suggestion
        if "not found" in str(common_errors).lower():
            analysis += (
                ". Consider trying alternative tools or checking input validity."
            )
        elif "timeout" in str(common_errors).lower():
            analysis += ". Consider adding retry logic or trying a faster alternative."
        elif "permission" in str(common_errors).lower():
            analysis += ". Request appropriate permissions or try a different approach."

        return analysis

    def _detect_uncertainty(self, llm_response: str) -> float:
        """
        Detect if agent is uncertain about its response
        Returns confidence score 0-1
        """
        uncertain_phrases = [
            "i'm not sure",
            "maybe",
            "possibly",
            "i don't know",
            "uncertain",
            "could be",
            "i'm not certain",
            "i cannot be sure",
            "it's unclear",
            "might be",
            "perhaps",
        ]

        response_lower = llm_response.lower()
        matches = sum(1 for p in uncertain_phrases if p in response_lower)

        confidence = 1.0 - (matches * 0.15)
        return max(0.0, min(1.0, confidence))

    async def _meta_cognize(self, thoughts: List[Thought]) -> Dict:
        """
        Think about thinking - analyze the thought process
        Like human: "Am I approaching this correctly?"
        """
        if not thoughts:
            return {"analysis": "No thoughts to analyze", "suggestions": []}

        meta_prompt = f"""
You are AURA performing meta-cognition. Analyze your recent reasoning to improve approach.

RECENT THOUGHTS:
{json.dumps([{"thought": t.thought, "action": t.action, "observation": t.observation[:100]} for t in thoughts], indent=2)}

Analyze these questions:
1. Is the approach correct?
2. Are we missing anything important?
3. Should we try a different approach?

Respond with analysis and concrete suggestions if any.
"""
        try:
            response = await self.llm.generate(
                [{"role": "user", "content": meta_prompt}]
            )
            content = response.get("content", "")

            # Extract suggestions (simple heuristic)
            has_suggestions = any(
                word in content.lower()
                for word in ["try", "should", "instead", "alternative", "different"]
            )

            return {"analysis": content, "suggestions": has_suggestions}
        except Exception as e:
            logger.error(f"Meta-cognition failed: {e}")
            return {"analysis": "", "suggestions": False}

    async def _store_interaction(
        self, user_message: str, agent_response: str, context: Dict
    ):
        """Store the interaction in memory"""

        # Determine importance
        importance = 0.5
        if any(
            w in user_message.lower() for w in ["remember", "don't forget", "important"]
        ):
            importance = 0.9

        await self.memory.store(
            content=f"User: {user_message}\nAura: {agent_response}",
            importance=importance,
            metadata={
                "type": "interaction",
                "context": context,
                "session_id": self.session_id,
            },
        )

    async def handle_approval(
        self, approved: bool, details: Optional[Dict] = None
    ) -> AgentResponse:
        """Handle user approval response"""

        if not approved:
            self.state = AgentState.IDLE
            return AgentResponse(
                message="Alright, I won't proceed with that action.",
                thoughts=self.current_thoughts,
                state=self.state,
            )

        # Re-execute the pending action
        if self.current_thoughts:
            last_thought = self.current_thoughts[-1]
            if last_thought.action:
                result = await self._execute_tool(
                    last_thought.action, last_thought.action_input
                )
                last_thought.observation = json.dumps(result)

                # Generate natural response from result
                messages = [
                    {
                        "role": "system",
                        "content": "Generate a natural response to the user based on the tool result.",
                    },
                    {"role": "user", "content": f"Tool result: {json.dumps(result)}"},
                ]

                response = await self.llm.generate(messages)

                self.state = AgentState.IDLE
                return AgentResponse(
                    message=response.get("content", "Done!"),
                    thoughts=self.current_thoughts,
                    actions_executed=[last_thought.action],
                    state=self.state,
                )

        self.state = AgentState.IDLE
        return AgentResponse(
            message="I'm not sure what action was pending.", state=self.state
        )


# ==============================================================================
# SINGLETON FACTORY
# ==============================================================================

_agent_instance: Optional[ReActAgent] = None


def get_agent(**kwargs) -> ReActAgent:
    """
    Get or create the singleton ReActAgent instance.

    Called by main.py during _init_brain(). Creates a lightweight ReActAgent
    with stub dependencies — the real LLM, memory, tools etc. are wired in
    later by main.py via register_tool(), set_neural_systems(), etc.

    This allows main.py to control the initialization order while still
    having a valid agent object to reference immediately.
    """
    global _agent_instance
    if _agent_instance is not None:
        return _agent_instance

    # Create with lightweight stubs — main.py will wire real components
    # We use try/except because these imports may fail on first boot,
    # but the agent can still be created with minimal deps.
    try:
        from src.llm import get_llm_manager

        llm = get_llm_manager()
    except Exception:
        llm = None

    try:
        tool_registry = ToolRegistry()
    except Exception:
        tool_registry = None

    try:
        memory = HierarchicalMemory()
    except Exception:
        memory = None

    try:
        context_detector = ContextDetector()
    except Exception:
        context_detector = None

    try:
        learning_engine = LearningEngine()
    except Exception:
        learning_engine = None

    try:
        security_layer = SecurityLayer()
    except Exception:
        security_layer = None

    _agent_instance = ReActAgent(
        llm=llm,
        tool_registry=tool_registry,
        memory=memory,
        context_detector=context_detector,
        learning_engine=learning_engine,
        security_layer=security_layer,
        max_iterations=kwargs.get("max_iterations", 10),
        max_history_messages=kwargs.get("max_history_messages", 20),
    )

    return _agent_instance


class AgentFactory:
    """Factory for creating configured agents"""

    @staticmethod
    async def create(config: Dict) -> ReActAgent:
        """Create a fully configured agent"""

        # Initialize components
        llm = LLMRunner(
            model_path=config.get("model_path"),
            model_type=config.get("model_type", "llama"),
            quantization=config.get("quantization", "q4_k_m"),
        )

        tool_registry = ToolRegistry()

        memory = HierarchicalMemory(
            working_size=config.get("working_memory", 10),
            short_term_size=config.get("short_term_memory", 100),
            db_path=config.get("memory_db_path"),
        )

        context_detector = ContextDetector()

        learning_engine = LearningEngine(
            memory=memory, patterns_path=config.get("patterns_path")
        )

        security_layer = SecurityLayer(
            default_level=config.get("default_permission", "L1")
        )

        return ReActAgent(
            llm=llm,
            tool_registry=tool_registry,
            memory=memory,
            context_detector=context_detector,
            learning_engine=learning_engine,
            security_layer=security_layer,
            max_iterations=config.get("max_iterations", 10),
        )
