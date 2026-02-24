"""
AURA v3 - Production Entry Point
Mobile-optimized, production-ready main module
"""

import asyncio
import logging
import signal
import sys
from typing import Optional
from enum import Enum

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of memory in AURA"""

    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    SKILL = "skill"
    ANCESTOR = "ancestor"


class AuraProduction:
    """
    Production AURA Main Class
    Initializes all services with proper dependency order
    Mobile-optimized: lightweight startup, lazy loading
    """

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self._initialized = False
        self._running = False

        # Core - THE BRAIN (LLM + Memory + Processing)
        self._agent_loop = None
        self._llm_manager = None
        self._neural_memory = None

        # NEW CORE SYSTEMS (from aura-v3 design)
        self._neuromorphic_engine = None
        self._mobile_power = None
        self._conversation = None
        self._user_profile = None
        self._personality = None
        self._context_provider = None

        # NEW: Proactivity Controller & Tool Orchestrator
        self._proactivity_controller = None
        self._tool_orchestrator = None

        # NEURAL SYSTEMS (AURA-Native Solutions)
        self._neural_validated_planner = None
        self._hebbian_self_correction = None
        self._neural_aware_router = None

        # ADDONS
        self._app_discovery = None
        self._termux_bridge = None
        self._tool_binding = None
        self._capability_gap = None

        # Services
        self._context_engine = None
        self._life_tracker = None
        self._proactive_engine = None
        self._dashboard = None
        self._task_context = None
        self._background_manager = None
        self._self_learning = None

        # NEW: Proactive Services (AURA-Native)
        self._proactive_event_tracker = None
        self._intelligent_call_manager = None
        self._proactive_life_explorer = None

        # LEARNING & SESSION (expanded)
        self._learning_engine = None
        self._session_manager = None

        # Utils
        self._health_monitor = None
        self._circuit_breaker = None
        self._graceful_shutdown = None
        self._error_recovery = None

        # SECURITY (new)
        self._authenticator = None
        self._privacy_manager = None
        self._permission_manager = None
        self._security_auditor = None

        # SOCIAL-LIFE MANAGER
        self._social_life_agent = None

        # HEALTHCARE AGENT
        self._healthcare_agent = None

        # INNER VOICE SYSTEM
        self._inner_voice = None

        # EXECUTION CONTROLLER (Stop/Kill System)
        self._execution_controller = None

        # RELATIONSHIP SYSTEM
        self._relationship_system = None

        # MOBILE SYSTEMS (optional - don't crash if unavailable)
        self._aura_space_server = None
        self._termux_widget_bridge = None
        self._character_sheet = None
        self._cinematic_moments = None

        # Mobile system config
        self._mobile_config = {
            "aura_space_enabled": True,
            "termux_widgets_enabled": True,
            "character_sheet_enabled": True,
            "cinematic_moments_enabled": True,
            "aura_space_port": 8080,
        }

    async def initialize(self):
        """Initialize all services in proper order"""
        if self._initialized:
            logger.warning("AURA already initialized")
            return

        logger.info("Initializing AURA v3...")

        try:
            # Phase 0: Core Brain (LLM + Memory)
            await self._init_brain()

            # Phase 0.05: Execution Control (Stop/Kill System)
            await self._init_execution_control()

            # Phase 0.1: Tools (depends on brain/agent)
            await self._init_tools()

            # Phase 0.5: Core Processing Engine (before everything)
            await self._init_core_engine()

            # Phase 0.6: Neural Systems (AURA-Native solutions)
            await self._init_neural_systems()

            # Phase 0.75: Security (before network/services)
            await self._init_security()

            # Phase 1: Core utilities (no dependencies)
            await self._init_utils()

            # Phase 2: Context & Session (foundational services)
            await self._init_context()
            await self._init_session()

            # Phase 3: Learning (depends on context)
            await self._init_learning()

            # Phase 4: Core services (utilities + context)
            await self._init_core_services()

            # Phase 5: Intelligence services (core)
            await self._init_intelligence()

            # Phase 5.5: NEW - Proactive Services (AURA-Native)
            await self._init_proactive_services()

            # Phase 6: NEW - Addons (depends on core)
            await self._init_addons()

            # Phase 7: NEW - Social-Life Manager
            await self._init_social_life()

            # Phase 7.5: NEW - Healthcare Agent
            await self._init_healthcare()

            # Phase 7.6: NEW - Inner Voice System
            await self._init_inner_voice()

            # Phase 8: UI/Interface services (intelligence)
            await self._init_interface()

            self._initialized = True
            logger.info("AURA v3 initialized successfully")

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            raise

    async def _init_brain(self):
        """Initialize the brain - LLM and memory"""
        from src.llm import get_llm_manager
        from src.agent import get_agent
        from src.memory import get_neural_memory
        from src.tools.handlers import ToolHandlers
        from src.tools.registry import ToolRegistry

        logger.info("Initializing AURA's brain...")

        self._llm_manager = get_llm_manager()
        self._neural_memory = get_neural_memory()

        # Initialize Memory Coordinator (NEW!)
        from src.memory.memory_coordinator import get_memory_coordinator

        self._memory_coordinator = get_memory_coordinator()
        self._memory_coordinator.set_memory_system(
            MemoryType.WORKING, self._neural_memory
        )

        self._agent_loop = get_agent()

        # Initialize Execution Controller (Stop/Kill System)
        from src.core.execution_control import get_execution_controller

        self._execution_controller = get_execution_controller()

        # Initialize Loop Detector (Infinite Loop Prevention)
        from src.core.loop_detector import get_loop_detector

        self._loop_detector = get_loop_detector()

        # Connect loop detector to execution controller for auto-stop
        self._execution_controller.set_loop_detector(self._loop_detector)

        logger.info("Brain initialized")

        # NEW: Initialize Relationship System (after brain)
        await self._init_relationship()

    async def _init_relationship(self):
        """Initialize the dynamic relationship system"""
        from src.agent import initialize_relationship_system

        logger.info("Initializing Relationship System...")

        self._relationship_system = await initialize_relationship_system()

        logger.info("Relationship System initialized")

    async def _init_execution_control(self):
        """Initialize execution controller for stop/kill functionality"""
        from src.core.execution_control import get_execution_controller

        logger.info("Initializing Execution Controller...")

        self._execution_controller = get_execution_controller()

        # Register cleanup handlers
        if self._neural_memory:
            self._execution_controller.register_cleanup_handler(
                "neural_memory",
                lambda: None,  # Memory handles its own cleanup
            )

        if self._session_manager:
            self._execution_controller.register_cleanup_handler(
                "session", self._session_manager.end_session
            )

        logger.info("Execution Controller initialized")

    async def _init_tools(self):
        """Initialize tool registry and handlers, wire to agent"""
        from src.tools.registry import ToolRegistry
        from src.tools.handlers import ToolHandlers
        import asyncio

        logger.info("Initializing tools...")

        self._tool_registry = ToolRegistry()

        tool_handlers = ToolHandlers()
        await tool_handlers.initialize()

        async def create_handler(tool_name: str):
            """Create async handler wrapper for each tool"""
            handler = getattr(tool_handlers, tool_name, None)
            if handler and asyncio.iscoroutinefunction(handler):
                return handler
            return None

        for tool_name, tool_def in self._tool_registry.tools.items():
            handler = getattr(tool_handlers, tool_name, None)
            if handler:
                self._agent_loop.register_tool(
                    name=tool_name,
                    handler=handler,
                    description=tool_def.description,
                    parameters=tool_def.parameters,
                )

        logger.info(f"Registered {len(self._agent_loop.tools)} tools with agent")

    async def _init_core_engine(self):
        """Initialize NEW core processing engine"""
        from src.core import (
            get_neuromorphic_engine,
            get_power_manager,
            get_conversation_engine,
            get_user_profiler,
            get_personality_engine,
        )
        from src.context import get_context_provider

        logger.info("Initializing core processing engine...")

        # Neuromorphic processing engine (event-driven)
        self._neuromorphic_engine = get_neuromorphic_engine()

        # Mobile power management
        self._mobile_power = get_power_manager()

        # Conversation engine (emotional, JARVIS-style)
        self._conversation = get_conversation_engine()

        # Deep user profiling
        self._user_profile = get_user_profiler("default")

        # Adaptive personality
        self._personality = get_personality_engine()

        # Real-time context provider
        self._context_provider = get_context_provider()

        logger.info("Core processing engine initialized")

    async def _init_neural_systems(self):
        """Initialize AURA-Native neural systems"""
        from src.core.neural_validated_planner import NeuralValidatedPlanner
        from src.core.hebbian_self_correction import HebbianSelfCorrector
        from src.core.neural_aware_router import NeuralAwareModelRouter

        logger.info("Initializing neural systems...")

        # Neural-validated planner (personalized planning using neural memory)
        self._neural_validated_planner = NeuralValidatedPlanner(
            neural_memory=self._neural_memory, llm_manager=self._llm_manager
        )

        # Hebbian self-correction (learning from successes/failures)
        self._hebbian_self_correction = HebbianSelfCorrector(
            neural_memory=self._neural_memory
        )

        # Neural-aware model router (context-aware model selection)
        self._neural_aware_router = NeuralAwareModelRouter(
            llm_manager=self._llm_manager, neural_memory=self._neural_memory
        )

        # Wire neural systems to agent loop
        if self._agent_loop:
            self._agent_loop.set_neural_systems(
                planner=self._neural_validated_planner,
                hebbian=self._hebbian_self_correction,
                router=self._neural_aware_router,
            )

        # NEW: Initialize Proactivity Controller & Tool Orchestrator
        await self._init_proactivity_and_orchestrator()

        logger.info("Neural systems initialized")

    async def _init_proactivity_and_orchestrator(self):
        """Initialize Proactivity Controller and Tool Orchestrator"""
        from src.core.proactivity_controller import get_proactivity_controller
        from src.core.tool_orchestrator import get_orchestrator

        logger.info("Initializing Proactivity Controller & Tool Orchestrator...")

        # Proactivity Controller - knows when NOT to act (safety first)
        self._proactivity_controller = get_proactivity_controller()
        self._proactivity_controller.neural_memory = self._neural_memory
        self._proactivity_controller.user_profile = self._user_profile

        # Tool Orchestrator - deterministic execution of JSON plans
        self._tool_orchestrator = get_orchestrator()
        self._tool_orchestrator.tool_registry = self._tool_registry

        # Register tool handlers with orchestrator
        if hasattr(self, "_agent_loop") and self._agent_loop:
            for tool_name, handler in self._agent_loop.tools.items():
                if handler and callable(handler):
                    self._tool_orchestrator.register_handler(tool_name, handler)

        # Wire orchestrator to agent loop
        if self._agent_loop:
            self._agent_loop.set_tool_orchestrator(self._tool_orchestrator)

        logger.info("Proactivity Controller & Tool Orchestrator initialized")

    async def _init_security(self):
        """Initialize security system"""
        from src.security import (
            get_authenticator,
            get_privacy_manager,
            get_permission_manager,
            get_security_auditor,
        )

        logger.info("Initializing security...")

        self._authenticator = get_authenticator()
        self._privacy_manager = get_privacy_manager()
        self._permission_manager = get_permission_manager()
        self._security_auditor = get_security_auditor()

        # Check if authentication is required
        if self._authenticator.has_auth_setup():
            logger.info("Authentication is configured")
        else:
            logger.info("No authentication configured - running in open mode")

        logger.info("Security initialized")

    async def _init_utils(self):
        """Initialize utility services"""
        from src.utils import (
            HealthMonitor,
            CircuitBreakerManager,
            GracefulShutdown,
            ErrorRecovery,
        )

        logger.info("Initializing utilities...")

        self._health_monitor = HealthMonitor()
        self._circuit_breaker = CircuitBreakerManager()
        self._graceful_shutdown = GracefulShutdown()
        self._error_recovery = ErrorRecovery()

        # Register shutdown handler
        self._graceful_shutdown.register_component(
            "health_monitor", self._health_monitor.stop, priority=10
        )

    async def _init_context(self):
        """Initialize real-time context provider"""
        logger.info("Initializing context provider...")

        await self._context_provider.start()

        logger.info("Context provider initialized")

    async def _init_session(self):
        """Initialize session manager"""
        from src.session import get_session_manager

        logger.info("Initializing session manager...")

        self._session_manager = get_session_manager()

        logger.info("Session manager initialized")

    async def _init_learning(self):
        """Initialize learning engine"""
        from src.learning import get_learning_engine

        logger.info("Initializing learning engine...")

        self._learning_engine = get_learning_engine()
        await self._learning_engine.start()

        logger.info("Learning engine initialized")

    async def _init_core_services(self):
        """Initialize core services"""
        from src.services.adaptive_context import AdaptiveContextEngine
        from src.services.life_tracker import LifeTracker
        from src.services.task_context import TaskContextPreservation

        logger.info("Initializing core services...")

        self._context_engine = AdaptiveContextEngine()
        self._life_tracker = LifeTracker()
        self._task_context = TaskContextPreservation()

    async def _init_intelligence(self):
        """Initialize intelligence services"""
        from src.services.proactive_engine import ProactiveEngine
        from src.services.self_learning import SelfLearningEngine

        logger.info("Initializing intelligence services...")

        self._self_learning = SelfLearningEngine()
        await self._self_learning.initialize()

        self._proactive_engine = ProactiveEngine(
            life_tracker=self._life_tracker,
            memory_system=self._neural_memory,
        )

    async def _init_proactive_services(self):
        """Initialize AURA-Native proactive services"""
        from src.services.proactive_event_tracker import ProactiveEventTracker
        from src.services.intelligent_call_manager import IntelligentCallManager
        from src.services.proactive_life_explorer import ProactiveLifeExplorer

        logger.info("Initializing proactive services...")

        # Proactive event tracker - auto-tracks dates/events
        self._proactive_event_tracker = ProactiveEventTracker(
            neural_memory=self._neural_memory,
            user_profile=self._user_profile,
        )
        await self._proactive_event_tracker.initialize()

        # Intelligent call manager - context-aware call handling
        self._intelligent_call_manager = IntelligentCallManager(
            neural_memory=self._neural_memory,
            termux_bridge=self._termux_bridge,
        )
        await self._intelligent_call_manager.initialize()

        # Proactive life explorer - explores/manages user's life
        self._proactive_life_explorer = ProactiveLifeExplorer(
            neural_memory=self._neural_memory,
            user_profile=self._user_profile,
            proactive_engine=self._proactive_engine,
        )
        await self._proactive_life_explorer.initialize()

        # Wire proactive services to the proactive engine
        if self._proactive_engine:
            self._proactive_engine.set_services(
                event_tracker=self._proactive_event_tracker,
                call_manager=self._intelligent_call_manager,
                life_explorer=self._proactive_life_explorer,
            )

        logger.info("Proactive services initialized")

    async def _init_addons(self):
        """Initialize addons system"""
        from src.addons.discovery import AppDiscovery
        from src.addons.termux_bridge import TermuxBridge
        from src.addons.tool_binding import AdaptiveToolBinder
        from src.addons.capability_gap import CapabilityGapHandler

        logger.info("Initializing addons...")

        # App discovery - finds installed apps
        self._app_discovery = AppDiscovery()

        # Termux bridge - controls Android
        self._termux_bridge = TermuxBridge()

        # Tool binding - creates LLM tool bindings
        self._tool_binding = AdaptiveToolBinder()

        # Capability gap - finds alternatives for missing features
        self._capability_gap = CapabilityGapHandler()

        # Initialize mobile UI systems
        await self._init_mobile_systems()

    async def _init_mobile_systems(self):
        """Initialize mobile UI and widget systems"""
        logger.info("Initializing mobile systems...")

        # Aura Space Server - Local UI server
        if self._mobile_config.get("aura_space_enabled", True):
            try:
                from src.mobile.aura_space_server import get_aura_space_server

                self._aura_space_server = get_aura_space_server()
                self._aura_space_server.aura = self
                logger.info("Aura Space Server initialized")
            except Exception as e:
                logger.warning(f"Aura Space Server unavailable: {e}")
                self._aura_space_server = None

        # Termux Widget Bridge - Floating bubbles, HUD
        if self._mobile_config.get("termux_widgets_enabled", True):
            try:
                from src.mobile.termux_widget_bridge import (
                    get_termux_bridge,
                    FloatingBubbleManager,
                    HUDOverlayManager,
                )

                self._termux_widget_bridge = get_termux_bridge()
                self._floating_bubble_manager = FloatingBubbleManager(
                    self._termux_widget_bridge
                )
                self._hud_overlay_manager = HUDOverlayManager(
                    self._termux_widget_bridge
                )
                logger.info(
                    f"Termux Widget Bridge initialized (available: {self._termux_widget_bridge.is_available()})"
                )
            except Exception as e:
                logger.warning(f"Termux Widget Bridge unavailable: {e}")
                self._termux_widget_bridge = None
                self._floating_bubble_manager = None
                self._hud_overlay_manager = None

        # Character Sheet - RPG-style profiles
        if self._mobile_config.get("character_sheet_enabled", True):
            try:
                from src.mobile.character_sheet import get_character_sheet_system

                self._character_sheet = get_character_sheet_system()
                logger.info("Character Sheet System initialized")
            except Exception as e:
                logger.warning(f"Character Sheet System unavailable: {e}")
                self._character_sheet = None

        # Cinematic Moments - Weekly recaps
        if self._mobile_config.get("cinematic_moments_enabled", True):
            try:
                from src.mobile.cinematic_moments import get_cinematic_moments_system

                self._cinematic_moments = get_cinematic_moments_system()
                # Link to character sheet for data
                if self._character_sheet:
                    self._cinematic_moments.character_sheet = self._character_sheet
                logger.info("Cinematic Moments System initialized")
            except Exception as e:
                logger.warning(f"Cinematic Moments System unavailable: {e}")
                self._cinematic_moments = None

        logger.info("Mobile systems initialized")

    async def _init_social_life(self):
        """Initialize social life agent"""
        from src.agents.social_life import get_social_life_agent

        logger.info("Initializing Social-Life Manager...")

        self._social_life_agent = get_social_life_agent()
        await self._social_life_agent.initialize()

        logger.info("Social-Life Manager initialized")

    async def _init_healthcare(self):
        """Initialize healthcare agent"""
        from src.agents.healthcare import HealthcareAgent
        from src.agents.coordinator import AgentCoordinator

        logger.info("Initializing Healthcare Agent...")

        try:
            coordinator = AgentCoordinator()
            self._healthcare_agent = HealthcareAgent(coordinator=coordinator)
            await self._healthcare_agent.initialize()
            logger.info("Healthcare Agent initialized")
        except Exception as e:
            logger.warning(f"Could not initialize Healthcare Agent: {e}")
            self._healthcare_agent = None

    async def _init_inner_voice(self):
        """Initialize inner voice system"""
        from src.services.inner_voice import InnerVoiceSystem

        logger.info("Initializing Inner Voice System...")

        self._inner_voice = InnerVoiceSystem(
            neural_memory=self._neural_memory,
            user_profile=self._user_profile,
        )
        await self._inner_voice.initialize()

        logger.info("Inner Voice System initialized")

        logger.info("Addons initialized")

    async def _init_interface(self):
        """Initialize interface services"""
        from src.services.dashboard import DashboardService
        from src.services.background_manager import BackgroundResourceManager

        logger.info("Initializing interface services...")

        self._dashboard = DashboardService()
        self._background_manager = BackgroundResourceManager()

    async def start(self):
        """Start AURA"""
        if not self._initialized:
            await self.initialize()

        logger.info("Starting AURA v3...")
        self._running = True

        # Start all services
        await self._context_provider.start()
        await self._context_engine.start()
        await self._life_tracker.start()
        await self._proactive_engine.start()
        await self._dashboard.start()
        await self._background_manager.start()
        await self._health_monitor.start()

        # Start proactive services (NEW!)
        if self._proactive_event_tracker:
            await self._proactive_event_tracker.start()
        if self._intelligent_call_manager:
            await self._intelligent_call_manager.start()
        if self._proactive_life_explorer:
            await self._proactive_life_explorer.start()

        # Start Social-Life Manager
        if self._social_life_agent:
            await self._social_life_agent.start()

        # Start Mobile Systems (background, non-blocking)
        await self._start_mobile_systems()

        # Create initial session
        from src.session import SessionType

        self._session_manager.create_session(session_type=SessionType.INTERACTION)

        logger.info("AURA v3 is now running!")

    async def _start_mobile_systems(self):
        """Start all mobile systems in background (non-blocking)"""
        # Start Aura Space Server
        if self._aura_space_server:
            try:
                port = self._mobile_config.get("aura_space_port", 8080)
                self._aura_space_server.port = port
                server_url = await self._aura_space_server.start()
                logger.info(f"Aura Space Server started at {server_url}")
            except Exception as e:
                logger.warning(f"Failed to start Aura Space Server: {e}")

    async def _stop_mobile_systems(self):
        """Stop all mobile systems gracefully"""
        # Stop Aura Space Server
        if self._aura_space_server:
            try:
                await self._aura_space_server.stop()
                logger.info("Aura Space Server stopped")
            except Exception as e:
                logger.warning(f"Error stopping Aura Space Server: {e}")

    async def stop(self):
        """Stop AURA gracefully"""
        logger.info("Stopping AURA v3...")
        self._running = False

        # End current session
        if self._session_manager:
            self._session_manager.end_session()

        # Stop context provider
        if self._context_provider:
            await self._context_provider.stop()

        # Stop learning engine
        if self._learning_engine:
            await self._learning_engine.stop()

        # Stop in reverse order
        if self._health_monitor:
            await self._health_monitor.stop()
        if self._background_manager:
            await self._background_manager.stop()
        if self._dashboard:
            await self._dashboard.stop()
        if self._proactive_engine:
            await self._proactive_engine.stop()
        if self._life_tracker:
            await self._life_tracker.stop()
        if self._social_life_agent:
            await self._social_life_agent.stop()

        # Stop proactive services (NEW!)
        if self._proactive_life_explorer:
            await self._proactive_life_explorer.stop()
        if self._intelligent_call_manager:
            await self._intelligent_call_manager.stop()
        if self._proactive_event_tracker:
            await self._proactive_event_tracker.stop()

        # Stop Mobile Systems
        await self._stop_mobile_systems()

        logger.info("AURA v3 stopped")

    async def process(self, user_input: str) -> str:
        """
        Process user input and return response
        Main entry point for user interaction
        THE BRAIN - Uses LLM-powered agent loop
        """
        if not self._running:
            return "AURA is not running. Call start() first."

        try:
            # Get current context for processing
            context = await self._context_provider.get_current_context()
            context_str = await self._context_provider.get_context_for_llm()

            # Get user profile for personalization
            user_context = (
                await self._user_profile.get_current_context()
                if self._user_profile
                else {}
            )

            # Get personality state for tone
            personality_state = (
                await self._personality.get_state() if self._personality else None
            )

            # Process through THE BRAIN - LLM-powered agent loop
            agent_response = await self._agent_loop.process(
                user_input,
                context=context_str,
                user_profile=user_context,
                personality=personality_state,
            )

            # Track relationship - help given when tools executed
            if self._relationship_system and hasattr(agent_response, "tools_used"):
                if agent_response.tools_used:
                    self._relationship_system.record_help_outcome(successful=True)

            # Record interaction for learning
            if self._learning_engine:
                await self._learning_engine.record_interaction(
                    input_text=user_input,
                    output_text=agent_response.message,
                    context={
                        "context": context_str,
                        "device": context.device.battery_level
                        if context.device
                        else None,
                        "location": context.location.place_name
                        if context.location
                        else None,
                    },
                    success=True,
                )

            # Add to session
            if self._session_manager:
                self._session_manager.add_turn(
                    user_input=user_input, aura_response=agent_response.message
                )

            # Apply personality and tone
            if self._personality and agent_response.message:
                final_response = await self._personality.format_response(
                    agent_response.message, personality_state
                )
            else:
                final_response = agent_response.message

            return final_response

        except Exception as e:
            logger.error(f"Error processing input: {e}")

            # Record failure for learning
            if self._learning_engine:
                await self._learning_engine.record_interaction(
                    input_text=user_input,
                    output_text="",
                    context={"error": str(e)},
                    success=False,
                )

            return f"I encountered an error processing your request: {str(e)}"

    def get_status(self) -> dict:
        """Get AURA status"""
        exec_status = None
        if self._execution_controller:
            try:
                exec_status = self._execution_controller.get_status()
            except:
                pass

        return {
            "running": self._running,
            "initialized": self._initialized,
            "core_systems": {
                "neuromorphic_engine": self._neuromorphic_engine is not None,
                "mobile_power": self._mobile_power is not None,
                "conversation": self._conversation is not None,
                "user_profile": self._user_profile is not None,
                "personality": self._personality is not None,
                "context_provider": self._context_provider is not None,
            },
            "addons": {
                "app_discovery": self._app_discovery is not None,
                "termux_bridge": self._termux_bridge is not None,
                "tool_binding": self._tool_binding is not None,
                "capability_gap": self._capability_gap is not None,
            },
            "services": {
                "context_engine": self._context_engine is not None,
                "life_tracker": self._life_tracker is not None,
                "proactive_engine": self._proactive_engine is not None,
                "dashboard": self._dashboard is not None,
                "background_manager": self._background_manager is not None,
            },
            "learning": {
                "learning_engine": self._learning_engine is not None,
                "session_manager": self._session_manager is not None,
            },
            "security": {
                "authenticator": self._authenticator is not None,
                "privacy_manager": self._privacy_manager is not None,
            },
            "execution_control": exec_status,
            "mobile_systems": {
                "aura_space_server": {
                    "enabled": self._mobile_config.get("aura_space_enabled", True),
                    "running": self._aura_space_server._running
                    if self._aura_space_server
                    else False,
                },
                "termux_widget_bridge": {
                    "enabled": self._mobile_config.get("termux_widgets_enabled", True),
                    "available": self._termux_widget_bridge.is_available()
                    if self._termux_widget_bridge
                    else False,
                },
                "character_sheet": {
                    "enabled": self._mobile_config.get("character_sheet_enabled", True),
                    "active": self._character_sheet is not None,
                },
                "cinematic_moments": {
                    "enabled": self._mobile_config.get(
                        "cinematic_moments_enabled", True
                    ),
                    "active": self._cinematic_moments is not None,
                },
            },
        }

    async def get_context(self) -> dict:
        """Get current real-time context"""
        if self._context_provider:
            return await self._context_provider.get_current_context()
        return {}

    async def get_user_profile(self) -> dict:
        """Get user profile"""
        if self._user_profile:
            return await self._user_profile.get_current_context()
        return {}

    async def get_personality_state(self) -> dict:
        """Get personality state"""
        if self._personality:
            return await self._personality.get_state()
        return {}

    async def get_session_summary(self) -> dict:
        """Get current session summary"""
        if self._session_manager:
            session = self._session_manager.get_session()
            if session:
                return {
                    "session_id": session.session_id,
                    "total_turns": session.total_turns,
                    "successful_turns": session.successful_turns,
                    "avg_response_time_ms": session.avg_response_time_ms,
                }
        return {}

    # ===== Mobile System Helpers =====

    async def update_mobile_state(self, **kwargs):
        """Update Aura state for mobile UI (Aura Space)"""
        if self._aura_space_server:
            self._aura_space_server.update_state(**kwargs)

    async def add_mobile_thought(self, thought: str, bubble_type: str = "normal"):
        """Add a thought to mobile UI thought stream"""
        if self._aura_space_server:
            self._aura_space_server.add_thought(thought, bubble_type)

    async def show_termux_notification(
        self, title: str, content: str, urgency: str = "normal"
    ):
        """Show a Termux notification"""
        if self._termux_widget_bridge:
            await self._termux_widget_bridge.show_notification(title, content, urgency)

    async def show_floating_bubble(self, status: str, thought: str = None):
        """Show floating bubble with Aura's status"""
        if self._floating_bubble_manager:
            await self._floating_bubble_manager.show_bubble(status, thought)

    async def show_hud(self, state: dict):
        """Show HUD overlay"""
        if self._hud_overlay_manager:
            await self._hud_overlay_manager.show_hud(state)

    def get_character_sheet(self) -> dict:
        """Get user character sheet"""
        if self._character_sheet:
            return self._character_sheet.get_user_sheet()
        return {}

    def get_aura_sheet(self) -> dict:
        """Get Aura's character sheet"""
        if self._character_sheet:
            return self._character_sheet.get_aura_sheet()
        return {}

    def update_user_attribute(self, attribute: str, value: float, evidence: str = ""):
        """Update a user attribute in character sheet"""
        if self._character_sheet:
            self._character_sheet.update_attribute(attribute, value, evidence)

    async def check_cinematic_moments(self):
        """Check and trigger cinematic moments"""
        if self._cinematic_moments:
            return await self._cinematic_moments.check_and_trigger_moments()
        return None

    async def trigger_milestone(self, milestone_type: str, details: dict):
        """Trigger a milestone cinematic moment"""
        if self._cinematic_moments:
            return await self._cinematic_moments.trigger_milestone(
                milestone_type, details
            )
        return None

    def set_mobile_config(self, **config):
        """Configure mobile systems at runtime"""
        self._mobile_config.update(config)


# Global instance
_aura_instance: Optional[AuraProduction] = None


async def get_aura() -> AuraProduction:
    """Get or create AURA instance"""
    global _aura_instance
    if _aura_instance is None:
        _aura_instance = AuraProduction()
    return _aura_instance


async def main():
    """Main entry point"""
    global _aura_instance

    aura = await get_aura()

    # Setup signal handlers
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}, shutting down...")
        asyncio.create_task(aura.stop())

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        await aura.start()

        # Keep running
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
    finally:
        await aura.stop()


if __name__ == "__main__":
    asyncio.run(main())
