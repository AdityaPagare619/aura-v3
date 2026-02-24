"""
AURA v3 Core Package
Neuromorphic processing engine and mobile-optimized core components
"""

from src.core.neuromorphic_engine import (
    NeuromorphicEngine,
    ResourceBudget,
    EventDrivenProcessor,
    NeuralEvent,
    ThermalState,
    MultiAgentOrchestrator,
    SubAgent,
    ProactivePlanner,
    ProactiveAction,
    get_neuromorphic_engine,
)

from src.core.mobile_power import (
    MobilePowerManager,
    MobileSensorManager,
    PowerMode,
    ScreenState,
    PowerConfig,
    SensorType,
    SensorReading,
    get_power_manager,
    get_sensor_manager,
)

from src.core.conversation import (
    EmotionalConversationEngine,
    AURAResponse,
    MessageContext,
    Tone,
    ConversationPhase,
    get_conversation_engine,
)

from src.core.user_profile import (
    DeepUserProfiler,
    PsychologicalProfile,
    CommunicationProfile,
    BehavioralProfile,
    EmotionalProfile,
    LifeContext,
    get_user_profiler,
)

from src.core.adaptive_personality import (
    AdaptivePersonalityEngine,
    AuraCoreIdentity,
    PersonalityState,
    PersonalityDimension,
    AuraOpinions,
    ReactionSystem,
    Reaction,
    get_personality_engine,
)

# AURA-Native Planning (Neural-Validated)
from src.core.neural_validated_planner import (
    NeuralValidatedPlanner,
    NeuralPatternValidator,
    NeuralValidationResult,
    ValidationResult,
    JSONPlan,
    create_neural_planner,
)

# AURA-Native Self-Correction (Hebbian)
from src.core.hebbian_self_correction import (
    HebbianSelfCorrector,
    HebbianCorrection,
    ActionOutcome,
    create_hebbian_corrector,
)

# AURA-Native Model Routing
from src.core.neural_aware_router import (
    NeuralAwareModelRouter,
    ModelTier,
    ModelConfig,
    NeuralDecision,
    create_neural_router,
)

# Loop Detection System
from src.core.loop_detector import (
    LoopDetector,
    LoopDetectionLevel,
    LoopType,
    LoopDetectionConfig,
    LoopDetectionResult,
    ActionFingerprint,
    ActionHasher,
    TextSimilarity,
    ResourceMonitor,
    IntegrationMixin,
    get_loop_detector,
    create_loop_detector,
)

# Execution Control (Stop/Kill System)
from src.core.execution_control import (
    ExecutionController,
    ExecutionState,
    StopLevel,
    AtomicActionType,
    AtomicAction,
    OperationTimeout,
    StopEvent,
    ExecutionSnapshot,
    AgentLoopIntegration,
    handle_stop_command,
    handle_force_stop_command,
    handle_kill_command,
    handle_execution_status_command,
    get_execution_controller,
)

__all__ = [
    # Neuromorphic Engine
    "NeuromorphicEngine",
    "ResourceBudget",
    "EventDrivenProcessor",
    "NeuralEvent",
    "ThermalState",
    "MultiAgentOrchestrator",
    "SubAgent",
    "ProactivePlanner",
    "ProactiveAction",
    "get_neuromorphic_engine",
    # Mobile Power & Sensors
    "MobilePowerManager",
    "MobileSensorManager",
    "PowerMode",
    "ScreenState",
    "PowerConfig",
    "SensorType",
    "SensorReading",
    "get_power_manager",
    "get_sensor_manager",
    # Conversation
    "EmotionalConversationEngine",
    "AURAResponse",
    "MessageContext",
    "Tone",
    "ConversationPhase",
    "get_conversation_engine",
    # User Profiling
    "DeepUserProfiler",
    "PsychologicalProfile",
    "CommunicationProfile",
    "BehavioralProfile",
    "EmotionalProfile",
    "LifeContext",
    "get_user_profiler",
    # Adaptive Personality
    "AdaptivePersonalityEngine",
    "AuraCoreIdentity",
    "PersonalityState",
    "PersonalityDimension",
    "AuraOpinions",
    "ReactionSystem",
    "Reaction",
    "get_personality_engine",
    # AURA-Native Neural-Validated Planning
    "NeuralValidatedPlanner",
    "NeuralPatternValidator",
    "NeuralValidationResult",
    "ValidationResult",
    "JSONPlan",
    "create_neural_planner",
    # AURA-Native Hebbian Self-Correction
    "HebbianSelfCorrector",
    "HebbianCorrection",
    "ActionOutcome",
    "create_hebbian_corrector",
    # AURA-Native Neural-Aware Router
    "NeuralAwareModelRouter",
    "ModelTier",
    "ModelConfig",
    "NeuralDecision",
    "create_neural_router",
    # Loop Detection System
    "LoopDetector",
    "LoopDetectionLevel",
    "LoopType",
    "LoopDetectionConfig",
    "LoopDetectionResult",
    "ActionFingerprint",
    "ActionHasher",
    "TextSimilarity",
    "ResourceMonitor",
    "IntegrationMixin",
    "get_loop_detector",
    "create_loop_detector",
    # Execution Control
    "ExecutionController",
    "ExecutionState",
    "StopLevel",
    "AtomicActionType",
    "AtomicAction",
    "OperationTimeout",
    "StopEvent",
    "ExecutionSnapshot",
    "AgentLoopIntegration",
    "handle_stop_command",
    "handle_force_stop_command",
    "handle_kill_command",
    "handle_execution_status_command",
    "get_execution_controller",
]
