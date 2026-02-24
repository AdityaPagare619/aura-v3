"""
AURA v3 Agent Package
LLM-powered reasoning brain
"""

from src.agent.loop import (
    AuraAgentLoop,
    AgentState,
    ReasoningStep,
    Thought,
    AgentContext,
    AgentResponse,
    get_agent,
)

from src.agent.app_explorer import (
    AppExplorer,
    ExplorationCommandHandler,
    Suggestion,
    SuggestionPriority,
    SecurityAlert,
    AppInfo,
    AppUsagePattern,
    CrossAppPattern,
    AnalysisType,
    get_app_explorer,
    get_exploration_handler,
    run_exploration,
)

from src.agent.daily_reporter import (
    DailyReporter,
    ReportType,
    ReportTime,
    ContentCategory,
    ReportPreferences,
    ReportHighlight,
    ReportData,
    get_daily_reporter,
)

from src.agent.personality_system import (
    PersonalitySystem,
    PersonalityCommands,
    MoodState,
    MoralPrinciple,
    get_personality_system,
)

from src.agent.relationship_system import (
    RelationshipSystem,
    RelationshipStage,
    RelationshipState,
    RelationshipMetrics,
    get_relationship_system,
    initialize_relationship_system,
)

from src.agent.interest_learner import (
    InterestCategory,
    Interest,
    InterestProfile,
    InterestDetector,
    get_interest_detector,
)

__all__ = [
    # Agent loop
    "AuraAgentLoop",
    "AgentState",
    "ReasoningStep",
    "Thought",
    "AgentContext",
    "AgentResponse",
    "get_agent",
    # App explorer
    "AppExplorer",
    "ExplorationCommandHandler",
    "Suggestion",
    "SuggestionPriority",
    "SecurityAlert",
    "AppInfo",
    "AppUsagePattern",
    "CrossAppPattern",
    "AnalysisType",
    "get_app_explorer",
    "get_exploration_handler",
    "run_exploration",
    # Daily reporter
    "DailyReporter",
    "ReportType",
    "ReportTime",
    "ContentCategory",
    "ReportPreferences",
    "ReportHighlight",
    "ReportData",
    "get_daily_reporter",
    # Relationship system
    "RelationshipSystem",
    "RelationshipStage",
    "RelationshipState",
    "RelationshipMetrics",
    "get_relationship_system",
    "initialize_relationship_system",
    # Interest learner
    "InterestCategory",
    "Interest",
    "InterestProfile",
    "InterestDetector",
    "get_interest_detector",
]
