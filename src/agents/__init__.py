"""
AURA v3 Agents Package
Multi-agent system for AURA
"""

from src.agents.coordinator import (
    AgentCoordinator,
    Agent,
    AgentType,
    AgentStatus,
    AgentTask,
    OrchestratorAgent,
    AnalyzerAgent,
    ExecutorAgent,
)

from src.agents.specialized_agents import (
    SocialMediaAnalyzerAgent,
    ShoppingAssistantAgent,
    ResearchAgent,
    AutomationAgent,
    create_specialized_agents,
    UserInterest,
    ShoppingIntent,
    ProductMatch,
    ShoppingUserProfile,
)

from src.agents.social_life import (
    SocialLifeAgent,
    get_social_life_agent,
    SocialAppAnalyzer,
    PatternRecognizer,
    RelationshipTracker,
    SocialInsights,
    EventManager,
    SocialPersonality,
)

from src.agents.healthcare import (
    HealthcareAgent,
    HealthDataAnalyzer,
    DietPlanner,
    FitnessTracker,
    HealthInsightsEngine,
    HealthcarePersonality,
    HealthProfile,
    MetricType,
    MealType,
    WorkoutType,
    HealthGoal,
    HealthInsight,
)

__all__ = [
    # Coordinator
    "AgentCoordinator",
    "Agent",
    "AgentType",
    "AgentStatus",
    "AgentTask",
    "OrchestratorAgent",
    "AnalyzerAgent",
    "ExecutorAgent",
    # Specialized
    "SocialMediaAnalyzerAgent",
    "ShoppingAssistantAgent",
    "ResearchAgent",
    "AutomationAgent",
    "create_specialized_agents",
    "UserInterest",
    "ShoppingIntent",
    "ProductMatch",
    "ShoppingUserProfile",
    # Social Life
    "SocialLifeAgent",
    "get_social_life_agent",
    "SocialAppAnalyzer",
    "PatternRecognizer",
    "RelationshipTracker",
    "SocialInsights",
    "EventManager",
    "SocialPersonality",
    # Healthcare
    "HealthcareAgent",
    "HealthDataAnalyzer",
    "DietPlanner",
    "FitnessTracker",
    "HealthInsightsEngine",
    "HealthcarePersonality",
    "HealthProfile",
    "MetricType",
    "MealType",
    "WorkoutType",
    "HealthGoal",
    "HealthInsight",
]
