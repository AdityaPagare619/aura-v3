"""
AURA v3 Character Sheet System
===============================

Manages the "character sheets" for:
1. User Sheet - Aura's understanding of the user
2. Aura Sheet - Aura's own capabilities and learning

This creates the RPG-like progression feel where users can see:
- Their tracked attributes
- Aura's current capabilities (like a tech tree)
- Progress and growth over time
"""

import json
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class AttributeType(Enum):
    """Types of user attributes Aura tracks"""

    FOCUS = "focus"
    ENERGY = "energy"
    RISK_TOLERANCE = "risk_tolerance"
    SOCIAL_LOAD = "social_load"
    PRIVACY_SENSITIVITY = "privacy_sensitivity"
    CURIOSITY = "curiosity"
    CONSISTENCY = "consistency"


class CapabilityStatus(Enum):
    """Status of Aura's capabilities"""

    LOCKED = "locked"
    TRAINING = "training"
    ACTIVE = "active"
    MASTERED = "mastered"


@dataclass
class UserAttribute:
    """A user attribute Aura tracks"""

    name: str
    value: float  # 0-10
    trend: str  # improving, declining, stable
    last_updated: str
    evidence: List[str] = field(default_factory=list)


@dataclass
class UserGoal:
    """A user goal Aura is helping with"""

    id: str
    title: str
    description: str
    progress: float  # 0-1
    created_at: str
    last_update: str


@dataclass
class AuraCapability:
    """A capability of Aura"""

    id: str
    name: str
    description: str
    status: CapabilityStatus
    training_progress: float  # 0-1
    last_trained: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)


@dataclass
class TraitDetected:
    """A trait Aura has detected about the user"""

    trait: str
    confidence: float
    evidence: List[str]
    detected_at: str


class CharacterSheetSystem:
    """
    Manages the "character sheets" for user and Aura.

    User Sheet:
    - Attributes (Focus, Energy, etc.)
    - Goals (short and long term)
    - Detected traits

    Aura Sheet:
    - Capabilities (like a tech tree)
    - Training progress
    - Next planned upgrades
    """

    def __init__(self):
        # User attributes (tracked by Aura)
        self.user_attributes: Dict[str, UserAttribute] = {
            "focus": UserAttribute("Focus", 5.0, "stable", datetime.now().isoformat()),
            "energy": UserAttribute(
                "Energy", 5.0, "stable", datetime.now().isoformat()
            ),
            "risk_tolerance": UserAttribute(
                "Risk Tolerance", 5.0, "stable", datetime.now().isoformat()
            ),
            "social_load": UserAttribute(
                "Social Load", 5.0, "stable", datetime.now().isoformat()
            ),
            "privacy_sensitivity": UserAttribute(
                "Privacy Sensitivity", 7.0, "stable", datetime.now().isoformat()
            ),
            "curiosity": UserAttribute(
                "Curiosity", 7.0, "stable", datetime.now().isoformat()
            ),
            "consistency": UserAttribute(
                "Consistency", 5.0, "stable", datetime.now().isoformat()
            ),
        }

        # User goals
        self.user_goals: List[UserGoal] = []

        # Detected traits
        self.detected_traits: List[TraitDetected] = []

        # Aura capabilities (tech tree)
        self.aura_capabilities: Dict[str, AuraCapability] = {
            "messaging": AuraCapability(
                id="messaging",
                name="Messaging",
                description="Send and manage messages",
                status=CapabilityStatus.ACTIVE,
                training_progress=1.0,
            ),
            "scheduling": AuraCapability(
                id="scheduling",
                name="Scheduling",
                description="Manage calendar and schedules",
                status=CapabilityStatus.ACTIVE,
                training_progress=1.0,
            ),
            "task_management": AuraCapability(
                id="task_management",
                name="Task Management",
                description="Organize and track tasks",
                status=CapabilityStatus.ACTIVE,
                training_progress=1.0,
            ),
            "health_tracking": AuraCapability(
                id="health_tracking",
                name="Health Tracking",
                description="Track health metrics and habits",
                status=CapabilityStatus.ACTIVE,
                training_progress=1.0,
            ),
            "social_automation": AuraCapability(
                id="social_automation",
                name="Social Automation",
                description="Manage social media and contacts",
                status=CapabilityStatus.TRAINING,
                training_progress=0.6,
            ),
            "app_automation": AuraCapability(
                id="app_automation",
                name="App Automation",
                description="Control and automate phone apps",
                status=CapabilityStatus.TRAINING,
                training_progress=0.4,
                dependencies=["messaging"],
            ),
            "self_improvement": AuraCapability(
                id="self_improvement",
                name="Self Improvement",
                description="Aura learns and improves itself",
                status=CapabilityStatus.TRAINING,
                training_progress=0.2,
                dependencies=["task_management"],
            ),
        }

        # Next planned upgrade
        self.next_upgrade: Optional[Dict] = None

    def update_attribute(self, attribute: str, value: float, evidence: str = ""):
        """Update a user attribute"""
        if attribute in self.user_attributes:
            old_value = self.user_attributes[attribute].value

            # Determine trend
            if value > old_value + 0.5:
                trend = "improving"
            elif value < old_value - 0.5:
                trend = "declining"
            else:
                trend = "stable"

            self.user_attributes[attribute] = UserAttribute(
                name=self.user_attributes[attribute].name,
                value=value,
                trend=trend,
                last_updated=datetime.now().isoformat(),
                evidence=self.user_attributes[attribute].evidence + [evidence]
                if evidence
                else [],
            )

    def add_goal(
        self, title: str, description: str, goal_type: str = "short_term"
    ) -> UserGoal:
        """Add a new user goal"""
        goal = UserGoal(
            id=f"goal_{len(self.user_goals) + 1}",
            title=title,
            description=description,
            progress=0.0,
            created_at=datetime.now().isoformat(),
            last_update=datetime.now().isoformat(),
        )
        self.user_goals.append(goal)
        return goal

    def update_goal_progress(self, goal_id: str, progress: float):
        """Update progress on a goal"""
        for goal in self.user_goals:
            if goal.id == goal_id:
                goal.progress = min(1.0, max(0.0, progress))
                goal.last_update = datetime.now().isoformat()
                break

    def detect_trait(self, trait: str, confidence: float, evidence: List[str]):
        """Record a detected trait about the user"""
        # Check if we already have this trait
        for existing in self.detected_traits:
            if existing.trait == trait:
                # Update if new evidence is stronger
                if confidence > existing.confidence:
                    existing.confidence = confidence
                    existing.evidence = evidence
                    existing.detected_at = datetime.now().isoformat()
                return

        # Add new trait
        self.detected_traits.append(
            TraitDetected(
                trait=trait,
                confidence=confidence,
                evidence=evidence,
                detected_at=datetime.now().isoformat(),
            )
        )

    def get_user_sheet(self) -> Dict:
        """Get the full user character sheet"""
        return {
            "attributes": {
                k: {
                    "name": v.name,
                    "value": v.value,
                    "trend": v.trend,
                    "last_updated": v.last_updated,
                }
                for k, v in self.user_attributes.items()
            },
            "goals": [asdict(g) for g in self.user_goals],
            "traits": [asdict(t) for t in self.detected_traits[-10:]],  # Last 10 traits
        }

    def get_aura_sheet(self) -> Dict:
        """Get Aura's character sheet"""
        # Calculate overall progress
        active_count = sum(
            1
            for c in self.aura_capabilities.values()
            if c.status == CapabilityStatus.ACTIVE
        )
        total_count = len(self.aura_capabilities)

        return {
            "capabilities": {
                k: {
                    "id": v.id,
                    "name": v.name,
                    "description": v.description,
                    "status": v.status.value,
                    "training_progress": v.training_progress,
                    "last_trained": v.last_trained,
                    "dependencies": v.dependencies,
                }
                for k, v in self.aura_capabilities.items()
            },
            "stats": {
                "active_capabilities": active_count,
                "total_capabilities": total_count,
                "overall_progress": active_count / total_count
                if total_count > 0
                else 0,
            },
            "next_upgrade": self.next_upgrade,
            "currently_training": [
                c.name
                for c in self.aura_capabilities.values()
                if c.status == CapabilityStatus.TRAINING
            ],
        }

    def set_next_upgrade(self, capability_id: str, description: str):
        """Set the next planned upgrade"""
        self.next_upgrade = {
            "capability_id": capability_id,
            "description": description,
            "planned_for": datetime.now().isoformat(),
        }

    def train_capability(self, capability_id: str, progress_delta: float):
        """Update training progress for a capability"""
        if capability_id in self.aura_capabilities:
            cap = self.aura_capabilities[capability_id]
            cap.training_progress = min(1.0, cap.training_progress + progress_delta)
            cap.last_trained = datetime.now().isoformat()

            # Check if ready to activate
            if cap.training_progress >= 1.0 and cap.status == CapabilityStatus.TRAINING:
                # Check dependencies
                deps_met = all(
                    self.aura_capabilities.get(
                        d, AuraCapability("", "", "", CapabilityStatus.LOCKED, 0)
                    ).status
                    == CapabilityStatus.ACTIVE
                    for d in cap.dependencies
                )
                if deps_met:
                    cap.status = CapabilityStatus.ACTIVE
                    logger.info(f"Capability activated: {capability_id}")


# Global instance
_sheet_system: Optional[CharacterSheetSystem] = None


def get_character_sheet_system() -> CharacterSheetSystem:
    """Get or create the character sheet system"""
    global _sheet_system
    if _sheet_system is None:
        _sheet_system = CharacterSheetSystem()
    return _sheet_system
