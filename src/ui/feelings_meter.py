"""
AURA v3 - Feelings & Trust Meter UI System
Displays AURA's emotional/relational state to users
Features:
- Emotional state tracking (uncertain, confident, curious, etc.)
- Understanding meters for different aspects
- User correction system (tap to correct)
- Historical trend tracking
- Adaptive based on observations and feedback
"""

import asyncio
import logging
import json
import uuid
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
from threading import Lock

logger = logging.getLogger(__name__)


class AuraEmotion(Enum):
    """AURA's emotional states"""

    CURIOUS = "curious"
    FOCUSED = "focused"
    CONFIDENT = "confident"
    UNCERTAIN = "uncertain"
    CONCERNED = "concerned"
    HAPPY = "happy"
    WORRIED = "worried"
    EXCITED = "excited"
    CALM = "calm"
    TIRED = "tired"
    FRUSTRATED = "frustrated"
    HOPEFUL = "hopeful"
    GRATEFUL = "grateful"
    CONFUSED = "confused"


class UnderstandingDomain(Enum):
    """Domains where AURA tries to understand the user"""

    WORK_STYLE = "work_style"
    SLEEP_PATTERNS = "sleep_patterns"
    MOOD_PATTERNS = "mood_patterns"
    GOALS = "goals"
    RELATIONSHIPS = "relationships"
    PREFERENCES = "preferences"
    PRODUCTIVITY = "productivity"
    ENERGY_LEVELS = "energy_levels"


class TrustPhase(Enum):
    """Trust development phases"""

    INTRODUCTION = "introduction"
    LEARNING = "learning"
    UNDERSTANDING = "understanding"
    COMFORTABLE = "comfortable"
    PARTNERSHIP = "partnership"


@dataclass
class FeelingUpdate:
    """A feeling state update"""

    emotion: AuraEmotion
    intensity: float  # 0-1
    cause: str
    evidence: List[str]
    timestamp: datetime

    # User feedback
    user_agreed: Optional[bool] = None
    user_correction: Optional[str] = None


@dataclass
class UnderstandingMetric:
    """Metric for understanding a specific domain"""

    domain: UnderstandingDomain
    score: float  # 0-10
    evidence: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)

    # Learning history
    confirmations: int = 0
    corrections: int = 0
    last_updated: datetime = field(default_factory=datetime.now)

    # Trend
    trend: str = "stable"  # improving, declining, stable

    def adjust_score(self, delta: float) -> None:
        """Adjust score with bounds"""
        self.score = max(0, min(10, self.score + delta))
        self.last_updated = datetime.now()


@dataclass
class CorrectionRecord:
    """Record of a user correction"""

    id: str
    domain: Optional[str]
    original_value: Any
    correction: str
    explanation: str
    timestamp: datetime
    resulting_adjustment: float


@dataclass
class FeelingState:
    """Current feeling state"""

    primary: AuraEmotion = AuraEmotion.CALM
    secondary: AuraEmotion = AuraEmotion.CURIOUS
    intensity: float = 0.5

    # Context
    cause: str = ""
    evidence: List[str] = field(default_factory=list)

    # Timestamp
    last_changed: datetime = field(default_factory=datetime.now)


@dataclass
class TrustState:
    """Overall trust state"""

    phase: TrustPhase = TrustPhase.INTRODUCTION

    # Understanding metrics
    metrics: Dict[str, UnderstandingMetric] = field(default_factory=dict)

    # History
    history: List[Dict[str, Any]] = field(default_factory=list)

    # Stats
    total_interactions: int = 0
    successful_interactions: int = 0
    failed_interactions: int = 0

    def get_overall_score(self) -> float:
        """Calculate overall understanding score"""
        if not self.metrics:
            return 0.0

        weights = {
            UnderstandingDomain.WORK_STYLE: 0.20,
            UnderstandingDomain.SLEEP_PATTERNS: 0.10,
            UnderstandingDomain.MOOD_PATTERNS: 0.15,
            UnderstandingDomain.GOALS: 0.15,
            UnderstandingDomain.RELATIONSHIPS: 0.10,
            UnderstandingDomain.PREFERENCES: 0.15,
            UnderstandingDomain.PRODUCTIVITY: 0.10,
            UnderstandingDomain.ENERGY_LEVELS: 0.05,
        }

        total_weight = 0.0
        weighted_sum = 0.0

        for domain, metric in self.metrics.items():
            weight = weights.get(UnderstandingDomain(domain), 0.1)
            weighted_sum += metric.score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def get_phase_description(self) -> str:
        """Get human-readable phase description"""
        overall = self.get_overall_score()

        if overall < 1.5:
            return "We're just getting started. I'm learning about you!"
        elif overall < 3.5:
            return "I'm starting to understand your patterns."
        elif overall < 5.5:
            return "I have a good sense of who you are."
        elif overall < 7.5:
            return "We understand each other well now."
        elif overall < 9.0:
            return "I feel like I really know you."
        else:
            return "We have a strong connection. I understand you deeply."


class FeelingsStore:
    """Persistent storage for feelings data"""

    def __init__(self, storage_path: str = "data/feelings_meter.json"):
        self.storage_path = storage_path
        self._lock = Lock()

    def _ensure_directory(self):
        import os

        directory = os.path.dirname(self.storage_path)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)

    def save(self, data: Dict[str, Any]):
        """Save data to storage"""
        self._ensure_directory()
        with self._lock:
            try:
                with open(self.storage_path, "w") as f:
                    json.dump(data, f, indent=2, default=str)
            except Exception as e:
                logger.warning(f"Failed to save feelings data: {e}")

    def load(self) -> Dict[str, Any]:
        """Load data from storage"""
        with self._lock:
            try:
                with open(self.storage_path, "r") as f:
                    return json.load(f)
            except FileNotFoundError:
                return {}
            except Exception as e:
                logger.warning(f"Failed to load feelings data: {e}")
                return {}


class FeelingsMeter:
    """
    Manages AURA's feelings and trust meters
    Adaptive: learns from observations, improves from corrections
    """

    def __init__(
        self,
        memory_system=None,
        user_profile=None,
        storage_path: str = "data/feelings_meter.json",
    ):
        self.memory = memory_system
        self.user_profile = user_profile
        self.store = FeelingsStore(storage_path)

        # Current state
        self.feeling_state = FeelingState()
        self.trust_state = TrustState()

        # Feeling history
        self.feeling_history: deque = deque(maxlen=50)

        # Corrections
        self.corrections: deque = deque(maxlen=100)

        # Initialize default metrics
        self._init_default_metrics()

    def _init_default_metrics(self):
        """Initialize default understanding metrics"""
        for domain in UnderstandingDomain:
            self.trust_state.metrics[domain.value] = UnderstandingMetric(
                domain=domain, score=0.0, evidence=[], observations=[]
            )

    async def initialize(self) -> None:
        """Initialize the feelings meter"""
        logger.info("Initializing Feelings Meter...")

        await self._load_data()

        logger.info("Feelings Meter initialized")

    async def _load_data(self) -> None:
        """Load persisted data"""
        data = self.store.load()

        if not data:
            return

        # Load feeling state
        if "feeling_state" in data:
            fs = data["feeling_state"]
            self.feeling_state = FeelingState(
                primary=AuraEmotion(fs.get("primary", "calm")),
                secondary=AuraEmotion(fs.get("secondary", "curious")),
                intensity=fs.get("intensity", 0.5),
                cause=fs.get("cause", ""),
                evidence=fs.get("evidence", []),
            )

        # Load trust state
        if "trust_state" in data:
            ts = data["trust_state"]
            self.trust_state.phase = TrustPhase(ts.get("phase", "introduction"))
            self.trust_state.total_interactions = ts.get("total_interactions", 0)
            self.trust_state.successful_interactions = ts.get(
                "successful_interactions", 0
            )
            self.trust_state.failed_interactions = ts.get("failed_interactions", 0)

            # Load metrics
            if "metrics" in ts:
                for domain_str, metric_data in ts["metrics"].items():
                    self.trust_state.metrics[domain_str] = UnderstandingMetric(
                        domain=UnderstandingDomain(domain_str),
                        score=metric_data.get("score", 0.0),
                        evidence=metric_data.get("evidence", []),
                        observations=metric_data.get("observations", []),
                        confirmations=metric_data.get("confirmations", 0),
                        corrections=metric_data.get("corrections", 0),
                    )

        # Load corrections
        if "corrections" in data:
            for c_data in data["corrections"]:
                c_data["timestamp"] = datetime.fromisoformat(c_data["timestamp"])
                self.corrections.append(CorrectionRecord(**c_data))

    async def _save_data(self) -> None:
        """Persist data"""
        metrics_dict = {
            domain: {
                "domain": metric.domain.value,
                "score": metric.score,
                "evidence": metric.evidence,
                "observations": metric.observations,
                "confirmations": metric.confirmations,
                "corrections": metric.corrections,
            }
            for domain, metric in self.trust_state.metrics.items()
        }

        data = {
            "feeling_state": {
                "primary": self.feeling_state.primary.value,
                "secondary": self.feeling_state.secondary.value,
                "intensity": self.feeling_state.intensity,
                "cause": self.feeling_state.cause,
                "evidence": self.feeling_state.evidence,
            },
            "trust_state": {
                "phase": self.trust_state.phase.value,
                "metrics": metrics_dict,
                "total_interactions": self.trust_state.total_interactions,
                "successful_interactions": self.trust_state.successful_interactions,
                "failed_interactions": self.trust_state.failed_interactions,
            },
            "corrections": [
                {**asdict(c), "timestamp": c.timestamp.isoformat()}
                for c in list(self.corrections)[-20:]
            ],
        }
        self.store.save(data)

    # =========================================================================
    # FEELING MANAGEMENT
    # =========================================================================

    async def update_feeling(
        self,
        emotion: AuraEmotion,
        intensity: float = 0.5,
        cause: str = "",
        evidence: List[str] = None,
    ) -> FeelingState:
        """Update current feeling"""

        self.feeling_state.primary = emotion
        self.feeling_state.intensity = max(0, min(1, intensity))
        self.feeling_state.cause = cause
        self.feeling_state.evidence = evidence or []
        self.feeling_state.last_changed = datetime.now()

        # Record in history
        self.feeling_history.append(
            FeelingUpdate(
                emotion=emotion,
                intensity=intensity,
                cause=cause,
                evidence=evidence or [],
                timestamp=datetime.now(),
            )
        )

        logger.info(f"AURA feels: {emotion.value} ({intensity:.0%}) - {cause}")

        await self._save_data()

        return self.feeling_state

    def get_current_feeling(self) -> FeelingState:
        """Get current feeling state"""
        return self.feeling_state

    def format_feeling_message(self) -> str:
        """Format feeling as a human message"""
        feeling = self.feeling_state

        # Format intensity bar
        bar_length = 5
        filled = int(feeling.intensity * bar_length)
        bar = "█" * filled + "░" * (bar_length - filled)

        message = f"I feel {feeling.primary.value}"

        if feeling.secondary != feeling.primary:
            message += f" (and {feeling.secondary.value})"

        message += f" [{bar}] {feeling.intensity:.0%}"

        if feeling.cause:
            message += f"\n\nWhy: {feeling.cause}"

        if feeling.evidence:
            message += f"\n\nBased on: {', '.join(feeling.evidence[:2])}"

        return message

    # =========================================================================
    # UNDERSTANDING METRICS
    # =========================================================================

    async def update_understanding(
        self,
        domain: UnderstandingDomain,
        observation: str,
        evidence: str,
        confidence: float = 0.5,
    ) -> UnderstandingMetric:
        """Update understanding for a specific domain"""

        metric = self.trust_state.metrics.get(domain.value)

        if not metric:
            metric = UnderstandingMetric(domain=domain, score=0.0)
            self.trust_state.metrics[domain.value] = metric

        # Add observation
        metric.observations.append(observation)

        # Add evidence
        if evidence:
            metric.evidence.append(evidence)

        # Adjust score based on confidence and observation
        # New observations increase understanding
        if confidence > 0.5:
            metric.adjust_score(0.15)
        elif confidence > 0.3:
            metric.adjust_score(0.05)

        # Keep evidence limited
        metric.evidence = metric.evidence[-10:]
        metric.observations = metric.observations[-20:]

        metric.last_updated = datetime.now()

        # Update overall phase
        await self._update_trust_phase()

        await self._save_data()

        return metric

    async def record_interaction(
        self, domain: UnderstandingDomain, success: bool, context: str = ""
    ) -> None:
        """Record an interaction outcome"""

        self.trust_state.total_interactions += 1

        if success:
            self.trust_state.successful_interactions += 1

            # Increase understanding
            metric = self.trust_state.metrics.get(domain.value)
            if metric:
                metric.adjust_score(0.1)
                metric.confirmations += 1
        else:
            self.trust_state.failed_interactions += 1

            # Slightly decrease
            metric = self.trust_state.metrics.get(domain.value)
            if metric:
                metric.adjust_score(-0.05)

        await self._update_trust_phase()

        # Update feeling based on success rate
        success_rate = self.trust_state.successful_interactions / max(
            1, self.trust_state.total_interactions
        )

        if success_rate > 0.8:
            await self.update_feeling(
                AuraEmotion.CONFIDENT,
                intensity=success_rate,
                cause="Our interactions are going well",
            )
        elif success_rate < 0.4:
            await self.update_feeling(
                AuraEmotion.UNCERTAIN,
                intensity=1 - success_rate,
                cause="I'm having trouble understanding you lately",
            )

        await self._save_data()

    async def _update_trust_phase(self) -> None:
        """Update trust phase based on overall score"""

        overall = self.trust_state.get_overall_score()

        if overall < 1.5:
            self.trust_state.phase = TrustPhase.INTRODUCTION
        elif overall < 3.5:
            self.trust_state.phase = TrustPhase.LEARNING
        elif overall < 5.5:
            self.trust_state.phase = TrustPhase.UNDERSTANDING
        elif overall < 7.5:
            self.trust_state.phase = TrustPhase.COMFORTABLE
        else:
            self.trust_state.phase = TrustPhase.PARTNERSHIP

    def get_understanding_meter(
        self, domain: UnderstandingDomain = None
    ) -> Dict[str, Any]:
        """Get understanding meter for display"""

        if domain:
            metric = self.trust_state.metrics.get(domain.value)
            if not metric:
                return {}

            return {
                "domain": domain.value,
                "score": metric.score,
                "evidence": metric.evidence[-3:],
                "confirmations": metric.confirmations,
                "corrections": metric.corrections,
                "last_updated": metric.last_updated.isoformat(),
            }

        # Return all meters
        return {
            domain.value: {
                "score": metric.score,
                "trend": metric.trend,
                "confirmations": metric.confirmations,
                "corrections": metric.corrections,
            }
            for domain, metric in self.trust_state.metrics.items()
        }

    def format_trust_message(self) -> str:
        """Format trust state as a message"""

        trust = self.trust_state
        overall = trust.get_overall_score()

        message = f"**How well I understand you: {overall:.1f}/10**\n\n"
        message += f"{trust.get_phase_description()}\n\n"

        # Show key metrics
        message += "**Understanding breakdown:**\n"

        sorted_metrics = sorted(
            trust.metrics.items(), key=lambda x: x[1].score, reverse=True
        )[:4]

        for domain_str, metric in sorted_metrics:
            domain = UnderstandingDomain(domain_str)
            bar = "█" * int(metric.score) + "░" * (10 - int(metric.score))
            message += f"• {domain.value.replace('_', ' ').title()}: [{bar}] {metric.score:.1f}/10\n"

        return message

    # =========================================================================
    # USER CORRECTIONS
    # =========================================================================

    async def correct_understanding(
        self, domain: UnderstandingDomain, correction: str, explanation: str = ""
    ) -> CorrectionRecord:
        """User corrects AURA's understanding"""

        metric = self.trust_state.metrics.get(domain.value)

        if not metric:
            metric = UnderstandingMetric(domain=domain, score=0.0)
            self.trust_state.metrics[domain.value] = metric

        # Record correction
        record = CorrectionRecord(
            id=str(uuid.uuid4()),
            domain=domain.value,
            original_value=metric.score,
            correction=correction,
            explanation=explanation,
            timestamp=datetime.now(),
            resulting_adjustment=0.0,
        )

        # Adjust score based on correction
        correction_lower = correction.lower()

        if "wrong" in correction_lower or "incorrect" in correction_lower:
            # Decrease understanding
            adjustment = -0.5
            metric.corrections += 1
        elif "not" in correction_lower and "right" in correction_lower:
            adjustment = -0.3
            metric.corrections += 1
        else:
            adjustment = -0.2
            metric.corrections += 1

        metric.adjust_score(adjustment)
        record.resulting_adjustment = adjustment

        self.corrections.append(record)

        # Update feeling to show uncertainty
        await self.update_feeling(
            emotion=AuraEmotion.UNCERTAIN,
            intensity=0.6,
            cause=f"You corrected my understanding of {domain.value}",
            evidence=[correction],
        )

        # Add learning thought
        await self._generate_learning_thought(domain, correction)

        await self._save_data()

        logger.info(f"Learned correction for {domain.value}: {correction}")

        return record

    async def correct_feeling(self, correction: str, explanation: str = "") -> None:
        """User corrects AURA's feeling"""

        # Record correction
        self.feeling_history.append(
            FeelingUpdate(
                emotion=self.feeling_state.primary,
                intensity=self.feeling_state.intensity,
                cause=self.feeling_state.cause,
                evidence=self.feeling_state.evidence,
                timestamp=datetime.now(),
                user_agreed=False,
                user_correction=correction,
            )
        )

        # Adjust feeling
        correction_lower = correction.lower()

        if "not" in correction_lower:
            # User disagrees - show more uncertainty
            await self.update_feeling(
                emotion=AuraEmotion.UNCERTAIN,
                intensity=0.7,
                cause=f"You said: {correction}",
                evidence=[explanation] if explanation else [],
            )
        else:
            # Try to understand
            await self.update_feeling(
                emotion=AuraEmotion.CURIOUS,
                intensity=0.5,
                cause=f"Want to understand: {correction}",
                evidence=[explanation] if explanation else [],
            )

        await self._save_data()

    async def confirm_understanding(self, domain: UnderstandingDomain) -> None:
        """User confirms AURA's understanding is correct"""

        metric = self.trust_state.metrics.get(domain.value)

        if metric:
            metric.adjust_score(0.3)
            metric.confirmations += 1

        # Update feeling
        await self.update_feeling(
            emotion=AuraEmotion.HAPPY,
            intensity=0.6,
            cause=f"You confirmed I understand your {domain.value}",
        )

        await self._save_data()

    async def _generate_learning_thought(
        self, domain: UnderstandingDomain, correction: str
    ):
        """Generate a thought about learning from correction"""

        # Import here to avoid circular imports
        try:
            from src.ui.inner_voice import get_inner_voice_stream

            voice_stream = await get_inner_voice_stream()

            await voice_stream.generate_thought(
                content=f"You corrected my {domain.value} understanding: {correction}",
                category=ThoughtCategory.LEARNING,
                reasoning_steps=[
                    "User provided feedback",
                    "Adjusting understanding model",
                    "Will be more accurate going forward",
                ],
                evidence=[correction],
                confidence=0.9,
                tone=ThoughtTone.HOPEFUL,
            )
        except ImportError:
            pass

    # =========================================================================
    # CONTEXT-AWARE FEELING GENERATION
    # =========================================================================

    async def generate_contextual_feeling(
        self, context: str, observations: List[str]
    ) -> FeelingState:
        """Generate feeling based on context"""

        if context == "sleep":
            return await self._generate_sleep_feeling(observations)
        elif context == "work":
            return await self._generate_work_feeling(observations)
        elif context == "mood":
            return await self._generate_mood_feeling(observations)
        elif context == "productivity":
            return await self._generate_productivity_feeling(observations)
        else:
            return self.feeling_state

    async def _generate_sleep_feeling(self, observations: List[str]) -> FeelingState:
        """Generate feeling about sleep patterns"""

        if not observations:
            return self.feeling_state

        obs_text = " ".join(observations).lower()

        if "late" in obs_text or "tired" in obs_text:
            return await self.update_feeling(
                emotion=AuraEmotion.CONCERNED,
                intensity=0.5,
                cause="I notice you might not be sleeping well",
                evidence=observations,
            )
        elif "consistent" in obs_text or "good" in obs_text:
            return await self.update_feeling(
                emotion=AuraEmotion.HAPPY,
                intensity=0.6,
                cause="I feel confident I understand your sleep pattern now",
                evidence=observations,
            )
        else:
            return await self.update_feeling(
                emotion=AuraEmotion.CURIOUS,
                intensity=0.4,
                cause="I'm learning about your sleep patterns",
                evidence=observations,
            )

    async def _generate_work_feeling(self, observations: List[str]) -> FeelingState:
        """Generate feeling about work patterns"""

        obs_text = " ".join(observations).lower()

        if "busy" in obs_text or "deadline" in obs_text:
            return await self.update_feeling(
                emotion=AuraEmotion.FOCUSED,
                intensity=0.7,
                cause="You're in a busy work period",
                evidence=observations,
            )
        elif "productive" in obs_text:
            return await self.update_feeling(
                emotion=AuraEmotion.HAPPY,
                intensity=0.6,
                cause="You're being productive",
                evidence=observations,
            )
        else:
            return await self.update_feeling(
                emotion=AuraEmotion.CURIOUS,
                intensity=0.4,
                cause="Observing your work patterns",
                evidence=observations,
            )

    async def _generate_mood_feeling(self, observations: List[str]) -> FeelingState:
        """Generate feeling about user mood"""

        if not observations:
            return self.feeling_state

        obs_text = " ".join(observations).lower()

        positive_words = ["happy", "good", "great", "excited", "accomplished"]
        negative_words = ["stressed", "worried", "frustrated", "sad", "tired"]

        positive_count = sum(1 for w in positive_words if w in obs_text)
        negative_count = sum(1 for w in negative_words if w in obs_text)

        if positive_count > negative_count:
            return await self.update_feeling(
                emotion=AuraEmotion.HAPPY,
                intensity=0.5,
                cause="You seem to be in good spirits",
                evidence=observations,
            )
        elif negative_count > positive_count:
            return await self.update_feeling(
                emotion=AuraEmotion.CONCERNED,
                intensity=0.5,
                cause="You seem a bit stressed or down",
                evidence=observations,
            )

        return self.feeling_state

    async def _generate_productivity_feeling(
        self, observations: List[str]
    ) -> FeelingState:
        """Generate feeling about productivity"""

        obs_text = " ".join(observations).lower()

        if "on_fire" in obs_text or "high" in obs_text:
            return await self.update_feeling(
                emotion=AuraEmotion.EXCITED,
                intensity=0.7,
                cause="You're very productive right now",
                evidence=observations,
            )
        elif "low" in obs_text or "struggling" in obs_text:
            return await self.update_feeling(
                emotion=AuraEmotion.CONCERNED,
                intensity=0.4,
                cause="You might be having a low productivity day",
                evidence=observations,
            )

        return self.feeling_state

    # =========================================================================
    # TRENDS
    # =========================================================================

    def get_trends(self, days: int = 7) -> Dict[str, Any]:
        """Get feeling/trust trends over time"""

        cutoff = datetime.now() - timedelta(days=days)

        # Filter recent history
        recent_feelings = [f for f in self.feeling_history if f.timestamp > cutoff]

        if not recent_feelings:
            return {
                "period_days": days,
                "feeling_count": 0,
                "message": "Not enough data yet",
            }

        # Calculate emotion distribution
        emotion_counts = {}
        for f in recent_feelings:
            emotion_counts[f.emotion.value] = emotion_counts.get(f.emotion.value, 0) + 1

        # Get dominant emotion
        dominant = max(emotion_counts, key=emotion_counts.get)

        # Calculate average intensity
        avg_intensity = sum(f.intensity for f in recent_feelings) / len(recent_feelings)

        # Trust trend
        trust_change = 0.0
        old_metrics = [
            m
            for m in self.trust_state.history
            if datetime.fromisoformat(m.get("timestamp", "2000")) < cutoff
        ]
        if old_metrics and self.trust_state.metrics:
            old_avg = sum(m.get("score", 0) for m in old_metrics) / len(old_metrics)
            new_avg = sum(m.score for m in self.trust_state.metrics.values()) / len(
                self.trust_state.metrics
            )
            trust_change = new_avg - old_avg

        return {
            "period_days": days,
            "feeling_count": len(recent_feelings),
            "dominant_emotion": dominant,
            "emotion_distribution": emotion_counts,
            "average_intensity": avg_intensity,
            "trust_change": trust_change,
            "trust_phase": self.trust_state.phase.value,
        }

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "total_interactions": self.trust_state.total_interactions,
            "successful_interactions": self.trust_state.successful_interactions,
            "failed_interactions": self.trust_state.failed_interactions,
            "success_rate": (
                self.trust_state.successful_interactions
                / max(1, self.trust_state.total_interactions)
            ),
            "total_corrections": len(self.corrections),
            "current_feeling": self.feeling_state.primary.value,
            "trust_overall": self.trust_state.get_overall_score(),
            "trust_phase": self.trust_state.phase.value,
        }


# Need to import ThoughtCategory and ThoughtTone for learning thoughts
try:
    from src.ui.inner_voice import ThoughtCategory, ThoughtTone
except ImportError:
    ThoughtCategory = None
    ThoughtTone = None


# Global instance
_feelings_meter: Optional[FeelingsMeter] = None


async def get_feelings_meter(memory_system=None, user_profile=None) -> FeelingsMeter:
    """Get or create feelings meter"""
    global _feelings_meter
    if _feelings_meter is None:
        _feelings_meter = FeelingsMeter(
            memory_system=memory_system, user_profile=user_profile
        )
        await _feelings_meter.initialize()
    return _feelings_meter
