"""
AURA v3 Learning Engine
Advanced self-learning system that improves from interactions
100% offline - learns locally without external APIs

This is AURA's "learning brain" - learns from every interaction
"""

import asyncio
import logging
import json
import os
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import hashlib

logger = logging.getLogger(__name__)


class LearningType(Enum):
    """Types of learning AURA performs"""

    PREFERENCE = "preference"  # User preferences
    BEHAVIORAL = "behavioral"  # Behavior patterns
    CONTEXTUAL = "contextual"  # Context-dependent
    LINGUISTIC = "linguistic"  # Language patterns
    EMOTIONAL = "emotional"  # Emotional responses
    CORRECTION = "correction"  # Learning from mistakes
    SUCCESS = "success"  # Learning from successes


class FeedbackType(Enum):
    """Types of user feedback"""

    EXPLICIT = "explicit"  # Direct feedback
    IMPLICIT = "implicit"  # Inferred from behavior
    CORRECTION = "correction"  # User correction
    REWARD = "reward"  # Positive reinforcement
    PUNISHMENT = "punishment"  # Negative reinforcement


@dataclass
class LearningEntry:
    """Single learning entry"""

    id: str
    learning_type: LearningType
    content: Dict[str, Any]
    confidence: float = 0.0
    times_applied: int = 0
    times_succeeded: int = 0
    times_failed: int = 0
    first_learned: datetime = field(default_factory=datetime.now)
    last_used: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class InteractionRecord:
    """Record of an interaction for learning"""

    id: str
    timestamp: datetime
    input_text: str
    output_text: str
    context: Dict[str, Any]
    success: bool
    feedback: Optional[str] = None
    correction: Optional[str] = None
    user_satisfaction: Optional[float] = None  # 0-1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Pattern:
    """Detected pattern in user behavior"""

    id: str
    pattern_type: str
    trigger: str  # What triggers this pattern
    action: str  # What action is taken
    frequency: float  # How often this occurs
    success_rate: float  # How often it succeeds
    context_dependencies: Dict[str, Any] = field(default_factory=dict)
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)


class PreferenceLearner:
    """
    Learns user preferences from interactions
    Tracks preferences across multiple dimensions
    """

    def __init__(self, storage_path: str = "data/learning/preferences"):
        self.storage_path = storage_path
        self._preferences: Dict[str, Dict] = defaultdict(dict)
        os.makedirs(storage_path, exist_ok=True)
        self._load_preferences()

    def _load_preferences(self):
        """Load saved preferences"""
        path = os.path.join(self.storage_path, "preferences.json")
        if os.path.exists(path):
            with open(path) as f:
                self._preferences = defaultdict(dict, json.load(f))

    def _save_preferences(self):
        """Save preferences"""
        path = os.path.join(self.storage_path, "preferences.json")
        with open(path, "w") as f:
            json.dump(dict(self._preferences), f, indent=2)

    def learn_preference(
        self, category: str, key: str, value: Any, confidence: float = 1.0
    ):
        """Learn a preference"""
        if category not in self._preferences:
            self._preferences[category] = {}

        existing = self._preferences[category].get(key)

        if existing:
            # Blend with existing preference
            old_confidence = existing.get("confidence", 0.5)
            old_value = existing.get("value")

            # Weighted average for numeric values
            if isinstance(value, (int, float)) and isinstance(old_value, (int, float)):
                new_value = (old_value * old_confidence + value * confidence) / (
                    old_confidence + confidence
                )
            else:
                # For non-numeric, use higher confidence value
                new_value = value if confidence > old_confidence else old_value

            new_confidence = min(old_confidence + confidence * 0.1, 1.0)

            self._preferences[category][key] = {
                "value": new_value,
                "confidence": new_confidence,
                "updated": datetime.now().isoformat(),
            }
        else:
            self._preferences[category][key] = {
                "value": value,
                "confidence": confidence,
                "learned": datetime.now().isoformat(),
            }

        self._save_preferences()

    def get_preference(self, category: str, key: str, default: Any = None) -> Any:
        """Get a preference"""
        pref = self._preferences.get(category, {}).get(key)
        if pref:
            return pref.get("value")
        return default

    def get_all_preferences(self, category: str = None) -> Dict:
        """Get all preferences"""
        if category:
            return self._preferences.get(category, {})
        return dict(self._preferences)


class BehavioralLearner:
    """
    Learns behavioral patterns from user actions
    """

    def __init__(self, storage_path: str = "data/learning/behavior"):
        self.storage_path = storage_path
        self._patterns: Dict[str, Pattern] = {}
        self._action_history: deque = deque(maxlen=10000)
        os.makedirs(storage_path, exist_ok=True)
        self._load_patterns()

    def _load_patterns(self):
        """Load learned patterns"""
        path = os.path.join(self.storage_path, "patterns.json")
        if os.path.exists(path):
            data = json.load(f)
            for p_data in data.get("patterns", []):
                p_data["first_seen"] = datetime.fromisoformat(p_data["first_seen"])
                p_data["last_seen"] = datetime.fromisoformat(p_data["last_seen"])
                self._patterns[p_data["id"]] = Pattern(**p_data)

    def _save_patterns(self):
        """Save patterns"""
        path = os.path.join(self.storage_path, "patterns.json")
        data = {
            "patterns": [
                {
                    **vars(p),
                    "first_seen": p.first_seen.isoformat(),
                    "last_seen": p.last_seen.isoformat(),
                }
                for p in self._patterns.values()
            ]
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def record_action(
        self, trigger: str, action: str, context: Dict = None, success: bool = True
    ):
        """Record an action for pattern learning"""
        entry = {
            "timestamp": datetime.now(),
            "trigger": trigger,
            "action": action,
            "context": context or {},
            "success": success,
        }
        self._action_history.append(entry)

        # Update or create pattern
        pattern_key = f"{trigger}->{action}"

        if pattern_key in self._patterns:
            pattern = self._patterns[pattern_key]
            pattern.frequency += 1
            pattern.last_seen = datetime.now()
            if success:
                pattern.success_rate = (
                    pattern.success_rate * (pattern.frequency - 1) + 1
                ) / pattern.frequency
            else:
                pattern.success_rate = (
                    pattern.success_rate * (pattern.frequency - 1) / pattern.frequency
                )
        else:
            self._patterns[pattern_key] = Pattern(
                id=pattern_key,
                pattern_type="action",
                trigger=trigger,
                action=action,
                frequency=1.0,
                success_rate=1.0 if success else 0.0,
                context_dependencies=context or {},
            )

        self._save_patterns()

    def get_best_action(self, trigger: str, context: Dict = None) -> Optional[str]:
        """Get the best action for a trigger"""
        candidates = [p for p in self._patterns.values() if p.trigger == trigger]

        if not candidates:
            return None

        # Score by success rate and recency
        now = datetime.now()
        best = None
        best_score = -1

        for p in candidates:
            recency = 1.0 / (
                1.0 + (now - p.last_seen).total_seconds() / 86400
            )  # Decay over days
            score = p.success_rate * 0.7 + recency * 0.3

            if context:
                # Check context match
                context_match = sum(
                    1 for k, v in context.items() if p.context_dependencies.get(k) == v
                ) / len(context)
                score *= 0.5 + context_match * 0.5

            if score > best_score:
                best_score = score
                best = p.action

        return best

    def get_patterns(self, min_frequency: float = 0.1) -> List[Pattern]:
        """Get all patterns above frequency threshold"""
        return [p for p in self._patterns.values() if p.frequency >= min_frequency]


class CorrectionLearner:
    """
    Learns from user corrections
    """

    def __init__(self, storage_path: str = "data/learning/corrections"):
        self.storage_path = storage_path
        self._corrections: Dict[str, Dict] = {}
        self._correction_patterns: Dict[str, str] = {}  # wrong -> right
        os.makedirs(storage_path, exist_ok=True)
        self._load_corrections()

    def _load_corrections(self):
        """Load corrections"""
        path = os.path.join(self.storage_path, "corrections.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
                self._corrections = data.get("corrections", {})
                self._correction_patterns = data.get("patterns", {})

    def _save_corrections(self):
        """Save corrections"""
        path = os.path.join(self.storage_path, "corrections.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "corrections": self._corrections,
                    "patterns": self._correction_patterns,
                },
                f,
                indent=2,
            )

    def learn_correction(self, original: str, correction: str, context: Dict = None):
        """Learn from a correction"""
        # Store the correction
        correction_id = hashlib.md5(f"{original}:{correction}".encode()).hexdigest()[
            :12
        ]

        self._corrections[correction_id] = {
            "original": original,
            "correction": correction,
            "context": context or {},
            "count": self._corrections.get(correction_id, {}).get("count", 0) + 1,
            "last_corrected": datetime.now().isoformat(),
        }

        # Extract pattern (what was wrong -> what was right)
        self._correction_patterns[original] = correction

        self._save_corrections()
        logger.info(f"Learned correction: {original[:30]}... -> {correction[:30]}...")

    def should_correct(self, text: str) -> Optional[str]:
        """Check if text matches a known correction"""
        # Exact match
        if text in self._correction_patterns:
            return self._correction_patterns[text]

        # Partial match
        for wrong, right in self._correction_patterns.items():
            if wrong in text:
                return text.replace(wrong, right)

        return None

    def get_correction_count(self) -> int:
        """Get total corrections learned"""
        return len(self._corrections)


class SuccessLearner:
    """
    Learns from successful interactions
    """

    def __init__(self, storage_path: str = "data/learning/success"):
        self.storage_path = storage_path
        self._successes: List[Dict] = []
        self._success_patterns: Dict[str, float] = {}  # pattern -> success rate
        os.makedirs(storage_path, exist_ok=True)
        self._load_successes()

    def _load_successes(self):
        """Load successes"""
        path = os.path.join(self.storage_path, "successes.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
                self._successes = data.get("successes", [])
                self._success_patterns = data.get("patterns", {})

    def _save_successes(self):
        """Save successes"""
        path = os.path.join(self.storage_path, "successes.json")
        with open(path, "w") as f:
            json.dump(
                {
                    "successes": self._successes[-1000:],  # Keep last 1000
                    "patterns": self._success_patterns,
                },
                f,
                indent=2,
            )

    def record_success(
        self,
        input_text: str,
        output_text: str,
        context: Dict = None,
        satisfaction: float = None,
    ):
        """Record a successful interaction"""
        success = {
            "timestamp": datetime.now().isoformat(),
            "input": input_text,
            "output": output_text,
            "context": context or {},
            "satisfaction": satisfaction,
        }

        self._successes.append(success)

        # Extract input patterns
        words = input_text.lower().split()
        for word in words:
            if len(word) > 3:
                if word not in self._success_patterns:
                    self._success_patterns[word] = 0.0
                self._success_patterns[word] = (
                    self._success_patterns[word] * 0.95 + 0.05
                )

        self._save_successes()

    def record_failure(self, input_text: str, output_text: str, context: Dict = None):
        """Record a failed interaction"""
        # Extract input patterns and decrease their success rate
        words = input_text.lower().split()
        for word in words:
            if len(word) > 3:
                if word not in self._success_patterns:
                    self._success_patterns[word] = 0.5  # Default
                self._success_patterns[word] = self._success_patterns[word] * 0.95

    def get_successful_approaches(self, input_keywords: List[str]) -> List[str]:
        """Get approaches that worked for similar inputs"""
        keywords = set(k.lower() for k in input_keywords if len(k) > 3)

        scored = []
        for success in self._successes[-100:]:  # Recent successes
            success_words = set(
                w.lower() for w in success["input"].split() if len(w) > 3
            )
            overlap = keywords & success_words
            if overlap:
                scored.append((len(overlap), success["output"]))

        # Return most relevant outputs
        scored.sort(reverse=True)
        return [s[1] for s in scored[:5]]


class LearningEngine:
    """
    Main learning engine - coordinates all learning subsystems
    """

    def __init__(self, storage_path: str = "data/learning"):
        self.storage_path = storage_path
        self._running = False

        # Learning subsystems
        self.preference_learner = PreferenceLearner(
            os.path.join(storage_path, "preferences")
        )
        self.behavioral_learner = BehavioralLearner(
            os.path.join(storage_path, "behavior")
        )
        self.correction_learner = CorrectionLearner(
            os.path.join(storage_path, "corrections")
        )
        self.success_learner = SuccessLearner(os.path.join(storage_path, "success"))

        # Interaction history
        self._interaction_history: deque = deque(maxlen=5000)

        # Learning callbacks
        self._callbacks: List[Callable] = []

    async def start(self):
        """Start learning engine"""
        self._running = True
        logger.info("Learning engine started")

    async def stop(self):
        """Stop learning engine"""
        self._running = False
        logger.info("Learning engine stopped")

    async def record_interaction(
        self,
        input_text: str,
        output_text: str,
        context: Dict = None,
        success: bool = True,
        user_feedback: str = None,
        correction: str = None,
        satisfaction: float = None,
    ):
        """Record an interaction for learning"""
        interaction = InteractionRecord(
            id=hashlib.md5(
                f"{input_text}:{datetime.now().isoformat()}".encode()
            ).hexdigest()[:12],
            timestamp=datetime.now(),
            input_text=input_text,
            output_text=output_text,
            context=context or {},
            success=success,
            feedback=user_feedback,
            correction=correction,
            user_satisfaction=satisfaction,
        )

        self._interaction_history.append(interaction)

        # Learn based on outcome
        if correction:
            # User corrected us - learn from it
            self.correction_learner.learn_correction(output_text, correction, context)

        if success:
            # Success - learn what worked
            self.success_learner.record_success(
                input_text, output_text, context, satisfaction
            )
            self.behavioral_learner.record_action(
                input_text, output_text, context, True
            )
        else:
            # Failure - learn what didn't work
            self.success_learner.record_failure(input_text, output_text, context)
            self.behavioral_learner.record_action(
                input_text, output_text, context, False
            )

        # Notify callbacks
        for callback in self._callbacks:
            try:
                await callback(interaction)
            except Exception as e:
                logger.error(f"Learning callback error: {e}")

    def apply_corrections(self, text: str) -> str:
        """Apply learned corrections to text"""
        correction = self.correction_learner.should_correct(text)
        if correction:
            logger.info(f"Applied correction: {text[:30]}... -> {correction[:30]}...")
            return correction
        return text

    def get_best_action(self, trigger: str, context: Dict = None) -> Optional[str]:
        """Get best action for trigger"""
        return self.behavioral_learner.get_best_action(trigger, context)

    def get_successful_approaches(self, input_text: str) -> List[str]:
        """Get successful approaches for similar input"""
        keywords = input_text.split()
        return self.success_learner.get_successful_approaches(keywords)

    def get_preference(self, category: str, key: str, default: Any = None) -> Any:
        """Get learned preference"""
        return self.preference_learner.get_preference(category, key, default)

    def learn_preference(
        self, category: str, key: str, value: Any, confidence: float = 0.5
    ):
        """Learn a preference"""
        self.preference_learner.learn_preference(category, key, value, confidence)

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics"""
        return {
            "total_interactions": len(self._interaction_history),
            "successful_interactions": sum(
                1 for i in self._interaction_history if i.success
            ),
            "corrections_learned": self.correction_learner.get_correction_count(),
            "patterns_learned": len(self.behavioral_learner._patterns),
            "preferences": {
                category: len(prefs)
                for category, prefs in self.preference_learner.get_all_preferences().items()
            },
        }

    def subscribe(self, callback: Callable):
        """Subscribe to learning events"""
        self._callbacks.append(callback)


# Global instance
_learning_engine: Optional[LearningEngine] = None


def get_learning_engine() -> LearningEngine:
    """Get or create learning engine"""
    global _learning_engine
    if _learning_engine is None:
        _learning_engine = LearningEngine()
    return _learning_engine
