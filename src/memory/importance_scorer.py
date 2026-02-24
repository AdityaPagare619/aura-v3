"""
Importance Scorer - Amygdala Analogue
Determines memory importance based on emotional, outcome, novelty factors
"""

import time
import hashlib
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from collections import OrderedDict


@dataclass
class Experience:
    """Input experience for scoring"""

    content: str
    emotional_valence: float = 0.0  # -1 (negative) to 1 (positive)
    outcome: Optional[str] = None
    context: Dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class ImportanceScorer:
    """
    Determines memory importance (Amygdala function)

    Factors:
    - Emotional impact (30%)
    - Outcome significance (25%)
    - Novelty (20%)
    - Repetition suppression (10%)
    - User relevance (15%)
    """

    WEIGHTS = {
        "emotional": 0.30,
        "outcome": 0.25,
        "novelty": 0.20,
        "relevance": 0.15,
        "repetition": 0.10,
    }

    def __init__(self, max_seen_content: int = 5000):
        self._seen_content: OrderedDict[str, float] = OrderedDict()
        self._max_seen_content = max_seen_content
        self._context_history: List[Dict] = []
        self._max_history = 1000

    def score(self, experience: Experience) -> float:
        """Calculate importance score 0-1"""

        emotional = self._emotional_weight(experience.emotional_valence)
        outcome = self._outcome_weight(experience.outcome)
        novelty = self._novelty_weight(experience.content)
        relevance = self._relevance_weight(experience.context)
        repetition = self._repetition_weight(experience.content)

        importance = (
            emotional * self.WEIGHTS["emotional"]
            + outcome * self.WEIGHTS["outcome"]
            + novelty * self.WEIGHTS["novelty"]
            + relevance * self.WEIGHTS["relevance"]
            + repetition * self.WEIGHTS["repetition"]
        )

        return min(1.0, max(0.0, importance))

    def _emotional_weight(self, valence: float) -> float:
        """Weight based on emotional valence"""
        # Strong emotions (positive or negative) increase importance
        return abs(valence)

    def _outcome_weight(self, outcome: Optional[str]) -> float:
        """Weight based on outcome significance"""
        if not outcome:
            return 0.0

        outcome_lower = outcome.lower()

        # Significant outcomes
        significant = [
            "success",
            "failed",
            "error",
            "complete",
            "achieved",
            "solved",
            "important",
            "critical",
            "warning",
            "urgent",
        ]

        for word in significant:
            if word in outcome_lower:
                return 0.8

        return 0.3

    def _novelty_weight(self, content: str) -> float:
        """Weight based on novelty"""
        content_hash = hashlib.md5(content.encode()).hexdigest()

        if content_hash in self._seen_content:
            age = time.time() - self._seen_content[content_hash]
            # Decay novelty over time
            days_old = age / 86400
            return max(0, 0.5 - days_old * 0.1)

        # New content - high novelty
        return 0.7

    def _relevance_weight(self, context: Dict) -> float:
        """Weight based on user relevance"""
        if not context:
            return 0.3

        # Explicit relevance flag
        if context.get("explicitly_important"):
            return 1.0

        # User-requested information
        if context.get("user_asked"):
            return 0.9

        # Task-related
        if context.get("task_related"):
            return 0.7

        return 0.4

    def _repetition_weight(self, content: str) -> float:
        """Weight based on repetition (suppress repeated content)"""
        # Count occurrences in recent history
        content_lower = content.lower()

        recent_count = 0
        for past in self._context_history[-100:]:
            if past.get("content", "").lower() == content_lower:
                recent_count += 1

        if recent_count == 0:
            return 1.0  # First occurrence
        elif recent_count == 1:
            return 0.6  # Second occurrence
        elif recent_count == 2:
            return 0.3  # Third occurrence
        else:
            return 0.1  # Repeated many times

    def update_history(self, content: str, context: Dict = None):
        """Update history for novelty/repetition tracking"""
        content_hash = hashlib.md5(content.encode()).hexdigest()

        # LRU eviction if at capacity
        if content_hash not in self._seen_content:
            while len(self._seen_content) >= self._max_seen_content:
                self._seen_content.popitem(last=False)

        self._seen_content[content_hash] = time.time()
        self._seen_content.move_to_end(content_hash)

        self._context_history.append(
            {"content": content, "timestamp": time.time(), "context": context or {}}
        )

        # Trim history
        if len(self._context_history) > self._max_history:
            self._context_history = self._context_history[-self._max_history :]

    def score_with_context(
        self,
        content: str,
        context: Dict = None,
        emotional_valence: float = 0.0,
        outcome: Optional[str] = None,
    ) -> float:
        """Score with full context"""
        experience = Experience(
            content=content,
            emotional_valence=emotional_valence,
            outcome=outcome,
            context=context or {},
        )

        score = self.score(experience)
        self.update_history(content, context)

        return score

    def get_stats(self) -> Dict:
        """Get scorer statistics"""
        return {
            "tracked_content": len(self._seen_content),
            "history_size": len(self._context_history),
        }
