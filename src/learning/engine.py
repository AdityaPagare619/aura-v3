"""
AURA Learning Engine - Learns from interactions to improve responses
"""

import json
import asyncio
import hashlib
import re
from datetime import datetime, time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, field, asdict


@dataclass
class IntentPattern:
    user_pattern: str
    tool_name: str
    parameters_template: Dict[str, Any]
    success_count: int = 0
    failure_count: int = 0
    last_used: Optional[str] = None
    confidence: float = 0.0


@dataclass
class ContactRecord:
    name: str
    priority_scores: Dict[str, float] = field(default_factory=dict)
    successful_contacts: int = 0
    failed_contacts: int = 0
    preferred_times: Dict[str, int] = field(default_factory=dict)
    call_delay: int = 0
    last_contact: Optional[str] = None


@dataclass
class ToolStats:
    success_count: int = 0
    failure_count: int = 0
    avg_execution_time: float = 0.0
    total_executions: int = 0


class LearningEngine:
    def __init__(
        self,
        memory=None,
        patterns_path: str = "data/patterns",
        min_confidence: float = 0.6,
        max_patterns: int = 1000,
    ):
        self.memory = memory
        self.patterns_path = Path(patterns_path)
        self.patterns_path.mkdir(parents=True, exist_ok=True)
        self.min_confidence = min_confidence
        self.max_patterns = max_patterns

        self.intent_patterns: Dict[str, IntentPattern] = {}
        self.contact_records: Dict[str, ContactRecord] = {}
        self.tool_stats: Dict[str, ToolStats] = {}

        self.intent_keywords: Dict[str, List[str]] = defaultdict(list)
        self.action_history: List[Dict[str, Any]] = []
        self.max_history = 1000

        self._initialized = False
        self._running = False

    async def initialize(self):
        if not self._initialized:
            await asyncio.get_running_loop().run_in_executor(None, self.load_patterns)
            self._initialized = True

    async def start(self):
        """Start the learning engine."""
        await self.initialize()
        self._running = True

    async def stop(self):
        """Stop the learning engine and save patterns."""
        if self._running:
            await self.save_all()
            self._running = False

    async def get_pattern_count(self) -> int:
        """Get count of learned patterns"""
        return (
            len(self.intent_patterns) + len(self.contact_records) + len(self.tool_stats)
        )

    async def save_all(self):
        """Save all patterns to disk"""
        self.save_patterns()

    def _normalize_text(self, text: str) -> str:
        return re.sub(r"\s+", " ", text.lower().strip())

    def _extract_intent_signature(self, message: str) -> str:
        normalized = self._normalize_text(message)
        normalized = re.sub(r"\b\d+\b", "{number}", normalized)
        normalized = re.sub(r"\b[a-z]+@[a-z]+\.[a-z]+\b", "{email}", normalized)
        words = normalized.split()
        intent_words = [w for w in words if len(w) > 2][:10]
        return " ".join(intent_words)

    def _extract_entities(self, message: str) -> Dict[str, Any]:
        entities = {}

        phone_match = re.search(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", message)
        if phone_match:
            entities["phone"] = phone_match.group()

        email_match = re.search(
            r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\b", message
        )
        if email_match:
            entities["email"] = email_match.group()

        time_match = re.search(r"\b(\d{1,2}):?(\d{2})?\s*(am|pm)?\b", message, re.I)
        if time_match:
            entities["time"] = time_match.group()

        number_match = re.search(r"\b(\d+)\b", message)
        if number_match:
            entities["number"] = number_match.group(1)

        return entities

    def _normalize_contact_name(self, name: str) -> str:
        return self._normalize_text(name).title()

    async def record_tool_use(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        result: Any,
        success: bool,
        execution_time: float = 0.0,
    ):
        await self.initialize()

        if tool_name not in self.tool_stats:
            self.tool_stats[tool_name] = ToolStats()

        stats = self.tool_stats[tool_name]
        stats.total_executions += 1
        if success:
            stats.success_count += 1
        else:
            stats.failure_count += 1

        if execution_time > 0:
            total_time = (
                stats.avg_execution_time * (stats.total_executions - 1) + execution_time
            )
            stats.avg_execution_time = total_time / stats.total_executions

        record = {
            "timestamp": datetime.now().isoformat(),
            "tool": tool_name,
            "parameters": parameters,
            "success": success,
            "execution_time": execution_time,
        }

        self.action_history.append(record)
        if len(self.action_history) > self.max_history:
            self.action_history = self.action_history[-self.max_history :]

        if "contact" in parameters:
            contact = self._normalize_contact_name(str(parameters["contact"]))
            if contact not in self.contact_records:
                self.contact_records[contact] = ContactRecord(name=contact)

            contact_rec = self.contact_records[contact]
            if success:
                contact_rec.successful_contacts += 1
            else:
                contact_rec.failed_contacts += 1
            contact_rec.last_contact = datetime.now().isoformat()

            hour = datetime.now().hour
            time_key = f"{hour:02d}"
            contact_rec.preferred_times[time_key] = (
                contact_rec.preferred_times.get(time_key, 0) + 1
            )

        await asyncio.get_running_loop().run_in_executor(None, self.save_patterns)

    def learn_intent_pattern(
        self,
        user_message: str,
        tool_used: str,
        parameters: Dict[str, Any],
        success: bool,
    ):
        intent_sig = self._extract_intent_signature(user_message)
        pattern_key = hashlib.md5(intent_sig.encode()).hexdigest()[:12]

        if pattern_key not in self.intent_patterns:
            template = {}
            for key, value in parameters.items():
                if isinstance(value, str):
                    template[key] = f"{{extracted_{key}}}"
                else:
                    template[key] = value

            self.intent_patterns[pattern_key] = IntentPattern(
                user_pattern=intent_sig,
                tool_name=tool_used,
                parameters_template=parameters,
            )

            for word in intent_sig.split():
                if word not in self.intent_keywords[pattern_key]:
                    self.intent_keywords[pattern_key].append(word)

        pattern = self.intent_patterns[pattern_key]
        if success:
            pattern.success_count += 1
        else:
            pattern.failure_count += 1

        total = pattern.success_count + pattern.failure_count
        if total > 0:
            pattern.confidence = pattern.success_count / total

        pattern.last_used = datetime.now().isoformat()

    def get_recommended_action(self, user_message: str) -> Optional[Dict[str, Any]]:
        if not self.intent_patterns:
            return None

        message_words = set(self._normalize_text(user_message).split())

        best_match: Optional[Tuple[str, float]] = None

        for pattern_key, keywords in self.intent_keywords.items():
            common_words = len(set(keywords) & message_words)
            if common_words > 0:
                pattern = self.intent_patterns[pattern_key]
                if pattern.confidence < 0.3:
                    continue

                score = common_words * pattern.confidence

                if best_match is None or score > best_match[1]:
                    best_match = (pattern_key, score)

        if best_match is None:
            return None

        pattern = self.intent_patterns[best_match[0]]
        if pattern.confidence < 0.5:
            return None

        entities = self._extract_entities(user_message)
        params = {}

        for key, value in pattern.parameters_template.items():
            if isinstance(value, str) and value.startswith("{extracted_"):
                entity_key = key
                if entity_key in entities:
                    params[key] = entities[entity_key]
                else:
                    quoted = re.search(rf'["\']([^"\']+)["\']', user_message)
                    if quoted:
                        params[key] = quoted.group(1)
            elif (
                isinstance(value, str) and value.startswith("{") and value.endswith("}")
            ):
                params[key] = None
            else:
                params[key] = value

        return {
            "tool": pattern.tool_name,
            "parameters": params,
            "confidence": pattern.confidence,
            "pattern_key": best_match[0],
        }

    def learn_contact_priority(
        self,
        contact: str,
        context: str,
        success: bool,
        priority_adjustment: float = 0.1,
    ):
        contact = self._normalize_contact_name(contact)
        context = self._normalize_text(context)

        if contact not in self.contact_records:
            self.contact_records[contact] = ContactRecord(name=contact)

        record = self.contact_records[contact]

        if context not in record.priority_scores:
            record.priority_scores[context] = 0.5

        if success:
            record.priority_scores[context] = min(
                1.0, record.priority_scores[context] + priority_adjustment
            )
        else:
            record.priority_scores[context] = max(
                0.0, record.priority_scores[context] - priority_adjustment
            )

    def get_contact_priority(self, contact: str, context: str) -> float:
        contact = self._normalize_contact_name(contact)
        context = self._normalize_text(context)

        if contact not in self.contact_records:
            return 0.5

        record = self.contact_records[contact]

        if context in record.priority_scores:
            return record.priority_scores[context]

        total = record.successful_contacts + record.failed_contacts
        if total > 0:
            return record.successful_contacts / total

        return 0.5

    def get_prioritized_contacts(self, contacts: List[str], context: str) -> List[str]:
        scored = [
            (contact, self.get_contact_priority(contact, context))
            for contact in contacts
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in scored]

    def learn_call_delay(self, contact: str, successful_delay: int):
        contact = self._normalize_contact_name(contact)

        if contact not in self.contact_records:
            self.contact_records[contact] = ContactRecord(name=contact)

        current = self.contact_records[contact].call_delay
        if current == 0:
            self.contact_records[contact].call_delay = successful_delay
        else:
            alpha = 0.3
            self.contact_records[contact].call_delay = int(
                alpha * successful_delay + (1 - alpha) * current
            )

    def get_call_delay(self, contact: str) -> int:
        contact = self._normalize_contact_name(contact)

        if contact in self.contact_records:
            return self.contact_records[contact].call_delay
        return 0

    def get_best_contact_time(self, contact: str) -> Optional[int]:
        contact = self._normalize_contact_name(contact)

        if contact not in self.contact_records:
            return None

        times = self.contact_records[contact].preferred_times
        if not times:
            return None

        best_hour = max(times.items(), key=lambda x: x[1])
        return int(best_hour[0])

    def get_tool_success_rate(self, tool_name: str) -> float:
        if tool_name not in self.tool_stats:
            return 0.5

        stats = self.tool_stats[tool_name]
        total = stats.success_count + stats.failure_count

        if total == 0:
            return 0.5

        return stats.success_count / total

    def get_tool_stats(self, tool_name: str) -> Optional[Dict[str, Any]]:
        if tool_name not in self.tool_stats:
            return None

        stats = self.tool_stats[tool_name]
        return {
            "success_rate": self.get_tool_success_rate(tool_name),
            "total_executions": stats.total_executions,
            "avg_execution_time": stats.avg_execution_time,
        }

    def get_most_successful_tools(
        self, min_executions: int = 5
    ) -> List[Tuple[str, float]]:
        results = []
        for tool_name, stats in self.tool_stats.items():
            if stats.total_executions >= min_executions:
                rate = self.get_tool_success_rate(tool_name)
                results.append((tool_name, rate))

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def save_patterns(self):
        try:
            intent_file = self.patterns_path / "intent_patterns.json"
            intent_data = {
                key: asdict(pattern) for key, pattern in self.intent_patterns.items()
            }
            with open(intent_file, "w", encoding="utf-8") as f:
                json.dump(intent_data, f, indent=2, ensure_ascii=False)

            contacts_file = self.patterns_path / "contact_records.json"
            contacts_data = {
                name: asdict(record) for name, record in self.contact_records.items()
            }
            with open(contacts_file, "w", encoding="utf-8") as f:
                json.dump(contacts_data, f, indent=2, ensure_ascii=False)

            tools_file = self.patterns_path / "tool_stats.json"
            tools_data = {
                name: asdict(stats) for name, stats in self.tool_stats.items()
            }
            with open(tools_file, "w", encoding="utf-8") as f:
                json.dump(tools_data, f, indent=2, ensure_ascii=False)

            keywords_file = self.patterns_path / "intent_keywords.json"
            with open(keywords_file, "w", encoding="utf-8") as f:
                json.dump(dict(self.intent_keywords), f, indent=2)

            history_file = self.patterns_path / "action_history.json"
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(self.action_history[-500:], f, indent=2)

            self._save_failures()
            self._save_strategy_improvements()
            self._save_fallback_history()

        except Exception as e:
            pass

    def load_patterns(self):
        try:
            intent_file = self.patterns_path / "intent_patterns.json"
            if intent_file.exists():
                with open(intent_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.intent_patterns = {
                        key: IntentPattern(**pattern) for key, pattern in data.items()
                    }

            contacts_file = self.patterns_path / "contact_records.json"
            if contacts_file.exists():
                with open(contacts_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.contact_records = {
                        name: ContactRecord(**record) for name, record in data.items()
                    }

            tools_file = self.patterns_path / "tool_stats.json"
            if tools_file.exists():
                with open(tools_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.tool_stats = {
                        name: ToolStats(**stats) for name, stats in data.items()
                    }

            keywords_file = self.patterns_path / "intent_keywords.json"
            if keywords_file.exists():
                with open(keywords_file, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    self.intent_keywords = defaultdict(list, loaded)

            history_file = self.patterns_path / "action_history.json"
            if history_file.exists():
                with open(history_file, "r", encoding="utf-8") as f:
                    self.action_history = json.load(f)

            self.load_failure_data()

        except Exception as e:
            pass

    def get_learning_summary(self) -> Dict[str, Any]:
        return {
            "intent_patterns_learned": len(self.intent_patterns),
            "contacts_tracked": len(self.contact_records),
            "tools_tracked": len(self.tool_stats),
            "total_actions_recorded": len(self.action_history),
            "high_confidence_patterns": sum(
                1 for p in self.intent_patterns.values() if p.confidence > 0.7
            ),
            "top_tools": self.get_most_successful_tools()[:5],
        }

    def clear_old_patterns(self, days_old: int = 30):
        cutoff = datetime.now().timestamp() - (days_old * 24 * 60 * 60)

        to_remove = []
        for key, pattern in self.intent_patterns.items():
            if pattern.last_used:
                last = datetime.fromisoformat(pattern.last_used).timestamp()
                if last < cutoff and pattern.confidence < 0.5:
                    to_remove.append(key)

        for key in to_remove:
            del self.intent_patterns[key]
            if key in self.intent_keywords:
                del self.intent_keywords[key]

    def export_learned_data(self) -> Dict[str, Any]:
        return {
            "intent_patterns": {k: asdict(v) for k, v in self.intent_patterns.items()},
            "contact_records": {k: asdict(v) for k, v in self.contact_records.items()},
            "tool_stats": {k: asdict(v) for k, v in self.tool_stats.items()},
            "exported_at": datetime.now().isoformat(),
        }

    def import_learned_data(self, data: Dict[str, Any], merge: bool = True):
        if not merge:
            self.intent_patterns.clear()
            self.contact_records.clear()
            self.tool_stats.clear()

        for key, pattern in data.get("intent_patterns", {}).items():
            self.intent_patterns[key] = IntentPattern(**pattern)

        for name, record in data.get("contact_records", {}).items():
            self.contact_records[name] = ContactRecord(**record)

        for name, stats in data.get("tool_stats", {}).items():
            self.tool_stats[name] = ToolStats(**stats)

    def record_failure(self, action: str, error: str, context: Dict, reflection: str):
        """
        Record a failure for later analysis
        Used for self-improvement and strategy refinement
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "error": error,
            "context": context,
            "reflection": reflection,
        }

        if not hasattr(self, "failure_history"):
            self.failure_history = []

        self.failure_history.append(record)
        if len(self.failure_history) > self.max_history:
            self.failure_history = self.failure_history[-self.max_history :]

        self._save_failures()

    def get_failures(self, pattern_key: str) -> List[Dict]:
        """
        Get all failures for a pattern
        Used for analyzing failure patterns and improving strategies
        """
        if not hasattr(self, "failure_history"):
            return []

        # Match by pattern_key if stored in context
        return [f for f in self.failure_history if f.get("pattern_key") == pattern_key]

    def get_failures_by_action(self, action: str) -> List[Dict]:
        """Get all failures for a specific action"""
        if not hasattr(self, "failure_history"):
            return []
        return [f for f in self.failure_history if f.get("action") == action]

    def update_strategy(self, pattern_key: str, analysis: str):
        """
        Update approach based on failure analysis
        Used for self-referential improvement
        """
        if not hasattr(self, "strategy_improvements"):
            self.strategy_improvements = {}

        self.strategy_improvements[pattern_key] = {
            "analysis": analysis,
            "updated_at": datetime.now().isoformat(),
            "improvement_count": self.strategy_improvements.get(pattern_key, {}).get(
                "improvement_count", 0
            )
            + 1,
        }

        self._save_strategy_improvements()

    def get_strategy(self, pattern_key: str) -> Optional[Dict]:
        """Get current strategy for a pattern"""
        if not hasattr(self, "strategy_improvements"):
            return None
        return self.strategy_improvements.get(pattern_key)

    def record_fallback_success(self, original: str, fallback: str):
        """
        Record successful fallback
        Used to learn which fallbacks work best
        """
        if not hasattr(self, "fallback_history"):
            self.fallback_history = defaultdict(list)

        self.fallback_history[original].append(
            {
                "fallback": fallback,
                "timestamp": datetime.now().isoformat(),
                "success": True,
            }
        )

        # Keep only recent fallbacks
        if len(self.fallback_history[original]) > 50:
            self.fallback_history[original] = self.fallback_history[original][-50:]

        self._save_fallback_history()

    def get_best_fallback(self, original: str) -> Optional[str]:
        """Get the most successful fallback for a failed tool"""
        if (
            not hasattr(self, "fallback_history")
            or original not in self.fallback_history
        ):
            return None

        fallbacks = self.fallback_history[original]
        fallback_counts = {}
        for record in fallbacks:
            fb = record.get("fallback")
            fallback_counts[fb] = fallback_counts.get(fb, 0) + 1

        if fallback_counts:
            return max(fallback_counts.items(), key=lambda x: x[1])[0]
        return None

    def _save_failures(self):
        """Persist failure history to disk"""
        try:
            failures_file = self.patterns_path / "failure_history.json"
            with open(failures_file, "w", encoding="utf-8") as f:
                json.dump(
                    self.failure_history[-500:]
                    if hasattr(self, "failure_history")
                    else [],
                    f,
                    indent=2,
                )
        except Exception as e:
            pass

    def _save_strategy_improvements(self):
        """Persist strategy improvements to disk"""
        try:
            strategy_file = self.patterns_path / "strategy_improvements.json"
            with open(strategy_file, "w", encoding="utf-8") as f:
                json.dump(
                    self.strategy_improvements
                    if hasattr(self, "strategy_improvements")
                    else {},
                    f,
                    indent=2,
                )
        except Exception as e:
            pass

    def _save_fallback_history(self):
        """Persist fallback history to disk"""
        try:
            fallback_file = self.patterns_path / "fallback_history.json"
            with open(fallback_file, "w", encoding="utf-8") as f:
                json.dump(
                    dict(self.fallback_history)
                    if hasattr(self, "fallback_history")
                    else {},
                    f,
                    indent=2,
                )
        except Exception as e:
            pass

    def load_failure_data(self):
        """Load failure and strategy data from disk"""
        try:
            failures_file = self.patterns_path / "failure_history.json"
            if failures_file.exists():
                with open(failures_file, "r", encoding="utf-8") as f:
                    self.failure_history = json.load(f)

            strategy_file = self.patterns_path / "strategy_improvements.json"
            if strategy_file.exists():
                with open(strategy_file, "r", encoding="utf-8") as f:
                    self.strategy_improvements = json.load(f)

            fallback_file = self.patterns_path / "fallback_history.json"
            if fallback_file.exists():
                with open(fallback_file, "r", encoding="utf-8") as f:
                    self.fallback_history = defaultdict(list, json.load(f))
        except Exception as e:
            pass


# Singleton instance
_learning_engine: Optional[LearningEngine] = None


def get_learning_engine(
    memory=None,
    patterns_path: str = None,
    min_confidence: float = 0.6,
    max_patterns: int = 1000,
) -> LearningEngine:
    """Get or create the singleton LearningEngine instance.

    Args:
        memory: Optional memory system reference.
        patterns_path: Path for pattern storage. Defaults to data/patterns.
        min_confidence: Minimum confidence threshold for patterns.
        max_patterns: Maximum number of patterns to store.

    Returns:
        The singleton LearningEngine instance.
    """
    global _learning_engine

    if _learning_engine is None:
        if patterns_path is None:
            # Default to data/patterns relative to project root
            project_root = Path(__file__).parent.parent.parent
            patterns_path = str(project_root / "data" / "patterns")

        _learning_engine = LearningEngine(
            memory=memory,
            patterns_path=patterns_path,
            min_confidence=min_confidence,
            max_patterns=max_patterns,
        )

    return _learning_engine


def reset_learning_engine():
    """Reset the singleton (useful for testing)."""
    global _learning_engine
    _learning_engine = None
