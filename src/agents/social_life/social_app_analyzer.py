"""
AURA v3 Social App Analyzer
Analyzes data from social apps (WhatsApp, Instagram, etc.)
100% offline - all processing is local
"""

import asyncio
import logging
import os
import json
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import hashlib
import statistics

logger = logging.getLogger(__name__)


class AppDataSource(Enum):
    """Supported social app data sources"""

    WHATSAPP = "whatsapp"
    INSTAGRAM = "instagram"
    LINKEDIN = "linkedin"
    SNAPCHAT = "snapchat"
    TELEGRAM = "telegram"
    TWITTER = "twitter"
    FACEBOOK = "facebook"
    MANUAL = "manual"


@dataclass
class MessagePattern:
    """Analysis of message patterns for a contact"""

    contact_name: str
    platform: str

    avg_response_time_minutes: float = 0.0
    message_count: int = 0
    avg_message_length: float = 0.0
    common_topics: List[str] = field(default_factory=list)
    communication_style: str = "unknown"

    last_message_time: Optional[datetime] = None
    first_message_time: Optional[datetime] = None

    sentiment_score: float = 0.5
    engagement_level: str = "unknown"

    response_times: List[float] = field(default_factory=list)
    message_timestamps: List[datetime] = field(default_factory=list)


@dataclass
class ResponseMetrics:
    """Response time metrics"""

    avg_response_minutes: float
    median_response_minutes: float
    min_response_minutes: float
    max_response_minutes: float
    std_deviation: float

    response_rate: float
    total_messages: int

    time_of_day_pattern: Dict[str, float] = field(default_factory=dict)


@dataclass
class PlatformAnalysis:
    """Analysis results for a single platform"""

    platform: str
    total_contacts: int
    total_messages: int
    avg_daily_messages: float
    most_active_hours: List[int] = field(default_factory=list)
    top_contacts: List[Dict] = field(default_factory=list)
    interest_distribution: Dict[str, float] = field(default_factory=dict)


class SocialAppAnalyzer:
    """
    Analyzes data from social apps
    Extracts patterns, metrics, and insights from message data
    Works offline - 100% private
    """

    def __init__(self, data_dir: str = "data/social_life/app_analysis"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._message_patterns: Dict[str, Dict[str, MessagePattern]] = {}
        self._raw_messages: Dict[str, List[Dict]] = {}

        self._analysis_cache: Dict[str, Any] = {}
        self._cache_ttl = 300

        self._max_messages_per_contact = 1000
        self._min_messages_for_analysis = 5

    async def initialize(self):
        """Initialize the analyzer"""
        logger.info("Initializing Social App Analyzer...")
        await self._load_cached_data()
        logger.info("Social App Analyzer initialized")

    async def _load_cached_data(self):
        """Load cached analysis data"""
        cache_file = self.data_dir / "message_patterns.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    data = json.load(f)
                    for contact_patterns in data.values():
                        for contact, pattern_data in contact_patterns.items():
                            if contact not in self._message_patterns:
                                self._message_patterns[contact] = {}
                            self._message_patterns[contact][
                                pattern_data["platform"]
                            ] = MessagePattern(
                                contact_name=pattern_data["contact_name"],
                                platform=pattern_data["platform"],
                                avg_response_time_minutes=pattern_data.get(
                                    "avg_response_time_minutes", 0.0
                                ),
                                message_count=pattern_data.get("message_count", 0),
                                avg_message_length=pattern_data.get(
                                    "avg_message_length", 0.0
                                ),
                                common_topics=pattern_data.get("common_topics", []),
                                communication_style=pattern_data.get(
                                    "communication_style", "unknown"
                                ),
                                sentiment_score=pattern_data.get(
                                    "sentiment_score", 0.5
                                ),
                                engagement_level=pattern_data.get(
                                    "engagement_level", "unknown"
                                ),
                            )
                logger.info(
                    f"Loaded patterns for {len(self._message_patterns)} contacts"
                )
            except Exception as e:
                logger.error(f"Error loading cached data: {e}")

    async def _save_cached_data(self):
        """Save analysis data to cache"""
        cache_file = self.data_dir / "message_patterns.json"
        try:
            data = {}
            for contact, patterns in self._message_patterns.items():
                data[contact] = {}
                for platform, pattern in patterns.items():
                    data[contact][platform] = {
                        "contact_name": pattern.contact_name,
                        "platform": pattern.platform,
                        "avg_response_time_minutes": pattern.avg_response_time_minutes,
                        "message_count": pattern.message_count,
                        "avg_message_length": pattern.avg_message_length,
                        "common_topics": pattern.common_topics,
                        "communication_style": pattern.communication_style,
                        "sentiment_score": pattern.sentiment_score,
                        "engagement_level": pattern.engagement_level,
                    }
            with open(cache_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cached data: {e}")

    async def add_message(
        self,
        contact_name: str,
        message: str,
        platform: str,
        timestamp: datetime,
        is_outgoing: bool,
    ):
        """Add a message for analysis"""
        platform_key = platform.lower()

        if platform_key not in self._raw_messages:
            self._raw_messages[platform_key] = []

        if contact_name not in self._raw_messages[platform_key]:
            self._raw_messages[platform_key][contact_name] = []

        message_data = {
            "contact": contact_name,
            "message": message,
            "timestamp": timestamp,
            "is_outgoing": is_outgoing,
            "platform": platform_key,
        }

        self._raw_messages[platform_key].append(message_data)

        if contact_name not in self._message_patterns:
            self._message_patterns[contact_name] = {}

        if platform_key not in self._message_patterns[contact_name]:
            self._message_patterns[contact_name][platform_key] = MessagePattern(
                contact_name=contact_name, platform=platform_key
            )

        pattern = self._message_patterns[contact_name][platform_key]
        pattern.message_count += 1
        pattern.message_timestamps.append(timestamp)

        if not pattern.first_message_time or timestamp < pattern.first_message_time:
            pattern.first_message_time = timestamp
        if not pattern.last_message_time or timestamp > pattern.last_message_time:
            pattern.last_message_time = timestamp

        words = message.split()
        current_avg = pattern.avg_message_length
        pattern.avg_message_length = (
            current_avg * (pattern.message_count - 1) + len(words)
        ) / pattern.message_count

        self._invalidate_cache(contact_name)

    async def analyze_message_pattern(
        self, contact_name: str, message: str, platform: str
    ) -> MessagePattern:
        """Analyze message pattern for a contact"""
        platform_key = platform.lower()

        if contact_name not in self._message_patterns:
            self._message_patterns[contact_name] = {}

        if platform_key not in self._message_patterns[contact_name]:
            self._message_patterns[contact_name][platform_key] = MessagePattern(
                contact_name=contact_name, platform=platform_key
            )

        pattern = self._message_patterns[contact_name][platform_key]

        pattern.common_topics = self._extract_topics(message, pattern.common_topics)

        await self._calculate_response_metrics(pattern)

        pattern.communication_style = self._determine_communication_style(pattern)

        pattern.engagement_level = self._calculate_engagement_level(pattern)

        await self._save_cached_data()

        return pattern

    def _extract_topics(self, message: str, existing_topics: List[str]) -> List[str]:
        """Extract topics from message"""
        topic_keywords = {
            "work": ["work", "job", "office", "meeting", "project", "client", "boss"],
            "family": ["family", "mom", "dad", "brother", "sister", "parents", "kids"],
            "friends": ["friend", "hangout", "party", "weekend", " outing"],
            "health": ["health", "gym", "exercise", "sick", "doctor", "fitness"],
            "travel": ["trip", "travel", "vacation", "flight", "hotel", "destination"],
            "food": ["food", "eat", "restaurant", "cook", "recipe", "dinner", "lunch"],
            "shopping": ["buy", "shop", "order", "purchase", "amazon", "sale"],
            "entertainment": ["movie", "music", "game", "show", "netflix", "series"],
            "news": ["news", "update", "happened", "heard", "read"],
            "plans": ["plan", "schedule", "tomorrow", "week", "upcoming"],
        }

        message_lower = message.lower()
        found_topics = []

        for topic, keywords in topic_keywords.items():
            if any(kw in message_lower for kw in keywords):
                found_topics.append(topic)

        combined_topics = list(set(existing_topics + found_topics))
        return combined_topics[:10]

    async def _calculate_response_metrics(self, pattern: MessagePattern):
        """Calculate response time metrics"""
        if len(pattern.message_timestamps) < 2:
            return

        timestamps = sorted(pattern.message_timestamps)
        response_times = []

        for i in range(1, len(timestamps)):
            diff = (timestamps[i] - timestamps[i - 1]).total_seconds() / 60
            if diff > 0 and diff < 1440:
                response_times.append(diff)

        if response_times:
            pattern.response_times = response_times
            pattern.avg_response_time_minutes = statistics.mean(response_times)

    def _determine_communication_style(self, pattern: MessagePattern) -> str:
        """Determine communication style based on patterns"""
        if pattern.avg_message_length < 5:
            return "brief"
        elif pattern.avg_message_length < 20:
            return "casual"
        elif pattern.avg_message_length < 50:
            return "detailed"
        else:
            return "elaborate"

    def _calculate_engagement_level(self, pattern: MessagePattern) -> str:
        """Calculate engagement level"""
        if not pattern.last_message_time:
            return "unknown"

        days_since = (datetime.now() - pattern.last_message_time).days

        if days_since > 30:
            return "low"
        elif days_since > 14:
            return "medium"
        elif days_since > 7:
            return "high"
        else:
            return "very_high"

    async def analyze_all_apps(self) -> Dict[str, PlatformAnalysis]:
        """Analyze all connected apps"""
        results = {}

        for platform, messages in self._raw_messages.items():
            if not messages:
                continue

            contact_stats = {}
            for msg in messages:
                contact = msg.get("contact", "unknown")
                if contact not in contact_stats:
                    contact_stats[contact] = {
                        "count": 0,
                        "timestamps": [],
                    }
                contact_stats[contact]["count"] += 1
                contact_stats[contact]["timestamps"].append(msg.get("timestamp"))

            total_messages = sum(s["count"] for s in contact_stats.values())

            timestamps = []
            for stats in contact_stats.values():
                timestamps.extend(stats["timestamps"])

            avg_daily = 0.0
            if timestamps:
                date_range = (max(timestamps) - min(timestamps)).days or 1
                avg_daily = total_messages / date_range

            most_active = self._get_most_active_hours(timestamps)

            top_contacts = sorted(
                [{"name": k, "messages": v["count"]} for k, v in contact_stats.items()],
                key=lambda x: x["messages"],
                reverse=True,
            )[:5]

            platform_analysis = PlatformAnalysis(
                platform=platform,
                total_contacts=len(contact_stats),
                total_messages=total_messages,
                avg_daily_messages=avg_daily,
                most_active_hours=most_active,
                top_contacts=top_contacts,
            )

            results[platform] = platform_analysis

        return results

    def _get_most_active_hours(self, timestamps: List[datetime]) -> List[int]:
        """Get most active hours"""
        hour_counts = {}
        for ts in timestamps:
            hour = ts.hour
            hour_counts[hour] = hour_counts.get(hour, 0) + 1

        sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
        return [h[0] for h in sorted_hours[:3]]

    async def get_contact_analysis(
        self, contact_name: str, platform: Optional[str] = None
    ) -> Dict[str, MessagePattern]:
        """Get analysis for a specific contact"""
        if contact_name not in self._message_patterns:
            return {}

        if platform:
            platform_key = platform.lower()
            if platform_key in self._message_patterns[contact_name]:
                return {
                    platform_key: self._message_patterns[contact_name][platform_key]
                }

        return self._message_patterns[contact_name]

    async def get_response_metrics(
        self, contact_name: str, platform: str
    ) -> Optional[ResponseMetrics]:
        """Get detailed response metrics for a contact"""
        patterns = await self.get_contact_analysis(contact_name, platform)
        pattern = patterns.get(platform.lower())

        if not pattern or not pattern.response_times:
            return None

        times = pattern.response_times

        return ResponseMetrics(
            avg_response_minutes=statistics.mean(times),
            median_response_minutes=statistics.median(times),
            min_response_minutes=min(times),
            max_response_minutes=max(times),
            std_deviation=statistics.stdev(times) if len(times) > 1 else 0,
            response_rate=1.0 if times else 0.0,
            total_messages=pattern.message_count,
            time_of_day_pattern={},
        )

    def _invalidate_cache(self, contact: str):
        """Invalidate cache for a contact"""
        cache_keys = [k for k in self._analysis_cache.keys() if contact in k]
        for key in cache_keys:
            del self._analysis_cache[key]

    async def get_social_activity_summary(self) -> Dict[str, Any]:
        """Get overall social activity summary"""
        total_contacts = len(self._message_patterns)
        total_messages = sum(
            p.message_count
            for patterns in self._message_patterns.values()
            for p in patterns.values()
        )

        recent_contacts = []
        for contact, patterns in self._message_patterns.items():
            latest = None
            for p in patterns.values():
                if p.last_message_time:
                    if not latest or p.last_message_time > latest:
                        latest = p.last_message_time
            if latest:
                recent_contacts.append({"contact": contact, "last_message": latest})

        recent_contacts.sort(key=lambda x: x["last_message"], reverse=True)

        return {
            "total_contacts": total_contacts,
            "total_messages": total_messages,
            "platforms_analyzed": list(self._raw_messages.keys()),
            "recent_contacts": recent_contacts[:10],
            "generated_at": datetime.now().isoformat(),
        }


_social_app_analyzer: Optional[SocialAppAnalyzer] = None


def get_social_app_analyzer() -> SocialAppAnalyzer:
    """Get or create social app analyzer"""
    global _social_app_analyzer
    if _social_app_analyzer is None:
        _social_app_analyzer = SocialAppAnalyzer()
    return _social_app_analyzer
