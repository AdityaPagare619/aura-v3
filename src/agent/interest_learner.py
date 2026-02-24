"""
AURA v3 Interest Learner
Discovers user interests from their data while respecting privacy
Supports multiple learning methods without accessing private content
"""

import asyncio
import logging
import json
import re
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from collections import defaultdict
import hashlib

logger = logging.getLogger(__name__)


class InterestCategory(Enum):
    """Categories of user interests"""

    DANCE = "dance"
    TECH = "tech"
    MUSIC = "music"
    SPORTS = "sports"
    GAMING = "gaming"
    READING = "reading"
    TRAVEL = "travel"
    FITNESS = "fitness"
    FOOD = "food"
    ART = "art"
    PHOTOGRAPHY = "photography"
    MOVIES = "movies"
    TV_SHOWS = "tv_shows"
    FASHION = "fashion"
    CARS = "cars"
    SCIENCE = "science"
    BUSINESS = "business"
    FINANCE = "finance"
    HEALTH = "health"
    SPIRITUALITY = "spirituality"
    NATURE = "nature"
    ANIMALS = "animals"
    NEWS = "news"
    POLITICS = "politics"
    HISTORY = "history"
    LANGUAGE = "language"
    CODING = "coding"
    DESIGN = "design"
    WRITING = "writing"
    PODCASTS = "podcasts"
    COMEDY = "comedy"

    @classmethod
    def get_all_categories(cls) -> List[str]:
        return [c.value for c in cls]


@dataclass
class Interest:
    """A discovered user interest with confidence score"""

    category: InterestCategory
    confidence: float  # 0.0 to 1.0
    evidence: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    first_detected: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    user_confirmed: Optional[bool] = (
        None  # None = not asked, True = confirmed, False = denied
    )

    def update_confidence(self, new_evidence: str, source: str):
        """Update confidence based on new evidence"""
        self.evidence.append(new_evidence)
        self.sources.append(source)
        self.last_updated = datetime.now()

        # Confidence grows with more evidence, capped at 0.95
        evidence_weight = min(len(self.evidence) * 0.1, 0.4)
        self.confidence = min(0.5 + evidence_weight, 0.95)


@dataclass
class InterestProfile:
    """Complete user interest profile"""

    interests: Dict[str, Interest] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    last_analyzed: Optional[datetime] = None
    analysis_sources: List[str] = field(default_factory=list)

    def add_interest(
        self, category: InterestCategory, confidence: float, evidence: str, source: str
    ):
        """Add or update an interest"""
        cat_name = category.value

        if cat_name in self.interests:
            self.interests[cat_name].update_confidence(evidence, source)
        else:
            self.interests[cat_name] = Interest(
                category=category,
                confidence=confidence,
                evidence=[evidence],
                sources=[source],
            )

    def get_top_interests(self, limit: int = 5) -> List[Interest]:
        """Get top interests by confidence"""
        sorted_interests = sorted(
            self.interests.values(), key=lambda x: x.confidence, reverse=True
        )
        return sorted_interests[:limit]

    def get_confirmed_interests(self) -> List[Interest]:
        """Get user-confirmed interests"""
        return [i for i in self.interests.values() if i.user_confirmed is True]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "interests": {
                k: {
                    "category": v.category.value,
                    "confidence": v.confidence,
                    "evidence": v.evidence,
                    "sources": v.sources,
                    "first_detected": v.first_detected.isoformat(),
                    "last_updated": v.last_updated.isoformat(),
                    "user_confirmed": v.user_confirmed,
                }
                for k, v in self.interests.items()
            },
            "created_at": self.created_at.isoformat(),
            "last_analyzed": self.last_analyzed.isoformat()
            if self.last_analyzed
            else None,
            "analysis_sources": self.analysis_sources,
        }


class InterestDetector:
    """
    Detects user interests from various data sources
    Respects privacy tiers - only analyzes public/sensitive data by default
    """

    KEYWORDS: Dict[InterestCategory, List[str]] = {
        InterestCategory.DANCE: [
            "dance",
            "dancing",
            "ballet",
            "hiphop",
            "salsa",
            "contemporary",
            "choreography",
            "step",
            "tap",
            "jazz",
            "ballroom",
        ],
        InterestCategory.TECH: [
            "tech",
            "technology",
            "gadget",
            "device",
            "ai",
            "software",
            "startup",
            "innovation",
            "robotics",
            "automation",
            "programming",
        ],
        InterestCategory.MUSIC: [
            "music",
            "song",
            "album",
            "concert",
            "singer",
            "band",
            "playlist",
            "spotify",
            "streaming",
            "instrument",
            "guitar",
            "piano",
            "drums",
        ],
        InterestCategory.SPORTS: [
            "sport",
            "sports",
            "football",
            "basketball",
            "tennis",
            "soccer",
            "baseball",
            "hockey",
            "volleyball",
            "cricket",
            "rugby",
        ],
        InterestCategory.GAMING: [
            "game",
            "gaming",
            "video game",
            "playstation",
            "xbox",
            "nintendo",
            "steam",
            "esports",
            "gamer",
            " twitch",
            "minecraft",
            "fortnite",
        ],
        InterestCategory.READING: [
            "book",
            "reading",
            "novel",
            "fiction",
            "non-fiction",
            "comic",
            "manga",
            "ebook",
            "kindle",
            "bibliophile",
            "literature",
        ],
        InterestCategory.TRAVEL: [
            "travel",
            "trip",
            "vacation",
            "destination",
            "flight",
            "hotel",
            "adventure",
            "explore",
            "backpacking",
            "tourism",
            "world",
        ],
        InterestCategory.FITNESS: [
            "fitness",
            "gym",
            "workout",
            "exercise",
            "training",
            "health",
            "muscle",
            "cardio",
            "yoga",
            "pilates",
            "crossfit",
        ],
        InterestCategory.FOOD: [
            "food",
            "cooking",
            "recipe",
            "restaurant",
            "chef",
            "cuisine",
            "baking",
            "vegetarian",
            "vegan",
            "foodie",
            "gourmet",
        ],
        InterestCategory.ART: [
            "art",
            "artist",
            "painting",
            "drawing",
            "sculpture",
            "gallery",
            "creative",
            "design",
            "illustration",
            "sketch",
            "canvas",
        ],
        InterestCategory.PHOTOGRAPHY: [
            "photo",
            "photography",
            "camera",
            "portrait",
            "landscape",
            "photographer",
            "editing",
            "lightroom",
            "canon",
            "nikon",
        ],
        InterestCategory.MOVIES: [
            "movie",
            "film",
            "cinema",
            "director",
            "actor",
            "actress",
            "hollywood",
            "netflix",
            "blockbuster",
            "indie",
        ],
        InterestCategory.TV_SHOWS: [
            "tv show",
            "series",
            "episode",
            "netflix",
            "hulu",
            "disney",
            "prime",
            "streaming",
            "season",
            "episode",
        ],
        InterestCategory.FASHION: [
            "fashion",
            "style",
            "clothing",
            "designer",
            "brand",
            "outfit",
            "trendy",
            "apparel",
            "shoes",
            "accessories",
        ],
        InterestCategory.CARS: [
            "car",
            "cars",
            "automotive",
            "vehicle",
            "driver",
            "racing",
            "motorcycle",
            "supercar",
            "luxury",
            "automobile",
        ],
        InterestCategory.SCIENCE: [
            "science",
            "scientific",
            "research",
            "physics",
            "chemistry",
            "biology",
            "astronomy",
            "space",
            "experiment",
            "discovery",
        ],
        InterestCategory.BUSINESS: [
            "business",
            "entrepreneur",
            "startup",
            "company",
            "management",
            "marketing",
            "sales",
            "strategy",
            "corporate",
        ],
        InterestCategory.FINANCE: [
            "finance",
            "investment",
            "stock",
            "trading",
            "crypto",
            "banking",
            "money",
            "wealth",
            "portfolio",
            "market",
            "bitcoin",
        ],
        InterestCategory.HEALTH: [
            "health",
            "medical",
            "wellness",
            "mental health",
            "nutrition",
            "healthy",
            "diet",
            "sleep",
            "stress",
            "mindfulness",
        ],
        InterestCategory.SPIRITUALITY: [
            "spirituality",
            "meditation",
            "yoga",
            "buddhism",
            "religion",
            "faith",
            "prayer",
            "mindfulness",
            "zen",
            "soul",
        ],
        InterestCategory.NATURE: [
            "nature",
            "outdoor",
            "hiking",
            "camping",
            "wildlife",
            "nature",
            "mountains",
            "beach",
            "forest",
            "environment",
            "eco",
        ],
        InterestCategory.ANIMALS: [
            "animal",
            "pet",
            "dog",
            "cat",
            "wildlife",
            "veterinary",
            "adoption",
            "puppy",
            "kitten",
            "zoo",
        ],
        InterestCategory.NEWS: [
            "news",
            "headline",
            "current events",
            "journalism",
            "media",
            "breaking",
            "report",
            "coverage",
        ],
        InterestCategory.POLITICS: [
            "politics",
            "political",
            "government",
            "policy",
            "election",
            "vote",
            "democrat",
            "republican",
            "campaign",
        ],
        InterestCategory.HISTORY: [
            "history",
            "historical",
            "war",
            "ancient",
            "century",
            "heritage",
            "museum",
            "archaeology",
            "past",
        ],
        InterestCategory.LANGUAGE: [
            "language",
            "learning",
            "spanish",
            "french",
            "german",
            "chinese",
            "japanese",
            "translation",
            "linguistics",
            "speak",
        ],
        InterestCategory.CODING: [
            "coding",
            "code",
            "programming",
            "developer",
            "software",
            "python",
            "javascript",
            "web development",
            "app",
            "software dev",
        ],
        InterestCategory.DESIGN: [
            "design",
            "designer",
            "graphic",
            "ui",
            "ux",
            "figma",
            "sketch",
            "creative",
            "brand",
            "visual",
        ],
        InterestCategory.WRITING: [
            "writing",
            "writer",
            "blog",
            "blogging",
            "content",
            "poetry",
            "journal",
            "author",
            "script",
            "story",
        ],
        InterestCategory.PODCASTS: [
            "podcast",
            "podcasting",
            "audio",
            "interview",
            "talk show",
            "episode",
            "spotify",
            "apple podcasts",
        ],
        InterestCategory.COMEDY: [
            "comedy",
            "funny",
            "standup",
            "humor",
            "joke",
            "meme",
            "satire",
            "improv",
            "comedian",
        ],
    }

    DIRECT_STATEMENTS = {
        "love": [
            "i love",
            "i'm into",
            "i'm really into",
            "i enjoy",
            "i'm passionate about",
        ],
        "hate": ["i hate", "i don't like", "i'm not into", "i don't enjoy"],
        "want": ["i want to", "i'm looking for", "i need to", "i've been wanting to"],
    }

    def __init__(self, data_dir: str = "data/interest_learner"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.profile = InterestProfile()
        self._load_profile()

        self._termux_bridge = None

    async def initialize(self):
        """Initialize the interest detector"""
        try:
            from src.addons.termux_bridge import TermuxBridge

            self._termux_bridge = TermuxBridge()
            await self._termux_bridge.check_availability()
        except Exception as e:
            logger.warning(f"Termux bridge not available: {e}")

        logger.info("Interest Detector initialized")

    def _load_profile(self):
        """Load saved interest profile"""
        profile_file = self.data_dir / "interest_profile.json"
        if profile_file.exists():
            try:
                with open(profile_file, "r") as f:
                    data = json.load(f)

                if "interests" in data:
                    for cat_name, interest_data in data["interests"].items():
                        try:
                            category = InterestCategory(cat_name)
                            self.profile.interests[cat_name] = Interest(
                                category=category,
                                confidence=interest_data.get("confidence", 0.5),
                                evidence=interest_data.get("evidence", []),
                                sources=interest_data.get("sources", []),
                                first_detected=datetime.fromisoformat(
                                    interest_data.get(
                                        "first_detected", datetime.now().isoformat()
                                    )
                                ),
                                last_updated=datetime.fromisoformat(
                                    interest_data.get(
                                        "last_updated", datetime.now().isoformat()
                                    )
                                ),
                                user_confirmed=interest_data.get("user_confirmed"),
                            )
                        except ValueError:
                            pass
                    if "created_at" in data:
                        self.profile.created_at = datetime.fromisoformat(
                            data["created_at"]
                        )
                    if "last_analyzed" in data and data["last_analyzed"]:
                        self.profile.last_analyzed = datetime.fromisoformat(
                            data["last_analyzed"]
                        )
                    if "analysis_sources" in data:
                        self.profile.analysis_sources = data["analysis_sources"]

                    logger.info(
                        f"Loaded interest profile with {len(self.profile.interests)} interests"
                    )
            except Exception as e:
                logger.error(f"Error loading interest profile: {e}")

    def _save_profile(self):
        """Save interest profile"""
        profile_file = self.data_dir / "interest_profile.json"
        try:
            with open(profile_file, "w") as f:
                json.dump(self.profile.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving interest profile: {e}")

    def analyze_text(self, text: str, source: str = "text") -> Dict[str, float]:
        """
        Analyze text to find interest signals
        Returns dict of category -> confidence boost
        """
        text_lower = text.lower()
        signals = {}

        for category, keywords in self.KEYWORDS.items():
            matches = sum(1 for kw in keywords if kw in text_lower)
            if matches > 0:
                boost = min(matches * 0.15, 0.5)
                signals[category.value] = boost

        if signals:
            for cat_name, boost in signals.items():
                try:
                    category = InterestCategory(cat_name)
                    self.profile.add_interest(category, 0.5 + boost, text[:100], source)
                except ValueError:
                    pass

        return signals

    def detect_direct_statements(self, text: str) -> List[Tuple[str, str]]:
        """
        Detect direct user statements about interests
        Returns list of (statement_type, category)
        """
        text_lower = text.lower()
        statements = []

        for category, keywords in self.KEYWORDS.items():
            for kw in keywords:
                for stmt_type, patterns in self.DIRECT_STATEMENTS.items():
                    for pattern in patterns:
                        if f"{pattern} {kw}" in text_lower:
                            statements.append((stmt_type, category.value))

        return statements

    async def analyze_file_names(
        self, directory: str = "/storage/emulated/0", max_files: int = 500
    ) -> Dict[str, float]:
        """
        Analyze file names in a directory to find interests
        Only analyzes file names, NOT content (privacy-safe)
        """
        signals = {}

        if not self._termux_bridge:
            return signals

        try:
            result = await self._termux_bridge.run_command(
                ["find", directory, "-maxdepth", "3", "-type", "f", "-name", "*"]
            )

            if not result.success:
                return signals

            files = result.stdout.strip().split("\n")[:max_files]

            for file_path in files:
                if not file_path:
                    continue

                file_name = Path(file_path).name.lower()

                for category, keywords in self.KEYWORDS.items():
                    for kw in keywords:
                        if kw in file_name:
                            signals[category.value] = signals.get(category.value, 0) + 1

            for cat_name, count in signals.items():
                try:
                    category = InterestCategory(cat_name)
                    boost = min(count * 0.1, 0.6)
                    self.profile.add_interest(
                        category,
                        0.4 + boost,
                        f"Found in {count} file names",
                        "file_analysis",
                    )
                    self.profile.analysis_sources.append("file_analysis")
                except ValueError:
                    pass

        except Exception as e:
            logger.error(f"Error analyzing files: {e}")

        return signals

    async def analyze_app_usage(self) -> Dict[str, float]:
        """
        Analyze app usage patterns to infer interests
        Uses installed apps list (public tier)
        """
        signals = {}

        if not self._termux_bridge:
            return signals

        try:
            result = await self._termux_bridge.run_command(
                ["pm", "list", "packages", "-3"]
            )

            if not result.success:
                return signals

            packages = result.stdout.lower()

            app_keywords = {
                "music": [
                    "spotify",
                    "youtube music",
                    "apple music",
                    "soundcloud",
                    "tidal",
                    "deezer",
                ],
                "gaming": ["game", "steam", "epic", "nintendo", "playstation", "xbox"],
                "fitness": ["fit", "gym", "workout", "health", "yoga", "strava"],
                "reading": ["kindle", "book", "read", "novel", "audible", "scribd"],
                "news": ["news", "cnn", "bbc", "nytimes", "reuters", "flipboard"],
                "tech": ["tech", "developer", "stack overflow", "github"],
                "social": ["instagram", "tiktok", "twitter", "facebook", "snapchat"],
                "video": ["netflix", "hulu", "disney", "prime", "youtube", "twitch"],
            }

            for interest, keywords in app_keywords.items():
                for kw in keywords:
                    if kw in packages:
                        signals[interest] = signals.get(interest, 0) + 1

            category_mapping = {
                "music": InterestCategory.MUSIC,
                "gaming": InterestCategory.GAMING,
                "fitness": InterestCategory.FITNESS,
                "reading": InterestCategory.READING,
                "news": InterestCategory.NEWS,
                "tech": InterestCategory.TECH,
                "social": InterestCategory.MUSIC,
                "video": InterestCategory.MOVIES,
            }

            for interest, count in signals.items():
                if interest in category_mapping:
                    category = category_mapping[interest]
                    boost = min(count * 0.2, 0.5)
                    self.profile.add_interest(
                        category,
                        0.4 + boost,
                        f"Found {count} related apps",
                        "app_usage",
                    )
                    self.profile.analysis_sources.append("app_usage")

        except Exception as e:
            logger.error(f"Error analyzing apps: {e}")

        return signals

    async def analyze_web_history(self, max_entries: int = 100) -> Dict[str, float]:
        """
        Analyze web search history to find interests (if available)
        Requires browser history access - public tier only
        """
        signals = {}

        if not self._termux_bridge:
            return signals

        try:
            chrome_history = "/data/data/com.android.chrome/app_chromecache"

            result = await self._termux_bridge.run_command(
                ["find", "/data/data", "-name", "history", "-type", "f"]
            )

            if not result.success:
                return signals

            logger.info("Web history analysis not fully implemented - requires root")
        except Exception as e:
            logger.error(f"Error analyzing web history: {e}")

        return signals

    async def analyze_conversation_history(
        self, messages: List[str]
    ) -> Dict[str, float]:
        """
        Analyze conversation history for interest signals
        Processes stored conversations (privacy-respecting)
        """
        signals = {}

        for message in messages:
            message_signals = self.analyze_text(message, "conversation")
            for cat, boost in message_signals.items():
                signals[cat] = signals.get(cat, 0) + boost

        return signals

    async def run_full_analysis(self):
        """Run all available analysis methods"""
        logger.info("Running full interest analysis...")

        await self.analyze_file_names()
        await self.analyze_app_usage()

        self.profile.last_analyzed = datetime.now()
        self._save_profile()

        logger.info(
            f"Interest analysis complete. Found {len(self.profile.interests)} interests"
        )

    def get_interest_suggestions(self, category: InterestCategory) -> List[str]:
        """Get content suggestions based on interest"""
        suggestions = {
            InterestCategory.DANCE: [
                "Search for dance tutorials on YouTube",
                "Find local dance classes",
                "Check dance competitions and events",
            ],
            InterestCategory.TECH: [
                "Search for latest tech news",
                "Find trending GitHub repositories",
                "Check tech industry updates",
            ],
            InterestCategory.MUSIC: [
                "Search for new music releases",
                "Find concerts near you",
                "Check music streaming recommendations",
            ],
            InterestCategory.SPORTS: [
                "Search for sports news",
                "Find local sports events",
                "Check game schedules",
            ],
            InterestCategory.GAMING: [
                "Search for gaming news",
                "Find trending games",
                "Check gaming communities",
            ],
            InterestCategory.READING: [
                "Search for book recommendations",
                "Find book clubs",
                "Check bestseller lists",
            ],
            InterestCategory.TRAVEL: [
                "Search for travel destinations",
                "Find flight deals",
                "Check travel blogs",
            ],
            InterestCategory.FITNESS: [
                "Search for workout routines",
                "Find fitness tips",
                "Check health news",
            ],
            InterestCategory.FOOD: [
                "Search for recipes",
                "Find restaurant recommendations",
                "Check food blogs",
            ],
        }

        return suggestions.get(category, ["Explore related content"])

    def confirm_interest(self, category: InterestCategory, confirmed: bool):
        """Mark interest as confirmed or denied by user"""
        cat_name = category.value

        if cat_name in self.profile.interests:
            self.profile.interests[cat_name].user_confirmed = confirmed
            if confirmed:
                self.profile.interests[cat_name].confidence = min(
                    self.profile.interests[cat_name].confidence + 0.2, 1.0
                )
            self._save_profile()

    def add_manual_interest(self, category: InterestCategory):
        """Add interest manually (user explicitly sets)"""
        cat_name = category.value

        if cat_name in self.profile.interests:
            self.profile.interests[cat_name].user_confirmed = True
            self.profile.interests[cat_name].confidence = 1.0
        else:
            self.profile.add_interest(category, 1.0, "Manually added by user", "manual")

        self._save_profile()

    def remove_interest(self, category: InterestCategory):
        """Remove an interest"""
        cat_name = category.value

        if cat_name in self.profile.interests:
            del self.profile.interests[cat_name]
            self._save_profile()

    def get_interest_summary(self) -> Dict[str, Any]:
        """Get summary of all interests"""
        top = self.profile.get_top_interests(10)
        confirmed = self.profile.get_confirmed_interests()

        return {
            "total_interests": len(self.profile.interests),
            "top_interests": [
                {
                    "category": i.category.value,
                    "confidence": i.confidence,
                    "user_confirmed": i.user_confirmed,
                    "evidence_count": len(i.evidence),
                }
                for i in top
            ],
            "confirmed_interests": [i.category.value for i in confirmed],
            "last_analyzed": self.profile.last_analyzed.isoformat()
            if self.profile.last_analyzed
            else None,
            "sources": list(set(self.profile.analysis_sources)),
        }

    def format_interests_for_display(self) -> str:
        """Format interests for CLI display"""
        summary = self.get_interest_summary()

        if summary["total_interests"] == 0:
            return "No interests discovered yet. Keep using AURA and I'll learn your interests!"

        lines = ["📊 Your Discovered Interests", "=" * 30, ""]

        for interest in summary["top_interests"]:
            cat = interest["category"].upper()
            conf = int(interest["confidence"] * 100)
            confirmed = "✅" if interest["user_confirmed"] else "⏳"

            lines.append(f"{confirmed} {cat}: {conf}% confidence")
            lines.append(f"   Evidence: {interest['evidence_count']} signals")

        if summary["confirmed_interests"]:
            lines.append("")
            lines.append("Confirmed: " + ", ".join(summary["confirmed_interests"]))

        lines.append("")
        lines.append(f"Last analyzed: {summary['last_analyzed'] or 'Never'}")
        lines.append(
            f"Sources: {', '.join(summary['sources']) if summary['sources'] else 'None'}"
        )

        return "\n".join(lines)

    def get_proactive_suggestions(self) -> List[Dict[str, Any]]:
        """
        Get proactive suggestions based on discovered interests
        Returns list of actions AURA can take proactively
        """
        suggestions = []
        confirmed = self.profile.get_confirmed_interests()

        for interest in confirmed:
            category = interest.category

            if category == InterestCategory.DANCE:
                suggestions.append(
                    {
                        "type": "research",
                        "topic": "dance tutorials and classes",
                        "interest": "dance",
                        "priority": "medium",
                    }
                )
                suggestions.append(
                    {
                        "type": "notify",
                        "topic": "dance events near you",
                        "interest": "dance",
                        "priority": "low",
                    }
                )

            elif category == InterestCategory.TECH:
                suggestions.append(
                    {
                        "type": "research",
                        "topic": "latest tech news",
                        "interest": "tech",
                        "priority": "high",
                    }
                )
                suggestions.append(
                    {
                        "type": "research",
                        "topic": "trending GitHub repositories",
                        "interest": "tech",
                        "priority": "medium",
                    }
                )

            elif category == InterestCategory.MUSIC:
                suggestions.append(
                    {
                        "type": "research",
                        "topic": "new music releases",
                        "interest": "music",
                        "priority": "medium",
                    }
                )
                suggestions.append(
                    {
                        "type": "research",
                        "topic": "concerts near you",
                        "interest": "music",
                        "priority": "low",
                    }
                )

            elif category == InterestCategory.SPORTS:
                suggestions.append(
                    {
                        "type": "research",
                        "topic": "sports news and scores",
                        "interest": "sports",
                        "priority": "high",
                    }
                )

            elif category == InterestCategory.GAMING:
                suggestions.append(
                    {
                        "type": "research",
                        "topic": "gaming news and reviews",
                        "interest": "gaming",
                        "priority": "medium",
                    }
                )

            elif category == InterestCategory.READING:
                suggestions.append(
                    {
                        "type": "research",
                        "topic": "book recommendations",
                        "interest": "reading",
                        "priority": "medium",
                    }
                )

            elif category == InterestCategory.TRAVEL:
                suggestions.append(
                    {
                        "type": "research",
                        "topic": "travel destinations and deals",
                        "interest": "travel",
                        "priority": "high",
                    }
                )

            elif category == InterestCategory.FITNESS:
                suggestions.append(
                    {
                        "type": "research",
                        "topic": "workout routines and fitness tips",
                        "interest": "fitness",
                        "priority": "high",
                    }
                )

            elif category == InterestCategory.FOOD:
                suggestions.append(
                    {
                        "type": "research",
                        "topic": "recipes and restaurant recommendations",
                        "interest": "food",
                        "priority": "medium",
                    }
                )

            elif category == InterestCategory.MOVIES:
                suggestions.append(
                    {
                        "type": "research",
                        "topic": "movie reviews and new releases",
                        "interest": "movies",
                        "priority": "medium",
                    }
                )

            elif category == InterestCategory.NEWS:
                suggestions.append(
                    {
                        "type": "research",
                        "topic": "breaking news",
                        "interest": "news",
                        "priority": "high",
                    }
                )

        return suggestions

    async def process_user_message(self, message: str) -> Optional[str]:
        """
        Process a user message for interest learning
        Returns a response if interest is detected
        """
        statements = self.detect_direct_statements(message)

        if not statements:
            return None

        for stmt_type, category in statements:
            if stmt_type == "love":
                try:
                    cat = InterestCategory(category)
                    self.profile.add_interest(
                        cat, 0.8, f"User said they love {category}", "direct_statement"
                    )
                    self._save_profile()
                    return f"I notice you love {category}! I'll keep that in mind and can help you discover more about it."
                except ValueError:
                    pass
            elif stmt_type == "hate":
                try:
                    cat = InterestCategory(category)
                    if category in self.profile.interests:
                        self.confirm_interest(cat, False)
                    return f"Noted - you're not interested in {category}. I'll avoid suggesting related content."
                except ValueError:
                    pass

        return None


_interest_detector: Optional[InterestDetector] = None


def get_interest_detector() -> InterestDetector:
    """Get or create interest detector"""
    global _interest_detector
    if _interest_detector is None:
        _interest_detector = InterestDetector()
    return _interest_detector
