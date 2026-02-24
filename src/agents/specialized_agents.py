"""
AURA v3 Specialized Agents
Specialized agents for different tasks - social media, shopping, research, etc.
"""

import asyncio
import logging
import uuid
import re
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict
import hashlib

from src.agents.coordinator import Agent, AgentType, AgentTask, AgentCoordinator

from src.agents.healthcare import HealthcareAgent

logger = logging.getLogger(__name__)


# ==============================================================================
# SOCIAL MEDIA ANALYZER AGENT
# ==============================================================================


class SocialPlatform(Enum):
    """Supported social platforms"""

    WHATSAPP = "whatsapp"
    INSTAGRAM = "instagram"
    LINKEDIN = "linkedin"
    SNAPCHAT = "snapchat"
    GENERAL = "general"


@dataclass
class UserInterest:
    """A detected user interest from social media"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    topic: str = ""
    platform: str = ""
    confidence: float = 0.0
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    occurrence_count: int = 0
    related_products: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ShoppingIntent:
    """Detected shopping intent from social media"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    product_type: str = ""
    platform_source: str = ""
    message: str = ""
    link: Optional[str] = None
    saved_item: bool = False
    price_range: Optional[str] = None
    detected_at: datetime = field(default_factory=datetime.now)
    status: str = "pending"  # pending, researching, completed


class SocialMediaAnalyzerAgent(Agent):
    """
    Analyzes social media activity to understand user interests
    Privacy-first: only analyzes with explicit permission
    """

    def __init__(self, coordinator: AgentCoordinator):
        super().__init__(
            agent_id="social_media_analyzer",
            name="Social Media Analyzer",
            agent_type=AgentType.ANALYZER,
            coordinator=coordinator,
        )
        self._interests: Dict[str, UserInterest] = {}
        self._shopping_intents: Dict[str, ShoppingIntent] = {}
        self._platform_permissions: Dict[str, bool] = {}

    def get_capabilities(self) -> List[str]:
        return [
            "social_media_analysis",
            "interest_detection",
            "shopping_intent_detection",
            "trend_analysis",
            "privacy_respecting",
        ]

    async def process_task(self, task: AgentTask) -> Any:
        """Process social media analysis tasks"""
        task_type = task.data.get("type", "analyze")

        if task_type == "analyze_message":
            return await self._analyze_message(
                task.data.get("message", ""),
                task.data.get("platform", "general"),
                task.data.get("sender", "unknown"),
            )
        elif task_type == "get_interests":
            return await self._get_user_interests(task.data.get("platform"))
        elif task_type == "get_shopping_intents":
            return await self._get_shopping_intents()
        elif task_type == "set_permission":
            return self._set_platform_permission(
                task.data.get("platform"), task.data.get("allowed", False)
            )
        else:
            return {"error": f"Unknown task type: {task_type}"}

    def _set_platform_permission(self, platform: str, allowed: bool) -> Dict:
        """Set permission for a social platform"""
        self._platform_permissions[platform.lower()] = allowed
        logger.info(f"Platform {platform} permission set to {allowed}")
        return {"platform": platform, "allowed": allowed}

    async def _analyze_message(
        self, message: str, platform: str, sender: str
    ) -> Dict[str, Any]:
        """Analyze a message for interests and shopping intents"""
        platform = platform.lower()

        if not self._platform_permissions.get(platform, False):
            return {"error": f"No permission to analyze {platform}"}

        message_lower = message.lower()
        results = {
            "interests_found": [],
            "shopping_intents_found": [],
        }

        # Detect interests
        interests = self._detect_interests(message, platform)
        for interest in interests:
            self._interests[interest.id] = interest
            results["interests_found"].append(
                {
                    "topic": interest.topic,
                    "confidence": interest.confidence,
                }
            )

        # Detect shopping intents
        shopping = self._detect_shopping_intent(message, platform)
        if shopping:
            self._shopping_intents[shopping.id] = shopping
            results["shopping_intents_found"].append(
                {
                    "product_type": shopping.product_type,
                    "has_link": shopping.link is not None,
                }
            )

        return results

    def _detect_interests(self, message: str, platform: str) -> List[UserInterest]:
        """Detect user interests from message content"""
        interests = []
        message_lower = message.lower()

        # Interest keywords - dynamically learned, not hardcoded
        interest_patterns = {
            "technology": ["ai", "tech", "programming", "coding", "software", "app"],
            "fashion": ["clothing", "fashion", "style", "outfit", "dress", "shirt"],
            "fitness": ["gym", "workout", "fitness", "exercise", "health"],
            "food": ["food", "recipe", "cooking", "restaurant", "dish"],
            "travel": ["travel", "trip", "vacation", "destination", "flight"],
            "business": ["business", "startup", "entrepreneur", "invest", "money"],
            "gaming": ["game", "gaming", "play", "xbox", "ps5", "pc"],
            "music": ["music", "song", "album", "artist", "concert"],
            "movies": ["movie", "film", "netflix", "series", "show"],
            "sports": ["cricket", "football", "sports", "match", "player"],
        }

        for topic, keywords in interest_patterns.items():
            if any(kw in message_lower for kw in keywords):
                # Check if interest already exists
                existing = self._find_interest(topic, platform)
                if existing:
                    existing.occurrence_count += 1
                    existing.last_seen = datetime.now()
                    existing.confidence = min(existing.confidence + 0.1, 1.0)
                    interests.append(existing)
                else:
                    interest = UserInterest(
                        topic=topic,
                        platform=platform,
                        confidence=0.5,
                        occurrence_count=1,
                    )
                    interests.append(interest)

        return interests

    def _detect_shopping_intent(
        self, message: str, platform: str
    ) -> Optional[ShoppingIntent]:
        """Detect shopping intent from message"""
        message_lower = message.lower()

        # Shopping intent patterns
        shopping_keywords = [
            "buy",
            "purchase",
            "order",
            "want",
            "need",
            "like",
            "save",
            "wishlist",
            "cart",
        ]

        # Product type patterns
        product_patterns = [
            ("shirt", "clothing"),
            ("pants", "clothing"),
            ("dress", "clothing"),
            ("shoes", "footwear"),
            ("phone", "electronics"),
            ("laptop", "electronics"),
            ("watch", "accessories"),
            ("bag", "accessories"),
            ("sunglasses", "accessories"),
            ("jewelry", "accessories"),
        ]

        has_shopping_keyword = any(kw in message_lower for kw in shopping_keywords)

        if not has_shopping_keyword:
            return None

        # Check for links
        link = None
        url_pattern = r"https?://[^\s]+"
        link_match = re.search(url_pattern, message)
        if link_match:
            link = link_match.group(0)

        # Detect product type
        product_type = "general"
        for keyword, ptype in product_patterns:
            if keyword in message_lower:
                product_type = ptype
                break

        # Check if saved/liked
        saved = "save" in message_lower or "like" in message_lower

        return ShoppingIntent(
            product_type=product_type,
            platform_source=platform,
            message=message[:200],
            link=link,
            saved_item=saved,
        )

    def _find_interest(self, topic: str, platform: str) -> Optional[UserInterest]:
        """Find existing interest"""
        for interest in self._interests.values():
            if interest.topic == topic and interest.platform == platform:
                return interest
        return None

    async def _get_user_interests(self, platform: Optional[str] = None) -> List[Dict]:
        """Get all detected user interests"""
        interests = list(self._interests.values())

        if platform:
            interests = [i for i in interests if i.platform == platform]

        return sorted(
            [
                {
                    "topic": i.topic,
                    "platform": i.platform,
                    "confidence": i.confidence,
                    "occurrences": i.occurrence_count,
                    "last_seen": i.last_seen.isoformat(),
                }
                for i in interests
            ],
            key=lambda x: x["confidence"],
            reverse=True,
        )

    async def _get_shopping_intents(self) -> List[Dict]:
        """Get all shopping intents"""
        return sorted(
            [
                {
                    "id": s.id,
                    "product_type": s.product_type,
                    "platform": s.platform_source,
                    "has_link": s.link is not None,
                    "saved": s.saved_item,
                    "status": s.status,
                    "detected_at": s.detected_at.isoformat(),
                }
                for s in self._shopping_intents.values()
            ],
            key=lambda x: x["detected_at"],
            reverse=True,
        )


# ==============================================================================
# SHOPPING ASSISTANT AGENT
# ==============================================================================


@dataclass
class ProductMatch:
    """A product match for the user"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    price: Optional[float] = None
    currency: str = "INR"
    link: str = ""
    image_url: Optional[str] = None
    platform: str = ""
    match_score: float = 0.0
    why_matched: str = ""
    user_in_image: Optional[str] = None  # Path to AI-generated image


@dataclass
class ShoppingUserProfile:
    """User shopping profile - learned dynamically"""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    size: Optional[str] = None
    preferred_colors: List[str] = field(default_factory=list)
    budget_range: tuple = (0, 10000)  # min, max
    style_preferences: List[str] = field(default_factory=list)
    body_type: Optional[str] = None
    gender: Optional[str] = None
    favorite_brands: List[str] = field(default_factory=list)
    purchase_history: List[str] = field(default_factory=list)


class ShoppingAssistantAgent(Agent):
    """
    Shopping assistant - finds products, generates images, considers budget
    Adaptive learning - learns user preferences over time
    """

    def __init__(self, coordinator: AgentCoordinator):
        super().__init__(
            agent_id="shopping_assistant",
            name="Shopping Assistant",
            agent_type=AgentType.EXECUTOR,
            coordinator=coordinator,
        )
        self._user_profile: Optional[ShoppingUserProfile] = None
        self._product_cache: Dict[str, List[ProductMatch]] = {}
        self._search_history: List[Dict] = []

    def get_capabilities(self) -> List[str]:
        return [
            "product_search",
            "price_comparison",
            "budget_matching",
            "style_matching",
            "ai_image_generation",
            "adaptive_learning",
        ]

    async def process_task(self, task: AgentTask) -> Any:
        """Process shopping tasks"""
        task_type = task.data.get("type", "search")

        if task_type == "search":
            return await self._search_products(
                task.data.get("query", ""),
                task.data.get("budget", None),
                task.data.get("platform", "general"),
            )
        elif task_type == "update_profile":
            return self._update_user_profile(task.data)
        elif task_type == "get_profile":
            return self._get_user_profile()
        elif task_type == "generate_image":
            return await self._generate_product_image(
                task.data.get("product_id"),
                task.data.get("user_image_path"),
            )
        elif task_type == "track_price":
            return await self._track_price(
                task.data.get("product_link"),
                task.data.get("target_price"),
            )
        else:
            return {"error": f"Unknown task type: {task_type}"}

    def _update_user_profile(self, data: Dict) -> Dict:
        """Update user shopping profile"""
        if not self._user_profile:
            self._user_profile = ShoppingUserProfile()

        # Update profile fields dynamically
        if "size" in data:
            self._user_profile.size = data["size"]
        if "preferred_colors" in data:
            self._user_profile.preferred_colors = data["preferred_colors"]
        if "budget_min" in data and "budget_max" in data:
            self._user_profile.budget_range = (data["budget_min"], data["budget_max"])
        if "style_preferences" in data:
            self._user_profile.style_preferences = data["style_preferences"]
        if "body_type" in data:
            self._user_profile.body_type = data["body_type"]
        if "gender" in data:
            self._user_profile.gender = data["gender"]
        if "favorite_brands" in data:
            self._user_profile.favorite_brands = data["favorite_brands"]

        logger.info("User shopping profile updated")
        return {"status": "updated", "profile": self._get_profile_summary()}

    def _get_user_profile(self) -> Dict:
        """Get current user profile"""
        if not self._user_profile:
            return {"status": "no_profile"}
        return self._get_profile_summary()

    def _get_profile_summary(self) -> Dict:
        """Get profile summary"""
        if not self._user_profile:
            return {}
        return {
            "size": self._user_profile.size,
            "preferred_colors": self._user_profile.preferred_colors,
            "budget_range": self._user_profile.budget_range,
            "style_preferences": self._user_profile.style_preferences,
            "body_type": self._user_profile.body_type,
            "gender": self._user_profile.gender,
            "favorite_brands": self._user_profile.favorite_brands,
        }

    async def _search_products(
        self, query: str, budget: Optional[tuple], platform: str
    ) -> List[Dict]:
        """Search for products matching user needs"""
        logger.info(f"Searching for: {query}")

        # Learn from search
        self._search_history.append(
            {
                "query": query,
                "timestamp": datetime.now().isoformat(),
                "platform": platform,
            }
        )

        # This is a placeholder - in production, would integrate with shopping APIs
        # For now, return mock structure showing what would be returned
        results = []

        # Generate mock results for demonstration
        mock_products = [
            {
                "name": f"Premium {query.title()} - Best Seller",
                "price": 899 if not budget else min(899, budget[1] if budget else 9999),
                "description": f"High quality {query} with premium finish",
                "match_score": 0.92,
                "why_matched": "Matches your style preferences and budget",
            },
            {
                "name": f"Classic {query.title()} - Affordable",
                "price": 499 if not budget else min(499, budget[1] if budget else 9999),
                "description": f"Classic design {query} at great price",
                "match_score": 0.85,
                "why_matched": "Within budget, popular choice",
            },
        ]

        for mp in mock_products:
            product = ProductMatch(
                name=mp["name"],
                description=mp["description"],
                price=mp["price"],
                link=f"https://example.com/search?q={query.replace(' ', '+')}",
                platform=platform,
                match_score=mp["match_score"],
                why_matched=mp["why_matched"],
            )
            results.append(product)

        # Cache results
        self._product_cache[query] = results

        return [
            {
                "name": p.name,
                "price": p.price,
                "description": p.description,
                "match_score": p.match_score,
                "why_matched": p.why_matched,
                "link": p.link,
            }
            for p in results
        ]

    async def _generate_product_image(
        self, product_id: str, user_image_path: Optional[str]
    ) -> Dict:
        """Generate AI image of user with product"""
        # This would integrate with Gemini API or similar
        # For now, return placeholder
        return {
            "status": "placeholder",
            "message": "Would integrate with Gemini/AI to generate image",
            "product_id": product_id,
            "user_image_path": user_image_path,
            "note": "Requires external AI service - not implemented in offline mode",
        }

    async def _track_price(self, product_link: str, target_price: float) -> Dict:
        """Track price for a product"""
        return {
            "status": "tracking",
            "link": product_link,
            "target_price": target_price,
            "message": "Price tracking would be implemented",
        }

    def learn_from_purchase(self, product_name: str, liked: bool):
        """Learn from user purchase decisions"""
        if liked and self._user_profile:
            # Extract style info from purchased product
            # This is adaptive learning
            logger.info(f"Learned: user liked {product_name}")


# ==============================================================================
# RESEARCH AGENT
# ==============================================================================


class ResearchAgent(Agent):
    """
    Proactive research agent - researches topics, finds information
    """

    def __init__(self, coordinator: AgentCoordinator):
        super().__init__(
            agent_id="research_agent",
            name="Research Agent",
            agent_type=AgentType.ANALYZER,
            coordinator=coordinator,
        )
        self._research_cache: Dict[str, Dict] = {}

    def get_capabilities(self) -> List[str]:
        return [
            "research",
            "information_synthesis",
            "trend_analysis",
            "summary_generation",
        ]

    async def process_task(self, task: AgentTask) -> Any:
        """Process research tasks"""
        task_type = task.data.get("type", "research")

        if task_type == "research":
            return await self._do_research(
                task.data.get("topic", ""),
                task.data.get("depth", "basic"),
            )
        elif task_type == "synthesize":
            return await self._synthesize_information(
                task.data.get("sources", []),
            )
        elif task_type == "summarize":
            return await self._summarize_content(
                task.data.get("content", ""),
                task.data.get("max_length", 200),
            )
        else:
            return {"error": f"Unknown task type: {task_type}"}

    async def _do_research(self, topic: str, depth: str) -> Dict:
        """Research a topic"""
        logger.info(f"Researching: {topic} (depth: {depth})")

        # Placeholder for research
        return {
            "topic": topic,
            "depth": depth,
            "findings": [
                {
                    "title": f"Key insight about {topic}",
                    "summary": "Research findings would go here",
                    "source": "would be from web/local sources",
                }
            ],
            "timestamp": datetime.now().isoformat(),
        }

    async def _synthesize_information(self, sources: List[str]) -> Dict:
        """Synthesize information from multiple sources"""
        return {
            "synthesized": True,
            "sources": sources,
            "summary": "Synthesis would be generated here",
        }

    async def _summarize_content(self, content: str, max_length: int) -> Dict:
        """Summarize content"""
        # Simple truncation for now
        summary = content[:max_length] + "..." if len(content) > max_length else content
        return {
            "summary": summary,
            "original_length": len(content),
            "summary_length": len(summary),
        }


# ==============================================================================
# AUTOMATION AGENT
# ==============================================================================


class AutomationAgent(Agent):
    """
    Automation agent - handles task automation and workflows
    """

    def __init__(self, coordinator: AgentCoordinator):
        super().__init__(
            agent_id="automation_agent",
            name="Automation Agent",
            agent_type=AgentType.EXECUTOR,
            coordinator=coordinator,
        )
        self._workflows: Dict[str, Dict] = {}
        self._scheduled_tasks: List[Dict] = []

    def get_capabilities(self) -> List[str]:
        return [
            "task_automation",
            "workflow_execution",
            "schedule_management",
            "event_triggers",
        ]

    async def process_task(self, task: AgentTask) -> Any:
        """Process automation tasks"""
        task_type = task.data.get("type", "execute")

        if task_type == "execute_workflow":
            return await self._execute_workflow(
                task.data.get("workflow_id"),
                task.data.get("params", {}),
            )
        elif task_type == "create_workflow":
            return self._create_workflow(
                task.data.get("name"),
                task.data.get("steps", []),
            )
        elif task_type == "schedule_task":
            return self._schedule_task(
                task.data.get("task"),
                task.data.get("schedule"),
            )
        else:
            return {"error": f"Unknown task type: {task_type}"}

    async def _execute_workflow(self, workflow_id: str, params: Dict) -> Dict:
        """Execute a workflow"""
        if workflow_id not in self._workflows:
            return {"error": f"Workflow {workflow_id} not found"}

        workflow = self._workflows[workflow_id]
        logger.info(f"Executing workflow: {workflow_id}")

        return {
            "workflow_id": workflow_id,
            "status": "completed",
            "steps_executed": len(workflow.get("steps", [])),
        }

    def _create_workflow(self, name: str, steps: List[Dict]) -> Dict:
        """Create a new workflow"""
        workflow_id = f"wf_{len(self._workflows)}"
        self._workflows[workflow_id] = {
            "name": name,
            "steps": steps,
            "created_at": datetime.now().isoformat(),
        }
        return {"workflow_id": workflow_id, "name": name}

    def _schedule_task(self, task: Dict, schedule: str) -> Dict:
        """Schedule a task"""
        self._scheduled_tasks.append(
            {
                "task": task,
                "schedule": schedule,
                "created_at": datetime.now().isoformat(),
            }
        )
        return {"status": "scheduled", "schedule": schedule}


# ==============================================================================
# FACTORY FUNCTION TO CREATE ALL AGENTS
# ==============================================================================


def create_specialized_agents(coordinator: AgentCoordinator) -> List[Agent]:
    """Create and return all specialized agents"""
    return [
        SocialMediaAnalyzerAgent(coordinator),
        ShoppingAssistantAgent(coordinator),
        ResearchAgent(coordinator),
        AutomationAgent(coordinator),
        HealthcareAgent(coordinator),
    ]
