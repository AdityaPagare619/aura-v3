"""
AURA v3 Relationship Tracker
Tracks relationships, important dates, and interaction history
100% offline - all data stored locally
"""

import asyncio
import logging
import json
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
import statistics

logger = logging.getLogger(__name__)


class RelationshipStrength(Enum):
    """Relationship strength levels"""

    ACQUAINTANCE = 1
    CASUAL = 2
    FRIEND = 3
    CLOSE = 4
    INTIMATE = 5


class ContactCategory(Enum):
    """Contact categories"""

    FAMILY = "family"
    FRIEND = "friend"
    WORK = "work"
    ACQUAINTANCE = "acquaintance"
    ROMANTIC = "romantic"
    OTHER = "other"


@dataclass
class Contact:
    """A tracked contact"""

    id: str
    name: str
    aliases: List[str] = field(default_factory=list)
    category: ContactCategory = ContactCategory.OTHER
    importance: float = 0.5
    trust_score: float = 0.5

    phone: Optional[str] = None
    email: Optional[str] = None
    platforms: List[str] = field(default_factory=list)

    notes: str = ""
    tags: Set[str] = field(default_factory=set)

    created_at: datetime = field(default_factory=datetime.now)
    last_interaction: Optional[datetime] = None


@dataclass
class Relationship:
    """Relationship with a contact"""

    contact: Contact
    strength: RelationshipStrength
    interaction_count: int = 0

    avg_message_length: float = 0.0
    response_time_avg: float = 0.0

    last_significant_interaction: Optional[datetime] = None
    first_interaction: Optional[datetime] = None

    important_dates: List[Dict] = field(default_factory=list)
    shared_interests: List[str] = field(default_factory=list)

    last_health_check: Optional[datetime] = None
    health_status: str = "healthy"


@dataclass
class Interaction:
    """A single interaction"""

    id: str
    contact_name: str
    timestamp: datetime

    interaction_type: str
    platform: str

    content_preview: str = ""
    sentiment: float = 0.5

    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)


class RelationshipTracker:
    """
    Tracks relationships and interactions
    Manages contact profiles and relationship health
    """

    def __init__(self, data_dir: str = "data/social_life/relationships"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self._contacts: Dict[str, Contact] = {}
        self._relationships: Dict[str, Relationship] = {}
        self._interactions: List[Interaction] = []

        self._max_interactions = 10000
        self._importance_threshold = 0.6
        self._reconnection_threshold_days = 30

    async def initialize(self):
        """Initialize relationship tracker"""
        logger.info("Initializing Relationship Tracker...")
        await self._load_contacts()
        await self._load_interactions()
        logger.info(
            f"Relationship Tracker initialized with {len(self._contacts)} contacts"
        )

    async def _load_contacts(self):
        """Load contacts from disk"""
        contacts_file = self.data_dir / "contacts.json"
        if contacts_file.exists():
            try:
                with open(contacts_file, "r") as f:
                    data = json.load(f)
                    for c_data in data.get("contacts", []):
                        c_data["created_at"] = datetime.fromisoformat(
                            c_data["created_at"]
                        )
                        if c_data.get("last_interaction"):
                            c_data["last_interaction"] = datetime.fromisoformat(
                                c_data["last_interaction"]
                            )
                        contact = Contact(**c_data)

                        strength = RelationshipStrength(c_data.get("strength", 2))
                        self._relationships[contact.id] = Relationship(
                            contact=contact,
                            strength=strength,
                            interaction_count=c_data.get("interaction_count", 0),
                        )
                        self._contacts[contact.id] = contact
                logger.info(f"Loaded {len(self._contacts)} contacts")
            except Exception as e:
                logger.error(f"Error loading contacts: {e}")

    async def _save_contacts(self):
        """Save contacts to disk"""
        contacts_file = self.data_dir / "contacts.json"
        try:
            data = {
                "contacts": [
                    {
                        **vars(c),
                        "created_at": c.created_at.isoformat(),
                        "last_interaction": c.last_interaction.isoformat()
                        if c.last_interaction
                        else None,
                        "strength": r.strength.value,
                        "interaction_count": r.interaction_count,
                    }
                    for idx, (c, r) in enumerate(
                        [
                            (
                                c,
                                self._relationships.get(
                                    c.id, Relationship(c, RelationshipStrength.CASUAL)
                                ),
                            )
                            for c in self._contacts.values()
                        ]
                    )
                ]
            }
            with open(contacts_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving contacts: {e}")

    async def _load_interactions(self):
        """Load interactions from disk"""
        interactions_file = self.data_dir / "interactions.json"
        if interactions_file.exists():
            try:
                with open(interactions_file, "r") as f:
                    data = json.load(f)
                    for i_data in data.get("interactions", []):
                        i_data["timestamp"] = datetime.fromisoformat(
                            i_data["timestamp"]
                        )
                        self._interactions.append(Interaction(**i_data))
                logger.info(f"Loaded {len(self._interactions)} interactions")
            except Exception as e:
                logger.error(f"Error loading interactions: {e}")

    async def _save_interactions(self):
        """Save interactions to disk"""
        interactions_file = self.data_dir / "interactions.json"
        try:
            data = {
                "interactions": [
                    {
                        **vars(i),
                        "timestamp": i.timestamp.isoformat(),
                    }
                    for i in self._interactions[-self._max_interactions :]
                ]
            }
            with open(interactions_file, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving interactions: {e}")

    async def add_contact(
        self,
        name: str,
        category: ContactCategory = ContactCategory.OTHER,
        platforms: List[str] = None,
        importance: float = 0.5,
        **kwargs,
    ) -> Contact:
        """Add a new contact"""
        contact_id = name.lower().replace(" ", "_")

        contact = Contact(
            id=contact_id,
            name=name,
            category=category,
            platforms=platforms or [],
            importance=importance,
            **kwargs,
        )

        self._contacts[contact_id] = contact

        strength = self._calculate_initial_strength(category)
        self._relationships[contact_id] = Relationship(
            contact=contact, strength=strength
        )

        await self._save_contacts()
        logger.info(f"Added contact: {name}")

        return contact

    def _calculate_initial_strength(
        self, category: ContactCategory
    ) -> RelationshipStrength:
        """Calculate initial relationship strength based on category"""
        strength_map = {
            ContactCategory.FAMILY: RelationshipStrength.CLOSE,
            ContactCategory.ROMANTIC: RelationshipStrength.INTIMATE,
            ContactCategory.FRIEND: RelationshipStrength.FRIEND,
            ContactCategory.WORK: RelationshipStrength.CASUAL,
            ContactCategory.ACQUAINTANCE: RelationshipStrength.ACQUAINTANCE,
            ContactCategory.OTHER: RelationshipStrength.ACQUAINTANCE,
        }
        return strength_map.get(category, RelationshipStrength.CASUAL)

    async def record_interaction(
        self,
        contact: str,
        message: str = "",
        platform: str = "unknown",
        interaction_type: str = "message",
        sentiment: float = 0.5,
        **metadata,
    ):
        """Record an interaction with a contact"""
        contact_id = contact.lower().replace(" ", "_")

        if contact_id not in self._contacts:
            await self.add_contact(contact, platforms=[platform])

        contact = self._contacts[contact_id]
        contact.last_interaction = datetime.now()

        interaction = Interaction(
            id=f"{contact_id}_{len(self._interactions)}",
            contact_name=contact.name,
            timestamp=datetime.now(),
            interaction_type=interaction_type,
            platform=platform,
            content_preview=message[:100] if message else "",
            sentiment=sentiment,
            metadata=metadata,
        )

        self._interactions.append(interaction)

        if len(self._interactions) > self._max_interactions:
            self._interactions = self._interactions[-self._max_interactions :]

        relationship = self._relationships.get(contact_id)
        if relationship:
            relationship.interaction_count += 1
            if not relationship.first_interaction:
                relationship.first_interaction = datetime.now()
            relationship.last_significant_interaction = datetime.now()

            self._update_relationship_health(relationship)

        await self._save_contacts()
        await self._save_interactions()

    def _update_relationship_health(self, relationship: Relationship):
        """Update relationship health based on recent interactions"""
        if not relationship.last_significant_interaction:
            relationship.health_status = "unknown"
            return

        days_since = (datetime.now() - relationship.last_significant_interaction).days

        if days_since < 7:
            relationship.health_status = "healthy"
        elif days_since < 14:
            relationship.health_status = "good"
        elif days_since < 30:
            relationship.health_status = "needs_attention"
        elif days_since < 60:
            relationship.health_status = "neglected"
        else:
            relationship.health_status = "at_risk"

    async def get_relationship(self, contact_name: str) -> Optional[Relationship]:
        """Get relationship for a contact"""
        contact_id = contact_name.lower().replace(" ", "_")
        return self._relationships.get(contact_id)

    async def get_all_relationships(self) -> List[Relationship]:
        """Get all relationships"""
        return list(self._relationships.values())

    async def get_relationship_summary(self) -> Dict[str, Any]:
        """Get relationship summary"""
        total = len(self._relationships)

        strength_dist = {}
        category_dist = {}
        health_dist = {}

        for rel in self._relationships.values():
            strength_dist[rel.strength.name] = (
                strength_dist.get(rel.strength.name, 0) + 1
            )
            category_dist[rel.contact.category.value] = (
                category_dist.get(rel.contact.category.value, 0) + 1
            )
            health_dist[rel.health_status] = health_dist.get(rel.health_status, 0) + 1

        important = [
            {"name": r.contact.name, "importance": r.contact.importance}
            for r in sorted(
                self._relationships.values(),
                key=lambda x: x.contact.importance,
                reverse=True,
            )[:10]
        ]

        return {
            "total_contacts": total,
            "strength_distribution": strength_dist,
            "category_distribution": category_dist,
            "health_distribution": health_dist,
            "important_contacts": important,
        }

    async def get_recent_interactions(
        self, contact_name: str = None, limit: int = 10
    ) -> List[Dict]:
        """Get recent interactions"""
        interactions = self._interactions

        if contact_name:
            contact_id = contact_name.lower().replace(" ", "_")
            interactions = [i for i in interactions if i.contact_name == contact_id]

        recent = sorted(interactions, key=lambda x: x.timestamp, reverse=True)[:limit]

        return [
            {
                "contact": i.contact_name,
                "type": i.interaction_type,
                "platform": i.platform,
                "timestamp": i.timestamp.isoformat(),
                "preview": i.content_preview,
            }
            for i in recent
        ]

    async def add_important_date(
        self,
        contact_name: str,
        date: datetime,
        description: str,
        recurring: bool = False,
    ):
        """Add an important date for a contact"""
        contact_id = contact_name.lower().replace(" ", "_")

        if contact_id not in self._contacts:
            await self.add_contact(contact_name)

        relationship = self._relationships.get(contact_id)
        if relationship:
            relationship.important_dates.append(
                {
                    "date": date,
                    "description": description,
                    "recurring": recurring,
                }
            )
            await self._save_contacts()

    async def suggest_reconnections(self) -> List[Dict[str, Any]]:
        """Suggest contacts to reconnect with"""
        suggestions = []
        now = datetime.now()

        for relationship in self._relationships.values():
            if not relationship.contact.last_interaction:
                continue

            days_since = (now - relationship.contact.last_interaction).days

            if (
                days_since > self._reconnection_threshold_days
                and relationship.health_status
                in ["needs_attention", "neglected", "at_risk"]
            ):
                suggestions.append(
                    {
                        "contact": relationship.contact.name,
                        "days_since": days_since,
                        "health": relationship.health_status,
                        "importance": relationship.contact.importance,
                        "reason": self._get_reconnection_reason(relationship),
                    }
                )

        suggestions.sort(
            key=lambda x: (x["importance"], -x["days_since"]), reverse=True
        )
        return suggestions[:10]

    def _get_reconnection_reason(self, relationship: Relationship) -> str:
        """Get reason for reconnection suggestion"""
        if relationship.contact.category == ContactCategory.FAMILY:
            return "Family connection important to maintain"
        elif relationship.contact.category == ContactCategory.WORK:
            return "Work relationship needs maintenance"
        elif relationship.health_status == "at_risk":
            return "Relationship at risk of fading"
        else:
            return "Haven't connected in a while"

    async def get_contact_by_name(self, name: str) -> Optional[Contact]:
        """Get contact by name"""
        contact_id = name.lower().replace(" ", "_")
        return self._contacts.get(contact_id)

    async def update_contact_importance(self, contact_name: str, importance: float):
        """Update contact importance"""
        contact_id = contact_name.lower().replace(" ", "_")
        if contact_id in self._contacts:
            self._contacts[contact_id].importance = max(0.0, min(1.0, importance))
            await self._save_contacts()

    async def search_contacts(self, query: str) -> List[Contact]:
        """Search contacts by name or tags"""
        query_lower = query.lower()
        results = []

        for contact in self._contacts.values():
            if query_lower in contact.name.lower():
                results.append(contact)
            elif any(query_lower in alias.lower() for alias in contact.aliases):
                results.append(contact)
            elif any(query_lower in tag.lower() for tag in contact.tags):
                results.append(contact)

        return results


_relationship_tracker: Optional[RelationshipTracker] = None


def get_relationship_tracker() -> RelationshipTracker:
    """Get or create relationship tracker"""
    global _relationship_tracker
    if _relationship_tracker is None:
        _relationship_tracker = RelationshipTracker()
    return _relationship_tracker
