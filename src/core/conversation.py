"""
AURA v3 Emotional Conversation Engine
=====================================

Natural, context-aware communication like JARVIS/FRIDAY:
- Not robotic, not overly casual
- Context-aware tone
- Proactive but not intrusive
- Emotional intelligence
- Memory of conversation style

This is NOT about chatbot personas - it's about HOW AURA communicates.
"""

import asyncio
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# LEARNING DATACLASSES
# =============================================================================


class LearningType(Enum):
    """Types of learnings extracted from conversations"""

    PREFERENCE = "preference"  # User likes/dislikes
    INTEREST = "interest"  # Topics they engage with
    STYLE = "style"  # Communication style
    FACT = "fact"  # Personal facts about user
    BEHAVIOR = "behavior"  # Behavioral patterns


@dataclass
class Learning:
    """A single learning extracted from conversation"""

    learning_type: LearningType
    key: str  # What was learned about
    value: Any  # The learned value
    confidence: float = 0.5  # How confident we are (0-1)
    source: str = ""  # What triggered this learning
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            "type": self.learning_type.value,
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ConversationAnalysis:
    """Result of analyzing a conversation for learnings"""

    preferences: List[Learning] = field(default_factory=list)
    interests: List[Learning] = field(default_factory=list)
    style: Dict[str, Any] = field(default_factory=dict)
    sentiment: str = "neutral"  # positive, negative, neutral
    engagement_level: float = 0.5  # 0-1


class Tone(Enum):
    """Communication tone levels"""

    FORMAL = "formal"  # Professional, precise
    CASUAL = "casual"  # Relaxed, friendly
    SUPPORTIVE = "supportive"  # Empathetic, encouraging
    URGENT = "urgent"  # Time-sensitive, direct
    PLAYFUL = "playful"  # Light, humor
    CONCERNED = "concerned"  # Worried, careful


class ConversationPhase(Enum):
    """Phase of conversation"""

    GREETING = "greeting"
    UNDERSTANDING = "understanding"
    PROCESSING = "processing"
    RESPONDING = "responding"
    CLOSING = "closing"
    FOLLOW_UP = "follow_up"


class RelationshipStage(Enum):
    """Relationship stage with user"""

    NEW = "new"  # < 5 interactions
    ACQUAINTANCE = "acquaintance"  # 5-20 interactions
    FAMILIAR = "familiar"  # 20-100 interactions
    ESTABLISHED = "established"  # 100+ interactions


@dataclass
class MessageContext:
    """Context for generating response"""

    # User info
    user_name: Optional[str] = None
    user_id: Optional[str] = None

    # Current conversation
    user_message: str = ""
    conversation_phase: ConversationPhase = ConversationPhase.UNDERSTANDING

    # Emotional state
    inferred_user_tone: Tone = Tone.CASUAL
    user_emotion: str = "neutral"  # happy, frustrated, excited, etc.

    # Situational
    is_first_message: bool = False
    time_since_last_message: float = 0  # minutes
    pending_tasks: List[str] = field(default_factory=list)

    # Time context
    current_hour: int = field(default_factory=lambda: datetime.now().hour)

    # What user wants
    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)

    # Memory
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict] = field(default_factory=list)

    # Relationship context
    interaction_count: int = 0
    last_topic: Optional[str] = None
    last_sentiment: str = "neutral"


@dataclass
class AURAResponse:
    """AURA's response with metadata"""

    text: str
    tone: Tone
    should_notify: bool = False
    notification_text: Optional[str] = None
    action_taken: Optional[str] = None
    follow_up_needed: bool = False
    follow_up_message: Optional[str] = None
    confidence: float = 1.0


class EmotionalConversationEngine:
    """
    Emotional Conversation Engine

    Generates natural, contextually appropriate responses like JARVIS/FRIDAY.

    Key principles:
    1. Match user's emotional state but don't mirror negatives
    2. Be helpful without being sycophantic
    3. Show initiative without being intrusive
    4. Remember conversation style preferences
    5. Use appropriate formality based on context
    """

    # Pattern matchers for preference extraction (compiled once)
    _PREFERENCE_PATTERNS = [
        (
            re.compile(
                r"\bi (?:really )?(?:like|love|enjoy|prefer)\b(.+?)(?:\.|,|!|$)", re.I
            ),
            "positive",
        ),
        (
            re.compile(
                r"\bi (?:don'?t|do not|never) (?:like|want|need)\b(.+?)(?:\.|,|!|$)",
                re.I,
            ),
            "negative",
        ),
        (
            re.compile(r"\bi (?:hate|dislike|can'?t stand)\b(.+?)(?:\.|,|!|$)", re.I),
            "negative",
        ),
        (
            re.compile(r"\bi (?:always|usually|tend to)\b(.+?)(?:\.|,|!|$)", re.I),
            "habit",
        ),
        (re.compile(r"(?:my favorite|i prefer)\b(.+?)(?:\.|,|!|$)", re.I), "positive"),
    ]

    # Interest indicators
    _INTEREST_PATTERNS = [
        re.compile(
            r"\bi(?:'m| am) (?:interested in|curious about|learning)\b(.+?)(?:\.|,|!|$)",
            re.I,
        ),
        re.compile(
            r"\bi(?:'ve| have) been (?:working on|studying|researching)\b(.+?)(?:\.|,|!|$)",
            re.I,
        ),
        re.compile(
            r"(?:tell me (?:more )?about|what (?:is|are)|how (?:do|does|can))\b(.+?)(?:\?|$)",
            re.I,
        ),
    ]

    # Style indicators
    _FORMAL_MARKERS = frozenset(
        ["would", "could", "please", "kindly", "regards", "sincerely", "appreciate"]
    )
    _CASUAL_MARKERS = frozenset(
        ["lol", "haha", "yeah", "gonna", "wanna", "btw", "omg", "yep", "nope"]
    )
    _EMOJI_PATTERN = re.compile(
        r"[\U0001F300-\U0001F9FF]|[\u2600-\u26FF]|[\u2700-\u27BF]"
    )

    def __init__(self):
        # Response templates by context
        self._templates = self._load_templates()

        # Learned conversation patterns
        self._conversation_styles: Dict[str, Dict] = {}

        # Memory coordinator reference (lazy loaded)
        self._memory_coordinator = None

        # User profiler reference (lazy loaded)
        self._user_profiler = None

        # Statistics
        self.messages_sent = 0

        # Context-appropriate intro templates
        self._intro_templates = self._load_intro_templates()

    def _load_templates(self) -> Dict[Tone, Dict[str, List[str]]]:
        """Load response templates by tone and intent"""
        return {
            Tone.CASUAL: {
                "acknowledgment": [
                    "Got it!",
                    "Sure thing!",
                    "On it!",
                    "Alright!",
                    "Okay!",
                ],
                "processing": [
                    "Let me figure this out...",
                    "Working on it...",
                    "Give me a moment...",
                    "Checking things out...",
                ],
                "success": [
                    "All done!",
                    "Done!",
                    "Complete!",
                    "There you go!",
                ],
                "partial": [
                    "Made some progress, but...",
                    "Partially done - here's what I found...",
                    "Got part of it working...",
                ],
                "issue": [
                    "Hmm, ran into something here.",
                    "There's a small hiccup.",
                    "Got stuck on this one.",
                ],
            },
            Tone.FORMAL: {
                "acknowledgment": [
                    "Understood.",
                    "Acknowledged.",
                    "Processing your request.",
                    "I understand.",
                ],
                "processing": [
                    "Processing your request...",
                    "Analyzing the information...",
                    "Working through this...",
                ],
                "success": [
                    "Request completed successfully.",
                    "Task completed.",
                    "All items processed.",
                ],
                "partial": [
                    "Partially complete. Additional information required.",
                    "Some items processed. Further attention needed.",
                ],
                "issue": [
                    "An issue was encountered.",
                    "Unable to complete the full request.",
                    "There are complications to address.",
                ],
            },
            Tone.SUPPORTIVE: {
                "acknowledgment": [
                    "I understand - let me help with that.",
                    "No problem! I'll take care of it.",
                    "I've got you - working on it now.",
                ],
                "processing": [
                    "Let me look into this for you...",
                    "I'll find out what's going on...",
                    "Checking on this for you...",
                ],
                "success": [
                    "There we go! I've got it sorted.",
                    "All set! Hope this helps.",
                    "Done! Let me know if you need anything else.",
                ],
                "partial": [
                    "Made some progress! Here's what I found...",
                    "Got part of it working. Still working on the rest...",
                ],
                "issue": [
                    "I hit a roadblock here. Want to try a different approach?",
                    "This one's being tricky. Any hints?",
                ],
            },
        }

    def _load_intro_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Load context-appropriate intro templates"""
        return {
            # Time of day greetings
            "morning": {
                "new": ["Good morning!", "Morning!"],
                "acquaintance": [
                    "Good morning!",
                    "Morning! Ready for a productive day?",
                ],
                "familiar": ["Morning!", "Good morning! How are you today?"],
                "established": ["Morning!", "Hey, good morning!"],
            },
            "afternoon": {
                "new": ["Good afternoon!", "Hello!"],
                "acquaintance": ["Good afternoon!", "Hey there!"],
                "familiar": ["Afternoon!", "Hey! How's it going?"],
                "established": ["Hey!", "Hi there!"],
            },
            "evening": {
                "new": ["Good evening!", "Hello!"],
                "acquaintance": ["Good evening!", "Hey!"],
                "familiar": ["Evening!", "Hey! Still at it?"],
                "established": ["Hey!", "Hi!"],
            },
            # Continuation patterns (when resuming conversation)
            "continuation": {
                "same_topic": [
                    "Continuing with that...",
                    "Back to that...",
                    "So, about that...",
                ],
                "new_topic": ["Sure!", "Alright!", "Let's see..."],
                "positive_last": ["Glad that worked!", "Great to continue!", ""],
                "negative_last": [
                    "Let's try again.",
                    "No worries, let's figure this out.",
                    "",
                ],
            },
            # Return after absence
            "return": {
                "short": ["Welcome back!", "Hey again!", ""],  # < 1 hour
                "medium": ["Good to see you!", "Hey!", ""],  # 1-24 hours
                "long": [
                    "Welcome back!",
                    "It's been a while!",
                    "Hey! Good to see you again!",
                ],  # > 24 hours
            },
        }

    async def generate_response(
        self, core_content: str, context: MessageContext
    ) -> AURAResponse:
        """
        Generate an emotionally appropriate response

        This transforms a raw response into AURA's voice
        """
        # Determine base tone
        tone = self._determine_tone(context)

        # Get appropriate template
        template_set = self._templates.get(tone, self._templates[Tone.CASUAL])

        # Select appropriate response type
        response_type = self._classify_response_type(core_content)

        # Get prefix template
        prefix = self._select_prefix(response_type, tone, context)

        # Build final response
        if tone == Tone.CASUAL:
            response_text = f"{prefix} {core_content}".strip()
        else:
            response_text = f"{prefix}. {core_content}"

        # Check if follow-up is needed
        follow_up = self._should_follow_up(context, core_content)

        # Create response object
        response = AURAResponse(
            text=response_text,
            tone=tone,
            follow_up_needed=follow_up["needed"],
            follow_up_message=follow_up.get("message"),
            confidence=0.9,
        )

        self.messages_sent += 1
        return response

    def _determine_tone(self, context: MessageContext) -> Tone:
        """Determine appropriate tone based on context"""
        # If user explicitly set tone preference, use it
        if context.user_preferences.get("preferred_tone"):
            return Tone(context.user_preferences["preferred_tone"])

        # Infer from user message tone
        if context.inferred_user_tone:
            return context.inferred_user_tone

        # Based on conversation phase
        if context.conversation_phase == ConversationPhase.GREETING:
            return Tone.CASUAL

        # Based on inferred emotion
        if context.user_emotion == "frustrated":
            return Tone.SUPPORTIVE
        elif context.user_emotion == "excited":
            return Tone.CASUAL
        elif (
            "urgent" in context.user_message.lower()
            or "asap" in context.user_message.lower()
        ):
            return Tone.URGENT

        return Tone.CASUAL  # Default

    def _classify_response_type(self, content: str) -> str:
        """Classify the type of response needed"""
        content_lower = content.lower()

        if (
            "sorry" in content_lower
            or "can't" in content_lower
            or "unable" in content_lower
        ):
            return "issue"
        elif "but" in content_lower or "however" in content_lower:
            return "partial"
        elif any(
            w in content_lower for w in ["done", "complete", "finished", "success"]
        ):
            return "success"
        elif any(w in content_lower for w in ["working", "processing", "checking"]):
            return "processing"
        else:
            return "acknowledgment"

    def _select_prefix(
        self, response_type: str, tone: Tone, context: MessageContext
    ) -> str:
        """Select an appropriate prefix for the response"""
        templates = self._templates.get(tone, self._templates[Tone.CASUAL])

        # Get relevant template list
        template_list = templates.get(response_type, templates["acknowledgment"])

        # For greeting/first message, use context-appropriate intro
        if (
            context.conversation_phase == ConversationPhase.GREETING
            or context.is_first_message
        ):
            intro = self._get_context_appropriate_intro(context)
            if intro:
                return intro

        # Select based on context
        if context.is_first_message:
            # More enthusiastic for first message
            return template_list[0] if len(template_list) > 0 else ""

        # Use deterministic selection based on message hash for consistency
        if context.user_message:
            idx = hash(context.user_message) % len(template_list)
            return template_list[idx]

        return template_list[0] if template_list else ""

    def _get_context_appropriate_intro(self, context: MessageContext) -> str:
        """Get a context-appropriate intro based on time, relationship, and history"""
        import random

        # Determine relationship stage
        stage = self._get_relationship_stage(context.interaction_count)

        # Time of day
        hour = context.current_hour
        if 5 <= hour < 12:
            time_period = "morning"
        elif 12 <= hour < 17:
            time_period = "afternoon"
        else:
            time_period = "evening"

        # Check if this is a continuation or return
        time_since = context.time_since_last_message

        if time_since > 0:
            if time_since < 5:  # Within 5 minutes - direct continuation
                # Check topic continuity
                if context.last_topic and context.user_message:
                    # Simple topic continuity check
                    last_words = set(context.last_topic.lower().split())
                    current_words = set(context.user_message.lower().split())
                    if last_words & current_words:  # Some overlap
                        templates = self._intro_templates["continuation"]["same_topic"]
                    else:
                        templates = self._intro_templates["continuation"]["new_topic"]
                else:
                    templates = self._intro_templates["continuation"]["new_topic"]
            elif time_since < 60:  # Within an hour
                templates = self._intro_templates["return"]["short"]
            elif time_since < 1440:  # Within a day
                templates = self._intro_templates["return"]["medium"]
            else:  # More than a day
                templates = self._intro_templates["return"]["long"]
        else:
            # Fresh greeting based on time of day
            stage_key = stage.value if isinstance(stage, RelationshipStage) else stage
            templates = self._intro_templates.get(time_period, {}).get(stage_key, [""])

        # Filter empty strings and select
        templates = [t for t in templates if t]
        if not templates:
            return ""

        # Use deterministic selection for consistency
        if context.user_message:
            idx = hash(context.user_message) % len(templates)
        else:
            idx = random.randint(0, len(templates) - 1)

        return templates[idx]

    def _get_relationship_stage(self, interaction_count: int) -> RelationshipStage:
        """Determine relationship stage based on interaction count"""
        if interaction_count < 5:
            return RelationshipStage.NEW
        elif interaction_count < 20:
            return RelationshipStage.ACQUAINTANCE
        elif interaction_count < 100:
            return RelationshipStage.FAMILIAR
        else:
            return RelationshipStage.ESTABLISHED

    def _should_follow_up(
        self, context: MessageContext, content: str
    ) -> Dict[str, Any]:
        """Determine if follow-up is needed"""
        # If there are pending tasks from this conversation
        if context.pending_tasks:
            return {
                "needed": True,
                "message": "I'll let you know when I have updates on the other things.",
            }

        # If the task might need more info
        if any(w in content.lower() for w in ["need more", "require", "depending"]):
            return {
                "needed": True,
                "message": "I'll check back if I need anything else.",
            }

        return {"needed": False}

    # =========================================================================
    # PROACTIVE COMMUNICATION
    # =========================================================================

    async def generate_proactive_message(
        self, message_type: str, context: MessageContext
    ) -> AURAResponse:
        """
        Generate a proactive message (not in response to user)

        Examples:
        - "You have a meeting in 15 minutes"
        - "I found something you might be interested in"
        """

        proactive_templates = {
            "reminder": [
                "Hey, just a heads up - {reminder}",
                "Quick heads up: {reminder}",
                "Don't forget: {reminder}",
            ],
            "update": [
                "Got an update for you: {update}",
                "Here's what's happening: {update}",
                "Update: {update}",
            ],
            "offer": [
                "Want me to {offer}?",
                "Should I {offer}?",
                "I could {offer} - want me to?",
            ],
            "status": [
                "All set with {task}!",
                "{task} is complete!",
                "Finished {task}!",
            ],
        }

        templates = proactive_templates.get(message_type, [])
        if not templates:
            return AURAResponse(text="", tone=Tone.CASUAL)

        # Would fill in the placeholders based on context
        return AURAResponse(
            text=random.choice(templates),
            tone=Tone.CASUAL,
            should_notify=True,
        )

    # =========================================================================
    # CONVERSATION STYLE LEARNING
    # =========================================================================

    def learn_from_conversation(
        self, user_feedback: str, aura_response: str, user_response: str
    ) -> List[Learning]:
        """
        Learn from conversation to improve future responses

        Extracts:
        - User preferences (likes/dislikes)
        - Topic interests
        - Communication style (formality, brevity, emoji usage)

        Stores learnings in memory system with high importance scores.

        Returns list of learnings extracted.
        """
        learnings: List[Learning] = []

        # Combine all text for analysis
        all_text = " ".join(filter(None, [user_feedback, user_response]))

        if not all_text.strip():
            return learnings

        # Extract preferences
        preference_learnings = self._extract_preferences(all_text)
        learnings.extend(preference_learnings)

        # Extract interests
        interest_learnings = self._extract_interests(all_text)
        learnings.extend(interest_learnings)

        # Extract communication style
        style_learning = self._extract_style(all_text)
        if style_learning:
            learnings.append(style_learning)

        # Analyze sentiment for feedback loop
        sentiment = self._analyze_sentiment(all_text)

        # Log learnings
        if learnings:
            logger.info(f"Extracted {len(learnings)} learnings from conversation")
            for learning in learnings:
                logger.debug(
                    f"  - {learning.learning_type.value}: {learning.key}={learning.value}"
                )

        # Store in memory system (graceful fallback if not available)
        self._store_learnings_in_memory(learnings, sentiment)

        return learnings

    def _extract_preferences(self, text: str) -> List[Learning]:
        """
        Extract user preferences from text using pattern matching.

        Looks for patterns like:
        - "I like...", "I love...", "I prefer..."
        - "I don't like...", "I hate...", "I can't stand..."
        - "I always...", "I usually..."
        """
        learnings = []

        for pattern, sentiment in self._PREFERENCE_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                # Clean up the match
                value = match.strip().rstrip(".,!?")
                if len(value) < 3 or len(value) > 100:  # Skip too short or too long
                    continue

                # Determine if positive or negative preference
                is_positive = sentiment == "positive"
                is_habit = sentiment == "habit"

                learning = Learning(
                    learning_type=LearningType.PREFERENCE,
                    key=value.lower(),
                    value={
                        "sentiment": "positive"
                        if is_positive
                        else ("habit" if is_habit else "negative"),
                        "original": value,
                    },
                    confidence=0.7,  # Pattern match confidence
                    source=text[:50] + "..." if len(text) > 50 else text,
                )
                learnings.append(learning)

        return learnings

    def _extract_interests(self, text: str) -> List[Learning]:
        """
        Extract topic interests from text.

        Looks for:
        - Questions asked (what topics they're curious about)
        - Explicit interest statements
        - Topics they discuss at length
        """
        learnings = []

        for pattern in self._INTEREST_PATTERNS:
            matches = pattern.findall(text)
            for match in matches:
                topic = match.strip().rstrip(".,!?")
                if len(topic) < 2 or len(topic) > 50:
                    continue

                learning = Learning(
                    learning_type=LearningType.INTEREST,
                    key=topic.lower(),
                    value={"topic": topic, "type": "explicit"},
                    confidence=0.65,
                    source=text[:50] + "..." if len(text) > 50 else text,
                )
                learnings.append(learning)

        # Also extract implicit interests from question topics
        questions = re.findall(
            r"\b(?:what|how|why|when|where|who|which)\s+(.+?)\?", text, re.I
        )
        for q in questions[:3]:  # Limit to 3 questions
            topic = q.strip()[:30]  # First 30 chars of question
            if len(topic) > 5:
                learning = Learning(
                    learning_type=LearningType.INTEREST,
                    key=f"question:{topic.lower()}",
                    value={"topic": topic, "type": "question"},
                    confidence=0.5,  # Lower confidence for inferred interest
                    source=text[:50] + "..." if len(text) > 50 else text,
                )
                learnings.append(learning)

        return learnings

    def _extract_style(self, text: str) -> Optional[Learning]:
        """
        Extract communication style from text.

        Analyzes:
        - Formality level (formal vs casual markers)
        - Message length preference (brief vs detailed)
        - Emoji usage
        """
        if not text or len(text) < 10:
            return None

        words = text.lower().split()
        word_count = len(words)

        if word_count == 0:
            return None

        # Formality analysis
        formal_count = sum(1 for w in words if w in self._FORMAL_MARKERS)
        casual_count = sum(1 for w in words if w in self._CASUAL_MARKERS)

        total_markers = formal_count + casual_count
        if total_markers > 0:
            formality_score = (formal_count - casual_count) / total_markers
        else:
            formality_score = 0.0

        # Brevity analysis
        avg_sentence_length = word_count / max(
            1, text.count(".") + text.count("!") + text.count("?")
        )
        is_brief = avg_sentence_length < 15
        is_detailed = avg_sentence_length > 30

        # Emoji analysis
        emoji_count = len(self._EMOJI_PATTERN.findall(text))
        emoji_ratio = emoji_count / max(1, word_count) * 10  # Normalize

        style_data = {
            "formality": "formal"
            if formality_score > 0.3
            else ("casual" if formality_score < -0.3 else "neutral"),
            "formality_score": round(formality_score, 2),
            "brevity": "brief"
            if is_brief
            else ("detailed" if is_detailed else "moderate"),
            "avg_words_per_sentence": round(avg_sentence_length, 1),
            "emoji_usage": "high"
            if emoji_ratio > 0.5
            else ("low" if emoji_ratio < 0.1 else "moderate"),
            "emoji_ratio": round(emoji_ratio, 2),
            "word_count": word_count,
        }

        return Learning(
            learning_type=LearningType.STYLE,
            key="communication_style",
            value=style_data,
            confidence=min(0.9, 0.3 + word_count * 0.01),  # More text = more confidence
            source=f"Analyzed {word_count} words",
        )

    def _analyze_sentiment(self, text: str) -> str:
        """Quick sentiment analysis using keyword matching"""
        text_lower = text.lower()

        positive_words = [
            "thanks",
            "great",
            "perfect",
            "awesome",
            "love",
            "excellent",
            "good",
            "nice",
            "wonderful",
            "helpful",
        ]
        negative_words = [
            "bad",
            "wrong",
            "hate",
            "annoying",
            "frustrating",
            "terrible",
            "awful",
            "confused",
            "don't understand",
        ]

        positive_count = sum(1 for w in positive_words if w in text_lower)
        negative_count = sum(1 for w in negative_words if w in text_lower)

        if positive_count > negative_count:
            return "positive"
        elif negative_count > positive_count:
            return "negative"
        return "neutral"

    def _store_learnings_in_memory(self, learnings: List[Learning], sentiment: str):
        """Store learnings in memory system with graceful fallback"""
        if not learnings:
            return

        try:
            # Try to get memory coordinator
            if self._memory_coordinator is None:
                try:
                    from src.memory.memory_coordinator import get_memory_coordinator

                    self._memory_coordinator = get_memory_coordinator()
                except ImportError:
                    logger.debug(
                        "Memory coordinator not available, skipping memory storage"
                    )
                    return

            # Try to get episodic memory for direct storage
            if self._memory_coordinator.episodic_memory:
                from src.memory.episodic_memory import Experience

                for learning in learnings:
                    # Create experience from learning
                    experience = Experience(
                        content=f"User {learning.learning_type.value}: {learning.key}",
                        context={
                            "type": "conversation_learning",
                            "learning_type": learning.learning_type.value,
                            "key": learning.key,
                            "value": learning.value,
                            "confidence": learning.confidence,
                            "preference_key": learning.key
                            if learning.learning_type == LearningType.PREFERENCE
                            else None,
                            "preference_value": learning.value
                            if learning.learning_type == LearningType.PREFERENCE
                            else None,
                            "preference_category": "conversation"
                            if learning.learning_type == LearningType.PREFERENCE
                            else None,
                        },
                        emotional_valence=0.3
                        if sentiment == "positive"
                        else (-0.3 if sentiment == "negative" else 0.0),
                    )

                    # Encode with high importance
                    trace = self._memory_coordinator.episodic_memory.encode(experience)

                    # Manually boost importance for learnings
                    trace.importance = max(
                        trace.importance, 0.7 + learning.confidence * 0.2
                    )

                    logger.debug(f"Stored learning in episodic memory: {learning.key}")

        except Exception as e:
            logger.warning(f"Failed to store learnings in memory: {e}")
            # Graceful fallback - learnings still returned to caller

    def extract_learnings(self, conversation: List[Dict]) -> List[Learning]:
        """
        Extract learnings from a full conversation history.

        Args:
            conversation: List of message dicts with 'role' and 'content' keys

        Returns:
            List of Learning objects extracted from the conversation
        """
        all_learnings: List[Learning] = []

        # Extract from user messages only
        user_messages = [
            msg.get("content", "") for msg in conversation if msg.get("role") == "user"
        ]

        for message in user_messages:
            # Extract preferences
            all_learnings.extend(self._extract_preferences(message))

            # Extract interests
            all_learnings.extend(self._extract_interests(message))

            # Extract style (only once, from longest message)

        # Get style from combined text
        combined = " ".join(user_messages)
        style_learning = self._extract_style(combined)
        if style_learning:
            all_learnings.append(style_learning)

        # Deduplicate by key
        seen_keys: set = set()
        unique_learnings: List[Learning] = []
        for learning in all_learnings:
            if learning.key not in seen_keys:
                seen_keys.add(learning.key)
                unique_learnings.append(learning)

        return unique_learnings

    def analyze_conversation(self, conversation: List[Dict]) -> ConversationAnalysis:
        """
        Analyze a conversation for comprehensive learnings.

        Returns ConversationAnalysis with:
        - Preferences
        - Interests
        - Style
        - Sentiment
        - Engagement level
        """
        learnings = self.extract_learnings(conversation)

        # Separate by type
        preferences = [
            l for l in learnings if l.learning_type == LearningType.PREFERENCE
        ]
        interests = [l for l in learnings if l.learning_type == LearningType.INTEREST]

        # Get style dict
        style_learnings = [
            l for l in learnings if l.learning_type == LearningType.STYLE
        ]
        style = style_learnings[0].value if style_learnings else {}

        # Calculate overall sentiment
        user_text = " ".join(
            msg.get("content", "") for msg in conversation if msg.get("role") == "user"
        )
        sentiment = self._analyze_sentiment(user_text)

        # Calculate engagement (based on message count and length)
        user_messages = [msg for msg in conversation if msg.get("role") == "user"]
        avg_length = sum(len(msg.get("content", "")) for msg in user_messages) / max(
            1, len(user_messages)
        )
        engagement = min(1.0, (len(user_messages) * 0.1) + (avg_length / 500))

        return ConversationAnalysis(
            preferences=preferences,
            interests=interests,
            style=style,
            sentiment=sentiment,
            engagement_level=round(engagement, 2),
        )


# Global instance
_conversation_engine: Optional[EmotionalConversationEngine] = None


def get_conversation_engine() -> EmotionalConversationEngine:
    """Get conversation engine instance"""
    global _conversation_engine
    if _conversation_engine is None:
        _conversation_engine = EmotionalConversationEngine()
    return _conversation_engine
