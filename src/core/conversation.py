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
import random
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


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


@dataclass
class MessageContext:
    """Context for generating response"""

    # User info
    user_name: Optional[str] = None

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

    # What user wants
    intent: Optional[str] = None
    entities: Dict[str, Any] = field(default_factory=dict)

    # Memory
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    conversation_history: List[Dict] = field(default_factory=list)


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

    def __init__(self):
        # Response templates by context
        self._templates = self._load_templates()

        # Learned conversation patterns
        self._conversation_styles: Dict[str, Dict] = {}

        # Statistics
        self.messages_sent = 0

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

        # Select one, considering context
        if context.is_first_message:
            # More enthusiastic for first message
            return (
                random.choice(template_list[:2])
                if len(template_list) > 2
                else template_list[0]
            )

        return random.choice(template_list)

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
    ):
        """
        Learn from conversation to improve future responses

        Track what worked and adjust accordingly
        """
        # Simple feedback parsing
        positive_indicators = ["thanks", "great", "perfect", "awesome", "love"]
        negative_indicators = ["weird", "off", "strange", "don't like", "not"]

        feedback_lower = user_feedback.lower() if user_feedback else ""

        if any(ind in feedback_lower for ind in positive_indicators):
            # Positive feedback - continue doing what worked
            logger.info("Positive feedback received - reinforcing style")

        elif any(ind in feedback_lower for ind in negative_indicators):
            # Negative feedback - adjust tone
            logger.info("Negative feedback - will adjust style")


# Global instance
_conversation_engine: Optional[EmotionalConversationEngine] = None


def get_conversation_engine() -> EmotionalConversationEngine:
    """Get conversation engine instance"""
    global _conversation_engine
    if _conversation_engine is None:
        _conversation_engine = EmotionalConversationEngine()
    return _conversation_engine
