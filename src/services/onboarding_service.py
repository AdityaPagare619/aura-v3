"""
AURA v3 Onboarding Service
===========================

Natural user onboarding through conversation:
- AURA introduces itself first on first contact
- Learns about user through NATURAL CHAT (not commands)
- Stores learned info in encrypted memory
- Integrates with DeepUserProfiler

Key principles:
1. AURA starts knowing NOTHING about user
2. User tells AURA about themselves naturally
3. Info gets stored properly in encrypted memory
4. AURA introduces itself first, then asks questions
"""

import asyncio
import logging
import json
import os
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class OnboardingPhase(Enum):
    """Phases of onboarding"""
    NEW_USER = "new_user"           # AURA introduces itself
    GETTING_STARTED = "getting_started"  # Ask basic info
    LEARNING_PREFERENCES = "learning_preferences"  # Learn communication style
    LEARNING_CONTEXT = "learning_context"  # Learn life context
    COMPLETE = "complete"           # Onboarding done


@dataclass
class OnboardingQuestion:
    """A question to ask user during onboarding"""
    question_id: str
    question_text: str
    parse_pattern: str  # Regex to extract info from response
    profile_field: str  # Which profile field to update
    follow_up: Optional[str] = None  # Optional follow-up question


@dataclass
class UserOnboardingState:
    """Current onboarding state for a user"""
    user_id: str
    phase: OnboardingPhase = OnboardingPhase.NEW_USER
    questions_asked: List[str] = field(default_factory=list)
    info_gathered: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    introduction_shown: bool = False


class OnboardingService:
    """
    Natural onboarding service for AURA
    
    Enables AURA to:
    1. Introduce itself on first contact
    2. Ask questions naturally to learn about user
    3. Store learned info in encrypted memory
    4. Flow user info to neural memory for persistence
    """

    # AURA's introduction message
    INTRODUCTION = """Hey there! I'm AURA - your personal AI assistant.

I don't know anything about you yet, so let me ask you a few questions so I can help you better.

{first_question}

(You can skip any question or answer in whatever way you like - I'm flexible!)"""

    # Questions to ask new users (in order)
    ONBOARDING_QUESTIONS = [
        OnboardingQuestion(
            question_id="name",
            question_text="First things first - what should I call you?",
            parse_pattern=r"(?:my name is|i'm|i am|call me|just|is|named)\s+([A-Za-z]+)",
            profile_field="display_name",
            follow_up="Got it! Any nickname or prefer something shorter?"
        ),
        OnboardingQuestion(
            question_id="how_to_help",
            question_text="What do you usually need help with? (Apps, tasks, reminders, or just chatting?)",
            parse_pattern=r"(apps|tasks|reminders|chatting|calendar|emails|shopping|health|work|study|all|everything)",
            profile_field="primary_needs",
            follow_up="That makes sense! Anything else?"
        ),
        OnboardingQuestion(
            question_id="communication_style",
            question_text="How do you like me to talk? Casual and friendly, or more formal and direct?",
            parse_pattern=r"(casual|friendly|formal|direct|professional|relaxed|chill|serious)",
            profile_field="communication_style",
            follow_up="Noted! Should I keep responses short and to the point, or more detailed?"
        ),
        OnboardingQuestion(
            question_id="detail_level",
            question_text="One more thing - do you prefer quick brief answers or detailed explanations?",
            parse_pattern=r"(brief|quick|short|concise|detailed|long|thorough|explanations)",
            profile_field="prefers_concise",
            follow_up="Perfect! That's all I need for now."
        ),
    ]

    def __init__(self, data_dir: str = "data/onboarding"):
        self.data_dir = data_dir
        self._states: Dict[str, UserOnboardingState] = {}
        
        # Dependencies (set after initialization)
        self._secure_storage = None
        self._neural_memory = None
        self._user_profiler = None
        
        # Load existing onboarding states
        self._load_states()

    async def initialize(self):
        """Initialize the onboarding service"""
        logger.info("Initializing Onboarding Service...")
        
        # Try to get secure storage for encrypted user data
        try:
            from src.core.security_layers import SecureStorage
            self._secure_storage = SecureStorage(
                self.data_dir, 
                encrypt_by_default=True
            )
            logger.info("Onboarding Service: Using encrypted storage")
        except ImportError:
            logger.warning("Onboarding Service: SecureStorage not available, using plain storage")
            os.makedirs(self.data_dir, exist_ok=True)
        
        logger.info("Onboarding Service initialized")

    def set_dependencies(
        self, 
        neural_memory=None, 
        user_profiler=None,
        secure_storage=None
    ):
        """Set dependencies for memory integration"""
        self._neural_memory = neural_memory
        self._user_profiler = user_profiler
        if secure_storage:
            self._secure_storage = secure_storage

    def is_onboarding_complete(self, user_id: str) -> bool:
        """Check if user has completed onboarding"""
        state = self._states.get(user_id)
        if not state:
            return False  # New user, not complete
        return state.phase == OnboardingPhase.COMPLETE

    def is_new_user(self, user_id: str) -> bool:
        """Check if this is a new user (no prior onboarding)"""
        return user_id not in self._states

    async def get_onboarding_response(
        self, 
        user_id: str, 
        user_message: str = ""
    ) -> Optional[Dict[str, Any]]:
        """
        Get AURA's onboarding response
        
        For new users:
        - Returns introduction if no message yet
        - Asks next question if user responded
        - Parses user response to extract info
        
        For returning users:
        - Returns None (onboarding complete)
        """
        # Get or create onboarding state
        state = self._states.get(user_id)
        
        if not state:
            # New user - start onboarding
            state = UserOnboardingState(
                user_id=user_id,
                phase=OnboardingPhase.NEW_USER
            )
            self._states[user_id] = state
            self._save_state(state)
            
            # Return introduction
            return {
                "type": "introduction",
                "message": self._get_introduction(),
                "phase": "new_user"
            }
        
        # Handle based on current phase
        if state.phase == OnboardingPhase.COMPLETE:
            return None  # Onboarding done
        
        # Process user's response and get next question
        if user_message:
            # Parse user's response to extract info
            await self._process_user_response(user_id, user_message)
        
        # Get next question to ask
        return await self._get_next_question(user_id)

    def _get_introduction(self) -> str:
        """Get AURA's introduction message"""
        first_question = self.ONBOARDING_QUESTIONS[0].question_text
        return self.INTRODUCTION.format(first_question=first_question)

    async def _process_user_response(self, user_id: str, message: str):
        """Process user's response and extract information"""
        state = self._states.get(user_id)
        if not state:
            return
        
        message_lower = message.lower()
        
        # Check if user wants to skip
        if any(ski
