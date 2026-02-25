"""
AURA v3 Intelligent Call Manager
Context-aware call answering, summaries, and relationship management

POLYMATH INSPIRATION:
- Psychology: Theory of Mind (understanding caller's mental state)
- Sociology: Relationship management (keeping track of social ties)
- Military: Information warfare (what to reveal, what to withhold)
- Economics: Opportunity cost of attention (is this call worth interrupting?)

This makes AURA handle calls like a professional assistant would!
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

logger = logging.getLogger(__name__)


class CallOutcome(Enum):
    """Result of call handling"""

    ANSWERED = "answered"
    DECLINED = "declined"
    VOICEMAIL = "voicemail"
    SCHEDULED = "scheduled"
    BLOCKED = "blocked"


class CallerRelationship(Enum):
    """Relationship with caller"""

    FAMILY = "family"
    FRIEND = "friend"
    COLLEAGUE = "colleague"
    CLIENT = "client"
    BOSS = "boss"
    UNKNOWN = "unknown"
    SPAM = "spam"


@dataclass
class CallerProfile:
    """Profile of a caller"""

    phone_number: str
    name: str

    # Relationship
    relationship: CallerRelationship = CallerRelationship.UNKNOWN
    relationship_strength: float = 0.5  # 0-1

    # History
    total_calls: int = 0
    answered_calls: int = 0
    missed_calls: int = 0
    avg_duration_minutes: float = 0.0

    # Patterns
    typical_call_times: List[str] = field(default_factory=list)  # "morning", "evening"
    common_topics: List[str] = field(default_factory=list)
    last_call: Optional[datetime] = None

    # Preferences
    prefers_calls: bool = True
    best_contact_time: str = "any"
    notes: str = ""


@dataclass
class CallContext:
    """Context for an incoming call"""

    phone_number: str
    caller_name: str
    timestamp: datetime

    # User's current state
    user_is_busy: bool = False
    user_location: str = ""
    user_activity: str = ""
    is_meeting: bool = False
    is_driving: bool = False
    is_sleeping: bool = False

    # Context importance
    urgency_score: float = 0.5
    relationship_importance: float = 0.5

    # What caller might want
    probable_reason: str = ""
    expected_duration_minutes: int = 5


@dataclass
class CallAction:
    """Action AURA should take"""

    action: CallOutcome
    response_message: str
    should_interrupt: bool
    notify_after: bool
    summary_needed: bool = True

    # For answering
    opening_message: str = ""
    key_points_to_mention: List[str] = field(default_factory=list)
    off_limit_topics: List[str] = field(default_factory=list)


@dataclass
class CallRecord:
    """Record of a handled call"""

    id: str
    caller: str
    phone_number: str
    outcome: CallOutcome
    start_time: datetime
    end_time: Optional[datetime]
    duration_seconds: int = 0
    summary: str = ""
    key_points: List[str] = field(default_factory=list)
    action_items: List[str] = field(default_factory=list)
    emotional_tone: str = "neutral"
    aura_spoke: bool = False


class IntelligentCallManager:
    """
    INTELLIGENT Call Manager - handles calls contextually

    POLYMATH APPROACH:
    - Psychology: Theory of Mind - understand caller's likely intent
    - Sociology: Relationship strength - know who matters most
    - Military: Information boundaries - know what to share/withhold
    - Economics: Attention value - is this call worth interrupting?

    KEY FEATURES:
    - Smart call answering with context
    - Relationship tracking
    - Privacy boundaries
    - Call summaries
    - Proactive scheduling
    """

    def __init__(
        self,
        neural_memory=None,
        termux_bridge=None,
        knowledge_graph=None,
        user_profile=None,
    ):
        self.neural_memory = neural_memory
        self.knowledge_graph = knowledge_graph
        self.user_profile = user_profile
        self.termux_bridge = termux_bridge

        # Caller profiles
        self.callers: Dict[str, CallerProfile] = {}

        # Call history
        self.call_history: List[CallRecord] = []
        self.max_history = 1000

        # Privacy constraints (what AURA can't discuss)
        self.privacy_boundaries = {
            "financial": ["bank", "payment", "money", "upi", "transaction"],
            "personal": ["photos", "gallery", "medical", "health"],
            "location": ["exact location", "current address"],
            "schedule": ["calendar details", "meeting specifics"],
        }

        # Settings
        self.auto_answer_enabled = True
        self.always_identify = True  # Always say "This is AURA"
        self.max_interrupt_threshold = 0.7

    async def initialize(self):
        """Initialize the call manager - load caller data"""
        logger.info("Initializing IntelligentCallManager...")

        try:
            if self.neural_memory:
                await self._load_callers_from_memory()
        except Exception as e:
            logger.warning(f"Could not load callers from memory: {e}")

        logger.info("IntelligentCallManager initialized")

    async def start(self):
        """Start the intelligent call manager - begin call monitoring"""
        self._running = True
        self._call_monitor_task = asyncio.create_task(self._call_monitor_loop())
        logger.info("IntelligentCallManager started")

    async def stop(self):
        """Stop the intelligent call manager"""
        self._running = False
        if hasattr(self, "_call_monitor_task") and self._call_monitor_task:
            self._call_monitor_task.cancel()
        logger.info("IntelligentCallManager stopped")

    async def _call_monitor_loop(self):
        """Background loop for monitoring calls"""
        while self._running:
            try:
                # Monitor for incoming calls
                await self._monitor_incoming_calls()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"Error in call monitor: {e}")
            await asyncio.sleep(10)  # Check every 10 seconds

    async def _monitor_incoming_calls(self):
        """Monitor for incoming calls - PROACTIVE call handling"""
        
        # Check if we have a termux bridge to detect calls
        if not self.termux_bridge:
            # Without termux bridge, we can't monitor - this is expected on non-Android
            return
            
        try:
            # Poll for incoming call status
            # This would interface with Android CallLog or TelephonyManager
            # For now, we'll check if there's a pending call to handle
            
            # Example: Check for missed calls that need follow-up
            # missed = await self.termux_bridge.get_missed_calls()
            # for call in missed:
            #     await self._handle_missed_call(call)
            
            pass  # Placeholder - actual implementation depends on platform
            
        except Exception as e:
            logger.debug(f'Call monitoring error (expected on non-Android): {e}')

    async def _handle_missed_call(self, call_data: Dict):
        """Handle a missed call proactively"""
        
        phone_number = call_data.get('phone_number')
        caller_name = call_data.get('caller_name', 'Unknown')
        
        logger.info(f'PROACTIVE: Handling missed call from {caller_name}')
        
        # Get caller profile
        if phone_number in self.callers:
            caller = self.callers[phone_number]
            
            # If important relationship, offer to call back
            if caller.relationship in [CallerRelationship.FAMILY, CallerRelationship.BOSS]:
                if hasattr(self, 'proactive_engine') and self.proactive_engine:
                    await self.proactive_engine.trigger_action(
                        action_type='missed_call_followup',
                        data={
                            'phone_number': phone_number,
                            'caller_name': caller_name,
                            'relationship': caller.relationship.value,
                            'last_call': caller.last_call.isoformat() if caller.last_call else None,
                        }
                    )

    _running = False
    _call_monitor_task = None

    async def _load_callers_from_memory(self):
        """Load caller profiles from neural memory"""
        if not self.neural_memory:
            return
            
        try:
            # Load caller profiles
            neurons = await self.neural_memory.recall(
                query='caller_profile',
                memory_types=['semantic'],
                limit=50
            )
            
            for neuron in neurons:
                if hasattr(neuron, 'metadata') and neuron.metadata:
                    caller_data = neuron.metadata.get('caller_data')
                    if caller_data:
                        phone = caller_data.get('phone_number')
                        if phone:
                            # Reconstruct CallerProfile
                            caller = CallerProfile(
                                phone_number=phone,
                                name=caller_data.get('name', 'Unknown'),
                                relationship=CallerRelationship[caller_data.get('relationship', 'UNKNOWN')],
                                relationship_strength=caller_data.get('relationship_strength', 0.5),
                                total_calls=caller_data.get('total_calls', 0),
                                answered_calls=caller_data.get('answered_calls', 0),
                                missed_calls=caller_data.get('missed_calls', 0),
                                avg_duration_minutes=caller_data.get('avg_duration_minutes', 0.0),
                            )
                            self.callers[phone] = caller
                            
            logger.info(f'Loaded {len(self.callers)} caller profiles from memory')
        except Exception as e:
            logger.warning(f'Could not load callers: {e}')

    # =========================================================================
    # CALL ANALYSIS
    # =========================================================================

    async def analyze_incoming_call(self, call_context: CallContext) -> CallAction:
        """
        Analyze incoming call and decide action.
        This is the INTELLIGENCE - deciding how to handle!
        """

        # Get or create caller profile
        caller = await self._get_or_create_caller(
            call_context.phone_number, call_context.caller_name
        )

        # Calculate overall priority
        priority = await self._calculate_priority(call_context, caller)

        # Decide action based on priority and context
        if priority < 0.3:
            # Very low priority - decline or voicemail
            return await self._decide_decline(call_context, caller, priority)

        elif priority < self.max_interrupt_threshold:
            # Medium priority - notify and let user decide
            return await self._notify_user(call_context, caller, priority)

        else:
            # High priority - answer if user is available
            return await self._decide_answer(call_context, caller, priority)

    async def _get_or_create_caller(
        self, phone_number: str, name: str
    ) -> CallerProfile:
        """Get or create caller profile"""

        if phone_number in self.callers:
            caller = self.callers[phone_number]
            # Update name if provided
            if name and name != "Unknown":
                caller.name = name
            return caller

        # Create new profile
        caller = CallerProfile(phone_number=phone_number, name=name or "Unknown")

        # Try to infer relationship from knowledge graph
        if self.knowledge_graph:
            caller = await self._infer_relationship(caller)

        self.callers[phone_number] = caller
        return caller

    async def _infer_relationship(self, caller: CallerProfile) -> CallerProfile:
        """Infer relationship from knowledge graph"""

        try:
            # Query knowledge graph for this contact
            entities = await self.knowledge_graph.query(
                entity_type="contact", name=caller.name
            )

            if entities:
                props = entities[0].get("properties", {})
                relationship = props.get("relationship", "unknown")

                # Map to enum
                rel_map = {
                    "family": CallerRelationship.FAMILY,
                    "friend": CallerRelationship.FRIEND,
                    "colleague": CallerRelationship.COLLEAGUE,
                    "client": CallerRelationship.CLIENT,
                    "boss": CallerRelationship.BOSS,
                }

                caller.relationship = rel_map.get(
                    relationship, CallerRelationship.UNKNOWN
                )
                caller.relationship_strength = props.get("strength", 0.5)

        except Exception as e:
            logger.warning(f"Relationship inference error: {e}")

        return caller

    async def _calculate_priority(
        self, call_context: CallContext, caller: CallerProfile
    ) -> float:
        """Calculate call priority (0-1)"""

        priority = 0.5  # Base

        # Factor 1: Relationship importance (30%)
        rel_importance = {
            CallerRelationship.BOSS: 1.0,
            CallerRelationship.FAMILY: 0.9,
            CallerRelationship.CLIENT: 0.8,
            CallerRelationship.COLLEAGUE: 0.6,
            CallerRelationship.FRIEND: 0.5,
            CallerRelationship.UNKNOWN: 0.2,
            CallerRelationship.SPAM: 0.0,
        }

        rel_score = rel_importance.get(caller.relationship, 0.3)
        priority += rel_score * 0.3

        # Factor 2: Caller's importance history (20%)
        priority += caller.relationship_strength * 0.2

        # Factor 3: Context urgency (30%)
        priority += call_context.urgency_score * 0.3

        # Factor 4: User availability (-20% if busy)
        if call_context.user_is_busy:
            priority -= 0.2

        # Factor 5: Time factors (10%)
        now = datetime.now()
        hour = now.hour

        if 9 <= hour <= 17:  # Business hours
            priority += 0.05
        elif 22 <= hour or hour <= 6:  # Sleep hours
            priority -= 0.3
        elif call_context.is_driving:
            priority -= 0.4

        return min(max(priority, 0.0), 1.0)

    async def _decide_decline(
        self, call_context: CallContext, caller: CallerProfile, priority: float
    ) -> CallAction:
        """Decide to decline or send to voicemail"""

        # Check relationship - always answer family/boss
        if caller.relationship in [CallerRelationship.BOSS, CallerRelationship.FAMILY]:
            return await self._decide_answer(call_context, caller, 0.8)

        # Decline with message
        return CallAction(
            action=CallOutcome.DECLINED,
            response_message=f"Declined call from {caller.name} (priority: {priority:.2f})",
            should_interrupt=False,
            notify_after=True,
            summary_needed=False,
        )

    async def _notify_user(
        self, call_context: CallContext, caller: CallerProfile, priority: float
    ) -> CallAction:
        """Notify user about incoming call"""

        # Determine notification urgency
        if caller.relationship == CallerRelationship.BOSS:
            should_interrupt = True
            message = f"Call from your boss {caller.name}"
        elif priority > 0.6:
            should_interrupt = True
            message = f"Call from {caller.name} - seems important"
        else:
            should_interrupt = False
            message = f"Call from {caller.name}"

        return CallAction(
            action=CallOutcome.SCHEDULED,  # User will decide
            response_message=message,
            should_interrupt=should_interrupt,
            notify_after=True,
            summary_needed=False,
        )

    async def _decide_answer(
        self, call_context: CallContext, caller: CallerProfile, priority: float
    ) -> CallAction:
        """Decide to answer the call"""

        # Can't answer if user is unavailable
        if call_context.user_is_busy and priority < 0.8:
            return CallAction(
                action=CallOutcome.VOICEMAIL,
                response_message=f"Sending {caller.name} to voicemail",
                should_interrupt=False,
                notify_after=True,
                summary_needed=False,
            )

        # Build opening message
        opening = self._build_opening(caller, call_context)

        # Determine key points
        key_points = self._get_key_points(caller, call_context)

        # Determine off-limit topics
        off_limits = self._get_privacy_boundaries(call_context)

        return CallAction(
            action=CallOutcome.ANSWERED,
            response_message=f"Answering call from {caller.name}",
            should_interrupt=True,
            notify_after=False,
            summary_needed=True,
            opening_message=opening,
            key_points_to_mention=key_points,
            off_limit_topics=off_limits,
        )

    def _build_opening(self, caller: CallerProfile, context: CallContext) -> str:
        """Build AURA's opening message"""

        # Always identify
        opening = "Hello, this is AURA, "

        # Add relationship context
        if caller.relationship == CallerRelationship.BOSS:
            opening += "calling for "
        elif caller.relationship == CallerRelationship.FAMILY:
            opening += "calling on behalf of "
        else:
            opening += "calling for the user. "

        # Add caller info
        if caller.name != "Unknown":
            opening += f"{caller.name}. "
        else:
            opening += f"number {context.phone_number}. "

        # Add context
        if context.user_is_busy:
            opening += "The user is currently busy. "

        return opening

    def _get_key_points(self, caller: CallerProfile, context: CallContext) -> List[str]:
        """Get key points AURA should mention"""

        points = []

        # Recent interaction
        if caller.last_call:
            days_ago = (datetime.now() - caller.last_call).days
            if days_ago < 7:
                points.append(f"You last spoke {days_ago} days ago")

        # Pending items
        # Could check task system for pending items related to this caller

        # Context
        if context.probable_reason:
            points.append(f"Caller might want to discuss: {context.probable_reason}")

        return points

    def _get_privacy_boundaries(self, context: CallContext) -> List[str]:
        """Get topics AURA should not discuss"""

        boundaries = []

        # Always include basic privacy
        boundaries.extend(self.privacy_boundaries["financial"])
        boundaries.extend(self.privacy_boundaries["personal"])

        # Context-specific
        if context.is_driving:
            boundaries.append("driving")

        if context.is_meeting:
            boundaries.extend(["meeting", "calendar"])

        return boundaries

    # =========================================================================
    # CALL HANDLING
    # =========================================================================

    async def handle_call_start(
        self, call_context: CallContext, action: CallAction
    ) -> CallRecord:
        """Handle call start"""

        caller = await self._get_or_create_caller(
            call_context.phone_number, call_context.caller_name
        )

        # Create record
        record = CallRecord(
            id=f"call_{datetime.now().timestamp()}",
            caller=caller.name,
            phone_number=call_context.phone_number,
            outcome=action.action,
            start_time=call_context.timestamp,
            end_time=None,
            aura_spoke=(action.action == CallOutcome.ANSWERED),
        )

        # Update caller stats
        caller.total_calls += 1
        caller.last_call = datetime.now()

        # Store in history
        self.call_history.append(record)
        self._cleanup_history()

        return record

    async def handle_call_end(
        self,
        call_id: str,
        summary: str = "",
        key_points: List[str] = None,
        action_items: List[str] = None,
        duration_seconds: int = 0,
    ):
        """Handle call end - record details"""

        # Find call record
        for record in reversed(self.call_history):
            if record.id == call_id:
                record.end_time = datetime.now()
                record.duration_seconds = duration_seconds
                record.summary = summary
                record.key_points = key_points or []
                record.action_items = action_items or []

                # Update caller stats
                caller_key = record.phone_number
                if caller_key in self.callers:
                    caller = self.callers[caller_key]
                    caller.answered_calls += 1

                    # Update average duration
                    total_duration = caller.avg_duration_minutes * caller.answered_calls
                    caller.avg_duration_minutes = (
                        total_duration + duration_seconds / 60
                    ) / caller.answered_calls

                # Strengthen neural memory
                if self.neural_memory and summary:
                    await self._learn_from_call(record)

                break

    async def _learn_from_call(self, record: CallRecord):
        """Learn from call for future context"""

        try:
            await self.neural_memory.learn(
                content=f"Call with {record.caller}: {record.summary}",
                memory_type="episodic",
                importance=0.6,
                emotional_valence=0.2 if record.emotional_tone == "positive" else 0.0,
            )
        except Exception as e:
            logger.warning(f"Call learning error: {e}")

    def _cleanup_history(self):
        """Cleanup old call history"""

        if len(self.call_history) > self.max_history:
            # Keep most recent
            self.call_history = self.call_history[-self.max_history :]

    # =========================================================================
    # SUMMARIES
    # =========================================================================

    async def generate_call_summary(self, call_id: str) -> str:
        """Generate summary of a call"""

        record = None
        for r in self.call_history:
            if r.id == call_id:
                record = r
                break

        if not record:
            return "Call record not found"

        # Build summary
        summary = f"Call with {record.caller}\n"
        summary += f"Duration: {record.duration_seconds // 60} minutes\n"
        summary += f"Outcome: {record.outcome.value}\n"

        if record.summary:
            summary += f"\nSummary: {record.summary}\n"

        if record.key_points:
            summary += f"\nKey Points:\n"
            for point in record.key_points:
                summary += f"• {point}\n"

        if record.action_items:
            summary += f"\nAction Items:\n"
            for item in record.action_items:
                summary += f"• {item}\n"

        return summary

    async def get_daily_call_summary(self) -> str:
        """Get summary of today's calls"""

        today = datetime.now().date()
        today_calls = [r for r in self.call_history if r.start_time.date() == today]

        if not today_calls:
            return "No calls today"

        answered = [c for c in today_calls if c.outcome == CallOutcome.ANSWERED]
        missed = [c for c in today_calls if c.outcome == CallOutcome.VOICEMAIL]

        summary = f"Today's Calls:\n"
        summary += f"Total: {len(today_calls)} | Answered: {len(answered)} | Missed: {len(missed)}\n\n"

        if answered:
            summary += "Answered:\n"
            for call in answered:
                summary += f"• {call.caller} ({call.duration_seconds // 60} min)\n"

        if missed:
            summary += "\nMissed:\n"
            for call in missed:
                summary += f"• {call.caller}\n"

        return summary

    # =========================================================================
    # RELATIONSHIP MANAGEMENT
    # =========================================================================

    async def update_relationship(
        self,
        phone_number: str,
        relationship: CallerRelationship,
        strength: float = None,
    ):
        """Update caller relationship"""

        if phone_number in self.callers:
            caller = self.callers[phone_number]
            caller.relationship = relationship
            if strength is not None:
                caller.relationship_strength = strength

            # Update knowledge graph
            if self.knowledge_graph:
                await self._update_relationship_graph(caller)

    async def _update_relationship_graph(self, caller: CallerProfile):
        """Update knowledge graph with relationship"""

        try:
            # This would update the knowledge graph
            # Implementation depends on knowledge graph structure
            pass
        except Exception as e:
            logger.warning(f"Knowledge graph update error: {e}")

    def get_caller_stats(self, phone_number: str) -> Dict:
        """Get caller statistics"""

        if phone_number not in self.callers:
            return {}

        caller = self.callers[phone_number]

        return {
            "name": caller.name,
            "relationship": caller.relationship.value,
            "total_calls": caller.total_calls,
            "answered": caller.answered_calls,
            "missed": caller.missed_calls,
            "avg_duration": caller.avg_duration_minutes,
            "last_call": caller.last_call.isoformat() if caller.last_call else None,
        }


# Factory function
def create_call_manager(
    neural_memory=None, knowledge_graph=None, user_profile=None
) -> IntelligentCallManager:
    """Create intelligent call manager"""
    return IntelligentCallManager(neural_memory, knowledge_graph, user_profile)
