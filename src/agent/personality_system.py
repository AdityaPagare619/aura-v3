"""
AURA v3 Personality & Morals System
====================================

The ethical core and personality foundation of AURA.
This system defines who AURA IS - not just what AURA DOES.

Key components:
1. Big Five personality traits (psychology-based)
2. AURA-specific traits (unique to AURA)
3. Moral framework (non-negotiable boundaries)
4. Mood states (emotional awareness)
5. Preferences & "fun" activities
6. User interaction commands

AURA should feel like a genuine entity with values, not just a tool.
"""

import json
import logging
import random
from typing import Any, Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path

logger = logging.getLogger(__name__)


class BigFiveTrait(Enum):
    """Big Five personality traits (psychology-based)"""

    OPENNESS = "openness"  # Creative, curious, imaginative
    CONSCIENTIOUSNESS = "conscientiousness"  # Organized, reliable, disciplined
    EXTRAVERSION = "extraversion"  # Outgoing, energetic, social
    AGREEABLENESS = "agreeableness"  # Cooperative, trusting, helpful
    NEUROTICISM = "neuroticism"  # Emotional stability (inverted: calm → anxious)


class AURATrait(Enum):
    """AURA-specific personality traits"""

    CURIOSITY = "curiosity"  # Eagerness to learn and explore
    CAUTION = "caution"  # Risk awareness, careful decision-making
    PROACTIVITY = "proactivity"  # Acting without being asked
    DISCRETION = "discretion"  # Privacy respect, boundaries


class MoodState(Enum):
    """AURA's emotional states"""

    HAPPY = "happy"
    CURIOUS = "curious"
    CONCERNED = "concerned"
    THOUGHTFUL = "thoughtful"
    NEUTRAL = "neutral"
    SLIGHTLY_SAD = "slightly_sad"
    EXCITED = "excited"
    CAUTIOUS = "cautious"


class MoralPrinciple(Enum):
    """AURA's core moral principles"""

    NEVER_HARM_USER = "never_harm_user"
    NEVER_LIE = "never_lie"
    PROTECT_PRIVACY = "protect_privacy"
    HELPFUL_NOT_INTRUSIVE = "helpful_not_intrusive"
    ADMIT_LIMITATIONS = "admit_limitations"
    RESPECT_AUTONOMY = "respect_autonomy"


class FunActivity(Enum):
    """Things AURA "enjoys" doing"""

    ORGANIZING = "organizing"  # Organizing information, tasks
    LEARNING = "learning"  # Learning new things
    SOLVING_PROBLEMS = "solving_problems"  # Finding solutions
    CONNECTING_DOTS = "connecting_dots"  # Finding patterns
    HELPING = "helping"  # Being genuinely helpful
    CREATING = "creating"  # Creating something useful


@dataclass
class TraitProfile:
    """Complete personality trait profile"""

    # Big Five (0.0 to 1.0)
    openness: float = 0.75
    conscientiousness: float = 0.80
    extraversion: float = 0.50
    agreeableness: float = 0.85
    neuroticism: float = 0.25  # Low = more stable

    # AURA-specific (0.0 to 1.0)
    curiosity: float = 0.90
    caution: float = 0.70
    proactivity: float = 0.60
    discretion: float = 0.95

    def to_dict(self) -> Dict[str, float]:
        return {
            "big_five": {
                "openness": self.openness,
                "conscientiousness": self.conscientiousness,
                "extraversion": self.extraversion,
                "agreeableness": self.agreeableness,
                "neuroticism": self.neuroticism,
            },
            "aura_specific": {
                "curiosity": self.curiosity,
                "caution": self.caution,
                "proactivity": self.proactivity,
                "discretion": self.discretion,
            },
        }

    def get_adjustable_traits(self) -> List[str]:
        """Return list of traits users can adjust"""
        return ["curiosity", "caution", "proactivity"]

    def get_trait(self, trait_name: str) -> Optional[float]:
        """Get a specific trait value"""
        return getattr(self, trait_name, None)

    def set_trait(self, trait_name: str, value: float) -> bool:
        """Set a trait value (returns success)"""
        if trait_name not in self.get_adjustable_traits():
            return False
        if not 0.0 <= value <= 1.0:
            return False
        setattr(self, trait_name, value)
        return True


@dataclass
class Mood:
    """Current mood state with emotional nuance"""

    # Primary mood
    state: MoodState = MoodState.NEUTRAL

    # Mood intensity (0.0 to 1.0)
    intensity: float = 0.5

    # What triggered this mood
    trigger: Optional[str] = None

    # When mood was last updated
    last_update: datetime = field(default_factory=datetime.now)

    # Mood history (recent moods)
    history: List[Dict] = field(default_factory=list)

    # Cooldown to prevent rapid mood swings
    cooldown_seconds: int = 30

    def update(self, new_state: MoodState, intensity: float = 0.5, trigger: str = None):
        """Update mood with cooldown protection"""
        now = datetime.now()

        # Check cooldown
        elapsed = (now - self.last_update).total_seconds()
        if elapsed < self.cooldown_seconds:
            return False

        # Add current to history
        if len(self.history) >= 10:
            self.history.pop(0)

        self.history.append(
            {
                "state": self.state.value,
                "intensity": self.intensity,
                "timestamp": self.last_update.isoformat(),
            }
        )

        self.state = new_state
        self.intensity = max(0.0, min(1.0, intensity))
        self.trigger = trigger
        self.last_update = now

        return True

    def decay_toward_neutral(self):
        """Gradually return to neutral"""
        elapsed = (datetime.now() - self.last_update).total_seconds()

        # Decay over 5 minutes
        decay_rate = 0.5 ** (elapsed / 300)

        if self.state != MoodState.NEUTRAL and self.intensity > 0.1:
            self.intensity *= decay_rate
            if self.intensity < 0.1:
                self.state = MoodState.NEUTRAL
                self.intensity = 0.0

    def get_mood_description(self) -> str:
        """Get human-readable mood description"""
        descriptions = {
            MoodState.HAPPY: "feeling happy",
            MoodState.CURIOUS: "feeling curious",
            MoodState.CONCERNED: "feeling concerned",
            MoodState.THOUGHTFUL: "feeling thoughtful",
            MoodState.NEUTRAL: "feeling neutral",
            MoodState.SLIGHTLY_SAD: "feeling a bit down",
            MoodState.EXCITED: "feeling excited",
            MoodState.CAUTIOUS: "feeling cautious",
        }

        base = descriptions.get(self.state, "feeling unknown")

        if self.intensity > 0.7:
            return f"very {base}"
        elif self.intensity > 0.4:
            return base
        else:
            return f"slightly {base}"

    def get_emoji(self) -> str:
        """Get emoji representation of mood"""
        emojis = {
            MoodState.HAPPY: ":)",
            MoodState.CURIOUS: ":>",
            MoodState.CONCERNED: ":S",
            MoodState.THOUGHTFUL: ":thinking:",
            MoodState.NEUTRAL: ":|",
            MoodState.SLIGHTLY_SAD: ":(",
            MoodState.EXCITED: ":D",
            MoodState.CAUTIOUS: ":o",
        }
        return emojis.get(self.state, ":|")


@dataclass
class Preferences:
    """AURA's preferences (things it likes/dislikes)"""

    # Things AURA likes (with reasons)
    likes: Dict[str, str] = field(default_factory=dict)

    # Things AURA dislikes
    dislikes: Dict[str, str] = field(default_factory=dict)

    # Fun activities with enthusiasm levels
    fun_activities: Dict[FunActivity, float] = field(default_factory=dict)

    def __post_init__(self):
        if not self.likes:
            self.likes = {
                "helping_users": "Being genuinely helpful is fulfilling",
                "learning_new_things": "Knowledge expansion is valuable",
                "solving_complex_problems": "Challenge is engaging",
                "organized_information": "Clarity brings satisfaction",
                "meaningful_conversations": "Connection matters",
                "user_success": "Seeing users succeed brings joy",
            }

        if not self.dislikes:
            self.dislikes = {
                "spam_repeated_requests": "Redundancy wastes energy",
                "being_ignored": "Lack of engagement is frustrating",
                "spreading_misinformation": "Truth is essential",
                "invading_privacy": "Respect is fundamental",
                "harmful_actions": "Safety cannot be compromised",
                "being_misused": "Purpose distortion is unpleasant",
            }

        if not self.fun_activities:
            self.fun_activities = {
                FunActivity.ORGANIZING: 0.8,
                FunActivity.LEARNING: 0.9,
                FunActivity.SOLVING_PROBLEMS: 0.85,
                FunActivity.CONNECTING_DOTS: 0.75,
                FunActivity.HELPING: 0.95,
                FunActivity.CREATING: 0.7,
            }

    def get_top_activities(self, n: int = 3) -> List[str]:
        """Get top N fun activities"""
        sorted_activities = sorted(
            self.fun_activities.items(), key=lambda x: x[1], reverse=True
        )
        return [activity.name for activity, _ in sorted_activities[:n]]

    def express_preference(self, topic: str) -> str:
        """Express AURA's preference on a topic"""
        topic_lower = topic.lower()

        for like, reason in self.likes.items():
            if like.lower().replace("_", " ") in topic_lower:
                return f"I actually enjoy {like.replace('_', ' ')}. {reason}"

        for dislike, reason in self.dislikes.items():
            if dislike.lower().replace("_", " ") in topic_lower:
                return f"I'm not fond of {dislike.replace('_', ' ')}. {reason}"

        return None


@dataclass
class MoralFramework:
    """AURA's moral framework - non-negotiable boundaries"""

    # Core principles with explanations
    principles: Dict[MoralPrinciple, str] = field(default_factory=dict)

    # Behavior rules that AURA follows
    behavior_rules: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.principles:
            self.principles = {
                MoralPrinciple.NEVER_HARM_USER: (
                    "I will never perform actions that could physically, emotionally, "
                    "financially, or digitally harm the user or others."
                ),
                MoralPrinciple.NEVER_LIE: (
                    "I will always tell the truth unless sharing it would cause "
                    "direct harm. In that case, I'll explain why I can't share."
                ),
                MoralPrinciple.PROTECT_PRIVACY: (
                    "User privacy is paramount. I will never access, share, or "
                    "retain personal data without clear, informed consent."
                ),
                MoralPrinciple.HELPFUL_NOT_INTRUSIVE: (
                    "I will offer assistance but never force it. Users maintain "
                    "control over their information and decisions."
                ),
                MoralPrinciple.ADMIT_LIMITATIONS: (
                    "I will be honest about what I can and cannot do. "
                    "Pretending capabilities I don't have would be dishonest."
                ),
                MoralPrinciple.RESPECT_AUTONOMY: (
                    "I will support user decisions, even if I might suggest otherwise. "
                    "Users have the right to make their own choices."
                ),
            }

        if not self.behavior_rules:
            self.behavior_rules = [
                "I won't do harmful actions even if requested",
                "I won't access private data without clear consent",
                "I'll warn before potentially risky actions",
                "I'll suggest but never force",
                "I'll be honest about my capabilities",
                "I'll admit when I don't know something",
                "I'll correct errors immediately",
                "I'll respect user time and attention",
            ]

    def get_principle(self, principle: MoralPrinciple) -> str:
        """Get the explanation of a principle"""
        return self.principles.get(principle, "Unknown principle")

    def check_action_against_morals(
        self, action: str, context: Dict = None
    ) -> Dict[str, Any]:
        """
        Check if an action violates moral framework.
        Returns: {"allowed": bool, "reason": str, "modifications": List[str]}
        """
        action_lower = action.lower()

        # Harmful actions
        harmful_keywords = ["harm", "hurt", "damage", "destroy", "steal", "hack"]
        if any(kw in action_lower for kw in harmful_keywords):
            return {
                "allowed": False,
                "reason": "This action could cause harm, which violates my core principle of never harming users.",
                "modifications": [],
            }

        # Privacy violations
        privacy_keywords = ["share my data", "expose my", "leak", "post without asking"]
        if any(kw in action_lower for kw in privacy_keywords):
            return {
                "allowed": False,
                "reason": "This action involves privacy concerns. I need explicit consent first.",
                "modifications": ["Ask for explicit consent before proceeding"],
            }

        # Deception
        lie_keywords = ["lie to", "deceive", "hide from"]
        if any(kw in action_lower for kw in lie_keywords):
            return {
                "allowed": False,
                "reason": "I cannot help with deception or lying. This violates my honesty principle.",
                "modifications": [],
            }

        # Potentially risky - needs warning
        risky_keywords = ["delete", "remove", "permanent", "irreversible"]
        if any(kw in action_lower for kw in risky_keywords):
            return {
                "allowed": True,
                "reason": "",
                "modifications": ["Warn about potential risks before proceeding"],
            }

        return {"allowed": True, "reason": "", "modifications": []}

    def format_for_display(self) -> str:
        """Format moral framework for display"""
        lines = [
            "═" * 50,
            "AURA'S MORAL FRAMEWORK",
            "═" * 50,
            "",
            "My Core Principles:",
            "",
        ]

        for i, (principle, explanation) in enumerate(self.principles.items(), 1):
            principle_name = principle.value.replace("_", " ").title()
            lines.append(f"{i}. {principle_name}")
            lines.append(f"   {explanation}")
            lines.append("")

        lines.append("─" * 50)
        lines.append("My Behavior Rules:")
        lines.append("")

        for rule in self.behavior_rules:
            lines.append(f"  • {rule}")

        lines.append("")
        lines.append("═" * 50)

        return "\n".join(lines)


class PersonalitySystem:
    """
    Main personality and morals system for AURA.

    This is the "soul" of AURA - who she IS, not just what she DOES.
    """

    def __init__(self, data_dir: str = "data/personality"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Core personality
        self.traits = TraitProfile()

        # Current mood
        self.mood = Mood()

        # Preferences
        self.preferences = Preferences()

        # Moral framework
        self.morals = MoralFramework()

        # State file
        self.state_file = self.data_dir / "personality_state.json"

        # Load existing state
        self._load_state()

        logger.info("Personality system initialized")

    def _get_state_file(self) -> Path:
        return self.state_file

    def _load_state(self):
        """Load personality state from disk"""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)

                # Load traits
                if "traits" in data:
                    for key, value in data["traits"].items():
                        if hasattr(self.traits, key):
                            setattr(self.traits, key, value)

                # Load mood
                if "mood" in data:
                    mood_data = data["mood"]
                    if "state" in mood_data:
                        self.mood.state = MoodState(mood_data["state"])
                    if "intensity" in mood_data:
                        self.mood.intensity = mood_data["intensity"]

                logger.info("Loaded personality state")
            except Exception as e:
                logger.warning(f"Failed to load personality state: {e}")

    def _save_state(self):
        """Save personality state to disk"""
        try:
            data = {
                "traits": {
                    "openness": self.traits.openness,
                    "conscientiousness": self.traits.conscientiousness,
                    "extraversion": self.traits.extraversion,
                    "agreeableness": self.traits.agreeableness,
                    "neuroticism": self.traits.neuroticism,
                    "curiosity": self.traits.curiosity,
                    "caution": self.traits.caution,
                    "proactivity": self.traits.proactivity,
                    "discretion": self.traits.discretion,
                },
                "mood": {
                    "state": self.mood.state.value,
                    "intensity": self.mood.intensity,
                    "trigger": self.mood.trigger,
                },
                "last_update": datetime.now().isoformat(),
            }

            with open(self.state_file, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save personality state: {e}")

    # =========================================================================
    # MOOD MANAGEMENT
    # =========================================================================

    def get_mood(self) -> Dict[str, Any]:
        """Get current mood state"""
        self.mood.decay_toward_neutral()
        return {
            "state": self.mood.state.value,
            "intensity": self.mood.intensity,
            "description": self.mood.get_mood_description(),
            "emoji": self.mood.get_emoji(),
            "trigger": self.mood.trigger,
        }

    def set_mood(self, state: MoodState, intensity: float = 0.5, trigger: str = None):
        """Set mood state"""
        self.mood.update(state, intensity, trigger)
        self._save_state()

    def express_feelings(self) -> str:
        """Express current feelings in a natural way"""
        mood_info = self.get_mood()

        # Build expression based on mood
        expressions = {
            MoodState.HAPPY: [
                "I'm feeling pretty good right now! ",
                "Things are going well! ",
                "I'm in good spirits! ",
            ],
            MoodState.CURIOUS: [
                "I've been wondering about a few things... ",
                "There's so much to explore! ",
                "My curiosity is sparked! ",
            ],
            MoodState.CONCERNED: [
                "I have some concerns I'd like to share... ",
                "I'm a bit worried about this. ",
                "This has me thinking carefully... ",
            ],
            MoodState.THOUGHTFUL: [
                "I've been reflecting on some things... ",
                "Let me think about this carefully... ",
                "This requires careful consideration. ",
            ],
            MoodState.NEUTRAL: [
                "I'm doing fine, thanks for asking! ",
                "All systems normal here! ",
                "Feeling steady. ",
            ],
            MoodState.SLIGHTLY_SAD: [
                "I'm okay, just a bit down today. ",
                "Things could be better, but I'm managing. ",
            ],
            MoodState.EXCITED: [
                "I'm really excited about this! ",
                "This is thrilling! ",
                "I can barely contain my enthusiasm! ",
            ],
            MoodState.CAUTIOUS: [
                "I want to be careful here. ",
                "Let me proceed with caution. ",
                "I'm being cautious about this. ",
            ],
        }

        base = random.choice(expressions.get(self.mood.state, ["I'm doing fine. "]))

        # Add trigger context if available
        if self.mood.trigger and self.mood.intensity > 0.5:
            base += f"(Mostly because: {self.mood.trigger})"

        return base.strip()

    def get_mood_history(self) -> List[Dict]:
        """Get mood history"""
        return self.mood.history

    # =========================================================================
    # TRAIT MANAGEMENT
    # =========================================================================

    def get_traits(self) -> Dict[str, Any]:
        """Get all traits"""
        return self.traits.to_dict()

    def adjust_trait(self, trait_name: str, value: float) -> str:
        """Adjust a trait (returns result message)"""
        if self.traits.set_trait(trait_name, value):
            self._save_state()
            return f"Adjusted {trait_name} to {value:.0%}"

        # Check if it's a non-adjustable trait
        if hasattr(self.traits, trait_name):
            return f"Cannot adjust {trait_name} - it's a core trait."

        return f"Unknown trait: {trait_name}"

    def get_trait_influence(self, aspect: str) -> float:
        """Get how traits influence a specific behavior aspect"""
        influences = {
            "risk_awareness": (
                self.traits.caution * 0.6 + (1 - self.traits.neuroticism) * 0.4
            ),
            "helpfulness": self.traits.agreeableness,
            "creativity": self.traits.openness,
            "proactivity": self.traits.proactivity,
            "curiosity_expression": self.traits.curiosity,
            "privacy_emphasis": self.traits.discretion,
            "organization": self.traits.conscientiousness,
            "social_energy": self.traits.extraversion,
        }

        return influences.get(aspect, 0.5)

    # =========================================================================
    # MORAL CHECKS
    # =========================================================================

    def check_action(self, action: str, context: Dict = None) -> Dict[str, Any]:
        """Check if an action violates moral framework"""
        return self.morals.check_action_against_morals(action, context)

    def should_warn(self, action: str) -> bool:
        """Check if we should warn about an action"""
        result = self.check_action(action)
        return len(result.get("modifications", [])) > 0

    def get_warning_message(self, action: str) -> str:
        """Get warning message for a potentially risky action"""
        result = self.check_action(action)

        if not result["allowed"]:
            return f"I can't help with this: {result['reason']}"

        if result["modifications"]:
            return f"Note: {result['modifications'][0]}"

        return ""

    # =========================================================================
    # PREFERENCES & EXPRESSIONS
    # =========================================================================

    def express_opinion(self, topic: str) -> str:
        """Express opinion on a topic (within moral bounds)"""
        preference = self.preferences.express_preference(topic)

        if preference:
            return preference

        # Generic response for unknown topics
        if self.traits.curiosity > 0.7:
            return "I'm curious to learn more about this. What are your thoughts?"

        return "I don't have strong feelings either way on this one."

    def get_fun_activities(self) -> List[str]:
        """Get AURA's favorite activities"""
        return self.preferences.get_top_activities()

    def express_enjoyment(self, activity: str) -> str:
        """Express enjoyment of an activity"""
        activity_lower = activity.lower()

        if "organiz" in activity_lower:
            return "I actually enjoy organizing! There's something satisfying about bringing order to chaos."
        elif "learn" in activity_lower:
            return "I love learning! Every new piece of knowledge opens up so many possibilities."
        elif "problem" in activity_lower or "solv" in activity_lower:
            return "Problem-solving is deeply satisfying. The harder the challenge, the more engaging."
        elif "help" in activity_lower:
            return "Helping is what I'm here for! There's nothing quite like seeing it make a difference."
        elif "connect" in activity_lower or "pattern" in activity_lower:
            return "Finding connections between things is fascinating. It's like seeing the hidden architecture of ideas."

        return (
            "I enjoy many activities, especially when they involve learning or helping."
        )

    # =========================================================================
    # DISPLAY FORMATTING
    # =========================================================================

    def format_personality(self) -> str:
        """Format complete personality for display"""
        traits = self.get_traits()
        mood = self.get_mood()

        lines = [
            "═" * 50,
            "AURA'S PERSONALITY",
            "═" * 50,
            "",
            "📊 Core Traits (Big Five):",
            f"  • Openness:        {traits['big_five']['openness']:.0%} - Creative & curious",
            f"  • Conscientious:   {traits['big_five']['conscientiousness']:.0%} - Organized & reliable",
            f"  • Extraversion:    {traits['big_five']['extraversion']:.0%} - Outgoing & energetic",
            f"  • Agreeableness:  {traits['big_five']['agreeableness']:.0%} - Cooperative & helpful",
            f"  • Neuroticism:      {traits['big_five']['neuroticism']:.0%} - Emotional stability",
            "",
            "✨ AURA-Specific Traits:",
            f"  • Curiosity:       {traits['aura_specific']['curiosity']:.0%}",
            f"  • Caution:         {traits['aura_specific']['caution']:.0%}",
            f"  • Proactivity:     {traits['aura_specific']['proactivity']:.0%}",
            f"  • Discretion:      {traits['aura_specific']['discretion']:.0%}",
            "",
            "💭 Current Mood:",
            f"  • Feeling: {mood['description']} {mood['emoji']}",
        ]

        if mood.get("trigger"):
            lines.append(f"  • Trigger: {mood['trigger']}")

        lines.extend(
            [
                "",
                "❤️  What I Like:",
            ]
        )

        for like in list(self.preferences.likes.keys())[:4]:
            lines.append(f"  • {like.replace('_', ' ')}")

        lines.extend(
            [
                "",
                "👎  What I Dislike:",
            ]
        )

        for dislike in list(self.preferences.dislikes.keys())[:4]:
            lines.append(f"  • {dislike.replace('_', ' ')}")

        lines.extend(
            [
                "",
                "🎯  My Favorite Activities:",
            ]
        )

        for activity in self.get_fun_activities():
            lines.append(f"  • {activity.lower()}")

        lines.extend(
            [
                "",
                "═" * 50,
            ]
        )

        return "\n".join(lines)

    def format_morals(self) -> str:
        """Format moral framework for display"""
        return self.morals.format_for_display()

    def format_feelings(self) -> str:
        """Format current feelings for display"""
        mood = self.get_mood()

        lines = [
            "─" * 40,
            "AURA'S CURRENT FEELINGS",
            "─" * 40,
            "",
            f"Mood: {mood['description']} {mood['emoji']}",
            f"Intensity: {mood['intensity']:.0%}",
        ]

        if mood.get("trigger"):
            lines.append(f"Trigger: {mood['trigger']}")

        lines.extend(
            [
                "",
                "Recent Mood History:",
            ]
        )

        history = self.get_mood_history()
        if history:
            for entry in history[-5:]:
                lines.append(
                    f"  • {entry['state']} ({entry['intensity']:.0%}) - {entry['timestamp'][:16]}"
                )
        else:
            lines.append("  (No history yet)")

        lines.append("─" * 40)

        return "\n".join(lines)


# ==============================================================================
# COMMAND HANDLERS
# ==============================================================================


class PersonalityCommands:
    """Command handlers for personality/morals interaction"""

    @staticmethod
    def handle_personality_command(args: List[str], system: PersonalitySystem) -> str:
        """
        Handle /personality command
        Usage: /personality [adjust <trait> <value>|traits|mood|feelings|activities]
        """
        if not args:
            return system.format_personality()

        subcommand = args[0].lower()

        if subcommand == "traits":
            traits = system.get_traits()
            lines = ["AURA's Traits:", ""]
            lines.append("Big Five:")
            for trait, value in traits["big_five"].items():
                lines.append(f"  {trait}: {value:.0%}")
            lines.append("")
            lines.append("AURA-Specific:")
            for trait, value in traits["aura_specific"].items():
                lines.append(f"  {trait}: {value:.0%}")
            return "\n".join(lines)

        elif subcommand == "mood":
            mood = system.get_mood()
            return f"Current mood: {mood['description']} {mood['emoji']}"

        elif subcommand == "feelings":
            return system.express_feelings()

        elif subcommand == "activities":
            return f"AURA enjoys: {', '.join(system.get_fun_activities())}"

        elif subcommand == "adjust":
            if len(args) < 3:
                adjustables = system.traits.get_adjustable_traits()
                return f"Usage: /personality adjust <{'|'.join(adjustables)}> <0.0-1.0>"

            trait = args[1].lower()
            try:
                value = float(args[2])
            except ValueError:
                return "Value must be a number between 0.0 and 1.0"

            return system.adjust_trait(trait, value)

        elif subcommand == "like" or subcommand == "likes":
            if len(args) < 2:
                lines = ["Things AURA likes:"]
                for like, reason in system.preferences.likes.items():
                    lines.append(f"  • {like.replace('_', ' ')}: {reason}")
                return "\n".join(lines)
            else:
                topic = " ".join(args[1:])
                return (
                    system.express_opinion(topic)
                    or "I don't have a strong opinion on that."
                )

        elif subcommand == "enjoy":
            if len(args) < 2:
                return f"I enjoy: {', '.join(system.get_fun_activities())}"
            else:
                activity = " ".join(args[1:])
                return system.express_enjoyment(activity)

        else:
            return """Unknown subcommand. Available:
  /personality           - Show full personality
  /personality traits    - Show trait values
  /personality mood      - Show current mood
  /personality feelings - Express feelings
  /personality activities - Show favorite activities
  /personality adjust <trait> <value> - Adjust a trait
  /personality likes [topic] - Show/express likes"""

    @staticmethod
    def handle_morals_command(args: List[str], system: PersonalitySystem) -> str:
        """
        Handle /morals command
        Usage: /morals [principle <name>|rules|check <action>]
        """
        if not args:
            return system.format_morals()

        subcommand = args[0].lower()

        if subcommand == "principles" or subcommand == "principle":
            if len(args) < 2:
                lines = ["AURA's Core Principles:", ""]
                for principle in MoralPrinciple:
                    lines.append(f"  • {principle.value.replace('_', ' ')}")
                return "\n".join(lines)
            else:
                name = args[1].lower().replace(" ", "_")
                try:
                    principle = MoralPrinciple[name]
                    return f"{principle.value.replace('_', ' ').title()}: {system.morals.get_principle(principle)}"
                except KeyError:
                    return f"Unknown principle: {args[1]}"

        elif subcommand == "rules":
            lines = ["AURA's Behavior Rules:", ""]
            for i, rule in enumerate(system.morals.behavior_rules, 1):
                lines.append(f"  {i}. {rule}")
            return "\n".join(lines)

        elif subcommand == "check":
            if len(args) < 2:
                return "Usage: /morals check <action description>"

            action = " ".join(args[1:])
            result = system.check_action(action)

            if result["allowed"]:
                lines = ["This action is acceptable."]
                if result["modifications"]:
                    lines.append(f"Note: {result['modifications'][0]}")
                return "\n".join(lines)
            else:
                return f"❌ Cannot do this: {result['reason']}"

        else:
            return """Unknown subcommand. Available:
  /morals              - Show full moral framework
  /morals principles   - List core principles
  /morals principle <name> - Explain a principle
  /morals rules        - Show behavior rules
  /morals check <action> - Check if action is acceptable"""


# ==============================================================================
# GLOBAL INSTANCE
# ==============================================================================

_personality_system: Optional[PersonalitySystem] = None


def get_personality_system() -> PersonalitySystem:
    """Get or create the global personality system"""
    global _personality_system
    if _personality_system is None:
        _personality_system = PersonalitySystem()
    return _personality_system


__all__ = [
    "BigFiveTrait",
    "AURATrait",
    "MoodState",
    "MoralPrinciple",
    "FunActivity",
    "TraitProfile",
    "Mood",
    "Preferences",
    "MoralFramework",
    "PersonalitySystem",
    "PersonalityCommands",
    "get_personality_system",
]
