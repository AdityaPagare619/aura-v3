"""
Tests for Conversation Learning System
======================================

Tests the real learning extraction from conversations:
- Preference extraction
- Interest extraction
- Communication style analysis
- Context-appropriate intros
- Memory integration
"""

import pytest
from datetime import datetime
from typing import List, Dict

from src.core.conversation import (
    EmotionalConversationEngine,
    MessageContext,
    ConversationPhase,
    Tone,
    LearningType,
    Learning,
    ConversationAnalysis,
    RelationshipStage,
)


class TestLearningExtraction:
    """Test extraction of learnings from conversation text"""

    @pytest.fixture
    def engine(self):
        return EmotionalConversationEngine()

    def test_extract_positive_preferences(self, engine):
        """Test extraction of positive preferences"""
        text = "I really like Python for scripting. I love automated testing."
        learnings = engine._extract_preferences(text)

        assert len(learnings) >= 2
        assert all(l.learning_type == LearningType.PREFERENCE for l in learnings)

        # Check that Python and testing were extracted
        keys = [l.key for l in learnings]
        assert any("python" in k for k in keys)
        assert any("testing" in k or "automated" in k for k in keys)

    def test_extract_negative_preferences(self, engine):
        """Test extraction of negative preferences"""
        text = "I don't like verbose output. I hate waiting for slow builds."
        learnings = engine._extract_preferences(text)

        assert len(learnings) >= 1
        # Check sentiment is negative
        for learning in learnings:
            assert learning.value.get("sentiment") in ["negative", "habit"]

    def test_extract_habits(self, engine):
        """Test extraction of habitual patterns"""
        text = "I always run tests before committing. I usually work late at night."
        learnings = engine._extract_preferences(text)

        assert len(learnings) >= 1
        habits = [l for l in learnings if l.value.get("sentiment") == "habit"]
        assert len(habits) >= 1

    def test_extract_interests_explicit(self, engine):
        """Test extraction of explicit interest statements"""
        text = (
            "I'm interested in machine learning. I'm curious about quantum computing."
        )
        learnings = engine._extract_interests(text)

        assert len(learnings) >= 1
        assert all(l.learning_type == LearningType.INTEREST for l in learnings)

    def test_extract_interests_from_questions(self, engine):
        """Test extraction of interests from questions asked"""
        text = "How does async/await work in Python? What is the best way to handle errors?"
        learnings = engine._extract_interests(text)

        # Should have question-based interests
        question_interests = [l for l in learnings if l.value.get("type") == "question"]
        assert len(question_interests) >= 1

    def test_extract_style_formal(self, engine):
        """Test extraction of formal communication style"""
        text = "I would appreciate if you could please provide the documentation. Thank you kindly."
        learning = engine._extract_style(text)

        assert learning is not None
        assert learning.learning_type == LearningType.STYLE
        assert learning.value.get("formality") == "formal"

    def test_extract_style_casual(self, engine):
        """Test extraction of casual communication style"""
        text = "yeah gonna do that later lol. btw can u check this out?"
        learning = engine._extract_style(text)

        assert learning is not None
        assert learning.value.get("formality") == "casual"

    def test_extract_style_emoji_usage(self, engine):
        """Test detection of emoji usage in style"""
        text = "That's great! 😊 Thanks so much! 👍"
        learning = engine._extract_style(text)

        assert learning is not None
        assert learning.value.get("emoji_usage") in ["moderate", "high"]

    def test_extract_style_brevity_brief(self, engine):
        """Test detection of brief communication style"""
        text = "Yes. Got it. Done. Thanks!"
        learning = engine._extract_style(text)

        assert learning is not None
        assert learning.value.get("brevity") == "brief"

    def test_extract_style_brevity_detailed(self, engine):
        """Test detection of detailed communication style"""
        text = """I was thinking about this problem for a while and I believe that 
        the best approach would be to first analyze the existing data structures 
        and then refactor them systematically to improve performance and 
        maintainability while ensuring backwards compatibility."""
        learning = engine._extract_style(text)

        assert learning is not None
        assert learning.value.get("brevity") == "detailed"


class TestLearnFromConversation:
    """Test the main learn_from_conversation method"""

    @pytest.fixture
    def engine(self):
        return EmotionalConversationEngine()

    def test_learn_returns_learnings(self, engine):
        """Test that learning extraction returns list of learnings"""
        learnings = engine.learn_from_conversation(
            user_feedback="Thanks, that's perfect!",
            aura_response="Here's the code you requested.",
            user_response="I really like how you formatted it. I prefer concise examples.",
        )

        assert isinstance(learnings, list)
        # Should have extracted at least a preference
        assert any(l.learning_type == LearningType.PREFERENCE for l in learnings)

    def test_learn_empty_input(self, engine):
        """Test handling of empty input"""
        learnings = engine.learn_from_conversation("", "", "")
        assert learnings == []

    def test_learn_confidence_scores(self, engine):
        """Test that learnings have reasonable confidence scores"""
        learnings = engine.learn_from_conversation(
            user_feedback="",
            aura_response="",
            user_response="I like Python. I'm interested in data science.",
        )

        for learning in learnings:
            assert 0.0 <= learning.confidence <= 1.0


class TestExtractLearningsFromConversation:
    """Test extraction from full conversation history"""

    @pytest.fixture
    def engine(self):
        return EmotionalConversationEngine()

    def test_extract_from_conversation_list(self, engine):
        """Test extracting learnings from conversation list"""
        conversation = [
            {"role": "user", "content": "I really like TypeScript over JavaScript."},
            {"role": "assistant", "content": "TypeScript has great type safety."},
            {"role": "user", "content": "How does it handle generics?"},
            {"role": "assistant", "content": "Here's how generics work..."},
            {"role": "user", "content": "I prefer concise examples. Thanks!"},
        ]

        learnings = engine.extract_learnings(conversation)

        # Should have preferences and interests
        assert len(learnings) > 0
        types = {l.learning_type for l in learnings}
        assert LearningType.PREFERENCE in types or LearningType.INTEREST in types

    def test_extract_deduplicates(self, engine):
        """Test that duplicate learnings are removed"""
        conversation = [
            {"role": "user", "content": "I like Python. I really like Python."},
            {"role": "assistant", "content": "Python is great!"},
            {"role": "user", "content": "I like Python programming."},
        ]

        learnings = engine.extract_learnings(conversation)

        # Should deduplicate similar learnings
        python_learnings = [l for l in learnings if "python" in l.key.lower()]
        assert len(python_learnings) <= 2  # Some deduplication should occur


class TestConversationAnalysis:
    """Test comprehensive conversation analysis"""

    @pytest.fixture
    def engine(self):
        return EmotionalConversationEngine()

    def test_analyze_returns_analysis_object(self, engine):
        """Test that analysis returns proper ConversationAnalysis"""
        conversation = [
            {"role": "user", "content": "I love functional programming!"},
            {"role": "assistant", "content": "FP has many benefits."},
            {"role": "user", "content": "What about immutability? That's great!"},
        ]

        analysis = engine.analyze_conversation(conversation)

        assert isinstance(analysis, ConversationAnalysis)
        assert isinstance(analysis.preferences, list)
        assert isinstance(analysis.interests, list)
        assert isinstance(analysis.style, dict)
        assert analysis.sentiment in ["positive", "negative", "neutral"]
        assert 0.0 <= analysis.engagement_level <= 1.0

    def test_analyze_sentiment_positive(self, engine):
        """Test positive sentiment detection"""
        conversation = [
            {"role": "user", "content": "Thanks! That's perfect and awesome!"},
            {"role": "assistant", "content": "Glad to help!"},
            {"role": "user", "content": "Great work, I love it!"},
        ]

        analysis = engine.analyze_conversation(conversation)
        assert analysis.sentiment == "positive"

    def test_analyze_sentiment_negative(self, engine):
        """Test negative sentiment detection"""
        conversation = [
            {"role": "user", "content": "This is wrong and frustrating."},
            {"role": "assistant", "content": "Let me fix that."},
            {"role": "user", "content": "I hate when this happens, it's terrible."},
        ]

        analysis = engine.analyze_conversation(conversation)
        assert analysis.sentiment == "negative"


class TestContextAppropriateIntros:
    """Test context-appropriate intro generation"""

    @pytest.fixture
    def engine(self):
        return EmotionalConversationEngine()

    def test_morning_greeting_new_user(self, engine):
        """Test morning greeting for new user"""
        context = MessageContext(
            user_message="Hello!",
            current_hour=9,
            interaction_count=2,
            is_first_message=True,
            conversation_phase=ConversationPhase.GREETING,
        )

        intro = engine._get_context_appropriate_intro(context)
        # Should be a morning greeting
        assert intro  # Not empty
        # New user should get more formal greeting
        assert any(w in intro.lower() for w in ["morning", "hello", "good"])

    def test_evening_greeting_established_user(self, engine):
        """Test evening greeting for established user"""
        context = MessageContext(
            user_message="Hi!",
            current_hour=20,
            interaction_count=150,
            is_first_message=True,
            conversation_phase=ConversationPhase.GREETING,
        )

        intro = engine._get_context_appropriate_intro(context)
        # Established users can get casual greetings
        assert intro  # Not empty

    def test_continuation_same_topic(self, engine):
        """Test continuation with same topic"""
        context = MessageContext(
            user_message="What about the Python code?",
            last_topic="Python implementation",
            time_since_last_message=2,  # 2 minutes ago
            interaction_count=50,
        )

        intro = engine._get_context_appropriate_intro(context)
        # Should be a continuation phrase or empty
        assert intro in [
            "Continuing with that...",
            "Back to that...",
            "So, about that...",
            "Sure!",
            "Alright!",
            "Let's see...",
            "",
        ]

    def test_return_after_long_absence(self, engine):
        """Test greeting after long absence"""
        context = MessageContext(
            user_message="Hey!",
            time_since_last_message=2880,  # 2 days
            interaction_count=30,
        )

        intro = engine._get_context_appropriate_intro(context)
        # Should acknowledge the return
        assert intro  # Should have some greeting


class TestRelationshipStage:
    """Test relationship stage determination"""

    @pytest.fixture
    def engine(self):
        return EmotionalConversationEngine()

    def test_new_user_stage(self, engine):
        """Test new user relationship stage"""
        stage = engine._get_relationship_stage(3)
        assert stage == RelationshipStage.NEW

    def test_acquaintance_stage(self, engine):
        """Test acquaintance relationship stage"""
        stage = engine._get_relationship_stage(10)
        assert stage == RelationshipStage.ACQUAINTANCE

    def test_familiar_stage(self, engine):
        """Test familiar relationship stage"""
        stage = engine._get_relationship_stage(50)
        assert stage == RelationshipStage.FAMILIAR

    def test_established_stage(self, engine):
        """Test established relationship stage"""
        stage = engine._get_relationship_stage(200)
        assert stage == RelationshipStage.ESTABLISHED


class TestSentimentAnalysis:
    """Test sentiment analysis"""

    @pytest.fixture
    def engine(self):
        return EmotionalConversationEngine()

    def test_positive_sentiment(self, engine):
        """Test positive sentiment detection"""
        result = engine._analyze_sentiment(
            "Great job! Thanks so much, this is perfect!"
        )
        assert result == "positive"

    def test_negative_sentiment(self, engine):
        """Test negative sentiment detection"""
        result = engine._analyze_sentiment("This is wrong and terrible, I hate it.")
        assert result == "negative"

    def test_neutral_sentiment(self, engine):
        """Test neutral sentiment detection"""
        result = engine._analyze_sentiment("Please run the build process.")
        assert result == "neutral"


class TestLearningToDict:
    """Test Learning dataclass serialization"""

    def test_learning_to_dict(self):
        """Test converting learning to dictionary"""
        learning = Learning(
            learning_type=LearningType.PREFERENCE,
            key="python",
            value={"sentiment": "positive", "original": "Python"},
            confidence=0.8,
            source="I like Python",
        )

        result = learning.to_dict()

        assert result["type"] == "preference"
        assert result["key"] == "python"
        assert result["value"]["sentiment"] == "positive"
        assert result["confidence"] == 0.8
        assert "timestamp" in result


class TestMemoryIntegration:
    """Test memory system integration (graceful fallback)"""

    @pytest.fixture
    def engine(self):
        return EmotionalConversationEngine()

    def test_store_learnings_graceful_fallback(self, engine):
        """Test that memory storage fails gracefully when not available"""
        learnings = [
            Learning(
                learning_type=LearningType.PREFERENCE,
                key="test",
                value={"sentiment": "positive"},
                confidence=0.5,
            )
        ]

        # Should not raise even if memory not available
        engine._store_learnings_in_memory(learnings, "positive")
        # If we get here without exception, test passes

    def test_learn_returns_learnings_without_memory(self, engine):
        """Test learning works even without memory system"""
        # Ensure memory coordinator is None
        engine._memory_coordinator = None

        learnings = engine.learn_from_conversation(
            user_feedback="Thanks!",
            aura_response="Here you go.",
            user_response="I like this approach.",
        )

        # Should still return learnings
        assert isinstance(learnings, list)
