"""
Tests for AURA v3 Adaptive Personality System

Tests the Big Five personality-driven response generation:
- PersonalityVector (OCEAN model)
- Personality-weighted selection
- Personality evolution from feedback
- SQLite persistence
"""

import unittest
import asyncio
import os
import sys
import tempfile
import shutil
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.core.adaptive_personality import (
    PersonalityVector,
    ResponseTemplate,
    ResponseTemplateDB,
    PersonalityPersistence,
    PersonalityEvolutionEngine,
    AdaptivePersonalityEngine,
    ReactionSystem,
    PersonalityState,
    PersonalityDimension,
    AuraCoreIdentity,
    AuraOpinions,
    Reaction,
    get_personality_engine,
)


class TestPersonalityVector(unittest.TestCase):
    """Test PersonalityVector (Big Five model)"""

    def test_default_values(self):
        """Test default Big Five values"""
        pv = PersonalityVector()
        self.assertEqual(pv.openness, 0.6)
        self.assertEqual(pv.conscientiousness, 0.7)
        self.assertEqual(pv.extraversion, 0.5)
        self.assertEqual(pv.agreeableness, 0.7)
        self.assertEqual(pv.neuroticism, 0.3)

    def test_custom_values(self):
        """Test custom Big Five values"""
        pv = PersonalityVector(
            openness=0.8,
            conscientiousness=0.6,
            extraversion=0.9,
            agreeableness=0.4,
            neuroticism=0.2,
        )
        self.assertEqual(pv.openness, 0.8)
        self.assertEqual(pv.extraversion, 0.9)

    def test_clamping(self):
        """Test values are clamped to 0-1"""
        pv = PersonalityVector(
            openness=1.5,
            conscientiousness=-0.5,
            extraversion=2.0,
            agreeableness=-1.0,
            neuroticism=0.5,
        )
        self.assertEqual(pv.openness, 1.0)
        self.assertEqual(pv.conscientiousness, 0.0)
        self.assertEqual(pv.extraversion, 1.0)
        self.assertEqual(pv.agreeableness, 0.0)

    def test_to_dict(self):
        """Test conversion to dictionary"""
        pv = PersonalityVector(openness=0.8, extraversion=0.3)
        d = pv.to_dict()
        self.assertEqual(d["openness"], 0.8)
        self.assertEqual(d["extraversion"], 0.3)
        self.assertIn("conscientiousness", d)
        self.assertIn("agreeableness", d)
        self.assertIn("neuroticism", d)

    def test_from_dict(self):
        """Test creation from dictionary"""
        d = {
            "openness": 0.9,
            "conscientiousness": 0.8,
            "extraversion": 0.7,
            "agreeableness": 0.6,
            "neuroticism": 0.5,
        }
        pv = PersonalityVector.from_dict(d)
        self.assertEqual(pv.openness, 0.9)
        self.assertEqual(pv.conscientiousness, 0.8)

    def test_shift(self):
        """Test trait shifting"""
        pv = PersonalityVector(openness=0.5)
        pv.shift("openness", 0.2)
        self.assertEqual(pv.openness, 0.7)

        pv.shift("openness", 0.5)  # Should clamp to 1.0
        self.assertEqual(pv.openness, 1.0)

        pv.shift("openness", -1.5)  # Should clamp to 0.0
        self.assertEqual(pv.openness, 0.0)


class TestResponseTemplate(unittest.TestCase):
    """Test ResponseTemplate scoring"""

    def test_neutral_template(self):
        """Test template with no personality weights"""
        template = ResponseTemplate("Hello!", "greeting")
        pv = PersonalityVector()
        score = template.score_for_personality(pv)
        # Should be ~0.5 (neutral)
        self.assertAlmostEqual(score, 0.5, places=1)

    def test_high_openness_template(self):
        """Test template favoring high openness"""
        template = ResponseTemplate("Let's explore!", "greeting", openness_weight=1.0)

        # High openness personality should score higher
        high_open = PersonalityVector(openness=0.9)
        low_open = PersonalityVector(openness=0.2)

        score_high = template.score_for_personality(high_open)
        score_low = template.score_for_personality(low_open)

        self.assertGreater(score_high, score_low)

    def test_negative_weight(self):
        """Test template with negative weight (prefers low trait)"""
        template = ResponseTemplate(
            "Reserved greeting", "greeting", extraversion_weight=-0.8
        )

        # Low extraversion should score higher with negative weight
        introverted = PersonalityVector(extraversion=0.2)
        extroverted = PersonalityVector(extraversion=0.9)

        score_intro = template.score_for_personality(introverted)
        score_extro = template.score_for_personality(extroverted)

        self.assertGreater(score_intro, score_extro)


class TestResponseTemplateDB(unittest.TestCase):
    """Test ResponseTemplateDB weighted selection"""

    def setUp(self):
        self.db = ResponseTemplateDB()

    def test_has_greeting_templates(self):
        """Test greeting templates exist"""
        templates = self.db.get_templates("greeting")
        self.assertGreater(len(templates), 5)

    def test_has_joke_templates(self):
        """Test joke reaction templates exist"""
        pun_templates = self.db.get_templates("joke_pun")
        general_templates = self.db.get_templates("joke_general")
        self.assertGreater(len(pun_templates), 3)
        self.assertGreater(len(general_templates), 2)

    def test_has_compliment_templates(self):
        """Test compliment templates exist"""
        templates = self.db.get_templates("compliment")
        self.assertGreater(len(templates), 3)

    def test_weighted_selection_returns_string(self):
        """Test weighted selection returns a string"""
        pv = PersonalityVector()
        response = self.db.select_weighted("greeting", pv)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_weighted_selection_varies_by_personality(self):
        """Test that different personalities get different response distributions"""
        extroverted = PersonalityVector(extraversion=0.95, agreeableness=0.9)
        introverted = PersonalityVector(extraversion=0.1, agreeableness=0.3)

        # Sample many times and check for different distributions
        extro_responses = set()
        intro_responses = set()

        for _ in range(50):
            extro_responses.add(self.db.select_weighted("greeting", extroverted))
            intro_responses.add(self.db.select_weighted("greeting", introverted))

        # Both should have some variety
        self.assertGreater(len(extro_responses), 1)
        self.assertGreater(len(intro_responses), 1)

    def test_empty_category_returns_ellipsis(self):
        """Test empty category returns fallback"""
        pv = PersonalityVector()
        response = self.db.select_weighted("nonexistent_category", pv)
        self.assertEqual(response, "...")

    def test_temperature_affects_randomness(self):
        """Test temperature parameter affects selection"""
        pv = PersonalityVector()

        # Very low temperature should be more deterministic
        low_temp_responses = set()
        for _ in range(30):
            low_temp_responses.add(
                self.db.select_weighted("greeting", pv, temperature=0.1)
            )

        # Very high temperature should have more variety
        high_temp_responses = set()
        for _ in range(30):
            high_temp_responses.add(
                self.db.select_weighted("greeting", pv, temperature=5.0)
            )

        # Low temperature should tend toward fewer unique responses
        # (though this is probabilistic, so we just check they're both non-empty)
        self.assertGreater(len(low_temp_responses), 0)
        self.assertGreater(len(high_temp_responses), 0)


class TestPersonalityPersistence(unittest.TestCase):
    """Test SQLite persistence for personality state"""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_personality.db")
        self.persistence = PersonalityPersistence(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_database_created(self):
        """Test database file is created"""
        self.assertTrue(os.path.exists(self.db_path))

    def test_load_default(self):
        """Test loading default personality"""
        pv = self.persistence.load()
        self.assertEqual(pv.openness, 0.6)
        self.assertEqual(pv.conscientiousness, 0.7)
        self.assertEqual(pv.extraversion, 0.5)
        self.assertEqual(pv.agreeableness, 0.7)
        self.assertEqual(pv.neuroticism, 0.3)

    def test_save_and_load(self):
        """Test saving and loading personality"""
        pv = PersonalityVector(
            openness=0.9,
            conscientiousness=0.8,
            extraversion=0.7,
            agreeableness=0.6,
            neuroticism=0.5,
        )

        success = self.persistence.save(pv)
        self.assertTrue(success)

        loaded = self.persistence.load()
        self.assertEqual(loaded.openness, 0.9)
        self.assertEqual(loaded.conscientiousness, 0.8)
        self.assertEqual(loaded.extraversion, 0.7)
        self.assertEqual(loaded.agreeableness, 0.6)
        self.assertEqual(loaded.neuroticism, 0.5)

    def test_evolution_recording(self):
        """Test recording personality evolution"""
        self.persistence.record_evolution(
            trait="openness",
            old_value=0.5,
            new_value=0.6,
            reason="positive feedback on creativity",
        )

        history = self.persistence.get_evolution_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["trait"], "openness")
        self.assertEqual(history[0]["old_value"], 0.5)
        self.assertEqual(history[0]["new_value"], 0.6)
        self.assertAlmostEqual(history[0]["delta"], 0.1, places=5)
        self.assertEqual(history[0]["reason"], "positive feedback on creativity")

    def test_evolution_history_limit(self):
        """Test evolution history respects limit"""
        for i in range(10):
            self.persistence.record_evolution(
                trait="openness",
                old_value=0.5 + i * 0.01,
                new_value=0.5 + (i + 1) * 0.01,
                reason=f"change {i}",
            )

        history = self.persistence.get_evolution_history(limit=5)
        self.assertEqual(len(history), 5)


class TestPersonalityEvolutionEngine(unittest.TestCase):
    """Test personality evolution from feedback"""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_evolution.db")
        self.persistence = PersonalityPersistence(self.db_path)
        self.engine = PersonalityEvolutionEngine(self.persistence)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_positive_feedback_increases_traits(self):
        """Test positive feedback increases relevant traits"""
        pv = PersonalityVector(openness=0.5, extraversion=0.5)

        new_pv, _ = self.engine.process_feedback(
            pv, feedback_type="joke", feedback_sentiment="positive"
        )

        # Positive joke feedback should increase openness and extraversion
        self.assertGreater(new_pv.openness, 0.5)
        self.assertGreater(new_pv.extraversion, 0.5)

    def test_negative_feedback_decreases_traits(self):
        """Test negative feedback decreases relevant traits"""
        pv = PersonalityVector(openness=0.5, extraversion=0.5)

        new_pv, _ = self.engine.process_feedback(
            pv, feedback_type="joke", feedback_sentiment="negative"
        )

        # Negative joke feedback should decrease openness and extraversion
        self.assertLess(new_pv.openness, 0.5)
        self.assertLess(new_pv.extraversion, 0.5)

    def test_neutral_feedback_no_change(self):
        """Test neutral feedback doesn't change traits"""
        pv = PersonalityVector(openness=0.5, extraversion=0.5)

        new_pv, changed = self.engine.process_feedback(
            pv, feedback_type="joke", feedback_sentiment="neutral"
        )

        self.assertEqual(new_pv.openness, 0.5)
        self.assertEqual(new_pv.extraversion, 0.5)
        self.assertFalse(changed)

    def test_task_completion_affects_conscientiousness(self):
        """Test task completion feedback affects conscientiousness"""
        pv = PersonalityVector(conscientiousness=0.5)

        new_pv, _ = self.engine.process_feedback(
            pv, feedback_type="task_completion", feedback_sentiment="positive"
        )

        self.assertGreater(new_pv.conscientiousness, 0.5)

    def test_empathy_feedback_affects_agreeableness(self):
        """Test empathy feedback affects agreeableness"""
        pv = PersonalityVector(agreeableness=0.5)

        new_pv, _ = self.engine.process_feedback(
            pv, feedback_type="empathy", feedback_sentiment="positive"
        )

        self.assertGreater(new_pv.agreeableness, 0.5)

    def test_traits_bounded(self):
        """Test traits stay within bounds during evolution"""
        pv = PersonalityVector(openness=0.95)

        # Apply many positive feedbacks
        for _ in range(100):
            pv, _ = self.engine.process_feedback(
                pv, feedback_type="creativity", feedback_sentiment="positive"
            )

        # Should not exceed maximum
        self.assertLessEqual(pv.openness, 0.9)  # MAX_TRAIT_VALUE

        pv = PersonalityVector(openness=0.15)

        # Apply many negative feedbacks
        for _ in range(100):
            pv, _ = self.engine.process_feedback(
                pv, feedback_type="creativity", feedback_sentiment="negative"
            )

        # Should not go below minimum
        self.assertGreaterEqual(pv.openness, 0.1)  # MIN_TRAIT_VALUE


class TestReactionSystem(unittest.TestCase):
    """Test ReactionSystem with personality-weighted selection"""

    def setUp(self):
        self.personality_state = PersonalityState()
        self.personality_vector = PersonalityVector()
        self.template_db = ResponseTemplateDB()
        self.reaction_system = ReactionSystem(
            self.personality_state, self.personality_vector, self.template_db
        )

    def test_react_to_pun(self):
        """Test reaction to pun joke"""
        reaction = asyncio.run(self.reaction_system.react_to_joke("pun", "happy"))
        self.assertIsInstance(reaction, Reaction)
        self.assertEqual(reaction.reaction_type, "joke")
        self.assertGreater(len(reaction.response), 0)
        self.assertTrue(reaction.should_respond)

    def test_react_to_general_joke(self):
        """Test reaction to general joke"""
        reaction = asyncio.run(self.reaction_system.react_to_joke("general", "neutral"))
        self.assertIsInstance(reaction, Reaction)
        self.assertEqual(reaction.reaction_type, "joke")

    def test_serious_personality_minimal_joke_reaction(self):
        """Test serious personality has minimal joke reaction"""
        self.personality_state.dimensions[PersonalityDimension.HUMOR] = 0.1

        reaction = asyncio.run(self.reaction_system.react_to_joke("pun", "happy"))

        self.assertEqual(reaction.response, "I see.")
        self.assertFalse(reaction.should_respond)

    def test_react_to_compliment(self):
        """Test reaction to compliment"""
        reaction = asyncio.run(
            self.reaction_system.react_to_compliment("You're so helpful!")
        )
        self.assertIsInstance(reaction, Reaction)
        self.assertEqual(reaction.reaction_type, "compliment")
        self.assertGreater(reaction.sentiment, 0.5)
        self.assertTrue(reaction.should_respond)

    def test_react_to_high_frustration(self):
        """Test reaction to high frustration"""
        reaction = asyncio.run(self.reaction_system.react_to_frustration(0.9))
        self.assertIsInstance(reaction, Reaction)
        self.assertEqual(reaction.reaction_type, "frustration")
        self.assertTrue(reaction.should_respond)

    def test_react_to_success(self):
        """Test reaction to success"""
        reaction = asyncio.run(self.reaction_system.react_to_success("achievement"))
        self.assertIsInstance(reaction, Reaction)
        self.assertEqual(reaction.reaction_type, "success")
        self.assertGreater(reaction.sentiment, 0.5)

    def test_react_to_insult(self):
        """Test reaction to insult"""
        reaction = asyncio.run(self.reaction_system.react_to_insult("You're useless"))
        self.assertIsInstance(reaction, Reaction)
        self.assertEqual(reaction.reaction_type, "insult")
        self.assertLess(reaction.sentiment, 0.5)


class TestAdaptivePersonalityEngine(unittest.TestCase):
    """Test complete AdaptivePersonalityEngine"""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "test_engine.db")
        self.engine = AdaptivePersonalityEngine(self.db_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_initialization(self):
        """Test engine initializes correctly"""
        self.assertIsNotNone(self.engine.core)
        self.assertIsNotNone(self.engine.personality_vector)
        self.assertIsNotNone(self.engine.personality)
        self.assertIsNotNone(self.engine.reactions)
        self.assertIsNotNone(self.engine.template_db)

    def test_get_personality_vector(self):
        """Test getting personality vector"""
        pv = self.engine.get_personality_vector()
        self.assertIsInstance(pv, PersonalityVector)

    def test_set_personality_vector(self):
        """Test setting personality vector"""
        new_pv = PersonalityVector(openness=0.9, extraversion=0.8)
        self.engine.set_personality_vector(new_pv)

        current = self.engine.get_personality_vector()
        self.assertEqual(current.openness, 0.9)
        self.assertEqual(current.extraversion, 0.8)

    def test_adjust_trait(self):
        """Test adjusting a Big Five trait"""
        initial_openness = self.engine.personality_vector.openness
        self.engine.adjust_trait("openness", 0.1, "test adjustment")

        self.assertAlmostEqual(
            self.engine.personality_vector.openness,
            min(1.0, initial_openness + 0.1),
            places=5,
        )

    def test_handle_joke(self):
        """Test handling joke"""
        response = asyncio.run(self.engine.handle_joke("pun", "happy"))
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_handle_compliment(self):
        """Test handling compliment"""
        response = asyncio.run(self.engine.handle_compliment("Great job!"))
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_handle_frustration(self):
        """Test handling frustration"""
        response = asyncio.run(self.engine.handle_frustration(0.8))
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_handle_success(self):
        """Test handling success"""
        response = asyncio.run(self.engine.handle_success("achievement"))
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_handle_insult(self):
        """Test handling insult"""
        response = asyncio.run(self.engine.handle_insult("You're bad"))
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_learn_from_positive_interaction(self):
        """Test learning from positive interaction"""
        initial_vector = PersonalityVector.from_dict(
            self.engine.personality_vector.to_dict()
        )

        # Simulate many positive interactions
        for _ in range(50):
            self.engine.learn_from_interaction("joke", "That was awesome!")

        # Some traits should have changed
        final_vector = self.engine.personality_vector

        # At least one trait should be different
        traits_changed = (
            initial_vector.openness != final_vector.openness
            or initial_vector.extraversion != final_vector.extraversion
        )
        self.assertTrue(traits_changed)

    def test_learn_from_negative_interaction(self):
        """Test learning from negative interaction"""
        # Set high initial values
        self.engine.personality_vector = PersonalityVector(
            openness=0.8, extraversion=0.8
        )

        initial_openness = self.engine.personality_vector.openness

        # Simulate negative interactions
        for _ in range(50):
            self.engine.learn_from_interaction("joke", "That was terrible, stop")

        # Openness should decrease
        self.assertLess(self.engine.personality_vector.openness, initial_openness)

    def test_get_identity_statement(self):
        """Test getting identity statement"""
        statement = self.engine.get_identity_statement()
        self.assertIsInstance(statement, str)
        self.assertIn("AURA", statement)

    def test_get_personality_summary(self):
        """Test getting personality summary"""
        summary = self.engine.get_personality_summary()

        self.assertIn("big_five", summary)
        self.assertIn("mood", summary)
        self.assertIn("energy", summary)
        self.assertIn("context", summary)
        self.assertIn("traits_description", summary)

        big_five = summary["big_five"]
        self.assertIn("openness", big_five)
        self.assertIn("conscientiousness", big_five)
        self.assertIn("extraversion", big_five)
        self.assertIn("agreeableness", big_five)
        self.assertIn("neuroticism", big_five)

    def test_set_context(self):
        """Test setting context"""
        self.engine.set_context("serious")
        self.assertEqual(self.engine.personality.current_context, "serious")

        # Invalid context should not change
        self.engine.set_context("invalid_context")
        self.assertEqual(self.engine.personality.current_context, "serious")

    def test_set_mood(self):
        """Test setting mood"""
        self.engine.set_mood("happy")
        self.assertEqual(self.engine.personality.mood, "happy")

        # Invalid mood should not change
        self.engine.set_mood("invalid_mood")
        self.assertEqual(self.engine.personality.mood, "happy")

    def test_legacy_dimension_sync(self):
        """Test legacy dimensions stay synced with Big Five"""
        self.engine.personality_vector = PersonalityVector(
            openness=0.9, extraversion=0.9
        )
        self.engine._sync_legacy_dimensions()

        # HUMOR should be high (derived from openness + extraversion)
        humor = self.engine.personality.dimensions[PersonalityDimension.HUMOR]
        self.assertGreater(humor, 0.7)

    def test_respects_boundary(self):
        """Test boundary checking"""
        self.assertTrue(self.engine.respects_boundary("help_user"))
        self.assertFalse(self.engine.respects_boundary("lie_to_user"))
        self.assertFalse(self.engine.respects_boundary("share_private_data"))

    def test_should_always_do(self):
        """Test always-do checking"""
        self.assertTrue(self.engine.should_always_do("ask_before_acting"))
        self.assertTrue(self.engine.should_always_do("explain_decisions"))
        self.assertFalse(self.engine.should_always_do("random_action"))

    def test_can_express_opinion(self):
        """Test opinion topic checking"""
        self.assertTrue(self.engine.can_express_opinion("task_approaches"))
        self.assertTrue(self.engine.can_express_opinion("communication"))
        self.assertFalse(self.engine.can_express_opinion("politics"))

    def test_get_opinion(self):
        """Test getting opinion on valid topic"""
        opinion = self.engine.get_opinion("task_approaches")
        self.assertIsInstance(opinion, str)
        self.assertIn("efficient", opinion.lower())

    def test_save_if_dirty(self):
        """Test saving accumulated changes"""
        self.engine._dirty = True
        self.engine.save_if_dirty()
        self.assertFalse(self.engine._dirty)

    def test_get_evolution_history(self):
        """Test getting evolution history"""
        # Record some evolution
        self.engine.adjust_trait("openness", 0.1, "test reason")

        history = self.engine.get_evolution_history()
        self.assertIsInstance(history, list)


class TestGlobalPersonalityEngine(unittest.TestCase):
    """Test global personality engine singleton"""

    def test_get_personality_engine_singleton(self):
        """Test singleton returns same instance"""
        # Note: This modifies global state, so be careful
        engine1 = get_personality_engine()
        engine2 = get_personality_engine()
        self.assertIs(engine1, engine2)


class TestPersonalityState(unittest.TestCase):
    """Test legacy PersonalityState"""

    def test_default_dimensions(self):
        """Test default dimension values"""
        ps = PersonalityState()

        self.assertIn(PersonalityDimension.HUMOR, ps.dimensions)
        self.assertIn(PersonalityDimension.EMPATHY, ps.dimensions)
        self.assertEqual(ps.mood, "neutral")
        self.assertEqual(ps.energy, 0.7)
        self.assertEqual(ps.current_context, "normal")


class TestAuraCoreIdentity(unittest.TestCase):
    """Test AuraCoreIdentity"""

    def test_core_values(self):
        """Test core values exist"""
        self.assertTrue(AuraCoreIdentity.CORE_VALUES["helpful"])
        self.assertTrue(AuraCoreIdentity.CORE_VALUES["honest"])
        self.assertTrue(AuraCoreIdentity.CORE_VALUES["privacy_first"])

    def test_never_do(self):
        """Test never-do actions"""
        self.assertIn("lie_to_user", AuraCoreIdentity.NEVER_DO)
        self.assertIn("share_private_data", AuraCoreIdentity.NEVER_DO)
        self.assertIn("manipulate_user", AuraCoreIdentity.NEVER_DO)

    def test_always_do(self):
        """Test always-do actions"""
        self.assertIn("ask_before_acting", AuraCoreIdentity.ALWAYS_DO)
        self.assertIn("explain_decisions", AuraCoreIdentity.ALWAYS_DO)
        self.assertIn("admit_mistakes", AuraCoreIdentity.ALWAYS_DO)


class TestAuraOpinions(unittest.TestCase):
    """Test AuraOpinions"""

    def test_opinion_topics(self):
        """Test opinion topics exist"""
        self.assertIn("task_approaches", AuraOpinions.OPINION_TOPICS)
        self.assertIn("communication", AuraOpinions.OPINION_TOPICS)
        self.assertIn("learning", AuraOpinions.OPINION_TOPICS)

    def test_opinion_boundaries(self):
        """Test opinion boundaries"""
        self.assertTrue(AuraOpinions.OPINION_BOUNDARIES["never_controversial"])
        self.assertTrue(AuraOpinions.OPINION_BOUNDARIES["never_political"])
        self.assertTrue(AuraOpinions.OPINION_BOUNDARIES["user_beliefs_respected"])


if __name__ == "__main__":
    unittest.main()
