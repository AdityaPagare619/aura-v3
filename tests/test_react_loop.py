"""
Tests for AURA v3 ReAct Agent Loop
"""

import unittest
import sys
import os
from datetime import datetime
from unittest.mock import Mock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.agent.loop import (
    ReActAgent,
    AgentState,
    Thought,
    AgentResponse,
    AgentFactory,
)


class TestAgentState(unittest.TestCase):
    def test_all_states(self):
        states = [
            AgentState.IDLE,
            AgentState.THINKING,
            AgentState.ACTING,
            AgentState.WAITING_APPROVAL,
            AgentState.WAITING_USER,
            AgentState.ERROR,
        ]
        self.assertEqual(len(states), 6)


class TestThought(unittest.TestCase):
    def test_create_thought(self):
        thought = Thought(
            thought="Test thought",
            action="test_action",
            action_input={"key": "value"},
            observation="test observation",
        )
        self.assertEqual(thought.thought, "Test thought")
        self.assertEqual(thought.action, "test_action")
        self.assertIsNotNone(thought.id)
        self.assertIsNotNone(thought.timestamp)


class TestAgentResponse(unittest.TestCase):
    def test_create_response(self):
        resp = AgentResponse(message="Hi", state=AgentState.IDLE)
        self.assertEqual(resp.message, "Hi")
        self.assertEqual(resp.state, AgentState.IDLE)

    def test_response_with_thoughts(self):
        thought = Thought(
            thought="Test", action="test", action_input={}, observation=""
        )
        resp = AgentResponse(message="Hi", state=AgentState.IDLE, thoughts=[thought])
        self.assertEqual(len(resp.thoughts), 1)


class TestAgentFactory(unittest.TestCase):
    def test_factory_exists(self):
        self.assertIsNotNone(AgentFactory)

    def test_factory_has_create_method(self):
        self.assertTrue(hasattr(AgentFactory, "create"))


if __name__ == "__main__":
    unittest.main()
