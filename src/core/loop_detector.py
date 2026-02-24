"""
AURA v3 Loop Detection System
============================

Prevents infinite loops and repeated actions in the agent execution loop.

Features:
- Pattern detection (same action, same thought, oscillating states, resource exhaustion)
- Action fingerprinting using hashes
- State tracking with configurable history
- Time-based detection for short-time loops
- Resource monitoring (CPU, memory, battery)
- Multiple response levels (WARNING, THROTTLE, STOP, KILL)
"""

import asyncio
import hashlib
import json
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class LoopDetectionLevel(Enum):
    """Response levels for loop detection"""

    NONE = "none"  # No loop detected
    WARNING = "warning"  # Notify user, ask for confirmation
    THROTTLE = "throttle"  # Slow down execution
    STOP = "stop"  # Graceful stop
    KILL = "kill"  # Emergency stop


class LoopType(Enum):
    """Types of loops that can be detected"""

    ACTION_REPEAT = "action_repeat"  # Same action repeated
    THOUGHT_REPEAT = "thought_repeat"  # Same thought repeated
    OSCILLATION = "oscillation"  # Oscillating between states
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # Resource exhaustion
    LLM_RESPONSE_REPEAT = "llm_response_repeat"  # Same LLM response
    TOOL_EXECUTION_LOOP = "tool_execution_loop"  # Tool execution loop


@dataclass
class LoopDetectionConfig:
    """Configuration for loop detection thresholds"""

    warning_threshold: int = 3  # Number of repeats before warning
    stop_threshold: int = 5  # Number of repeats before stopping
    kill_threshold: int = 10  # Number of repeats before emergency kill

    time_window_seconds: float = 60.0  # Time window for time-based detection
    max_history_size: int = 100  # Maximum history size

    action_similarity_threshold: float = 0.9  # Similarity threshold for actions
    thought_similarity_threshold: float = 0.8  # Similarity threshold for thoughts

    cpu_warning_threshold: float = 90.0  # CPU usage warning %
    cpu_stop_threshold: float = 98.0  # CPU usage stop %
    memory_warning_threshold: float = 90.0  # Memory usage warning %
    memory_stop_threshold: float = 98.0  # Memory usage stop %
    battery_warning_threshold: int = 10  # Battery warning level %
    battery_stop_threshold: int = 5  # Battery stop level %

    enable_time_based: bool = True  # Enable time-based detection
    enable_resource_monitoring: bool = True  # Enable resource monitoring
    enable_action_fingerprinting: bool = True  # Enable action fingerprinting


@dataclass
class LoopDetectionResult:
    """Result of loop detection"""

    level: LoopDetectionLevel
    loop_type: Optional[LoopType]
    details: str
    repeated_count: int
    suggested_action: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActionFingerprint:
    """Fingerprint of an action for comparison"""

    action_hash: str
    action_type: str
    parameters: Dict[str, Any]
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StateHistoryEntry:
    """Entry in state history"""

    state: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


class ActionHasher:
    """Utility class for creating action fingerprints"""

    @staticmethod
    def create_fingerprint(
        action_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Create a hash fingerprint for an action"""
        components = [action_type]

        if parameters:
            sorted_params = json.dumps(parameters, sort_keys=True)
            components.append(sorted_params)

        if context:
            sorted_context = json.dumps(context, sort_keys=True)
            components.append(sorted_context)

        combined = "|".join(components)
        return hashlib.sha256(combined.encode()).hexdigest()[:16]

    @staticmethod
    def actions_similar(fp1: str, fp2: str, threshold: float = 0.9) -> bool:
        """Check if two action fingerprints are similar"""
        if fp1 == fp2:
            return True

        matching = sum(1 for a, b in zip(fp1, fp2) if a == b)
        similarity = matching / max(len(fp1), len(fp2), 1)
        return similarity >= threshold


class TextSimilarity:
    """Utility class for text similarity comparison"""

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"[^\w\s]", "", text)
        return text

    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Calculate similarity between two texts (0-1)"""
        if not text1 or not text2:
            return 0.0

        norm1 = TextSimilarity.normalize_text(text1)
        norm2 = TextSimilarity.normalize_text(text2)

        if norm1 == norm2:
            return 1.0

        words1 = set(norm1.split())
        words2 = set(norm2.split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union) if union else 0.0


class ResourceMonitor:
    """Monitor system resources for loop detection"""

    def __init__(self):
        self._last_check: Optional[datetime] = None
        self._cache_ttl: float = 5.0  # Cache for 5 seconds
        self._cpu_usage: float = 0.0
        self._memory_usage: float = 0.0

    async def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage"""
        now = datetime.now()

        if (
            self._last_check
            and (now - self._last_check).total_seconds() < self._cache_ttl
        ):
            return self._cpu_usage

        try:
            import psutil

            self._cpu_usage = psutil.cpu_percent(interval=0.1)
            self._last_check = now
        except ImportError:
            try:
                import subprocess

                result = subprocess.run(
                    "top -bn1 | grep 'Cpu(s)' | awk '{print $2}'",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0 and result.stdout.strip():
                    self._cpu_usage = float(result.stdout.strip().replace("%", ""))
            except:
                self._cpu_usage = 0.0

        return self._cpu_usage

    async def get_memory_usage(self) -> float:
        """Get current memory usage percentage"""
        now = datetime.now()

        if (
            self._last_check
            and (now - self._last_check).total_seconds() < self._cache_ttl
        ):
            return self._memory_usage

        try:
            import psutil

            self._memory_usage = psutil.virtual_memory().percent
            self._last_check = now
        except ImportError:
            try:
                import subprocess

                result = subprocess.run(
                    "free | grep Mem | awk '{print ($3/$2) * 100.0}'",
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                if result.returncode == 0 and result.stdout.strip():
                    self._memory_usage = float(result.stdout.strip())
            except:
                self._memory_usage = 0.0

        return self._memory_usage

    async def get_battery_level(self) -> int:
        """Get battery level percentage"""
        try:
            import subprocess

            result = subprocess.run(
                "termux-battery-status",
                shell=True,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                data = json.loads(result.stdout)
                return int(data.get("percentage", 100))
        except:
            pass

        return 100  # Assume full battery if unavailable


class LoopDetector:
    """
    Main loop detection system for AURA v3.

    Detects and prevents:
    - Same action repeated N times
    - Same thought repeated
    - Oscillating between 2 states
    - Resource exhaustion patterns
    - LLM response loops
    - Tool execution loops
    """

    def __init__(self, config: Optional[LoopDetectionConfig] = None):
        self.config = config or LoopDetectionConfig()

        self._action_history: deque = deque(maxlen=self.config.max_history_size)
        self._thought_history: deque = deque(maxlen=self.config.max_history_size)
        self._state_history: deque = deque(maxlen=self.config.max_history_size)
        self._llm_response_history: deque = deque(maxlen=self.config.max_history_size)

        self._resource_monitor = ResourceMonitor()

        self._callbacks: Dict[LoopDetectionLevel, List[Callable]] = {
            LoopDetectionLevel.WARNING: [],
            LoopDetectionLevel.THROTTLE: [],
            LoopDetectionLevel.STOP: [],
            LoopDetectionLevel.KILL: [],
        }

        self._is_throttled: bool = False
        self._throttle_until: Optional[datetime] = None
        self._throttle_delay: float = 1.0

        self._last_warning_time: Optional[datetime] = None
        self._consecutive_loops: int = 0
        self._total_loops_detected: int = 0

        self._execution_paused: bool = False
        self._emergency_stop: bool = False

    def register_callback(
        self, level: LoopDetectionLevel, callback: Callable[[LoopDetectionResult], None]
    ):
        """Register a callback for a specific detection level"""
        if level != LoopDetectionLevel.NONE:
            self._callbacks[level].append(callback)

    def _execute_callbacks(self, result: LoopDetectionResult):
        """Execute all callbacks for the detection level"""
        callbacks = self._callbacks.get(result.level, [])
        for callback in callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    async def check_action_loop(
        self,
        action_type: str,
        parameters: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> LoopDetectionResult:
        """Check if an action is being repeated in a loop"""

        fingerprint = ActionHasher.create_fingerprint(action_type, parameters, context)
        entry = ActionFingerprint(
            action_hash=fingerprint,
            action_type=action_type,
            parameters=parameters or {},
            timestamp=datetime.now(),
            context=context or {},
        )

        self._action_history.append(entry)

        same_actions = [a for a in self._action_history if a.action_hash == fingerprint]
        repeat_count = len(same_actions)

        if repeat_count >= self.config.kill_threshold:
            self._consecutive_loops += 1
            self._total_loops_detected += 1

            result = LoopDetectionResult(
                level=LoopDetectionLevel.KILL,
                loop_type=LoopType.ACTION_REPEAT,
                details=f"Action '{action_type}' repeated {repeat_count} times - EMERGENCY STOP",
                repeated_count=repeat_count,
                suggested_action="emergency_stop",
                metadata={
                    "action_type": action_type,
                    "parameters": parameters,
                    "fingerprint": fingerprint,
                },
            )

            self._emergency_stop = True
            self._execute_callbacks(result)
            return result

        if repeat_count >= self.config.stop_threshold:
            self._consecutive_loops += 1
            self._total_loops_detected += 1

            result = LoopDetectionResult(
                level=LoopDetectionLevel.STOP,
                loop_type=LoopType.ACTION_REPEAT,
                details=f"Action '{action_type}' repeated {repeat_count} times - graceful stop",
                repeated_count=repeat_count,
                suggested_action="graceful_stop",
                metadata={
                    "action_type": action_type,
                    "parameters": parameters,
                    "fingerprint": fingerprint,
                },
            )

            self._execute_callbacks(result)
            return result

        if repeat_count >= self.config.warning_threshold:
            result = LoopDetectionResult(
                level=LoopDetectionLevel.WARNING,
                loop_type=LoopType.ACTION_REPEAT,
                details=f"Action '{action_type}' repeated {repeat_count} times - WARNING",
                repeated_count=repeat_count,
                suggested_action="confirm_continue",
                metadata={
                    "action_type": action_type,
                    "parameters": parameters,
                    "fingerprint": fingerprint,
                },
            )

            self._last_warning_time = datetime.now()
            self._execute_callbacks(result)
            return result

        if self.config.enable_time_based:
            recent_same = [
                a
                for a in self._action_history
                if a.action_hash == fingerprint
                and (datetime.now() - a.timestamp).total_seconds()
                < self.config.time_window_seconds
            ]

            if len(recent_same) >= 3:
                result = LoopDetectionResult(
                    level=LoopDetectionLevel.THROTTLE,
                    loop_type=LoopType.ACTION_REPEAT,
                    details=f"Action '{action_type}' repeated {len(recent_same)} times in {self.config.time_window_seconds}s - THROTTLING",
                    repeated_count=len(recent_same),
                    suggested_action="throttle",
                    metadata={"action_type": action_type, "time_based": True},
                )

                self._enable_throttle()
                self._execute_callbacks(result)
                return result

        return LoopDetectionResult(
            level=LoopDetectionLevel.NONE,
            loop_type=None,
            details="No action loop detected",
            repeated_count=repeat_count,
            suggested_action=None,
        )

    async def check_thought_loop(
        self,
        thought_content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> LoopDetectionResult:
        """Check if a thought is being repeated"""

        entry = {
            "content": thought_content,
            "timestamp": datetime.now(),
            "context": context or {},
        }

        self._thought_history.append(entry)

        similar_thoughts = [
            t
            for t in self._thought_history
            if TextSimilarity.calculate_similarity(t["content"], thought_content)
            >= self.config.thought_similarity_threshold
        ]

        repeat_count = len(similar_thoughts)

        if repeat_count >= self.config.kill_threshold:
            self._consecutive_loops += 1
            self._total_loops_detected += 1

            result = LoopDetectionResult(
                level=LoopDetectionLevel.KILL,
                loop_type=LoopType.THOUGHT_REPEAT,
                details=f"Similar thought repeated {repeat_count} times - EMERGENCY STOP",
                repeated_count=repeat_count,
                suggested_action="emergency_stop",
                metadata={"thought_preview": thought_content[:100]},
            )

            self._emergency_stop = True
            self._execute_callbacks(result)
            return result

        if repeat_count >= self.config.stop_threshold:
            self._consecutive_loops += 1
            self._total_loops_detected += 1

            result = LoopDetectionResult(
                level=LoopDetectionLevel.STOP,
                loop_type=LoopType.THOUGHT_REPEAT,
                details=f"Similar thought repeated {repeat_count} times - graceful stop",
                repeated_count=repeat_count,
                suggested_action="graceful_stop",
                metadata={"thought_preview": thought_content[:100]},
            )

            self._execute_callbacks(result)
            return result

        if repeat_count >= self.config.warning_threshold:
            result = LoopDetectionResult(
                level=LoopDetectionLevel.WARNING,
                loop_type=LoopType.THOUGHT_REPEAT,
                details=f"Similar thought repeated {repeat_count} times - WARNING",
                repeated_count=repeat_count,
                suggested_action="confirm_continue",
                metadata={"thought_preview": thought_content[:100]},
            )

            self._last_warning_time = datetime.now()
            self._execute_callbacks(result)
            return result

        return LoopDetectionResult(
            level=LoopDetectionLevel.NONE,
            loop_type=None,
            details="No thought loop detected",
            repeated_count=repeat_count,
            suggested_action=None,
        )

    async def check_state_oscillation(
        self,
        new_state: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> LoopDetectionResult:
        """Check if states are oscillating between two values"""

        entry = StateHistoryEntry(
            state=new_state,
            timestamp=datetime.now(),
            metadata=metadata or {},
        )

        self._state_history.append(entry)

        if len(self._state_history) < 4:
            return LoopDetectionResult(
                level=LoopDetectionLevel.NONE,
                loop_type=None,
                details="Not enough state history for oscillation detection",
                repeated_count=0,
                suggested_action=None,
            )

        recent_states = list(self._state_history)[-6:]

        oscillating = True
        for i in range(len(recent_states) - 2):
            if (
                recent_states[i].state
                == recent_states[i + 2].state
                != recent_states[i + 1].state
            ):
                continue
            else:
                oscillating = False
                break

        if oscillating:
            unique_states = len(set(s.state for s in recent_states))

            if unique_states == 2:
                repeat_count = len(recent_states) // 2

                if repeat_count >= self.config.kill_threshold:
                    self._consecutive_loops += 1
                    self._total_loops_detected += 1

                    result = LoopDetectionResult(
                        level=LoopDetectionLevel.KILL,
                        loop_type=LoopType.OSCILLATION,
                        details=f"State oscillation detected between 2 states - EMERGENCY STOP",
                        repeated_count=repeat_count,
                        suggested_action="emergency_stop",
                        metadata={"states": [s.state for s in recent_states]},
                    )

                    self._emergency_stop = True
                    self._execute_callbacks(result)
                    return result

                if repeat_count >= self.config.stop_threshold:
                    self._consecutive_loops += 1
                    self._total_loops_detected += 1

                    result = LoopDetectionResult(
                        level=LoopDetectionLevel.STOP,
                        loop_type=LoopType.OSCILLATION,
                        details=f"State oscillation detected between 2 states - graceful stop",
                        repeated_count=repeat_count,
                        suggested_action="graceful_stop",
                        metadata={"states": [s.state for s in recent_states]},
                    )

                    self._execute_callbacks(result)
                    return result

                if repeat_count >= self.config.warning_threshold:
                    result = LoopDetectionResult(
                        level=LoopDetectionLevel.WARNING,
                        loop_type=LoopType.OSCILLATION,
                        details=f"State oscillation detected between 2 states - WARNING",
                        repeated_count=repeat_count,
                        suggested_action="confirm_continue",
                        metadata={"states": [s.state for s in recent_states]},
                    )

                    self._last_warning_time = datetime.now()
                    self._execute_callbacks(result)
                    return result

        return LoopDetectionResult(
            level=LoopDetectionLevel.NONE,
            loop_type=None,
            details="No state oscillation detected",
            repeated_count=0,
            suggested_action=None,
        )

    async def check_llm_response_loop(
        self,
        response_text: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> LoopDetectionResult:
        """Check if LLM is returning the same or similar responses"""

        entry = {
            "response": response_text,
            "timestamp": datetime.now(),
            "context": context or {},
        }

        self._llm_response_history.append(entry)

        similar_responses = [
            r
            for r in self._llm_response_history
            if TextSimilarity.calculate_similarity(r["response"], response_text)
            >= self.config.action_similarity_threshold
        ]

        repeat_count = len(similar_responses)

        if repeat_count >= self.config.kill_threshold:
            self._consecutive_loops += 1
            self._total_loops_detected += 1

            result = LoopDetectionResult(
                level=LoopDetectionLevel.KILL,
                loop_type=LoopType.LLM_RESPONSE_REPEAT,
                details=f"LLM response repeated {repeat_count} times - EMERGENCY STOP",
                repeated_count=repeat_count,
                suggested_action="emergency_stop",
                metadata={"response_preview": response_text[:100]},
            )

            self._emergency_stop = True
            self._execute_callbacks(result)
            return result

        if repeat_count >= self.config.stop_threshold:
            self._consecutive_loops += 1
            self._total_loops_detected += 1

            result = LoopDetectionResult(
                level=LoopDetectionLevel.STOP,
                loop_type=LoopType.LLM_RESPONSE_REPEAT,
                details=f"LLM response repeated {repeat_count} times - graceful stop",
                repeated_count=repeat_count,
                suggested_action="graceful_stop",
                metadata={"response_preview": response_text[:100]},
            )

            self._execute_callbacks(result)
            return result

        if repeat_count >= self.config.warning_threshold:
            result = LoopDetectionResult(
                level=LoopDetectionLevel.WARNING,
                loop_type=LoopType.LLM_RESPONSE_REPEAT,
                details=f"LLM response repeated {repeat_count} times - WARNING",
                repeated_count=repeat_count,
                suggested_action="confirm_continue",
                metadata={"response_preview": response_text[:100]},
            )

            self._last_warning_time = datetime.now()
            self._execute_callbacks(result)
            return result

        return LoopDetectionResult(
            level=LoopDetectionLevel.NONE,
            loop_type=None,
            details="No LLM response loop detected",
            repeated_count=repeat_count,
            suggested_action=None,
        )

    async def check_resource_exhaustion(self) -> LoopDetectionResult:
        """Check for resource exhaustion patterns"""

        if not self.config.enable_resource_monitoring:
            return LoopDetectionResult(
                level=LoopDetectionLevel.NONE,
                loop_type=None,
                details="Resource monitoring disabled",
                repeated_count=0,
                suggested_action=None,
            )

        cpu_usage = await self._resource_monitor.get_cpu_usage()
        memory_usage = await self._resource_monitor.get_memory_usage()
        battery_level = await self._resource_monitor.get_battery_level()

        if cpu_usage >= self.config.cpu_stop_threshold:
            result = LoopDetectionResult(
                level=LoopDetectionLevel.KILL,
                loop_type=LoopType.RESOURCE_EXHAUSTION,
                details=f"CPU usage at {cpu_usage:.1f}% - EMERGENCY STOP",
                repeated_count=0,
                suggested_action="emergency_stop",
                metadata={
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "battery_level": battery_level,
                },
            )

            self._emergency_stop = True
            self._execute_callbacks(result)
            return result

        if memory_usage >= self.config.memory_stop_threshold:
            result = LoopDetectionResult(
                level=LoopDetectionLevel.KILL,
                loop_type=LoopType.RESOURCE_EXHAUSTION,
                details=f"Memory usage at {memory_usage:.1f}% - EMERGENCY STOP",
                repeated_count=0,
                suggested_action="emergency_stop",
                metadata={
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "battery_level": battery_level,
                },
            )

            self._emergency_stop = True
            self._execute_callbacks(result)
            return result

        if battery_level <= self.config.battery_stop_threshold:
            result = LoopDetectionResult(
                level=LoopDetectionLevel.STOP,
                loop_type=LoopType.RESOURCE_EXHAUSTION,
                details=f"Battery at {battery_level}% - graceful stop",
                repeated_count=0,
                suggested_action="graceful_stop",
                metadata={
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "battery_level": battery_level,
                },
            )

            self._execute_callbacks(result)
            return result

        if cpu_usage >= self.config.cpu_warning_threshold:
            result = LoopDetectionResult(
                level=LoopDetectionLevel.WARNING,
                loop_type=LoopType.RESOURCE_EXHAUSTION,
                details=f"CPU usage at {cpu_usage:.1f}% - WARNING",
                repeated_count=0,
                suggested_action="confirm_continue",
                metadata={
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "battery_level": battery_level,
                },
            )

            self._execute_callbacks(result)
            return result

        if memory_usage >= self.config.memory_warning_threshold:
            result = LoopDetectionResult(
                level=LoopDetectionLevel.WARNING,
                loop_type=LoopType.RESOURCE_EXHAUSTION,
                details=f"Memory usage at {memory_usage:.1f}% - WARNING",
                repeated_count=0,
                suggested_action="confirm_continue",
                metadata={
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "battery_level": battery_level,
                },
            )

            self._execute_callbacks(result)
            return result

        if battery_level <= self.config.battery_warning_threshold:
            result = LoopDetectionResult(
                level=LoopDetectionLevel.THROTTLE,
                loop_type=LoopType.RESOURCE_EXHAUSTION,
                details=f"Battery at {battery_level}% - THROTTLING",
                repeated_count=0,
                suggested_action="throttle",
                metadata={
                    "cpu_usage": cpu_usage,
                    "memory_usage": memory_usage,
                    "battery_level": battery_level,
                },
            )

            self._enable_throttle()
            self._execute_callbacks(result)
            return result

        return LoopDetectionResult(
            level=LoopDetectionLevel.NONE,
            loop_type=None,
            details="Resources OK",
            repeated_count=0,
            suggested_action=None,
            metadata={
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "battery_level": battery_level,
            },
        )

    async def check_tool_execution_loop(
        self,
        tool_name: str,
        parameters: Optional[Dict[str, Any]] = None,
        result: Optional[str] = None,
    ) -> LoopDetectionResult:
        """Check for tool execution loops"""

        context = {"parameters": parameters, "result": result[:100] if result else None}
        return await self.check_action_loop(
            action_type=f"tool:{tool_name}", parameters=parameters, context=context
        )

    def _enable_throttle(self):
        """Enable throttling mode"""
        self._is_throttled = True
        self._throttle_until = datetime.now() + timedelta(seconds=self._throttle_delay)

    async def wait_if_throttled(self):
        """Wait if execution is throttled"""
        if self._is_throttled and self._throttle_until:
            wait_time = (self._throttle_until - datetime.now()).total_seconds()
            if wait_time > 0:
                logger.info(f"Throttled, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
            self._is_throttled = False
            self._throttle_until = None

    def set_throttle_delay(self, delay: float):
        """Set throttle delay in seconds"""
        self._throttle_delay = delay

    def reset(self):
        """Reset the loop detector state"""
        self._action_history.clear()
        self._thought_history.clear()
        self._state_history.clear()
        self._llm_response_history.clear()

        self._is_throttled = False
        self._throttle_until = None
        self._last_warning_time = None
        self._consecutive_loops = 0
        self._execution_paused = False
        self._emergency_stop = False

        logger.info("Loop detector state reset")

    def pause_execution(self):
        """Pause execution"""
        self._execution_paused = True
        logger.warning("Execution paused by loop detector")

    def resume_execution(self):
        """Resume execution"""
        self._execution_paused = False
        logger.info("Execution resumed")

    def get_status(self) -> Dict[str, Any]:
        """Get current loop detector status"""
        return {
            "is_throttled": self._is_throttled,
            "throttle_until": self._throttle_until.isoformat()
            if self._throttle_until
            else None,
            "consecutive_loops": self._consecutive_loops,
            "total_loops_detected": self._total_loops_detected,
            "execution_paused": self._execution_paused,
            "emergency_stop": self._emergency_stop,
            "action_history_size": len(self._action_history),
            "thought_history_size": len(self._thought_history),
            "state_history_size": len(self._state_history),
            "llm_response_history_size": len(self._llm_response_history),
        }

    def should_continue(self) -> bool:
        """Check if execution should continue"""
        if self._emergency_stop:
            logger.error("Emergency stop triggered - not continuing")
            return False

        if self._execution_paused:
            logger.warning("Execution paused - not continuing")
            return False

        return True


class IntegrationMixin:
    """Mixin for integrating loop detection with agent execution"""

    def __init__(self):
        self._loop_detector: Optional[LoopDetector] = None

    def set_loop_detector(self, detector: LoopDetector):
        """Set the loop detector instance"""
        self._loop_detector = detector

    async def check_before_action(
        self, action_type: str, parameters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Check if action should proceed"""
        if not self._loop_detector:
            return True

        if not self._loop_detector.should_continue():
            return False

        await self._loop_detector.wait_if_throttled()

        result = await self._loop_detector.check_action_loop(action_type, parameters)

        if result.level == LoopDetectionLevel.NONE:
            return True

        if result.level in [
            LoopDetectionLevel.WARNING,
            LoopDetectionLevel.THROTTLE,
        ]:
            return True

        return False

    async def check_before_think(self, thought: str) -> bool:
        """Check if thought processing should proceed"""
        if not self._loop_detector:
            return True

        if not self._loop_detector.should_continue():
            return False

        result = await self._loop_detector.check_thought_loop(thought)

        if result.level == LoopDetectionLevel.NONE:
            return True

        if result.level in [
            LoopDetectionLevel.WARNING,
            LoopDetectionLevel.THROTTLE,
        ]:
            return True

        return False


_loop_detector_instance: Optional[LoopDetector] = None


def get_loop_detector(config: Optional[LoopDetectionConfig] = None) -> LoopDetector:
    """Get or create loop detector instance"""
    global _loop_detector_instance
    if _loop_detector_instance is None:
        _loop_detector_instance = LoopDetector(config)
    return _loop_detector_instance


def create_loop_detector(config: Optional[LoopDetectionConfig] = None) -> LoopDetector:
    """Create a new loop detector instance"""
    return LoopDetector(config)


__all__ = [
    "LoopDetector",
    "LoopDetectionLevel",
    "LoopType",
    "LoopDetectionConfig",
    "LoopDetectionResult",
    "ActionFingerprint",
    "ActionHasher",
    "TextSimilarity",
    "ResourceMonitor",
    "IntegrationMixin",
    "get_loop_detector",
    "create_loop_detector",
]
