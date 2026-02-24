# AURA v3 API Documentation

## Overview

The AURA v3 API provides programmatic access to all AURA functionality. This document covers the core APIs, their methods, and usage examples.

---

## Table of Contents

1. [Core Classes](#core-classes)
2. [Initialization](#initialization)
3. [Processing](#processing)
4. [Configuration](#configuration)
5. [Utilities](#utilities)

---

## Core Classes

### AuraProduction

Main entry point for AURA functionality.

```python
from src.main import AuraProduction, get_aura
```

#### Methods

##### `__init__(self, config_path: Optional[str] = None)`

Initialize AURA with optional config path.

```python
aura = AuraProduction(config_path="/path/to/config.yaml")
```

##### `async initialize(self)`

Initialize all AURA components. Must be called before using other methods.

```python
await aura.initialize()
```

##### `async start(self)`

Start AURA services.

```python
await aura.start()
```

##### `async stop(self)`

Gracefully stop AURA services.

```python
await aura.stop()
```

##### `async process(self, user_input: str) -> str`

Process user input and return response.

```python
response = await aura.process("Hello, how are you?")
print(response)
```

##### `get_status(self) -> dict`

Get current AURA status.

```python
status = aura.get_status()
print(status)
# {
#   "running": True,
#   "initialized": True,
#   "core_systems": {...},
#   "services": {...},
#   ...
# }
```

---

## Initialization

### get_aura()

Get or create the global AURA instance.

```python
from src.main import get_aura

aura = await get_aura()
```

---

## Core Systems

### NeuromorphicEngine

Event-driven processing engine.

```python
from src.core import get_neuromorphic_engine

engine = get_neuromorphic_engine()
await engine.start()
```

#### Methods

##### `emit_user_message(self, message: str)`

Emit a user message for processing.

```python
await engine.emit_user_message("What's the weather?")
```

##### `get_status(self) -> Dict`

Get engine status.

```python
status = engine.get_status()
```

---

### MobilePowerManager

Battery and power management.

```python
from src.core import get_power_manager

power_manager = get_power_manager()
```

#### Methods

##### `get_power_mode(self) -> PowerMode`

Get current power mode.

```python
mode = power_manager.get_power_mode()
print(mode.value)  # "balanced", "power_save", etc.
```

##### `get_processing_limits(self) -> Dict`

Get processing limits based on power mode.

```python
limits = power_manager.get_processing_limits()
# {
#   "max_tokens": 1024,
#   "tick_rate": 2.0,
#   "allow_background": True,
#   "allow_proactive": True
# }
```

##### `should_process(self, task_priority: int) -> bool`

Check if task should run based on power mode.

```python
should_run = power_manager.should_process(task_priority=7)
```

---

### DeepUserProfiler

User profile and preference learning.

```python
from src.core import get_user_profiler

profiler = get_user_profiler()
```

#### Methods

##### `async get_current_context(self) -> Dict`

Get current user context.

```python
context = await profiler.get_current_context()
```

---

### AdaptivePersonalityEngine

Adaptive personality system.

```python
from src.core import get_personality_engine

personality = get_personality_engine()
```

#### Methods

##### `async get_state(self) -> Dict`

Get current personality state.

```python
state = await personality.get_state()
```

##### `async format_response(self, response: str, state: Dict) -> str`

Format response based on personality.

```python
formatted = await personality.format_response("Hello!", state)
```

---

### ContextProvider

Real-time context gathering.

```python
from src.context import get_context_provider

context_provider = get_context_provider()
await context_provider.start()
```

#### Methods

##### `async get_current_context(self) -> Dict`

Get current context snapshot.

```python
context = await context_provider.get_current_context()
```

##### `async get_context_for_llm(self) -> str`

Get context formatted for LLM.

```python
context_str = await context_provider.get_context_for_llm()
```

---

## Services

### LearningEngine

Interaction learning system.

```python
from src.learning import get_learning_engine

learning = get_learning_engine()
await learning.start()
```

#### Methods

##### `async record_interaction(...)`

Record an interaction for learning.

```python
await learning.record_interaction(
    input_text="Hello",
    output_text="Hi there!",
    context={"location": "home"},
    success=True
)
```

---

### SessionManager

Conversation session management.

```python
from src.session import get_session_manager, SessionType

session_manager = get_session_manager()
session_manager.create_session(session_type=SessionType.INTERACTION)
```

---

### LifeTracker

Life pattern tracking.

```python
from src.services import LifeTracker

tracker = LifeTracker()
await tracker.start()
```

---

### ProactiveEngine

Proactive assistance.

```python
from src.services import ProactiveEngine

proactive = ProactiveEngine(
    life_tracker=tracker,
    memory_system=memory
)
await proactive.start()
```

---

## Configuration

### Loading Configuration

```python
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)
```

### Configuration Schema

```yaml
core:
  llm:
    provider: str           # LLM provider: llama.cpp, ollama, transformers
    model_path: str        # Path to model file
    max_tokens: int        # Maximum tokens in response
    temperature: float     # Sampling temperature (0.0-2.0)
    n_gpu_layers: int      # GPU layers (for GGML)
    n_ctx: int             # Context window size
    preload: bool          # Preload model at startup

power:
  full_power_threshold: int    # % for full power
  balanced_threshold: int     # % for balanced
  power_save_threshold: int   # % for power save
  critical_threshold: int     # % for critical
  max_tokens_full: int        # Max tokens in full power
  max_tokens_balanced: int    # Max tokens in balanced
  max_tokens_save: int        # Max tokens in power save
  max_tokens_ultra: int      # Max tokens in ultra save

privacy:
  log_level: str          # DEBUG, INFO, WARNING, ERROR
  store_conversations: bool
  anonymize_data: bool
  allow_telemetry: bool

features:
  voice_enabled: bool
  voice_language: str
  proactive_mode: bool
  background_tasks: bool
  notifications: bool

security:
  require_auth: bool
  auth_method: str        # pin, biometric
  session_timeout: int    # seconds
  max_failed_attempts: int
```

---

## Utilities

### HealthMonitor

System health monitoring.

```python
from src.utils import HealthMonitor

monitor = HealthMonitor()
await monitor.start()

# Get health status
status = await monitor.get_health()
```

### CircuitBreakerManager

Circuit breaker for fault tolerance.

```python
from src.utils import CircuitBreakerManager

cb = CircuitBreakerManager()
```

### GracefulShutdown

Graceful shutdown handling.

```python
from src.utils import GracefulShutdown

shutdown = GracefulShutdown()
shutdown.register_component("my_service", stop_callback)
```

---

## Error Handling

AURA uses standard Python exceptions. Wrap API calls in try/except:

```python
try:
    response = await aura.process("Hello")
except Exception as e:
    print(f"Error: {e}")
    # Handle error appropriately
```

---

## Examples

### Basic Usage

```python
import asyncio
from src.main import AuraProduction

async def main():
    # Create and initialize
    aura = AuraProduction()
    await aura.initialize()
    await aura.start()
    
    # Process input
    response = await aura.process("What's the time?")
    print(response)
    
    # Get status
    print(aura.get_status())
    
    # Shutdown
    await aura.stop()

asyncio.run(main())
```

### With Custom Configuration

```python
import asyncio
from src.main import AuraProduction

async def main():
    aura = AuraProduction(config_path="/path/to/custom_config.yaml")
    await aura.initialize()
    await aura.start()
    
    # ... use aura ...
    
    await aura.stop()

asyncio.run(main())
```

### Accessing Core Systems

```python
import asyncio
from src.main import AuraProduction

async def main():
    aura = AuraProduction()
    await aura.initialize()
    
    # Access power manager
    power_mode = aura._mobile_power.get_power_mode()
    print(f"Power mode: {power_mode.value}")
    
    # Access user profiler
    user_context = await aura._user_profile.get_current_context()
    print(f"User context: {user_context}")

asyncio.run(main())
```

---

## Rate Limits

There are no rate limits for local processing. However, power management may throttle processing when:
- Battery is low
- Device is thermal throttling
- Screen is off (background mode)

---

## Thread Safety

AURA is designed to be async-first and is not thread-safe. Always use from the main async event loop.

---

## Version

Current version: 3.0.0

See CHANGELOG.md for version history.
