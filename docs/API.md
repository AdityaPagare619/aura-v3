# AURA v3 API Documentation

## Overview

The AURA v3 API provides programmatic access to all AURA functionality. This document covers the core APIs, their methods, and usage examples.

**Version**: 3.0.0

---

## Table of Contents

1. [Core Classes](#core-classes)
   - [AuraProduction](#auraproduction)
2. [LLM System](#llm-system)
   - [LLMManager](#llmmanager)
   - [ProductionLLM](#productionllm)
   - [ModelRegistry](#modelregistry)
3. [Memory System](#memory-system)
   - [NeuralMemory](#neuralmemory)
   - [MemoryCoordinator](#memorycoordinator)
   - [Memory Types](#memory-types)
4. [Tool System](#tool-system)
   - [ToolRegistry](#toolregistry)
   - [ToolExecutor](#toolexecutor)
   - [ToolHandlers](#toolhandlers)
   - [Tool Registration API](#tool-registration-api)
5. [REST API](#rest-api)
   - [Authentication](#authentication)
   - [Endpoints](#endpoints)
   - [Rate Limiting](#rate-limiting)
6. [Agent Loop Hooks](#agent-loop-hooks)
7. [Configuration](#configuration)
8. [Core Systems](#core-systems)
9. [Examples](#examples)

---

## Core Classes

### AuraProduction

Main entry point for AURA functionality with tiered initialization.

```python
from src.main import AuraProduction, get_aura
```

#### Tiered Loading System

AURA uses a 3-tier loading system for optimal startup performance:

| Tier | Components | Timing |
|------|-----------|--------|
| Tier 0 | Config, logging, power manager | Immediate |
| Tier 1 | Core systems (memory, LLM, personality) | ~2s |
| Tier 2 | Services (context, learning, proactive) | Background |

#### Constructor

```python
def __init__(
    self,
    config_path: Optional[str] = None,
    minimal: bool = False
)
```

**Parameters:**
- `config_path`: Path to YAML configuration file (default: `config.yaml`)
- `minimal`: If `True`, skip non-essential components

**Example:**
```python
aura = AuraProduction(config_path="/path/to/config.yaml")
```

#### Methods

##### `async initialize()`

Initialize all AURA components using tiered loading.

```python
await aura.initialize()
# Tier 0: Config, logging, power → immediate
# Tier 1: Memory, LLM, personality → ~2s
# Tier 2: Context, learning, proactive → background
```

##### `async start()`

Start AURA services and begin processing.

```python
await aura.start()
```

##### `async stop()`

Gracefully stop all AURA services.

```python
await aura.stop()
```

##### `async process(user_input: str, stream: bool = False) -> Union[str, AsyncGenerator]`

Process user input and return response.

**Parameters:**
- `user_input`: The user's message
- `stream`: If `True`, return an async generator for streaming

**Returns:** Response string or async generator

```python
# Standard response
response = await aura.process("Hello, how are you?")
print(response)

# Streaming response
async for chunk in await aura.process("Tell me a story", stream=True):
    print(chunk, end="", flush=True)
```

##### `get_status() -> dict`

Get current AURA status including all subsystems.

```python
status = aura.get_status()
# {
#   "running": True,
#   "initialized": True,
#   "tier": 2,
#   "core_systems": {
#     "llm": {"loaded": True, "model": "..."},
#     "memory": {"ready": True, "neurons": 1000},
#     "personality": {"state": "friendly"}
#   },
#   "services": {
#     "context_provider": True,
#     "learning_engine": True,
#     "proactive_engine": False
#   },
#   "power_mode": "balanced",
#   "uptime_seconds": 3600
# }
```

##### `async process_with_tools(user_input: str) -> str`

Process input with tool execution enabled.

```python
response = await aura.process_with_tools("What's the weather like?")
# AURA may call weather tools automatically
```

##### `get_inner_voice_state() -> dict`

Get current inner voice/thinking state.

```python
state = aura.get_inner_voice_state()
# {"thoughts": [...], "active": True, "last_update": "..."}
```

##### `get_feeling_state() -> dict`

Get current emotional/feeling state.

```python
feeling = aura.get_feeling_state()
# {"mood": "curious", "energy": 0.8, "engagement": 0.9}
```

##### `get_trust_level() -> float`

Get current trust level with user.

```python
trust = aura.get_trust_level()
# 0.85 (range 0.0-1.0)
```

#### Global Instance

```python
from src.main import get_aura

# Get or create global AURA instance
aura = await get_aura()
```

---

## LLM System

### LLMManager

Manages STT (Speech-to-Text), TTS (Text-to-Speech), and LLM models.

```python
from src.llm.manager import LLMManager
```

#### Constructor

```python
def __init__(
    self,
    config: Optional[Dict] = None,
    lazy_load: bool = True
)
```

**Parameters:**
- `config`: LLM configuration dictionary
- `lazy_load`: Defer model loading until first use

#### Methods

##### `async initialize()`

Initialize LLM manager and optionally preload models.

```python
manager = LLMManager(config)
await manager.initialize()
```

##### `async generate(prompt: str, **kwargs) -> str`

Generate text from prompt.

**Parameters:**
- `prompt`: Input prompt
- `max_tokens`: Maximum tokens to generate (default: 512)
- `temperature`: Sampling temperature (default: 0.7)
- `stop`: Stop sequences
- `stream`: Enable streaming

```python
response = await manager.generate(
    prompt="Explain quantum computing",
    max_tokens=256,
    temperature=0.8
)
```

##### `async generate_stream(prompt: str, **kwargs) -> AsyncGenerator[str, None]`

Stream generated text.

```python
async for chunk in manager.generate_stream("Write a poem"):
    print(chunk, end="")
```

##### `async transcribe(audio_data: bytes) -> str`

Transcribe audio to text (STT).

```python
text = await manager.transcribe(audio_bytes)
```

##### `async synthesize(text: str) -> bytes`

Synthesize text to speech (TTS).

```python
audio = await manager.synthesize("Hello, world!")
```

##### `get_status() -> dict`

Get LLM manager status.

```python
status = manager.get_status()
# {"llm_loaded": True, "stt_loaded": False, "tts_loaded": True, ...}
```

##### `async unload()`

Unload all models to free memory.

```python
await manager.unload()
```

---

### ProductionLLM

Production-grade LLM with model registry, benchmarks, and thermal monitoring.

```python
from src.llm.production_llm import ProductionLLM
```

#### Constructor

```python
def __init__(
    self,
    config: Optional[Dict] = None,
    models_dir: str = "models",
    lazy_load: bool = True
)
```

#### Methods

##### `async load_model(model_name: str, force: bool = False) -> bool`

Load a specific model from the registry.

```python
success = await llm.load_model("phi-3-mini-4k", force=True)
```

##### `async generate(prompt: str, **kwargs) -> str`

Generate text with production optimizations.

**Parameters:**
- `prompt`: Input prompt
- `max_tokens`: Maximum output tokens
- `temperature`: Sampling temperature (0.0-2.0)
- `top_p`: Nucleus sampling parameter
- `top_k`: Top-k sampling parameter
- `repeat_penalty`: Repetition penalty
- `stop`: Stop sequences list
- `grammar`: GBNF grammar for constrained generation

```python
response = await llm.generate(
    prompt="Write a haiku about coding",
    max_tokens=64,
    temperature=0.9,
    stop=["\n\n"]
)
```

##### `async benchmark(prompt: str, iterations: int = 5) -> dict`

Benchmark generation performance.

```python
results = await llm.benchmark("Test prompt", iterations=10)
# {
#   "tokens_per_second": 45.2,
#   "time_to_first_token": 0.12,
#   "total_time": 2.3,
#   "memory_mb": 4096
# }
```

##### `get_thermal_status() -> dict`

Get thermal monitoring status.

```python
thermal = llm.get_thermal_status()
# {"temperature_c": 65, "throttling": False, "fan_speed": 2500}
```

##### `async switch_model(model_name: str) -> bool`

Hot-switch to a different model.

```python
await llm.switch_model("llama-3-8b")
```

##### `get_model_info() -> dict`

Get current model information.

```python
info = llm.get_model_info()
# {
#   "name": "phi-3-mini-4k",
#   "parameters": "3.8B",
#   "context_length": 4096,
#   "quantization": "Q4_K_M"
# }
```

---

### ModelRegistry

Registry for managing available models and their specifications.

```python
from src.llm.production_llm import ModelRegistry, ModelSpec
```

#### ModelSpec Dataclass

```python
@dataclass
class ModelSpec:
    name: str                    # Model identifier
    path: str                    # Path to model file
    context_length: int          # Maximum context window
    parameters: str              # Parameter count (e.g., "7B")
    quantization: str            # Quantization type (e.g., "Q4_K_M")
    recommended_gpu_layers: int  # Suggested GPU layers
    min_memory_mb: int           # Minimum RAM required
    capabilities: List[str]      # ["chat", "code", "reasoning"]
    default_temperature: float   # Default sampling temp
    stop_tokens: List[str]       # Default stop sequences
```

#### Methods

##### `register(spec: ModelSpec)`

Register a new model.

```python
registry = ModelRegistry()
registry.register(ModelSpec(
    name="my-custom-model",
    path="models/custom.gguf",
    context_length=8192,
    parameters="7B",
    quantization="Q5_K_M",
    recommended_gpu_layers=35,
    min_memory_mb=6000,
    capabilities=["chat", "code"],
    default_temperature=0.7,
    stop_tokens=["<|end|>"]
))
```

##### `get(name: str) -> Optional[ModelSpec]`

Get model specification by name.

```python
spec = registry.get("phi-3-mini-4k")
```

##### `list_models() -> List[str]`

List all registered model names.

```python
models = registry.list_models()
# ["phi-3-mini-4k", "llama-3-8b", "mistral-7b", ...]
```

##### `get_best_for_capability(capability: str, max_memory_mb: int) -> Optional[ModelSpec]`

Find best model for a capability within memory constraints.

```python
spec = registry.get_best_for_capability("code", max_memory_mb=8000)
```

---

## Memory System

### NeuralMemory

Biologically-inspired memory system with neurons, synapses, and memory clusters.

```python
from src.memory.neural_memory import NeuralMemory, MemoryType
```

#### Memory Architecture

```
NeuralMemory
├── Neurons (individual memory units)
│   ├── content: str
│   ├── embedding: np.ndarray
│   ├── activation: float (0.0-1.0)
│   ├── memory_type: MemoryType
│   └── metadata: dict
├── Synapses (connections between neurons)
│   ├── source_id: str
│   ├── target_id: str
│   └── weight: float
└── MemoryClusters (grouped related memories)
    ├── centroid: np.ndarray
    ├── neuron_ids: List[str]
    └── label: str
```

#### MemoryType Enum

```python
class MemoryType(Enum):
    WORKING = "working"      # Short-term, immediate context
    EPISODIC = "episodic"    # Personal experiences, events
    SEMANTIC = "semantic"    # Facts, knowledge, concepts
    SKILL = "skill"          # Learned procedures, how-to
    ANCESTOR = "ancestor"    # Core personality, base knowledge
```

#### Constructor

```python
def __init__(
    self,
    embedding_model: Optional[str] = None,
    max_neurons: int = 10000,
    decay_rate: float = 0.01,
    cluster_threshold: float = 0.7
)
```

#### Methods

##### `async store(content: str, memory_type: MemoryType, metadata: dict = None) -> str`

Store a new memory and return its neuron ID.

```python
memory = NeuralMemory()
await memory.initialize()

neuron_id = await memory.store(
    content="User prefers concise responses",
    memory_type=MemoryType.SEMANTIC,
    metadata={"source": "feedback", "confidence": 0.9}
)
```

##### `async recall(query: str, memory_type: Optional[MemoryType] = None, limit: int = 10) -> List[dict]`

Recall memories relevant to a query.

**Parameters:**
- `query`: Search query
- `memory_type`: Filter by memory type (optional)
- `limit`: Maximum results

**Returns:** List of memory dicts with `content`, `similarity`, `activation`, `metadata`

```python
memories = await memory.recall(
    query="user preferences",
    memory_type=MemoryType.SEMANTIC,
    limit=5
)
for m in memories:
    print(f"{m['content']} (similarity: {m['similarity']:.2f})")
```

##### `async recall_recent(limit: int = 10, memory_type: Optional[MemoryType] = None) -> List[dict]`

Recall most recent memories.

```python
recent = await memory.recall_recent(limit=5, memory_type=MemoryType.EPISODIC)
```

##### `async strengthen(neuron_id: str, amount: float = 0.1)`

Strengthen a memory (increase activation).

```python
await memory.strengthen(neuron_id, amount=0.2)
```

##### `async weaken(neuron_id: str, amount: float = 0.1)`

Weaken a memory (decrease activation).

```python
await memory.weaken(neuron_id, amount=0.1)
```

##### `async connect(source_id: str, target_id: str, weight: float = 0.5)`

Create or update a synapse between neurons.

```python
await memory.connect(neuron_id_1, neuron_id_2, weight=0.8)
```

##### `async get_connected(neuron_id: str, min_weight: float = 0.3) -> List[dict]`

Get neurons connected to a given neuron.

```python
connected = await memory.get_connected(neuron_id, min_weight=0.5)
```

##### `async decay()`

Apply decay to all neurons (called periodically).

```python
await memory.decay()
# Reduces activation of all neurons by decay_rate
```

##### `async consolidate()`

Consolidate memories: cluster, prune weak, merge duplicates.

```python
await memory.consolidate()
```

##### `async cluster_memories()`

Automatically cluster related memories.

```python
clusters = await memory.cluster_memories()
# Returns list of MemoryCluster objects
```

##### `get_stats() -> dict`

Get memory system statistics.

```python
stats = memory.get_stats()
# {
#   "total_neurons": 5432,
#   "total_synapses": 12000,
#   "total_clusters": 45,
#   "by_type": {
#     "working": 50,
#     "episodic": 2000,
#     "semantic": 3000,
#     "skill": 300,
#     "ancestor": 82
#   },
#   "avg_activation": 0.45,
#   "memory_mb": 256
# }
```

##### `async save(path: str)`

Save memory state to disk.

```python
await memory.save("memory_state.pkl")
```

##### `async load(path: str)`

Load memory state from disk.

```python
await memory.load("memory_state.pkl")
```

---

### MemoryCoordinator

Integrates all memory systems (neural, episodic, semantic, working).

```python
from src.memory.memory_coordinator import MemoryCoordinator
```

#### Constructor

```python
def __init__(
    self,
    neural_memory: Optional[NeuralMemory] = None,
    config: Optional[Dict] = None
)
```

#### Methods

##### `async initialize()`

Initialize all memory subsystems.

```python
coordinator = MemoryCoordinator()
await coordinator.initialize()
```

##### `async store_interaction(user_input: str, response: str, context: dict = None)`

Store a conversation interaction across memory systems.

```python
await coordinator.store_interaction(
    user_input="What's the capital of France?",
    response="Paris is the capital of France.",
    context={"topic": "geography", "successful": True}
)
```

##### `async get_relevant_context(query: str, max_items: int = 10) -> str`

Get relevant context for a query, formatted for LLM.

```python
context = await coordinator.get_relevant_context(
    query="Tell me about France",
    max_items=5
)
# Returns formatted string of relevant memories
```

##### `async get_working_memory() -> List[dict]`

Get current working memory contents.

```python
working = await coordinator.get_working_memory()
```

##### `async clear_working_memory()`

Clear working memory (start fresh context).

```python
await coordinator.clear_working_memory()
```

##### `async consolidate_all()`

Consolidate all memory systems.

```python
await coordinator.consolidate_all()
```

---

### Memory Types

#### Working Memory

Short-term memory for immediate context.

```python
# Store in working memory
await memory.store(
    "Current conversation about Python",
    MemoryType.WORKING,
    {"expires_in": 300}  # 5 minutes
)

# Automatically cleared or consolidated
```

#### Episodic Memory

Personal experiences and events.

```python
# Store an episodic memory
await memory.store(
    "User mentioned they're learning to code on 2024-01-15",
    MemoryType.EPISODIC,
    {"date": "2024-01-15", "emotion": "positive"}
)
```

#### Semantic Memory

Facts, knowledge, and concepts.

```python
# Store semantic knowledge
await memory.store(
    "Python is a programming language created by Guido van Rossum",
    MemoryType.SEMANTIC,
    {"category": "programming", "confidence": 1.0}
)
```

#### Skill Memory

Learned procedures and how-to knowledge.

```python
# Store a skill
await memory.store(
    "To format Python code, use black formatter: black filename.py",
    MemoryType.SKILL,
    {"tool": "black", "domain": "python"}
)
```

#### Ancestor Memory

Core personality and foundational knowledge.

```python
# Ancestor memories are typically pre-loaded
await memory.store(
    "I am AURA, an AI assistant focused on being helpful and honest",
    MemoryType.ANCESTOR,
    {"immutable": True}
)
```

---

## Tool System

### ToolRegistry

Registry for tool definitions using JSON Schema.

```python
from src.tools.registry import ToolRegistry, ToolDefinition
```

#### ToolDefinition Dataclass

```python
@dataclass
class ToolDefinition:
    name: str                      # Tool identifier
    description: str               # Human-readable description
    category: str                  # Category: communication, app_control, etc.
    parameters: Dict[str, Any]     # JSON Schema for parameters
    required: List[str]            # Required parameter names
    handler: str                   # Handler method name
    requires_confirmation: bool    # Needs user confirmation
    risk_level: str                # low, medium, high
    timeout_seconds: int           # Execution timeout
    enabled: bool                  # Is tool enabled
```

#### Tool Categories

| Category | Description | Examples |
|----------|-------------|----------|
| `communication` | Messaging and calls | send_message, make_call |
| `app_control` | App management | open_app, close_app |
| `screen` | Screen interaction | screenshot, tap, swipe |
| `information` | Data retrieval | get_weather, search_web |
| `utility` | System utilities | set_timer, calculate |

#### Constructor

```python
def __init__(self, handlers: Optional[object] = None)
```

#### Methods

##### `register(definition: ToolDefinition)`

Register a new tool.

```python
registry = ToolRegistry()

registry.register(ToolDefinition(
    name="get_weather",
    description="Get current weather for a location",
    category="information",
    parameters={
        "type": "object",
        "properties": {
            "location": {
                "type": "string",
                "description": "City name or coordinates"
            },
            "units": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "default": "celsius"
            }
        }
    },
    required=["location"],
    handler="handle_get_weather",
    requires_confirmation=False,
    risk_level="low",
    timeout_seconds=10,
    enabled=True
))
```

##### `get(name: str) -> Optional[ToolDefinition]`

Get a tool by name.

```python
tool = registry.get("get_weather")
```

##### `list_tools(category: Optional[str] = None, enabled_only: bool = True) -> List[ToolDefinition]`

List registered tools.

```python
# All enabled tools
all_tools = registry.list_tools()

# Tools by category
info_tools = registry.list_tools(category="information")

# Including disabled
all_including_disabled = registry.list_tools(enabled_only=False)
```

##### `get_tools_for_llm() -> List[dict]`

Get tools formatted for LLM function calling.

```python
tools_json = registry.get_tools_for_llm()
# Returns list of OpenAI-compatible function definitions
```

##### `enable(name: str)` / `disable(name: str)`

Enable or disable a tool.

```python
registry.disable("send_message")  # Disable for safety
registry.enable("send_message")   # Re-enable
```

##### `set_risk_filter(max_risk: str)`

Filter tools by risk level.

```python
registry.set_risk_filter("medium")  # Only low and medium risk tools
```

---

### ToolExecutor

Executes tools with validation, timeout, and error handling.

```python
from src.tools.registry import ToolExecutor
```

#### Constructor

```python
def __init__(
    self,
    registry: ToolRegistry,
    handlers: object,
    require_confirmation_callback: Optional[Callable] = None
)
```

#### Methods

##### `async execute(name: str, parameters: dict) -> dict`

Execute a tool by name with parameters.

**Returns:** `{"success": bool, "result": Any, "error": Optional[str]}`

```python
executor = ToolExecutor(registry, handlers)

result = await executor.execute(
    "get_weather",
    {"location": "San Francisco", "units": "celsius"}
)

if result["success"]:
    print(result["result"])
else:
    print(f"Error: {result['error']}")
```

##### `async execute_with_retry(name: str, parameters: dict, max_retries: int = 3) -> dict`

Execute with automatic retry on failure.

```python
result = await executor.execute_with_retry(
    "search_web",
    {"query": "Python tutorials"},
    max_retries=3
)
```

##### `validate_parameters(name: str, parameters: dict) -> Tuple[bool, Optional[str]]`

Validate parameters against tool schema.

```python
valid, error = executor.validate_parameters("get_weather", {"location": "NYC"})
if not valid:
    print(f"Invalid parameters: {error}")
```

---

### ToolHandlers

Actual implementations of tool functionality.

```python
from src.tools.handlers import ToolHandlers
```

#### Available Handlers

##### Communication

```python
# Send a message
await handlers.handle_send_message({
    "recipient": "John",
    "message": "Hello!",
    "app": "sms"  # sms, whatsapp, telegram
})

# Make a call
await handlers.handle_make_call({
    "contact": "John",
    "type": "voice"  # voice, video
})
```

##### App Control

```python
# Open an app
await handlers.handle_open_app({
    "app_name": "Chrome"
})

# Close an app
await handlers.handle_close_app({
    "app_name": "Chrome"
})

# List running apps
result = await handlers.handle_list_apps({})
```

##### Screen

```python
# Take screenshot
screenshot = await handlers.handle_screenshot({
    "save_path": "/tmp/screen.png"  # optional
})

# Tap at coordinates
await handlers.handle_tap({
    "x": 500,
    "y": 300
})

# Swipe gesture
await handlers.handle_swipe({
    "start_x": 500, "start_y": 800,
    "end_x": 500, "end_y": 200,
    "duration_ms": 300
})
```

##### Information

```python
# Get weather
weather = await handlers.handle_get_weather({
    "location": "New York"
})

# Search web
results = await handlers.handle_search_web({
    "query": "Python async tutorial",
    "num_results": 5
})

# Get time
time_info = await handlers.handle_get_time({
    "timezone": "America/New_York"
})
```

##### Utility

```python
# Set timer
await handlers.handle_set_timer({
    "duration_seconds": 300,
    "label": "Break time"
})

# Calculate
result = await handlers.handle_calculate({
    "expression": "sqrt(144) + 10^2"
})

# Set reminder
await handlers.handle_set_reminder({
    "message": "Call mom",
    "time": "2024-01-20T15:00:00"
})
```

---

### Tool Registration API

#### Registering Custom Tools

```python
from src.tools.registry import ToolRegistry, ToolDefinition

# 1. Create registry
registry = ToolRegistry()

# 2. Define your tool
my_tool = ToolDefinition(
    name="my_custom_tool",
    description="Does something custom",
    category="utility",
    parameters={
        "type": "object",
        "properties": {
            "input": {
                "type": "string",
                "description": "Input value"
            },
            "option": {
                "type": "boolean",
                "default": False
            }
        }
    },
    required=["input"],
    handler="handle_my_custom_tool",
    requires_confirmation=False,
    risk_level="low",
    timeout_seconds=30,
    enabled=True
)

# 3. Register it
registry.register(my_tool)
```

#### Creating Custom Handlers

```python
class MyCustomHandlers:
    async def handle_my_custom_tool(self, params: dict) -> dict:
        input_val = params["input"]
        option = params.get("option", False)
        
        # Your logic here
        result = process_input(input_val, option)
        
        return {
            "success": True,
            "result": result
        }

# Attach to registry
handlers = MyCustomHandlers()
registry = ToolRegistry(handlers=handlers)
executor = ToolExecutor(registry, handlers)
```

#### Dynamic Tool Registration

```python
# Register tools from a config file
import yaml

with open("tools_config.yaml") as f:
    tools_config = yaml.safe_load(f)

for tool_conf in tools_config["tools"]:
    registry.register(ToolDefinition(**tool_conf))
```

---

## REST API

AURA provides a REST API server for external integrations.

```python
from src.api.server import AuraAPIServer, SecurityConfig
```

### Authentication

#### Security Configuration

```python
security_config = SecurityConfig(
    require_auth=True,
    auth_token="your-secret-token",  # Or None for auto-generated
    localhost_only=True,             # Bind to 127.0.0.1 only
    rate_limit_requests=30,          # Max requests
    rate_limit_window_seconds=60,    # Time window
    allowed_origins=["http://localhost:3000"]  # CORS origins
)
```

#### Bearer Token Authentication

Protected endpoints require the `Authorization` header:

```bash
curl -X POST http://localhost:8080/api/chat \
  -H "Authorization: Bearer your-secret-token" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello"}'
```

### Server Setup

```python
from src.api.server import AuraAPIServer

server = AuraAPIServer(
    aura=aura_instance,
    host="127.0.0.1",
    port=8080,
    security_config=security_config
)

await server.start()
```

### Endpoints

#### Public Endpoints (No Auth Required)

##### `GET /api/status`

Get AURA status.

**Response:**
```json
{
  "status": "running",
  "version": "3.0.0",
  "uptime_seconds": 3600,
  "initialized": true
}
```

##### `GET /api/inner-voice`

Get inner voice/thinking state.

**Response:**
```json
{
  "thoughts": ["Processing context...", "Formulating response..."],
  "active": true,
  "last_update": "2024-01-15T10:30:00Z"
}
```

##### `GET /api/inner-voice/logs`

Get inner voice logs.

**Query Parameters:**
- `limit`: Number of entries (default: 50)
- `since`: ISO timestamp filter

**Response:**
```json
{
  "logs": [
    {"timestamp": "...", "thought": "...", "type": "reasoning"},
    ...
  ]
}
```

##### `GET /api/feeling`

Get emotional state.

**Response:**
```json
{
  "mood": "curious",
  "energy": 0.8,
  "engagement": 0.9,
  "sentiment": "positive"
}
```

##### `GET /api/trust`

Get trust level.

**Response:**
```json
{
  "trust_level": 0.85,
  "factors": {
    "interaction_count": 150,
    "positive_feedback": 0.9,
    "time_known_days": 30
  }
}
```

##### `GET /api/auth/status`

Check authentication status.

**Response:**
```json
{
  "authenticated": false,
  "auth_required": true
}
```

#### Protected Endpoints (Auth Required)

##### `POST /api/chat`

Send a message to AURA.

**Request:**
```json
{
  "message": "What's the weather like?",
  "stream": false,
  "context": {
    "conversation_id": "abc123"
  }
}
```

**Response:**
```json
{
  "response": "I'd be happy to check the weather for you...",
  "conversation_id": "abc123",
  "tools_used": ["get_weather"],
  "processing_time_ms": 250
}
```

**Streaming Response** (if `stream: true`):
```
data: {"chunk": "I'd be ", "done": false}
data: {"chunk": "happy to ", "done": false}
data: {"chunk": "help!", "done": true}
```

##### `POST /api/feedback`

Submit feedback on a response.

**Request:**
```json
{
  "conversation_id": "abc123",
  "rating": 5,
  "comment": "Very helpful response",
  "type": "positive"
}
```

**Response:**
```json
{
  "success": true,
  "message": "Feedback recorded"
}
```

### Rate Limiting

The API implements rate limiting per client:

```python
# Default: 30 requests per 60 seconds
rate_limiter = RateLimiter(
    max_requests=30,
    window_seconds=60
)
```

**Rate Limit Headers:**
```
X-RateLimit-Limit: 30
X-RateLimit-Remaining: 25
X-RateLimit-Reset: 1705312800
```

**Rate Limited Response (429):**
```json
{
  "error": "Rate limit exceeded",
  "retry_after": 45
}
```

---

## Agent Loop Hooks

AURA provides hooks for customizing the agent processing loop.

### Available Hooks

```python
from src.core import AgentHooks

class MyHooks(AgentHooks):
    async def on_input_received(self, input_text: str) -> str:
        """Called when user input is received. Can modify input."""
        # Log, filter, or transform input
        return input_text
    
    async def on_context_gathered(self, context: dict) -> dict:
        """Called after context is gathered. Can modify context."""
        # Add custom context
        context["custom_data"] = get_custom_data()
        return context
    
    async def on_tools_selected(self, tools: List[str]) -> List[str]:
        """Called when tools are selected. Can filter tools."""
        # Remove sensitive tools
        return [t for t in tools if t != "send_message"]
    
    async def on_tool_executed(self, tool_name: str, result: dict):
        """Called after a tool executes."""
        # Log tool usage
        log_tool_usage(tool_name, result)
    
    async def on_response_generated(self, response: str) -> str:
        """Called before response is returned. Can modify response."""
        # Filter, format, or log
        return response
    
    async def on_memory_stored(self, memory_type: str, content: str):
        """Called when memory is stored."""
        # Track memory operations
        pass
    
    async def on_error(self, error: Exception, stage: str):
        """Called when an error occurs."""
        # Custom error handling
        log_error(error, stage)
```

### Registering Hooks

```python
from src.main import AuraProduction

aura = AuraProduction()
await aura.initialize()

# Register hooks
aura.register_hooks(MyHooks())

# Or register individual hook functions
aura.on_input_received(my_input_handler)
aura.on_response_generated(my_response_handler)
```

### Hook Execution Order

1. `on_input_received` - Input preprocessing
2. `on_context_gathered` - Context augmentation
3. `on_tools_selected` - Tool filtering
4. `on_tool_executed` - Per-tool callback (multiple times)
5. `on_memory_stored` - Memory tracking
6. `on_response_generated` - Response postprocessing

---

## Configuration

### Configuration File (config.yaml)

```yaml
# AURA v3 Configuration

# Core LLM Settings
core:
  llm:
    provider: llama.cpp           # llama.cpp, ollama, transformers
    model_path: models/phi-3.gguf # Path to model file
    max_tokens: 512               # Maximum response tokens
    temperature: 0.7              # Sampling temperature (0.0-2.0)
    top_p: 0.9                    # Nucleus sampling
    top_k: 40                     # Top-k sampling
    repeat_penalty: 1.1           # Repetition penalty
    n_gpu_layers: -1              # GPU layers (-1 = all)
    n_ctx: 4096                   # Context window size
    n_batch: 512                  # Batch size
    preload: true                 # Preload model at startup
    
  stt:
    enabled: true
    model: whisper-small          # whisper-tiny, whisper-small, whisper-medium
    language: en
    
  tts:
    enabled: true
    model: piper                  # piper, coqui
    voice: en_US-lessac-medium

# Memory Settings
memory:
  neural:
    max_neurons: 10000
    decay_rate: 0.01
    cluster_threshold: 0.7
    embedding_model: all-MiniLM-L6-v2
    
  persistence:
    enabled: true
    path: data/memory
    autosave_interval: 300        # seconds
    
  consolidation:
    enabled: true
    interval: 3600                # seconds
    min_activation: 0.1           # prune below this

# Power Management
power:
  full_power_threshold: 80        # Battery % for full power
  balanced_threshold: 50          # Battery % for balanced
  power_save_threshold: 20        # Battery % for power save
  critical_threshold: 10          # Battery % for critical
  
  max_tokens_full: 1024
  max_tokens_balanced: 512
  max_tokens_save: 256
  max_tokens_ultra: 128
  
  thermal_throttle_temp: 80       # Celsius

# Privacy Settings
privacy:
  log_level: INFO                 # DEBUG, INFO, WARNING, ERROR
  store_conversations: true
  anonymize_data: false
  allow_telemetry: false
  local_only: true                # No external API calls

# Feature Toggles
features:
  voice_enabled: true
  voice_language: en-US
  proactive_mode: true
  background_tasks: true
  notifications: true
  inner_voice: true
  tools_enabled: true

# Security Settings
security:
  require_auth: true
  auth_method: token              # token, pin, biometric
  session_timeout: 3600           # seconds
  max_failed_attempts: 5
  api_localhost_only: true

# API Server
api:
  enabled: true
  host: 127.0.0.1
  port: 8080
  rate_limit_requests: 30
  rate_limit_window: 60

# Logging
logging:
  level: INFO
  file: logs/aura.log
  max_size_mb: 100
  backup_count: 5
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
```

### Loading Configuration

```python
import yaml
from src.main import AuraProduction

# Default config
aura = AuraProduction()

# Custom config path
aura = AuraProduction(config_path="/path/to/config.yaml")

# Load config manually
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Access config values
llm_config = config["core"]["llm"]
memory_config = config["memory"]
```

### Environment Variables

Configuration can be overridden with environment variables:

```bash
export AURA_LLM_MODEL_PATH=/path/to/model.gguf
export AURA_LLM_MAX_TOKENS=1024
export AURA_API_PORT=9000
export AURA_LOG_LEVEL=DEBUG
```

---

## Core Systems

### NeuromorphicEngine

Event-driven processing engine.

```python
from src.core import get_neuromorphic_engine

engine = get_neuromorphic_engine()
await engine.start()

# Emit events
await engine.emit_user_message("Hello")
status = engine.get_status()
```

### MobilePowerManager

Battery and power management.

```python
from src.core import get_power_manager

power = get_power_manager()
mode = power.get_power_mode()         # PowerMode.BALANCED
limits = power.get_processing_limits() # {"max_tokens": 512, ...}
should_run = power.should_process(priority=7)
```

### DeepUserProfiler

User profile and preference learning.

```python
from src.core import get_user_profiler

profiler = get_user_profiler()
context = await profiler.get_current_context()
preferences = await profiler.get_preferences()
```

### AdaptivePersonalityEngine

Adaptive personality system.

```python
from src.core import get_personality_engine

personality = get_personality_engine()
state = await personality.get_state()
formatted = await personality.format_response("Hi!", state)
```

### ContextProvider

Real-time context gathering.

```python
from src.context import get_context_provider

context_provider = get_context_provider()
await context_provider.start()
context = await context_provider.get_current_context()
llm_context = await context_provider.get_context_for_llm()
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

### With Streaming

```python
import asyncio
from src.main import AuraProduction

async def main():
    aura = AuraProduction()
    await aura.initialize()
    await aura.start()
    
    # Stream response
    print("AURA: ", end="")
    async for chunk in await aura.process("Tell me a story", stream=True):
        print(chunk, end="", flush=True)
    print()
    
    await aura.stop()

asyncio.run(main())
```

### Custom Tool Registration

```python
import asyncio
from src.main import AuraProduction
from src.tools.registry import ToolRegistry, ToolDefinition, ToolExecutor

class CustomHandlers:
    async def handle_greet(self, params: dict) -> dict:
        name = params.get("name", "friend")
        return {"success": True, "result": f"Hello, {name}!"}

async def main():
    aura = AuraProduction()
    await aura.initialize()
    
    # Get tool registry
    registry = aura._tool_registry
    
    # Register custom tool
    registry.register(ToolDefinition(
        name="greet",
        description="Greet someone by name",
        category="utility",
        parameters={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "Name to greet"}
            }
        },
        required=[],
        handler="handle_greet",
        requires_confirmation=False,
        risk_level="low",
        timeout_seconds=5,
        enabled=True
    ))
    
    # Add handler
    handlers = CustomHandlers()
    executor = ToolExecutor(registry, handlers)
    
    result = await executor.execute("greet", {"name": "World"})
    print(result)  # {"success": True, "result": "Hello, World!"}

asyncio.run(main())
```

### Memory Operations

```python
import asyncio
from src.memory.neural_memory import NeuralMemory, MemoryType

async def main():
    memory = NeuralMemory()
    await memory.initialize()
    
    # Store memories
    id1 = await memory.store(
        "User likes Python programming",
        MemoryType.SEMANTIC,
        {"confidence": 0.9}
    )
    
    id2 = await memory.store(
        "Had a conversation about web scraping yesterday",
        MemoryType.EPISODIC,
        {"date": "2024-01-14"}
    )
    
    # Connect related memories
    await memory.connect(id1, id2, weight=0.7)
    
    # Recall memories
    results = await memory.recall("Python", limit=5)
    for r in results:
        print(f"- {r['content']} (sim: {r['similarity']:.2f})")
    
    # Get connected memories
    connected = await memory.get_connected(id1)
    print(f"Connected to '{memory.get(id1).content}':")
    for c in connected:
        print(f"  - {c['content']}")
    
    # Save state
    await memory.save("memory_state.pkl")

asyncio.run(main())
```

### REST API Client

```python
import aiohttp
import asyncio

async def main():
    base_url = "http://localhost:8080"
    token = "your-secret-token"
    
    async with aiohttp.ClientSession() as session:
        # Check status (no auth needed)
        async with session.get(f"{base_url}/api/status") as resp:
            status = await resp.json()
            print(f"Status: {status}")
        
        # Send message (auth required)
        headers = {"Authorization": f"Bearer {token}"}
        payload = {"message": "Hello AURA!", "stream": False}
        
        async with session.post(
            f"{base_url}/api/chat",
            json=payload,
            headers=headers
        ) as resp:
            result = await resp.json()
            print(f"Response: {result['response']}")
        
        # Get inner voice
        async with session.get(f"{base_url}/api/inner-voice") as resp:
            voice = await resp.json()
            print(f"Thoughts: {voice['thoughts']}")

asyncio.run(main())
```

### Agent Loop Hooks

```python
import asyncio
from src.main import AuraProduction
from src.core import AgentHooks

class LoggingHooks(AgentHooks):
    async def on_input_received(self, input_text: str) -> str:
        print(f"[INPUT] {input_text}")
        return input_text
    
    async def on_response_generated(self, response: str) -> str:
        print(f"[OUTPUT] {response[:100]}...")
        return response
    
    async def on_tool_executed(self, tool_name: str, result: dict):
        print(f"[TOOL] {tool_name}: {result['success']}")

async def main():
    aura = AuraProduction()
    await aura.initialize()
    
    # Register hooks
    aura.register_hooks(LoggingHooks())
    
    await aura.start()
    
    response = await aura.process("What's the weather in NYC?")
    # [INPUT] What's the weather in NYC?
    # [TOOL] get_weather: True
    # [OUTPUT] The weather in New York City is...
    
    await aura.stop()

asyncio.run(main())
```

---

## Error Handling

AURA uses standard Python exceptions. Common exceptions:

```python
from src.core.exceptions import (
    AuraError,           # Base exception
    InitializationError, # Startup failures
    LLMError,            # LLM-related errors
    MemoryError,         # Memory system errors
    ToolError,           # Tool execution errors
    AuthenticationError, # API auth failures
    RateLimitError       # Rate limiting
)

try:
    response = await aura.process("Hello")
except LLMError as e:
    print(f"LLM Error: {e}")
except MemoryError as e:
    print(f"Memory Error: {e}")
except AuraError as e:
    print(f"AURA Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
```

---

## Thread Safety

AURA is designed to be async-first and is **not thread-safe**. Always use from the main async event loop:

```python
# Correct
asyncio.run(main())

# Incorrect - don't call from multiple threads
# threading.Thread(target=lambda: asyncio.run(aura.process("Hi"))).start()
```

For concurrent requests, use async concurrency:

```python
async def handle_multiple():
    tasks = [
        aura.process("Question 1"),
        aura.process("Question 2"),
        aura.process("Question 3"),
    ]
    results = await asyncio.gather(*tasks)
    return results
```

---

## Version

Current version: **3.0.0**

See [CHANGELOG.md](CHANGELOG.md) for version history.

---

## License

AURA v3 is proprietary software. See LICENSE for details.
