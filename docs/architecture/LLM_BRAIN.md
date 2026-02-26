# LLM Brain Architecture

> **The LLM as AURA's Cognitive Core**

This document describes how the Large Language Model serves as AURA v3's "brain" - the central intelligence that orchestrates all reasoning, decision-making, and tool coordination.

---

## Table of Contents

1. [Conceptual Overview](#conceptual-overview)
2. [Architecture Layers](#architecture-layers)
3. [Local vs Cloud Model Selection](#local-vs-cloud-model-selection)
4. [Model Registry & Recommendations](#model-registry--recommendations)
5. [Prompt Engineering Patterns](#prompt-engineering-patterns)
6. [Context Injection from Memory](#context-injection-from-memory)
7. [Tool Calling Flow](#tool-calling-flow)
8. [Response Parsing](#response-parsing)
9. [Fallback Strategies](#fallback-strategies)
10. [Token Budget Management](#token-budget-management)
11. [Performance & Monitoring](#performance--monitoring)
12. [File Reference](#file-reference)

---

## Conceptual Overview

AURA's LLM is not just a text generator - it's the **cognitive orchestrator** that:

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER REQUEST                              │
└─────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      LLM BRAIN (ReAct Loop)                      │
│  ┌─────────┐   ┌─────────┐   ┌─────────────┐   ┌─────────────┐  │
│  │ THOUGHT │ → │ ACTION  │ → │ OBSERVATION │ → │  RESPONSE   │  │
│  │(Reason) │   │ (Tool)  │   │  (Result)   │   │  (Answer)   │  │
│  └─────────┘   └─────────┘   └─────────────┘   └─────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              ┌─────────┐ ┌─────────┐ ┌─────────┐
              │ Memory  │ │  Tools  │ │ Context │
              │ System  │ │ (20+)   │ │ Window  │
              └─────────┘ └─────────┘ └─────────┘
```

**Key Responsibilities:**
- **Intent Understanding**: Parse natural language into actionable intents
- **Tool Selection**: Choose appropriate tools for each task
- **Parameter Extraction**: Extract tool parameters from conversation
- **Multi-step Reasoning**: Chain multiple tools for complex tasks
- **Error Recovery**: Self-reflect and retry with different strategies
- **Response Synthesis**: Generate human-friendly responses

---

## Architecture Layers

### Layer 1: ProductionLLM (`src/llm/production_llm.py`)
The canonical, production-grade LLM backend supporting multiple inference engines.

```python
class ProductionLLM:
    """
    Production-grade LLM with:
    - 4 backends: LLAMA_CPP, GPT4ALL, OLLAMA, TRANSFORMERS
    - Real memory calculations
    - Thermal throttling awareness
    - Streaming generation
    - Fallback chains
    """
```

### Layer 2: LLMManager (`src/llm/manager.py`)
Higher-level manager handling model lifecycle and multi-modal support.

```python
class LLMManager:
    """
    Manages:
    - LLM model loading/unloading
    - STT (Whisper) integration
    - TTS (Piper) integration
    - Unified interface for all model types
    """
```

### Layer 3: LLMRunner (`src/llm/__init__.py`)
Backward-compatible wrapper providing simple API for agent loop.

```python
class LLMRunner:
    async def generate(self, messages: List[Dict]) -> Dict:
        """Returns: {"content": str, "usage": dict, "model": str}"""
    
    async def generate_with_tools(self, messages, tools) -> Dict:
        """Tool-aware generation with JSON schema injection"""
```

### Layer 4: AgentLoop (`src/agent/loop.py`)
The ReAct reasoning loop that uses LLM for decision-making.

```python
class AgentLoop:
    async def run(self, user_input: str) -> str:
        # ReAct: Thought → Action → Observation → loop
```

---

## Local vs Cloud Model Selection

AURA is designed as a **local-first** assistant. All LLM inference runs on-device.

### Supported Backends

| Backend | Library | Best For | GPU Support |
|---------|---------|----------|-------------|
| `LLAMA_CPP` | llama-cpp-python | Quantized GGUF models | CUDA, Metal, ROCm |
| `GPT4ALL` | gpt4all | Easy setup, CPU-optimized | Limited |
| `OLLAMA` | ollama | Model management, API | Full |
| `TRANSFORMERS` | transformers | HuggingFace models | Full |

### Backend Selection Logic

```python
# From production_llm.py - Backend priority
def _get_available_backend(self) -> Optional[LLMBackend]:
    # Priority: LLAMA_CPP > OLLAMA > GPT4ALL > TRANSFORMERS
    for backend in [LLMBackend.LLAMA_CPP, LLMBackend.OLLAMA, 
                    LLMBackend.GPT4ALL, LLMBackend.TRANSFORMERS]:
        if self._check_backend_available(backend):
            return backend
    return None
```

### Why Local-First?
1. **Privacy**: User data never leaves device
2. **Offline**: Works without internet
3. **Speed**: No network latency
4. **Cost**: No API fees
5. **Control**: Full model customization

---

## Model Registry & Recommendations

### Default Model Registry

```python
DEFAULT_MODEL_REGISTRY = {
    "qwen2.5-1.5b-q4": {
        "name": "Qwen 2.5 1.5B Q4",
        "size_gb": 1.0,
        "context_length": 32768,
        "quantization": "Q4_K_M",
        "capabilities": ["chat", "reasoning", "tool_use"],
        "recommended_ram_gb": 4,
        "performance_tier": "balanced"
    },
    "llama-3.2-1b-q4": {
        "name": "Llama 3.2 1B Q4", 
        "size_gb": 0.7,
        "context_length": 8192,
        "quantization": "Q4_K_M",
        "capabilities": ["chat", "tool_use"],
        "recommended_ram_gb": 2,
        "performance_tier": "fast"
    },
    "deepseek-r1-1.5b-q4": {
        "name": "DeepSeek-R1 1.5B Q4",
        "size_gb": 1.0,
        "context_length": 16384,
        "quantization": "Q4_K_M",
        "capabilities": ["chat", "reasoning", "tool_use", "math"],
        "recommended_ram_gb": 4,
        "performance_tier": "reasoning"
    }
}
```

### Model Recommendation Engine

The system recommends models based on device capabilities:

```python
def get_recommended_models(self) -> List[Dict[str, Any]]:
    """
    Recommends models based on:
    - Available RAM (real-time check)
    - Thermal state (throttling awareness)
    - Required capabilities
    - Performance tier preference
    """
    available_ram = self._get_available_ram_gb()
    recommendations = []
    
    for model_id, spec in self.model_registry.items():
        required_ram = self._calculate_required_ram(spec)
        if required_ram <= available_ram * 0.8:  # 80% safety margin
            recommendations.append({
                "model_id": model_id,
                "fits_in_ram": True,
                "headroom_gb": available_ram - required_ram
            })
    
    return sorted(recommendations, key=lambda x: x["headroom_gb"], reverse=True)
```

### RAM Calculation Formula

```python
def _calculate_required_ram(self, model_spec: Dict) -> float:
    """
    Formula: model_size * quant_factor + context_memory + overhead
    
    Where:
    - quant_factor: Q4=0.5, Q5=0.625, Q8=1.0, FP16=2.0
    - context_memory: context_length * 4 bytes * 2 (KV cache)
    - overhead: 10% for runtime allocations
    """
    base_size = model_spec["size_gb"]
    quant_factor = QUANTIZATION_FACTORS.get(model_spec["quantization"], 1.0)
    context_bytes = model_spec["context_length"] * 4 * 2
    context_gb = context_bytes / (1024**3)
    
    total = (base_size * quant_factor) + context_gb
    return total * 1.1  # 10% overhead
```

---

## Prompt Engineering Patterns

### System Prompt Structure

```python
SYSTEM_PROMPT = """You are AURA, an intelligent AI assistant running locally on the user's device.

CURRENT CONTEXT:
- Date/Time: {current_datetime}
- Location: {user_location}
- Device: {device_info}

CAPABILITIES:
You have access to these tools:
{tool_descriptions}

RESPONSE FORMAT:
You must respond in JSON format:
- For tool calls: {{"type": "tool_call", "tool": "tool_name", "parameters": {{}}, "reasoning": "why"}}
- For responses: {{"type": "response", "content": "your response"}}

GUIDELINES:
1. Think step-by-step before acting
2. Use tools when needed, respond directly when you can
3. Be concise but helpful
4. If uncertain, ask for clarification
"""
```

### Tool Schema Injection

Tools are injected as JSON schemas:

```python
def _format_tools_for_prompt(self, tools: List[Dict]) -> str:
    """
    Formats tools as:
    
    AVAILABLE TOOLS:
    
    1. send_message
       Description: Send SMS or messaging app message
       Parameters:
         - recipient (required): Contact name or phone number
         - message (required): Message content
         - app (optional): "sms", "whatsapp", "telegram"
    
    2. search_web
       ...
    """
```

### ReAct Prompt Pattern

The agent loop uses ReAct (Reasoning + Acting):

```python
REACT_PROMPT = """
Based on the conversation, decide your next action.

THINK: Analyze what the user wants and what information you have.
ACT: Either use a tool or respond directly.

Previous attempts (if any):
{previous_attempts}

Remember:
- If a tool failed, try an alternative approach
- If you have enough information, respond directly
- Always explain your reasoning
"""
```

---

## Context Injection from Memory

### Memory Integration Points

```python
class AgentLoop:
    async def _build_context(self, user_input: str) -> List[Dict]:
        """
        Builds context from multiple sources:
        1. System prompt with current state
        2. Relevant memories (semantic search)
        3. Recent conversation history
        4. Tool results from current session
        """
        context = []
        
        # 1. System prompt
        context.append({
            "role": "system",
            "content": self._build_system_prompt()
        })
        
        # 2. Memory injection (if memory system available)
        if self.memory:
            relevant = await self.memory.search(user_input, limit=5)
            if relevant:
                context.append({
                    "role": "system", 
                    "content": f"RELEVANT MEMORIES:\n{self._format_memories(relevant)}"
                })
        
        # 3. Conversation history (trimmed)
        context.extend(self._trim_messages(self.messages))
        
        # 4. Current user input
        context.append({"role": "user", "content": user_input})
        
        return context
```

### Memory Search Integration

```python
# Semantic search for relevant context
relevant_memories = await memory.search(
    query=user_input,
    limit=5,
    threshold=0.7,  # Similarity threshold
    types=["fact", "preference", "conversation"]
)

# Format for injection
memory_context = "\n".join([
    f"- {m['content']} (confidence: {m['confidence']:.0%})"
    for m in relevant_memories
])
```

---

## Tool Calling Flow

### Complete Flow Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│                         TOOL CALLING FLOW                         │
└──────────────────────────────────────────────────────────────────┘

User: "Send a message to John saying I'll be late"
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. LLM REASONING                                                 │
│    Input: User message + Tool schemas + Context                  │
│    Output: JSON decision                                         │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. RESPONSE PARSING                                              │
│    {                                                             │
│      "type": "tool_call",                                        │
│      "tool": "send_message",                                     │
│      "parameters": {                                             │
│        "recipient": "John",                                      │
│        "message": "I'll be late"                                 │
│      },                                                          │
│      "reasoning": "User wants to send SMS to John"               │
│    }                                                             │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. SECURITY CHECK                                                │
│    - Is tool in allowed list?                                    │
│    - Are parameters safe?                                        │
│    - Does user have permission?                                  │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. TOOL EXECUTION                                                │
│    result = await tool_registry.execute(                         │
│        "send_message",                                           │
│        {"recipient": "John", "message": "I'll be late"}          │
│    )                                                             │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. OBSERVATION INJECTION                                         │
│    Add to context: {"role": "tool", "content": result}           │
└─────────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. LLM SYNTHESIS                                                 │
│    Generate human-friendly response based on tool result         │
│    "I've sent your message to John letting him know you'll       │
│     be late."                                                    │
└─────────────────────────────────────────────────────────────────┘
```

### Tool Execution Code

```python
async def _execute_tool(self, tool_name: str, parameters: Dict) -> Dict:
    """Execute a tool with security checks and error handling."""
    
    # 1. Security check
    if not self._is_tool_allowed(tool_name):
        return {"success": False, "error": f"Tool '{tool_name}' not allowed"}
    
    # 2. Get tool from registry
    tool = self.tool_registry.get(tool_name)
    if not tool:
        return {"success": False, "error": f"Tool '{tool_name}' not found"}
    
    # 3. Validate parameters
    validation = tool.validate_parameters(parameters)
    if not validation.valid:
        return {"success": False, "error": validation.error}
    
    # 4. Execute with timeout
    try:
        async with asyncio.timeout(30):  # 30 second timeout
            result = await tool.execute(**parameters)
        return {"success": True, "result": result}
    except asyncio.TimeoutError:
        return {"success": False, "error": "Tool execution timed out"}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

---

## Response Parsing

### JSON Response Format

The LLM must respond in one of two formats:

```python
# Tool call format
{
    "type": "tool_call",
    "tool": "tool_name",
    "parameters": {
        "param1": "value1",
        "param2": "value2"
    },
    "reasoning": "Explanation of why this tool was chosen"
}

# Direct response format
{
    "type": "response",
    "content": "The response to show the user"
}
```

### Parsing Implementation

```python
def _parse_llm_response(self, response: str) -> Dict:
    """
    Parse LLM response with multiple fallback strategies.
    """
    # 1. Try direct JSON parse
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        pass
    
    # 2. Extract JSON from markdown code blocks
    json_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', response)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass
    
    # 3. Find JSON object in text
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass
    
    # 4. Fallback: treat as direct response
    return {
        "type": "response",
        "content": response.strip()
    }
```

### Confidence Scoring

```python
def _calculate_confidence(self, response: str) -> float:
    """
    Calculate confidence based on uncertain phrases.
    """
    uncertain_phrases = [
        "i think", "maybe", "probably", "not sure",
        "might be", "could be", "possibly", "i believe",
        "it seems", "appears to"
    ]
    
    response_lower = response.lower()
    uncertainty_count = sum(
        1 for phrase in uncertain_phrases 
        if phrase in response_lower
    )
    
    # Base confidence 1.0, reduce by 0.1 for each uncertain phrase
    confidence = max(0.3, 1.0 - (uncertainty_count * 0.1))
    return confidence
```

---

## Fallback Strategies

### Tool Fallback Chains

When a tool fails, the system tries alternatives:

```python
FALLBACK_STRATEGIES = {
    # If get_contacts fails, try these alternatives
    "get_contacts": ["search_contacts", "list_contacts"],
    
    # If send_sms fails, try other messaging
    "send_sms": ["send_message", "send_notification"],
    
    # If web_search fails, try different engines  
    "search_web": ["search_duckduckgo", "search_brave"],
    
    # If get_location fails
    "get_location": ["get_last_known_location", "ask_user_location"],
    
    # Calendar fallbacks
    "get_calendar_events": ["list_events", "search_calendar"],
}
```

### Self-Reflection on Failure

```python
async def _handle_tool_failure(self, tool_name: str, error: str) -> str:
    """
    When a tool fails, ask LLM to reflect and suggest alternatives.
    """
    reflection_prompt = f"""
    The tool '{tool_name}' failed with error: {error}
    
    Please reflect on this failure and either:
    1. Try an alternative tool from: {FALLBACK_STRATEGIES.get(tool_name, [])}
    2. Ask the user for more information
    3. Provide a helpful response explaining the limitation
    
    What would you like to do?
    """
    
    # Add failure to context for learning
    self.messages.append({
        "role": "system",
        "content": f"TOOL FAILURE: {tool_name} - {error}"
    })
    
    # Get LLM's reflection
    response = await self.llm.generate([
        *self.messages,
        {"role": "user", "content": reflection_prompt}
    ])
    
    return response["content"]
```

### Model Fallback Chain

```python
class ProductionLLM:
    def __init__(self):
        self.fallback_chain = [
            "qwen2.5-1.5b-q4",      # Primary: balanced
            "llama-3.2-1b-q4",      # Fallback 1: smaller/faster
            "phi-3-mini-q4",        # Fallback 2: even smaller
        ]
    
    async def generate_with_fallback(self, prompt: str) -> str:
        """Try each model in fallback chain until one succeeds."""
        for model_id in self.fallback_chain:
            try:
                if self._can_load_model(model_id):
                    await self._ensure_model_loaded(model_id)
                    return await self._generate(prompt)
            except Exception as e:
                logger.warning(f"Model {model_id} failed: {e}, trying next...")
                continue
        
        raise RuntimeError("All models in fallback chain failed")
```

---

## Token Budget Management

### Context Window Management

```python
class AgentLoop:
    def __init__(self):
        self.max_context_tokens = 4096    # Default context budget
        self.max_history_messages = 20     # Message history limit
        self.reserved_tokens = 512         # Reserved for response
    
    def _trim_messages(self, messages: List[Dict]) -> List[Dict]:
        """
        Trim message history to fit context window.
        
        Strategy:
        1. Always keep system prompt
        2. Always keep last user message
        3. Keep most recent messages up to limit
        4. Summarize older messages if needed
        """
        if len(messages) <= self.max_history_messages:
            return messages
        
        # Keep system prompt (first message)
        system_msgs = [m for m in messages if m["role"] == "system"]
        
        # Keep recent messages
        recent = messages[-self.max_history_messages:]
        
        # Combine
        return system_msgs + recent
```

### Token Counting

```python
def _estimate_tokens(self, text: str) -> int:
    """
    Estimate token count (rough approximation).
    More accurate: use tiktoken or model's tokenizer.
    """
    # Rough estimate: ~4 characters per token for English
    return len(text) // 4

def _get_context_usage(self, messages: List[Dict]) -> Dict:
    """Get current context window usage."""
    total_chars = sum(len(m.get("content", "")) for m in messages)
    estimated_tokens = self._estimate_tokens(total_chars)
    
    return {
        "used_tokens": estimated_tokens,
        "max_tokens": self.max_context_tokens,
        "available_tokens": self.max_context_tokens - estimated_tokens,
        "usage_percent": (estimated_tokens / self.max_context_tokens) * 100
    }
```

### Streaming for Perceived Speed

```python
async def generate_streaming(self, prompt: str) -> AsyncIterator[str]:
    """
    Stream tokens for faster perceived response.
    
    Benefits:
    - User sees response immediately
    - Can cancel early if off-track
    - Better UX for long responses
    """
    async for token in self.model.generate_stream(prompt):
        yield token
        
        # Check for stop conditions
        if self._should_stop_generation(token):
            break
```

---

## Performance & Monitoring

### Latency Tracking

```python
class ProductionLLM:
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "total_tokens_generated": 0,
            "average_latency_ms": 0,
            "p95_latency_ms": 0,
            "errors": 0
        }
        self._latencies = []  # Rolling window
    
    async def generate(self, prompt: str) -> str:
        start = time.perf_counter()
        
        try:
            result = await self._generate_internal(prompt)
            
            # Track metrics
            latency_ms = (time.perf_counter() - start) * 1000
            self._latencies.append(latency_ms)
            self._latencies = self._latencies[-100:]  # Keep last 100
            
            self.metrics["total_requests"] += 1
            self.metrics["average_latency_ms"] = sum(self._latencies) / len(self._latencies)
            self.metrics["p95_latency_ms"] = sorted(self._latencies)[int(len(self._latencies) * 0.95)]
            
            return result
        except Exception as e:
            self.metrics["errors"] += 1
            raise
```

### Thermal Throttling Awareness

```python
def _check_thermal_state(self) -> str:
    """
    Check device thermal state for mobile optimization.
    
    Returns: "nominal", "fair", "serious", "critical"
    """
    # Platform-specific thermal checking
    if platform.system() == "Linux" and os.path.exists("/sys/class/thermal"):
        # Read thermal zones
        for zone in Path("/sys/class/thermal").glob("thermal_zone*"):
            temp = int((zone / "temp").read_text()) / 1000
            if temp > 80:
                return "critical"
            elif temp > 70:
                return "serious"
            elif temp > 60:
                return "fair"
    
    return "nominal"

def _adjust_for_thermal(self, params: Dict) -> Dict:
    """Adjust generation parameters based on thermal state."""
    thermal = self._check_thermal_state()
    
    if thermal == "critical":
        params["max_tokens"] = min(params.get("max_tokens", 512), 256)
        params["num_threads"] = max(1, params.get("num_threads", 4) // 2)
    elif thermal == "serious":
        params["max_tokens"] = min(params.get("max_tokens", 512), 384)
    
    return params
```

### Power Consumption Estimation

```python
def estimate_power_consumption(self, model_spec: Dict, duration_s: float) -> float:
    """
    Estimate power consumption in watt-hours.
    
    Factors:
    - Model size (larger = more compute)
    - Quantization (lower = less compute)
    - GPU vs CPU (GPU typically more power)
    - Duration of inference
    """
    # Base power draw estimates (watts)
    base_power = {
        "cpu_idle": 5,
        "cpu_inference": 25,
        "gpu_inference": 50,
    }
    
    # Adjust for model size
    size_factor = model_spec["size_gb"] / 2.0  # Normalized to 2GB
    
    # Calculate watt-hours
    power_w = base_power["cpu_inference"] * size_factor
    power_wh = (power_w * duration_s) / 3600
    
    return power_wh
```

---

## File Reference

| File | Purpose | Lines |
|------|---------|-------|
| `src/llm/production_llm.py` | Canonical LLM backend with multi-engine support | 1633 |
| `src/llm/manager.py` | LLM manager with STT/TTS integration | 568 |
| `src/llm/__init__.py` | LLMRunner wrapper for agent loop | 256 |
| `src/llm/real_llm.py` | Deprecated, replaced by ProductionLLM | 355 |
| `src/agent/loop.py` | ReAct agent loop using LLM | 959 |

---

## Summary

The LLM Brain architecture provides:

1. **Local-First Design**: All inference runs on-device for privacy and offline capability
2. **Multi-Backend Support**: LLAMA_CPP, GPT4ALL, OLLAMA, TRANSFORMERS
3. **Smart Model Selection**: Automatic recommendations based on device capabilities
4. **ReAct Pattern**: Thought → Action → Observation loop for complex reasoning
5. **Robust Parsing**: Multiple fallback strategies for JSON response parsing
6. **Graceful Degradation**: Fallback chains for both tools and models
7. **Context Management**: Smart trimming to fit context windows
8. **Performance Aware**: Latency tracking, thermal throttling, power estimation

This architecture enables AURA to function as a capable, privacy-respecting AI assistant that adapts to the user's hardware while providing intelligent orchestration of all available tools and capabilities.
