# AURA Multi-Agent Framework

AURA uses a distributed subagent architecture for parallel processing and specialized tasks.

## Architecture

```
AURA (Main Agent)
├── Neural Processing Agent
│   ├── NeuralValidatedPlanner
│   ├── HebbianSelfCorrector
│   └── NeuralAwareRouter
├── Proactive Services Agent
│   ├── EventTracker
│   ├── CallManager
│   └── LifeExplorer
├── Learning Agent
│   ├── SelfLearningEngine
│   └── MemoryConsolidation
├── Context Agent
│   ├── ContextProvider
│   └── AdaptiveContext
├── SocialLife Agent
│   ├── RelationshipTracker
│   ├── PatternRecognizer
│   └── SocialInsights
├── Healthcare Agent
│   ├── HealthAnalyzer
│   ├── DietPlanner
│   └── FitnessTracker
└── Voice Agent
    ├── STT (Speech-to-Text)
    ├── TTS (Text-to-Speech)
    └── InnerVoice
```

## Usage

```python
# Initialize AURA with all agents
aura = await get_aura()
await aura.initialize()

# Process with parallel agents
response = await aura.process("Remind me about mom's birthday")

# AURA automatically:
# 1. Uses context agent to understand schedule
# 2. Activates proactive event tracker
# 3. Stores memory with neural memory
# 4. Plans reminder with neural planner
# 5. Shows inner voice reasoning
```

## Subagent Communication

Agents communicate via message bus:

```python
from src.agents.message_bus import MessageBus

bus = MessageBus()

# Agent publishes event
bus.publish("event_detected", {
    "type": "birthday",
    "entity": "mom",
    "date": "2026-02-23"
})

# Other agents subscribe
@bus.subscribe("event_detected")
async def on_birthday(event):
    # Trigger proactive response
    await proactive_engine.schedule_reminder(event)
```

## Parallel Processing

```python
# Run multiple agents in parallel
async def process_complex_request(user_input):
    tasks = [
        context_agent.analyze(user_input),
        memory_agent.retrieve_relevant(user_input),
        personality_agent.get_state(),
        proactive_agent.check_triggers(user_input)
    ]
    
    results = await asyncio.gather(*tasks)
    
    # Synthesize results
    return await synthesizer.combine(results)
```

## Agent Specialization

### 1. Neural Processing Agent
- Validates plans against learned patterns
- Routes to optimal model tier
- Applies Hebbian learning from outcomes

### 2. Proactive Services Agent
- Tracks events automatically
- Manages call handling
- Explores life domains

### 3. Learning Agent
- Records interactions
- Consolidates memories
- Adapts to user patterns

### 4. Context Agent
- Gathers real-time context
- Detects user state
- Provides relevant information

### 5. SocialLife Agent
- Tracks relationships
- Analyzes social patterns
- Provides insights

### 6. Healthcare Agent
- Analyzes health data
- Plans diet/fitness
- Generates insights

### 7. Voice Agent
- STT/TTS processing
- Inner voice expression
- Audio feedback

## Task Distribution

```python
class AgentCoordinator:
    """Coordinates work across agents"""
    
    async def delegate_task(self, task, context):
        # Analyze task type
        task_type = self.classify_task(task)
        
        # Route to appropriate agent(s)
        if task_type == "planning":
            return await self.neural_agent.create_plan(task, context)
        elif task_type == "proactive":
            return await self.proactive_agent.evaluate(task)
        elif task_type == "learning":
            return await self.learning_agent.record(task)
        elif task_type == "health":
            return await self.healthcare_agent.analyze(task)
        
        # Complex tasks need multiple agents
        elif task_type == "complex":
            return await self.coordinate_agents(task, context)
```

## Error Handling

Each agent has isolated error handling:

```python
try:
    result = await agent.execute(task)
except AgentTimeout:
    # Retry with timeout
    result = await agent.execute(task, timeout=60)
except AgentError as e:
    # Log and fallback
    logger.error(f"Agent {agent.name} failed: {e}")
    result = await fallback_agent.execute(task)
```

## Monitoring

Each agent reports health:

```python
agent_status = {
    "neural_agent": "healthy",
    "proactive_agent": "healthy", 
    "learning_agent": "healthy",
    "context_agent": "degraded",  # High latency
    "healthcare_agent": "healthy"
}
```
