# AURA v3 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build AURA v3 - a 100% offline, privacy-first personal AI assistant for Android/Termux that acts like Iron Man's Friday, with mobile-optimized SLMs, dynamic app discovery, healthcare and social-life management, and sub-agents with personalities.

**Architecture:** Tool-First Architecture with SLM generating JSON plans, deterministic orchestrator executing. GraphPilot for offline app topology mapping. SQLite state machine + Zvec for local vector RAG + Graphiti for temporal knowledge graphs. Multi-model pipeline with dynamic model swapping based on task complexity.

**Tech Stack:** 
- Models: LFM2.5-1.2B-Thinking (primary router), Phi-4-mini-flash-reasoning (advanced reasoning), DeepSeek-R1-Distill-Qwen-7B (expert swap)
- Memory: SQLite, Zvec, Graphiti
- Perception: OmniParser V2
- Execution: bitnet.cpp, ExecuTorch via Android NDK
- NPU: Speculative decoding with sd.npu

---

## Executive Summary

This plan transforms AURA v3 from a concept into a production-ready mobile AI assistant based on 2026 edge AI research. The implementation follows a 6-phase approach over 18 months, prioritizing:
1. **Privacy-first** - Zero external APIs, complete offline operation
2. **Mobile-native** - Optimized for 4GB RAM Android/Termux
3. **Adaptive personality** - Unique differentiator per simulation insights
4. **Setup simplicity** - Addressing the #1 complaint from simulation

## Key Research Integrations

| Research Finding | Implementation |
|----------------|----------------|
| Tool-First Architecture | SLM generates JSON plans, deterministic orchestrator executes |
| GraphPilot (70% latency reduction) | Offline app topology mapping with knowledge graphs |
| Memory: SQLite + Zvec + Graphiti | State machine + vector RAG + temporal knowledge |
| Model Selection 2026 | LFM2.5-1.2B (router), Phi-4-mini (reasoning), DeepSeek-R1-7B (expert) |
| OmniParser V2 (60% latency reduction) | Visual grounding for screen parsing |
| NPU Speculative Decoding (3.81x speedup) | sd.npu implementation |
| ExPO | Self-Explanation Policy Optimization for reasoning |

---

## Phase 1: Foundation (Months 1-3)

### Overview
Establish core infrastructure, mobile-first build system, and basic SLM pipeline.

### 1.1 Mobile Build System Setup

**Files:**
- Create: `build/android/setup.sh`
- Create: `build/android/termux_packages.txt`
- Create: `build/android/Dockerfile.crosscompile`
- Modify: `src/main.py` (add mobile detection)

**Step 1: Create Termux package manifest**

```bash
# termux_packages.txt
# Core AI Runtime
ai-engine
python
python-pip
git
curl
wget

# ML Dependencies
numpy
scipy
torch (via pip)
tokenizers

# System
termux-api
termux-tts
termux-speech-to-text

# Utilities
openssh
tmux
termux-tools
```

**Step 2: Create cross-compilation Docker setup**

```dockerfile
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y \
    python3 python3-pip git curl wget \
    aarch64-linux-gnu gcc-aarch64-linux-gnu \
    g++-aarch64-linux-gnu cmake
WORKDIR /build
COPY . .
RUN python3 setup.py build --target=android
```

**Step 3: Add mobile detection to main.py**

```python
import platform
import os

def is_mobile():
    """Detect if running on Android/Termux"""
    if platform.system() == "Linux":
        if os.path.exists("/data/data/com.termux"):
            return True
    return False

def get_device_profile():
    """Get device capabilities"""
    if is_mobile():
        return {
            "ram_limit_mb": 4096,
            "npu_available": detect_npu(),
            "storage_limit_mb": 2048,
            "battery_monitoring": True
        }
    return {"ram_limit_mb": 16000, "npu_available": False}
```

**Step 4: Verify mobile detection**

Run: `python -c "from main import is_mobile, get_device_profile; print(is_mobile(), get_device_profile())"`
Expected: `True {'ram_limit_mb': 4096, ...}` on Termux

---

### 1.2 SLM Pipeline Implementation

**Files:**
- Create: `src/llm/router.py`
- Create: `src/llm/model_manager.py`
- Create: `src/llm/quantization.py`
- Create: `config/models.yaml`

**Step 1: Define model configuration**

```yaml
# config/models.yaml
models:
  router:
    name: "LiquidAI/LFM2.5-1.2B-Thinking"
    params: 1.2
    memory_mb: 900
    quantization: 4bit
    max_context: 32k
    capabilities:
      - planning
      - tool_routing
      - simple_reasoning
  
  reasoner:
    name: "microsoft/phi-4-mini-flash-reasoning"
    params: 3.8
    memory_mb: 2400
    quantization: 4bit
    max_context: 64k
    capabilities:
      - complex_reasoning
      - math
      - code_generation
  
  expert:
    name: "deepseek/DeepSeek-R1-Distill-Qwen-7B"
    params: 7
    memory_mb: 4800
    quantization: 4bit
    max_context: 32k
    capabilities:
      - advanced_reasoning
      - debugging
      - complex_analysis
```

**Step 2: Implement model manager**

```python
# src/llm/model_manager.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from typing import Optional, Dict, Any
import threading
import gc

class ModelManager:
    def __init__(self, config_path: str = "config/models.yaml"):
        self.config = self._load_config(config_path)
        self.active_model = None
        self.active_tokenizer = None
        self.loaded_models: Dict[str, Any] = {}
        self.lock = threading.Lock()
        
    def _load_config(self, path: str) -> Dict:
        with open(path) as f:
            return yaml.safe_load(f)
    
    def load_model(self, model_type: str, force: bool = False) -> None:
        """Load model into memory, with dynamic swapping"""
        with self.lock:
            if model_type in self.loaded_models and not force:
                return
                
            # Check memory availability
            available = self._get_available_memory_mb()
            required = self.config['models'][model_type]['memory_mb']
            
            if available < required:
                self._unload_nonessential_models(required)
                
            # Load new model
            model_config = self.config['models'][model_type]
            self.loaded_models[model_type] = {
                'model': self._load_quantized_model(model_config['name']),
                'tokenizer': AutoTokenizer.from_pretrained(
                    model_config['name'], 
                    trust_remote_code=True
                ),
                'config': model_config
            }
    
    def _load_quantized_model(self, model_name: str):
        """Load model with 4-bit quantization"""
        # Use bitsandbytes or gguf format
        from transformers import BitsAndBytesConfig
        
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        
        return AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True
        )
    
    def generate(self, prompt: str, model_type: str = "router", **kwargs) -> str:
        """Generate with specified model"""
        if model_type not in self.loaded_models:
            self.load_model(model_type)
            
        model_data = self.loaded_models[model_type]
        inputs = model_data['tokenizer'](prompt, return_tensors="pt").to("cuda")
        
        outputs = model_data['model'].generate(
            **inputs,
            max_new_tokens=kwargs.get('max_tokens', 512),
            temperature=kwargs.get('temperature', 0.7),
            do_sample=True
        )
        
        return model_data['tokenizer'].decode(outputs[0], skip_special_tokens=True)
    
    def _get_available_memory_mb(self) -> int:
        """Get available RAM in MB"""
        import psutil
        return psutil.virtual_memory().available // (1024 * 1024)
    
    def _unload_nonessential_models(self, required_mb: int):
        """Unload models to free memory"""
        for model_type in ['expert', 'reasoner']:
            if model_type in self.loaded_models:
                del self.loaded_models[model_type]
                gc.collect()
                torch.cuda.empty_cache()
```

**Step 3: Implement router**

```python
# src/llm/router.py
from .model_manager import ModelManager
import json
import re

class AURARouter:
    def __init__(self, model_manager: ModelManager):
        self.mm = model_manager
        self.tool_schema = self._load_tool_schema()
    
    def _load_tool_schema(self) -> dict:
        """Load available tools and their schemas"""
        # Simplified - full implementation in Phase 2
        return {
            "tools": [
                {"name": "send_message", "params": {"recipient": "string", "message": "string"}},
                {"name": "check_calendar", "params": {"date": "string"}},
                {"name": "query_health", "params": {"metric": "string", "timeframe": "string"}},
            ]
        }
    
    def route(self, user_input: str) -> dict:
        """Route user request to appropriate action plan"""
        
        system_prompt = f"""You are AURA, an intelligent assistant. 
Analyze the user request and create a JSON plan using only these tools:
{json.dumps(self.tool_schema, indent=2)}

Respond ONLY with valid JSON in this format:
{{
    "reasoning": "Brief explanation of the plan",
    "action": "tool_name",
    "params": {{"param1": "value1"}},
    "confidence": 0.0-1.0
}}

If no tool fits, respond with:
{{"reasoning": "...", "action": "respond_directly", "params": {{"message": "..."}}, "confidence": 0.5}}"""
        
        response = self.mm.generate(
            prompt=f"{system_prompt}\n\nUser: {user_input}",
            model_type="router"
        )
        
        return self._parse_json_plan(response)
    
    def _parse_json_plan(self, response: str) -> dict:
        """Extract JSON plan from model response"""
        try:
            # Find JSON in response
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except json.JSONDecodeError:
            pass
        
        return {
            "reasoning": "Parse failed, responding directly",
            "action": "respond_directly",
            "params": {"message": response},
            "confidence": 0.3
        }
```

---

### 1.3 SQLite State Machine

**Files:**
- Create: `src/memory/state_machine.py`
- Create: `src/memory/schemas/state_machine.sql`
- Create: `tests/memory/test_state_machine.py`

**Step 1: Define state machine schema**

```sql
-- src/memory/schemas/state_machine.sql
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    state TEXT NOT NULL DEFAULT 'idle',
    context TEXT,
    user_id TEXT
);

CREATE TABLE IF NOT EXISTS message_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    timestamp INTEGER NOT NULL,
    processed INTEGER DEFAULT 0,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS task_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    task_type TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    priority INTEGER DEFAULT 0,
    created_at INTEGER NOT NULL,
    updated_at INTEGER NOT NULL,
    result TEXT,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS execution_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    task_id INTEGER,
    action TEXT NOT NULL,
    result TEXT,
    timestamp INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);
```

**Step 2: Implement state machine**

```python
# src/memory/state_machine.py
import sqlite3
import json
import time
import uuid
from typing import Optional, Dict, List, Any
from contextlib import contextmanager
import threading

class AURAStateMachine:
    def __init__(self, db_path: str = "data/aura_state.db"):
        self.db_path = db_path
        self.lock = threading.Lock()
        self._init_db()
    
    def _init_db(self):
        """Initialize database with schema"""
        with self._get_connection() as conn:
            with open("src/memory/schemas/state_machine.sql") as f:
                conn.executescript(f.read())
            conn.commit()
    
    @contextmanager
    def _get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    def create_session(self, user_id: Optional[str] = None) -> str:
        """Create new session"""
        session_id = str(uuid.uuid4())
        now = int(time.time())
        
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO sessions (id, created_at, updated_at, state, user_id)
                   VALUES (?, ?, ?, 'idle', ?)""",
                (session_id, now, now, user_id)
            )
            conn.commit()
        
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get session state"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                "SELECT * FROM sessions WHERE id = ?", (session_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def update_session_state(self, session_id: str, state: str, context: Optional[Dict] = None):
        """Update session state"""
        now = int(time.time())
        context_json = json.dumps(context) if context else None
        
        with self._get_connection() as conn:
            conn.execute(
                """UPDATE sessions 
                   SET state = ?, context = ?, updated_at = ?
                   WHERE id = ?""",
                (state, context_json, now, session_id)
            )
            conn.commit()
    
    def queue_message(self, session_id: str, role: str, content: str):
        """Add message to queue"""
        now = int(time.time())
        
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO message_queue (session_id, role, content, timestamp)
                   VALUES (?, ?, ?, ?)""",
                (session_id, role, content, now)
            )
            conn.commit()
    
    def get_conversation_history(self, session_id: str, limit: int = 50) -> List[Dict]:
        """Get conversation history"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """SELECT role, content, timestamp 
                   FROM message_queue 
                   WHERE session_id = ?
                   ORDER BY timestamp DESC
                   LIMIT ?""",
                (session_id, limit)
            )
            return [dict(row) for row in cursor.fetchall()]
    
    def create_task(self, session_id: str, task_type: str, payload: Dict, priority: int = 0) -> int:
        """Create task in queue"""
        now = int(time.time())
        
        with self._get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO task_queue (session_id, task_type, payload, priority, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (session_id, task_type, json.dumps(payload), priority, now, now)
            )
            conn.commit()
            return cursor.lastrowid
    
    def get_pending_tasks(self, session_id: str, limit: int = 10) -> List[Dict]:
        """Get pending tasks ordered by priority"""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """SELECT * FROM task_queue 
                   WHERE session_id = ? AND status = 'pending'
                   ORDER BY priority DESC, created_at ASC
                   LIMIT ?""",
                (session_id, limit)
            )
            rows = cursor.fetchall()
            tasks = []
            for row in rows:
                task = dict(row)
                task['payload'] = json.loads(task['payload'])
                tasks.append(task)
            return tasks
    
    def complete_task(self, task_id: int, result: Any):
        """Mark task as completed"""
        now = int(time.time())
        
        with self._get_connection() as conn:
            conn.execute(
                """UPDATE task_queue 
                   SET status = 'completed', result = ?, updated_at = ?
                   WHERE id = ?""",
                (json.dumps(result), now, task_id)
            )
            conn.commit()
    
    def log_action(self, session_id: str, action: str, result: Any, task_id: Optional[int] = None):
        """Log execution action"""
        now = int(time.time())
        
        with self._get_connection() as conn:
            conn.execute(
                """INSERT INTO execution_logs (session_id, task_id, action, result, timestamp)
                   VALUES (?, ?, ?, ?, ?)""",
                (session_id, task_id, action, json.dumps(result), now)
            )
            conn.commit()
```

**Step 3: Write state machine tests**

```python
# tests/memory/test_state_machine.py
import pytest
import tempfile
import os
from src.memory.state_machine import AURAStateMachine

@pytest.fixture
def state_machine():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        yield AURAStateMachine(db_path)

def test_create_session(state_machine):
    session_id = state_machine.create_session("test_user")
    assert session_id is not None
    assert len(session_id) == 36  # UUID format

def test_session_lifecycle(state_machine):
    session_id = state_machine.create_session()
    
    # Check initial state
    session = state_machine.get_session(session_id)
    assert session['state'] == 'idle'
    
    # Update state
    state_machine.update_session_state(session_id, "processing", {"step": 1})
    session = state_machine.get_session(session_id)
    assert session['state'] == "processing"
    assert session['context'] == '{"step": 1}'

def test_message_queue(state_machine):
    session_id = state_machine.create_session()
    
    state_machine.queue_message(session_id, "user", "Hello")
    state_machine.queue_message(session_id, "assistant", "Hi there")
    
    history = state_machine.get_conversation_history(session_id)
    assert len(history) == 2
    assert history[0]['content'] == "Hi there"

def test_task_queue(state_machine):
    session_id = state_machine.create_session()
    
    task_id = state_machine.create_task(
        session_id, "health_check", {"metric": "heart_rate"}, priority=5
    )
    
    pending = state_machine.get_pending_tasks(session_id)
    assert len(pending) == 1
    assert pending[0]['payload']['metric'] == "heart_rate"
    
    state_machine.complete_task(task_id, {"status": "normal", "value": 72})
    
    completed = state_machine.get_pending_tasks(session_id)
    assert len(completed) == 0
```

**Step 4: Run tests**

Run: `pytest tests/memory/test_state_machine.py -v`
Expected: All tests pass

---

### 1.4 Basic Orchestrator

**Files:**
- Create: `src/core/orchestrator.py`
- Create: `src/core/tool_executor.py`

**Step 1: Implement tool executor**

```python
# src/core/tool_executor.py
from typing import Dict, Any, Callable
import json
import asyncio
from abc import ABC, abstractmethod

class Tool(ABC):
    @abstractmethod
    def get_name(self) -> str:
        pass
    
    @abstractmethod
    def get_schema(self) -> Dict:
        pass
    
    @abstractmethod
    async def execute(self, **params) -> Dict[str, Any]:
        pass

class ToolExecutor:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
        self._register_builtin_tools()
    
    def _register_builtin_tools(self):
        """Register core tools"""
        from src.tools.builtin import (
            SendMessageTool, 
            QueryHealthTool,
            QueryCalendarTool,
            GeneralKnowledgeTool
        )
        
        for tool_class in [SendMessageTool, QueryHealthTool, QueryCalendarTool, GeneralKnowledgeTool]:
            tool = tool_class()
            self.register(tool)
    
    def register(self, tool: Tool):
        self.tools[tool.get_name()] = tool
    
    def get_tool_schema(self) -> Dict:
        return {
            "tools": [tool.get_schema() for tool in self.tools.values()]
        }
    
    async def execute(self, tool_name: str, params: Dict) -> Dict[str, Any]:
        if tool_name not in self.tools:
            return {"error": f"Tool '{tool_name}' not found"}
        
        try:
            result = await self.tools[tool_name].execute(**params)
            return {"success": True, "result": result}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def execute_plan(self, plan: Dict) -> Dict[str, Any]:
        """Execute a complete plan"""
        action = plan.get('action')
        params = plan.get('params', {})
        
        if action == 'respond_directly':
            return {"type": "direct", "message": params.get('message')}
        
        return await self.execute(action, params)
```

**Step 2: Implement orchestrator**

```python
# src/core/orchestrator.py
from src.llm.router import AURARouter
from src.core.tool_executor import ToolExecutor
from src.memory.state_machine import AURAStateMachine
from typing import Dict, Any, Optional
import asyncio
import logging

logger = logging.getLogger(__name__)

class AURAOrchestrator:
    def __init__(
        self,
        router: AURARouter,
        executor: ToolExecutor,
        state_machine: AURAStateMachine
    ):
        self.router = router
        self.executor = executor
        self.state_machine = state_machine
    
    async def process_request(
        self,
        user_input: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Main request processing pipeline"""
        
        # Create session if needed
        if not session_id:
            session_id = self.state_machine.create_session()
        
        # Get conversation context
        history = self.state_machine.get_conversation_history(session_id)
        
        # Update state to processing
        self.state_machine.update_session_state(
            session_id, "processing", 
            {"step": "analyzing_request"}
        )
        
        # Route the request
        plan = self.router.route(user_input)
        
        # Log the plan
        self.state_machine.log_action(
            session_id, "plan_created", plan
        )
        
        # Execute the plan
        result = await self.executor.execute_plan(plan)
        
        # Store messages
        self.state_machine.queue_message(session_id, "user", user_input)
        if result.get('type') == 'direct':
            self.state_machine.queue_message(
                session_id, "assistant", result.get('message', '')
            )
        
        # Update state back to idle
        self.state_machine.update_session_state(session_id, "idle")
        
        return {
            "session_id": session_id,
            "result": result,
            "plan": plan
        }
    
    async def process_background_task(self, task: Dict) -> Dict[str, Any]:
        """Process background task from queue"""
        task_id = task['id']
        task_type = task['task_type']
        payload = task['payload']
        
        logger.info(f"Processing background task {task_id}: {task_type}")
        
        # Route without user input
        plan = self.router.route_for_task(task_type, payload)
        result = await self.executor.execute_plan(plan)
        
        # Complete task
        self.state_machine.complete_task(task_id, result)
        
        return result
```

---

### 1.5 Phase 1 Resource Estimate

| Component | Effort | Dependencies |
|-----------|--------|--------------|
| Mobile Build System | 2 weeks | None |
| SLM Pipeline | 3 weeks | Mobile Build |
| State Machine | 2 weeks | None |
| Basic Orchestrator | 2 weeks | SLM Pipeline, State Machine |
| **Total** | **9 weeks** | |

---

## Phase 2: Memory Architecture (Months 4-6)

### Overview
Implement the biologically-inspired memory architecture: Zvec for vector RAG, Graphiti for temporal knowledge graphs, and the user profiling system.

### 2.1 Zvec Vector Database Integration

**Files:**
- Create: `src/memory/vector_store.py`
- Create: `src/memory/rag_pipeline.py`
- Modify: `config/models.yaml` (add embedding config)

**Step 1: Install and configure Zvec**

```bash
# In Termux
pip install zvec
# or compile from source if not available
```

**Step 2: Implement vector store**

```python
# src/memory/vector_store.py
import zvec
import numpy as np
import os
from typing import List, Dict, Any, Optional
import json

class AURAVectorStore:
    def __init__(self, storage_path: str = "data/vectors"):
        self.storage_path = storage_path
        os.makedirs(storage_path, exist_ok=True)
        
        # Initialize vector store
        self.store = zvec.VectorStore(
            dimension=384,  # Standard embedding dimension
            metric="cosine",
            storage_path=os.path.join(storage_path, "vectors.zvec")
        )
        
        # Metadata store
        self.metadata_path = os.path.join(storage_path, "metadata.json")
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        if os.path.exists(self.metadata_path):
            with open(self.metadata_path) as f:
                return json.load(f)
        return {"documents": {}}
    
    def _save_metadata(self):
        with open(self.metadata_path, 'w') as f:
            json.dump(self.metadata, f)
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadata: Optional[List[Dict]] = None
    ) -> List[int]:
        """Add documents to vector store"""
        ids = []
        
        for i, (text, embedding) in enumerate(zip(texts, embeddings)):
            doc_id = self.store.add(embedding)
            ids.append(doc_id)
            
            # Store metadata
            meta = metadata[i] if metadata else {}
            meta['text'] = text
            meta['id'] = doc_id
            self.metadata['documents'][str(doc_id)] = meta
        
        self._save_metadata()
        return ids
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        filter_fn: Optional[callable] = None
    ) -> List[Dict]:
        """Search for similar documents"""
        results = self.store.search(query_embedding, top_k)
        
        documents = []
        for doc_id, score in results:
            if filter_fn and not filter_fn(self.metadata['documents'].get(str(doc_id), {})):
                continue
            
            doc = self.metadata['documents'].get(str(doc_id), {})
            documents.append({
                'id': doc_id,
                'score': score,
                'text': doc.get('text', ''),
                'metadata': {k: v for k, v in doc.items() if k != 'text'}
            })
        
        return documents
    
    def delete(self, doc_id: int):
        """Delete document"""
        self.store.delete(doc_id)
        if str(doc_id) in self.metadata['documents']:
            del self.metadata['documents'][str(doc_id)]
            self._save_metadata()
```

**Step 2: Implement RAG pipeline**

```python
# src/memory/rag_pipeline.py
import numpy as np
from typing import List, Dict, Any, Optional
from src.memory.vector_store import AURAVectorStore
from src.llm.model_manager import ModelManager

class RAGPipeline:
    def __init__(
        self,
        vector_store: AURAVectorStore,
        model_manager: ModelManager
    ):
        self.vector_store = vector_store
        self.mm = model_manager
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Get embeddings using small embedding model"""
        # Use a lightweight embedding model
        # In production, use a mobile-optimized model like
        # sentence-transformers/all-MiniLM-L6-v2
        from sentence_transformers import SentenceTransformer
        
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        embeddings = model.encode(texts)
        return embeddings
    
    def index_documents(
        self,
        texts: List[str],
        metadata: Optional[List[Dict]] = None,
        source: str = "user"
    ):
        """Index documents for retrieval"""
        embeddings = self._get_embeddings(texts)
        self.vector_store.add_documents(texts, embeddings, metadata)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        source_filter: Optional[str] = None
    ) -> List[Dict]:
        """Retrieve relevant documents"""
        query_embedding = self._get_embeddings([query])[0]
        
        def filter_fn(doc):
            if source_filter and doc.get('source') != source_filter:
                return False
            return True
        
        return self.vector_store.search(
            query_embedding, 
            top_k, 
            filter_fn=filter_fn
        )
    
    def augment_prompt(self, user_query: str, context_window: int = 3) -> str:
        """Augment prompt with retrieved context"""
        docs = self.retrieve(user_query, top_k=context_window)
        
        if not docs:
            return user_query
        
        context = "\n\n".join([
            f"Context [{i+1}]: {doc['text']}"
            for i, doc in enumerate(docs)
        ])
        
        return f"""Based on relevant context:

{context}

User Query: {user_query}

Provide a response that considers the above context."""
```

---

### 2.2 Graphiti Temporal Knowledge Graph

**Files:**
- Create: `src/memory/knowledge_graph.py`
- Create: `src/memory/persona_tracker.py`

**Step 1: Implement knowledge graph**

```python
# src/memory/knowledge_graph.py
import sqlite3
import json
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Entity:
    id: Optional[int]
    type: str
    name: str
    properties: Dict
    created_at: int
    valid_from: int
    valid_to: Optional[int]  # None = currently valid

@dataclass
class Relationship:
    id: Optional[int]
    source_id: int
    target_id: int
    relation_type: str
    properties: Dict
    created_at: int
    valid_from: int
    valid_to: Optional[int]

class GraphitiKnowledgeGraph:
    """Temporal knowledge graph with validity intervals"""
    
    def __init__(self, db_path: str = "data/knowledge_graph.db"):
        self.db_path = db_path
        self._init_db()
    
    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS entities (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    properties TEXT,
                    created_at INTEGER NOT NULL,
                    valid_from INTEGER NOT NULL,
                    valid_to INTEGER,
                    UNIQUE(type, name, valid_from)
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS relationships (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    target_id INTEGER NOT NULL,
                    relation_type TEXT NOT NULL,
                    properties TEXT,
                    created_at INTEGER NOT NULL,
                    valid_from INTEGER NOT NULL,
                    valid_to INTEGER,
                    FOREIGN KEY (source_id) REFERENCES entities(id),
                    FOREIGN KEY (target_id) REFERENCES entities(id)
                )
            """)
            
            conn.execute("""
                CREATE INDEX idx_entities_type ON entities(type)
            """)
            conn.execute("""
                CREATE INDEX idx_entities_name ON entities(name)
            """)
            conn.execute("""
                CREATE INDEX idx_relationships_source ON relationships(source_id)
            """)
            
            conn.commit()
    
    def add_entity(
        self,
        entity_type: str,
        name: str,
        properties: Optional[Dict] = None,
        valid_from: Optional[int] = None,
        valid_to: Optional[int] = None
    ) -> int:
        """Add entity with temporal validity"""
        now = int(time.time())
        props = json.dumps(properties or {})
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO entities (type, name, properties, created_at, valid_from, valid_to)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (entity_type, name, props, now, valid_from or now, valid_to))
            conn.commit()
            return cursor.lastrowid
    
    def update_entity(
        self,
        entity_id: int,
        new_properties: Dict,
        valid_from: Optional[int] = None
    ):
        """Update entity - invalidates old version, creates new"""
        now = int(time.time())
        
        with sqlite3.connect(self.db_path) as conn:
            # Invalidate old version
            conn.execute("""
                UPDATE entities 
                SET valid_to = ? 
                WHERE id = ? AND valid_to IS NULL
            """, (now - 1, entity_id))
            
            # Get old entity data
            cursor = conn.execute(
                "SELECT type, name FROM entities WHERE id = ?", (entity_id,)
            )
            row = cursor.fetchone()
            
            # Create new version
            conn.execute("""
                INSERT INTO entities (type, name, properties, created_at, valid_from, valid_to)
                VALUES (?, ?, ?, ?, ?, NULL)
            """, (row[0], row[1], json.dumps(new_properties), now, valid_from or now))
            
            conn.commit()
    
    def get_entity_at_time(
        self,
        entity_type: str,
        name: str,
        timestamp: Optional[int] = None
    ) -> Optional[Entity]:
        """Get entity state at specific time"""
        ts = timestamp or int(time.time())
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM entities
                WHERE type = ? AND name = ?
                AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)
                ORDER BY valid_from DESC
                LIMIT 1
            """, (entity_type, name, ts, ts))
            
            row = cursor.fetchone()
            if row:
                return Entity(
                    id=row['id'],
                    type=row['type'],
                    name=row['name'],
                    properties=json.loads(row['properties']),
                    created_at=row['created_at'],
                    valid_from=row['valid_from'],
                    valid_to=row['valid_to']
                )
            return None
    
    def add_relationship(
        self,
        source_id: int,
        target_id: int,
        relation_type: str,
        properties: Optional[Dict] = None,
        valid_from: Optional[int] = None,
        valid_to: Optional[int] = None
    ) -> int:
        """Add temporal relationship"""
        now = int(time.time())
        props = json.dumps(properties or {})
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO relationships 
                (source_id, target_id, relation_type, properties, created_at, valid_from, valid_to)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (source_id, target_id, relation_type, props, now, valid_from or now, valid_to))
            conn.commit()
            return cursor.lastrowid
    
    def query(
        self,
        entity_type: Optional[str] = None,
        relation_type: Optional[str] = None,
        source_id: Optional[int] = None,
        target_id: Optional[int] = None,
        timestamp: Optional[int] = None
    ) -> List[Dict]:
        """Query knowledge graph"""
        ts = timestamp or int(time.time())
        
        query = "SELECT * FROM entities WHERE 1=1"
        params = []
        
        if entity_type:
            query += " AND type = ?"
            params.append(entity_type)
        
        query += " AND valid_from <= ? AND (valid_to IS NULL OR valid_to > ?)"
        params.extend([ts, ts])
        
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
```

---

### 2.3 User Profiling System

**Files:**
- Create: `src/memory/user_profile.py`
- Create: `src/agents/personality.py`

**Step 1: Implement user profiling**

```python
# src/memory/user_profile.py
from src.memory.knowledge_graph import GraphitiKnowledgeGraph
from typing import Dict, Any, List
import json

class UserProfiler:
    """Builds and maintains user profile over time"""
    
    def __init__(self, knowledge_graph: GraphitiKnowledgeGraph):
        self.kg = knowledge_graph
    
    def update_from_interaction(
        self,
        user_id: str,
        user_input: str,
        assistant_response: str,
        context: Dict = None
    ):
        """Update user profile based on interaction"""
        now = int(time.time())
        
        # Extract preferences from conversation
        preferences = self._extract_preferences(user_input, assistant_response)
        
        # Update knowledge graph
        user_entity = self.kg.get_entity_at_time("user", user_id)
        
        if user_entity:
            # Update existing profile
            current_props = user_entity.properties
            current_props.update(preferences)
            self.kg.update_entity(user_entity.id, current_props)
        else:
            # Create new user entity
            self.kg.add_entity(
                "user",
                user_id,
                {
                    **preferences,
                    "created_at": now,
                    "interaction_count": 1
                }
            )
    
    def _extract_preferences(
        self,
        user_input: str,
        response: str
    ) -> Dict:
        """Extract preferences from text using simple heuristics"""
        preferences = {}
        
        # Diet preferences
        diet_keywords = {
            "vegetarian": ["vegetarian", "veggie", "no meat"],
            "vegan": ["vegan", "no animal products"],
            "keto": ["keto", "low carb", "high fat"],
            "paleo": ["paleo", "whole foods"],
        }
        
        for diet, keywords in diet_keywords.items():
            if any(k in user_input.lower() for k in keywords):
                preferences["diet"] = diet
        
        # Health goals
        goal_keywords = {
            "weight_loss": ["lose weight", "slim", "diet"],
            "muscle": ["muscle", "strength", "gain"],
            "endurance": ["run", "cardio", " stamina"],
            "general": ["healthy", "wellness"]
        }
        
        for goal, keywords in goal_keywords.items():
            if any(k in user_input.lower() for k in keywords):
                preferences["health_goal"] = goal
        
        # Communication preferences
        if "?" in user_input:
            preferences["last_question_time"] = int(time.time())
        
        return preferences
    
    def get_user_profile(self, user_id: str) -> Dict:
        """Get current user profile"""
        entity = self.kg.get_entity_at_time("user", user_id)
        return entity.properties if entity else {}
    
    def get_user_history(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get user's historical profile states"""
        # Query all valid versions
        return self.kg.query(entity_type="user")
```

---

### 2.4 Phase 2 Resource Estimate

| Component | Effort | Dependencies |
|-----------|--------|--------------|
| Zvec Integration | 2 weeks | Phase 1 |
| RAG Pipeline | 2 weeks | Zvec |
| Graphiti Knowledge Graph | 3 weeks | Phase 1 |
| User Profiling | 2 weeks | Graphiti |
| **Total** | **9 weeks** | |

---

## Phase 3: Dynamic App Discovery & Tool System (Months 7-9)

### Overview
Implement dynamic app discovery (GraphPilot), OmniParser integration, and the tool binding system.

### 3.1 GraphPilot Implementation

**Files:**
- Create: `src/tools/graphpilot.py`
- Create: `src/tools/app_discovery.py`
- Create: `src/tools/binding_generator.py`

**Step 1: Implement app discovery**

```python
# src/tools/app_discovery.py
import os
import json
import subprocess
from typing import Dict, List, Optional
import asyncio

class AppDiscovery:
    """Discovers installed apps and their capabilities"""
    
    def __init__(self):
        self.discovered_apps: Dict = {}
        self.app_capabilities: Dict = {}
    
    def discover_android_apps(self) -> List[Dict]:
        """Discover installed Android apps via pm command"""
        if not self._is_android():
            return []
        
        try:
            # Get package list
            result = subprocess.run(
                ["pm", "list", "packages", "-3"],  # Third-party apps
                capture_output=True,
                text=True,
                timeout=30
            )
            
            packages = []
            for line in result.stdout.splitlines():
                if line.startswith("package:"):
                    pkg = line.replace("package:", "").strip()
                    packages.append(pkg)
            
            return [self._analyze_app(pkg) for pkg in packages[:50]]
        except Exception as e:
            print(f"Error discovering apps: {e}")
            return []
    
    def _analyze_app(self, package: str) -> Dict:
        """Analyze app capabilities"""
        try:
            # Get app info
            result = subprocess.run(
                ["dumpsys", "package", package],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            # Extract activities
            activities = []
            for line in result.stdout.splitlines():
                if "Activity Resolver" in line:
                    break
                if "activity " in line.lower():
                    activities.append(line.strip())
            
            return {
                "package": package,
                "activities": activities[:20],  # Limit
                "discovered_at": int(time.time())
            }
        except:
            return {"package": package, "error": "Cannot analyze"}
    
    def _is_android(self) -> bool:
        """Check if running on Android"""
        return os.path.exists("/data/data/com.termux")
```

**Step 2: Implement GraphPilot knowledge graph**

```python
# src/tools/graphpilot.py
from src.memory.knowledge_graph import GraphitiKnowledgeGraph
from typing import Dict, List, Any, Optional
import json

class GraphPilot:
    """Graph-based offline app topology mapping"""
    
    def __init__(self, knowledge_graph: GraphitiKnowledgeGraph):
        self.kg = knowledge_graph
    
    def build_app_graph(self, app_info: Dict):
        """Build knowledge graph for an app"""
        
        package = app_info['package']
        
        # Create app node
        app_id = self.kg.add_entity(
            "app",
            package,
            {
                "package": package,
                "activities": app_info.get('activities', []),
                "discovered_at": app_info.get('discovered_at')
            }
        )
        
        # Create page nodes for activities
        for i, activity in enumerate(app_info.get('activities', [])):
            page_id = self.kg.add_entity(
                "page",
                f"{package}:{activity}",
                {
                    "app": package,
                    "activity": activity,
                    "index": i
                }
            )
            
            # Connect app to pages
            self.kg.add_relationship(
                app_id,
                page_id,
                "has_page"
            )
    
    def build_transition_graph(
        self,
        app_package: str,
        transitions: List[Dict]
    ):
        """Build transition rules between pages"""
        
        # Get all pages for app
        pages = self.kg.query(entity_type="page")
        app_pages = [p for p in pages if p.get('properties', {}).get('app') == app_package]
        
        page_map = {p['name']: p['id'] for p in app_pages}
        
        for transition in transitions:
            from_page = page_map.get(transition.get('from'))
            to_page = page_map.get(transition.get('to'))
            
            if from_page and to_page:
                self.kg.add_relationship(
                    from_page,
                    to_page,
                    "transitions_to",
                    {
                        "action": transition.get('action'),
                        "trigger": transition.get('trigger')
                    }
                )
    
    def get_navigation_path(
        self,
        app_package: str,
        from_page: str,
        to_page: str
    ) -> Optional[List[Dict]]:
        """Get shortest navigation path between pages"""
        
        # Use BFS to find path
        pages = self.kg.query(entity_type="page")
        app_pages = {p['name']: p for p in pages if p.get('properties', {}).get('app') == app_package}
        
        if from_page not in app_pages or to_page not in app_pages:
            return None
        
        # Simplified BFS
        from_id = app_pages[from_page]['id']
        to_id = app_pages[to_page]['id']
        
        visited = set()
        queue = [(from_id, [])]
        
        while queue:
            current, path = queue.pop(0)
            
            if current == to_id:
                return path
            
            if current in visited:
                continue
            
            visited.add(current)
            
            # Get relationships
            rels = self.kg.query(relation_type="transitions_to")
            for rel in rels:
                if rel['source_id'] == current:
                    queue.append((rel['target_id'], path + [rel]))
        
        return None
    
    def generate_action_plan(
        self,
        app_package: str,
        target_action: str,
        current_screen: str
    ) -> Dict:
        """Generate action plan for app operation"""
        
        # Find target page
        pages = self.kg.query(entity_type="page")
        target_page = None
        
        for page in pages:
            props = page.get('properties', {})
            if props.get('app') == app_package and target_action in props.get('activity', ''):
                target_page = page['name']
                break
        
        if not target_page:
            return {
                "reasoning": "Target action not found in app graph",
                "requires_visual": True,
                "action": None
            }
        
        # Find path
        path = self.get_navigation_path(app_package, current_screen, target_page)
        
        if path:
            return {
                "reasoning": "Found navigation path in app graph",
                "requires_visual": False,
                "actions": [{"action": r['properties']['action']} for r in path]
            }
        
        return {
            "reasoning": "No pre-mapped path, need visual exploration",
            "requires_visual": True,
            "action": None
        }
```

---

### 3.2 OmniParser Integration

**Files:**
- Create: `src/perception/omni_parser.py`
- Create: `src/perception/screen_parser.py`

**Step 1: Implement OmniParser wrapper**

```python
# src/perception/omni_parser.py
import subprocess
import json
import os
from typing import Dict, List, Any, Optional
import numpy as np
from PIL import Image
import time

class OmniParserWrapper:
    """Wrapper for OmniParser V2 for screen parsing"""
    
    def __init__(
        self,
        model_path: str = "models/omniparser-v2",
        device: str = "cpu"
    ):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.is_loaded = False
    
    def load_model(self):
        """Load OmniParser model"""
        if self.is_loaded:
            return
        
        try:
            from transformers import AutoProcessor, AutoModelForVision2Seq
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_path,
                trust_remote_code=True
            )
            self.model = AutoModelForVision2Seq.from_pretrained(
                self.model_path,
                trust_remote_code=True
            ).to(self.device)
            
            self.is_loaded = True
        except Exception as e:
            print(f"Failed to load OmniParser: {e}")
            # Fallback to simpler implementation
            self._setup_fallback()
    
    def _setup_fallback(self):
        """Setup fallback screen parser using accessibility"""
        self.is_loaded = False
        print("Using fallback screen parser")
    
    def parse_screen(
        self,
        screenshot_path: str,
        use_npu: bool = True
    ) -> Dict[str, Any]:
        """Parse screen and extract interactive elements"""
        
        if not self.is_loaded:
            return self._fallback_parse(screenshot_path)
        
        start_time = time.time()
        
        try:
            # Load image
            image = Image.open(screenshot_path).convert("RGB")
            
            # Process through model
            inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=512
                )
            
            # Parse output
            result = self.processor.batch_decode(
                outputs,
                skip_special_tokens=True
            )[0]
            
            parsed = json.loads(result)
            
            return {
                "success": True,
                "elements": parsed.get("elements", []),
                "latency_ms": (time.time() - start_time) * 1000,
                "method": "omniparser"
            }
            
        except Exception as e:
            return self._fallback_parse(screenshot_path)
    
    def _fallback_parse(self, screenshot_path: str) -> Dict:
        """Fallback using Android accessibility"""
        
        # Try using uiautomator dump
        try:
            result = subprocess.run(
                ["uiautomator", "dump", "/sdcard/window_dump.xml"],
                capture_output=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Parse XML
                return self._parse_uiautomator_xml("/sdcard/window_dump.xml")
        except:
            pass
        
        return {
            "success": False,
            "elements": [],
            "error": "Could not parse screen",
            "method": "fallback"
        }
    
    def _parse_uiautomator_xml(self, xml_path: str) -> Dict:
        """Parse uiautomator XML output"""
        import xml.etree.ElementTree as ET
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            elements = []
            for node in root.iter():
                bounds = node.get('bounds')
                if bounds:
                    elements.append({
                        "type": node.get('class', 'unknown'),
                        "text": node.get('text', ''),
                        "content-desc": node.get('content-desc', ''),
                        "bounds": bounds,
                        "clickable": node.get('clickable', 'false') == 'true'
                    })
            
            return {
                "success": True,
                "elements": elements,
                "method": "uiautomator"
            }
        except Exception as e:
            return {"success": False, "elements": [], "error": str(e)}
```

---

### 3.3 Dynamic Tool Binding System

**Files:**
- Create: `src/tools/binding_manager.py`
- Create: `src/tools/dynamic_tool.py`

**Step 1: Implement binding manager**

```python
# src/tools/binding_manager.py
from typing import Dict, Any, List, Optional, Callable
import json
import asyncio
from dataclasses import dataclass

@dataclass
class ToolBinding:
    name: str
    description: str
    parameters: Dict[str, Any]
    executor: Callable
    app_package: Optional[str] = None
    discovered: bool = False
    confidence: float = 0.0

class BindingManager:
    """Manages dynamic tool bindings from app discovery"""
    
    def __init__(self):
        self.bindings: Dict[str, ToolBinding] = {}
        self.template_bindings = self._load_templates()
    
    def _load_templates(self) -> Dict:
        """Load tool binding templates"""
        return {
            "messaging": {
                "schema": {
                    "send_message": {
                        "params": {
                            "recipient": {"type": "string"},
                            "message": {"type": "string"}
                        }
                    }
                },
                "apps": ["com.whatsapp", "com.telegram", "com.signal"]
            },
            "calendar": {
                "schema": {
                    "check_calendar": {
                        "params": {
                            "date": {"type": "string"},
                            "time_range": {"type": "string"}
                        }
                    }
                },
                "apps": ["com.google.android.calendar", "com.android.calendar"]
            },
            "health": {
                "schema": {
                    "query_health": {
                        "params": {
                            "metric": {"type": "string"},
                            "timeframe": {"type": "string"}
                        }
                    }
                },
                "apps": ["com.google.android.apps.healthdata"]
            }
        }
    
    def create_binding_from_app(
        self,
        app_package: str,
        activities: List[str]
    ) -> Optional[ToolBinding]:
        """Create tool binding from discovered app"""
        
        # Match to template
        for template_name, template in self.template_bindings.items():
            if app_package in template['apps']:
                return ToolBinding(
                    name=template_name,
                    description=f"Interface for {app_package}",
                    parameters=template['schema'][template_name]['params'],
                    executor=self._create_executor(template_name, app_package),
                    app_package=app_package,
                    discovered=True,
                    confidence=0.8
                )
        
        # Generic binding for unknown apps
        return ToolBinding(
            name=f"app_{app_package.split('.')[-1]}",
            description=f"Generic interface for {app_package}",
            parameters={},
            executor=self._create_generic_executor(app_package),
            app_package=app_package,
            discovered=True,
            confidence=0.5
        )
    
    def _create_executor(self, template: str, app_package: str):
        """Create executor function for template"""
        
        async def execute(**params):
            # Template-specific execution logic
            if template == "messaging":
                return await self._execute_messaging(app_package, params)
            elif template == "calendar":
                return await self._execute_calendar(app_package, params)
            elif template == "health":
                return await self._execute_health(app_package, params)
        
        return execute
    
    async def _execute_messaging(self, app_package: str, params: Dict) -> Dict:
        """Execute messaging action"""
        # Use Android intents
        return {
            "success": True,
            "action": "open_app_with_intent",
            "package": app_package,
            "params": params
        }
    
    async def _execute_calendar(self, app_package: str, params: Dict) -> Dict:
        """Execute calendar query"""
        return {
            "success": True,
            "action": "query_calendar_content",
            "package": app_package,
            "params": params
        }
    
    async def _execute_health(self, app_package: str, params: Dict) -> Dict:
        """Execute health data query"""
        return {
            "success": True,
            "action": "query_health_data",
            "package": app_package,
            "params": params
        }
    
    def register_binding(self, binding: ToolBinding):
        """Register a tool binding"""
        self.bindings[binding.name] = binding
    
    def get_binding(self, name: str) -> Optional[ToolBinding]:
        """Get tool binding by name"""
        return self.bindings.get(name)
    
    def get_all_bindings(self) -> List[ToolBinding]:
        """Get all registered bindings"""
        return list(self.bindings.values())
```

---

### 3.4 Phase 3 Resource Estimate

| Component | Effort | Dependencies |
|-----------|--------|--------------|
| App Discovery | 2 weeks | Phase 1 |
| GraphPilot | 3 weeks | App Discovery, Phase 1 |
| OmniParser | 2 weeks | None |
| Tool Binding | 2 weeks | GraphPilot |
| **Total** | **9 weeks** | |

---

## Phase 4: Sub-Agent System (Months 10-12)

### Overview
Implement specialized sub-agents (healthcare, social-life) with their own personalities that learn from interactions.

### 4.1 Healthcare Agent

**Files:**
- Create: `src/agents/healthcare_agent.py`
- Create: `src/agents/health_data.py`
- Create: `config/health_prompts.yaml`

**Step 1: Define healthcare agent**

```python
# src/agents/healthcare_agent.py
from src.agents.base_agent import BaseAgent
from typing import Dict, Any, List, Optional
import json
from datetime import datetime, timedelta

class HealthcareAgent(BaseAgent):
    """Healthcare assistant agent with personality"""
    
    def __init__(self, model_manager, memory_system):
        super().__init__(
            name="Dr. AURA",
            personality="""You are Dr. AURA, a caring and knowledgeable healthcare assistant. 
            You provide health insights, diet recommendations, and fitness tracking.
            Always emphasize consulting healthcare professionals for serious concerns.
            Be encouraging but never make definitive medical diagnoses.""",
            model_manager=model_manager,
            memory_system=memory_system
        )
        
        self.health_data = HealthDataManager(memory_system)
    
    async def process_request(self, user_input: str, context: Dict) -> Dict[str, Any]:
        """Process health-related request"""
        
        # Analyze intent
        intent = await self._analyze_intent(user_input)
        
        if intent == "track_metric":
            return await self._handle_tracking(user_input, context)
        elif intent == "diet_query":
            return await self._handle_diet(user_input, context)
        elif intent == "fitness":
            return await self._handle_fitness(user_input, context)
        elif intent == "insight":
            return await self._generate_insight(user_input, context)
        
        return await super().process_request(user_input, context)
    
    async def _analyze_intent(self, user_input: str) -> str:
        """Analyze health intent"""
        
        keywords = {
            "track_metric": ["track", "log", "record", "today", "measurement"],
            "diet_query": ["eat", "food", "meal", "diet", "calories", "nutrition"],
            "fitness": ["exercise", "workout", "run", "walk", "fitness", "training"],
            "insight": ["how am i", "trend", "分析", "insight", "recommend"]
        }
        
        input_lower = user_input.lower()
        for intent, kwds in keywords.items():
            if any(k in input_lower for k in kwds):
                return intent
        
        return "general"
    
    async def _handle_tracking(self, user_input: str, context: Dict) -> Dict:
        """Handle metric tracking"""
        
        # Extract metric and value
        metric = self._extract_metric(user_input)
        value = self._extract_value(user_input)
        
        if metric and value:
            self.health_data.log_metric(
                context['user_id'],
                metric,
                value,
                context.get('timestamp')
            )
            
            return {
                "type": "health_tracking",
                "message": f"Logged {metric}: {value}",
                "metric": metric,
                "value": value
            }
        
        return {
            "type": "clarification",
            "message": "What metric would you like to track? (e.g., heart rate, weight, steps)"
        }
    
    async def _handle_diet(self, user_input: str, context: Dict) -> Dict:
        """Handle diet queries"""
        
        user_profile = self.memory_system.get_user_profile(context['user_id'])
        diet_preference = user_profile.get('diet', 'no_restriction')
        
        # Get recent meals
        recent_meals = self.health_data.get_recent_meals(
            context['user_id'],
            days=3
        )
        
        # Generate recommendation
        prompt = f"""User has {diet_preference} diet preference.
        
Recent meals: {json.dumps(recent_meals)}

User query: {user_input}

Provide a helpful diet suggestion or nutritional insight."""
        
        response = await self.model_manager.generate(prompt, model_type="reasoner")
        
        return {
            "type": "diet_recommendation",
            "message": response,
            "diet": diet_preference
        }
    
    async def _handle_fitness(self, user_input: str, context: Dict) -> Dict:
        """Handle fitness queries"""
        
        # Get fitness data
        fitness_data = self.health_data.get_fitness_summary(
            context['user_id'],
            days=7
        )
        
        return {
            "type": "fitness_summary",
            "data": fitness_data,
            "message": self._format_fitness_message(fitness_data)
        }
    
    async def _generate_insight(self, user_input: str, context: Dict) -> Dict:
        """Generate health insights"""
        
        user_profile = self.memory_system.get_user_profile(context['user_id'])
        
        # Get trend data
        trends = self.health_data.get_trends(
            context['user_id'],
            metric=user_profile.get('health_goal', 'steps'),
            days=30
        )
        
        return {
            "type": "health_insight",
            "trends": trends,
            "message": self._format_trends_message(trends)
        }
    
    def _extract_metric(self, text: str) -> Optional[str]:
        """Extract health metric from text"""
        metrics = {
            "heart rate": "heart_rate",
            "pulse": "heart_rate",
            "weight": "weight",
            "blood pressure": "blood_pressure",
            "steps": "steps",
            "calories": "calories",
            "sleep": "sleep_hours",
            "water": "water_intake"
        }
        
        text_lower = text.lower()
        for key, metric in metrics.items():
            if key in text_lower:
                return metric
        return None
    
    def _extract_value(self, text: str) -> Optional[float]:
        """Extract numeric value from text"""
        import re
        numbers = re.findall(r'\d+\.?\d*', text)
        if numbers:
            return float(numbers[0])
        return None
```

---

### 4.2 Social-Life Agent

**Files:**
- Create: `src/agents/social_life_agent.py`
- Create: `src/agents/social_analyzer.py`

**Step 2: Define social-life agent**

```python
# src/agents/social_life_agent.py
from src.agents.base_agent import BaseAgent
from typing import Dict, Any, List
import json
from datetime import datetime, timedelta

class SocialLifeAgent(BaseAgent):
    """Social life management agent"""
    
    def __init__(self, model_manager, memory_system):
        super().__init__(
            name="Social AURA",
            personality="""You are Social AURA, a thoughtful friend who helps manage 
            your social life. You remember important dates, suggest meetups based on 
            patterns, and provide gentle insights about social habits.
            Be supportive but never intrusive about social matters.""",
            model_manager=model_manager,
            memory_system=memory_system
        )
        
        self.social_data = SocialDataManager(memory_system)
    
    async def process_request(self, user_input: str, context: Dict) -> Dict[str, Any]:
        """Process social-life request"""
        
        intent = await self._analyze_intent(user_input)
        
        if intent == "reminder":
            return await self._handle_reminder(user_input, context)
        elif intent == "insight":
            return await self._handle_insight(user_input, context)
        elif intent == "suggestion":
            return await self._handle_suggestion(user_input, context)
        elif intent == "pattern":
            return await self._handle_pattern(user_input, context)
        
        return await super().process_request(user_input, context)
    
    async def _analyze_intent(self, user_input: str) -> str:
        """Analyze social intent"""
        
        keywords = {
            "reminder": ["remind", "remember", "anniversary", "birthday", "follow up"],
            "insight": ["how", "分析", "insight", "habit", "pattern"],
            "suggestion": ["suggest", "recommend", "should", "could", "meetup"],
            "pattern": ["pattern", "usually", "always", "never", "often"]
        }
        
        input_lower = user_input.lower()
        for intent, kwds in keywords.items():
            if any(k in input_lower for k in kwds):
                return intent
        
        return "general"
    
    async def _handle_reminder(self, user_input: str, context: Dict) -> Dict:
        """Handle reminder requests"""
        
        # Extract person and date
        person = self._extract_person(user_input)
        date = self._extract_date(user_input)
        
        if person:
            # Store reminder
            self.social_data.add_reminder(
                context['user_id'],
                person,
                date,
                user_input
            )
            
            return {
                "type": "reminder_set",
                "message": f"I'll remind you about {person} on {date}",
                "person": person,
                "date": date
            }
        
        return {
            "type": "clarification",
            "message": "Who would you like me to remind you about?"
        }
    
    async def _handle_insight(self, user_input: str, context: Dict) -> Dict:
        """Generate social insights"""
        
        # Analyze social patterns
        patterns = self.social_data.analyze_social_patterns(
            context['user_id'],
            days=30
        )
        
        return {
            "type": "social_insight",
            "patterns": patterns,
            "message": self._format_insights(patterns)
        }
    
    async def _handle_suggestion(self, user_input: str, context: Dict) -> Dict:
        """Generate suggestions based on patterns"""
        
        # Get user patterns
        patterns = self.social_data.analyze_social_patterns(
            context['user_id'],
            days=30
        )
        
        # Generate suggestion
        suggestion = await self._generate_suggestion(patterns, context)
        
        return {
            "type": "suggestion",
            "message": suggestion
        }
    
    def _extract_person(self, text: str) -> Optional[str]:
        """Extract person name from text"""
        # Simple extraction - in production use NER
        import re
        names = re.findall(r'([A-Z][a-z]+ [A-Z][a-z]+)', text)
        return names[0] if names else None
    
    def _extract_date(self, text: str) -> Optional[str]:
        """Extract date from text"""
        # Simple extraction - in production use date parsing
        import re
        date_patterns = [
            r'\d{1,2}/\d{1,2}',
            r'\d{1,2}-\d{1,2}',
            r'next week',
            r'tomorrow'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, text)
            if match:
                return match.group()
        
        return None
```

---

### 4.3 Agent Orchestration

**Files:**
- Create: `src/agents/agent_manager.py`
- Create: `src/agents/base_agent.py`

**Step 3: Implement agent manager**

```python
# src/agents/agent_manager.py
from typing import Dict, Any, Optional, List
from src.agents.healthcare_agent import HealthcareAgent
from src.agents.social_life_agent import SocialLifeAgent
from src.agents.base_agent import BaseAgent
import logging

logger = logging.getLogger(__name__)

class AgentManager:
    """Manages multiple sub-agents with routing"""
    
    def __init__(self, model_manager, memory_system):
        self.model_manager = model_manager
        self.memory_system = memory_system
        
        # Initialize agents
        self.agents: Dict[str, BaseAgent] = {
            "healthcare": HealthcareAgent(model_manager, memory_system),
            "social": SocialLifeAgent(model_manager, memory_system),
        }
        
        self.default_agent = BaseAgent(
            name="AURA",
            personality="""You are AURA, a helpful and intelligent assistant. 
            You are like Iron Man's Friday - proactive, smart, and always ready to help.
            You have access to healthcare and social life agents for specialized tasks.""",
            model_manager=model_manager,
            memory_system=memory_system
        )
    
    async def route_request(
        self,
        user_input: str,
        context: Dict
    ) -> Dict[str, Any]:
        """Route request to appropriate agent"""
        
        # Detect required agent
        agent_type = self._detect_agent(user_input)
        
        if agent_type and agent_type in self.agents:
            logger.info(f"Routing to {agent_type} agent")
            return await self.agents[agent_type].process_request(
                user_input, context
            )
        
        # Use default agent
        return await self.default_agent.process_request(user_input, context)
    
    def _detect_agent(self, user_input: str) -> Optional[str]:
        """Detect which agent to route to"""
        
        healthcare_keywords = [
            "health", "fitness", "exercise", "diet", "weight",
            "heart", "sleep", "nutrition", "workout", "run",
            "calories", "steps", "medical", "doctor"
        ]
        
        social_keywords = [
            "social", "friend", "meet", "remind", "birthday",
            "message", "call", "connect", "hang out", "invite",
            "party", "dinner", "coffee", "plan"
        ]
        
        input_lower = user_input.lower()
        
        healthcare_score = sum(1 for k in healthcare_keywords if k in input_lower)
        social_score = sum(1 for k in social_keywords if k in input_lower)
        
        if healthcare_score > social_score and healthcare_score > 0:
            return "healthcare"
        elif social_score > healthcare_score and social_score > 0:
            return "social"
        
        return None
    
    def get_agent(self, name: str) -> Optional[BaseAgent]:
        """Get specific agent"""
        return self.agents.get(name)
    
    def get_all_agents(self) -> List[BaseAgent]:
        """Get all available agents"""
        return list(self.agents.values()) + [self.default_agent]
```

---

### 4.4 Phase 4 Resource Estimate

| Component | Effort | Dependencies |
|-----------|--------|--------------|
| Base Agent Class | 1 week | Phase 1-2 |
| Healthcare Agent | 3 weeks | Base Agent |
| Social-Life Agent | 3 weeks | Base Agent |
| Agent Manager | 2 weeks | Healthcare, Social |
| **Total** | **9 weeks** | |

---

## Phase 5: Optimization & Performance (Months 13-15)

### Overview
Implement NPU optimizations, speculative decoding, ExPO training, and battery management.

### 5.1 NPU Speculative Decoding

**Files:**
- Create: `src/optimization/speculative_decode.py`
- Create: `src/optimization/npu_coordinator.py`

**Step 1: Implement speculative decoding**

```python
# src/optimization/speculative_decode.py
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import threading

class SpeculativeDecoder:
    """NPU-coordinated speculative decoding"""
    
    def __init__(self, model_manager):
        self.mm = model_manager
        self.draft_model = None
        self.target_model = None
        self.npu_available = self._detect_npu()
    
    def _detect_npu(self) -> bool:
        """Detect NPU availability"""
        import os
        return os.path.exists("/dev/npu") or os.path.exists("/sys/class/npu")
    
    def load_models(self, draft_name: str, target_name: str):
        """Load draft and target models"""
        # Draft model: smaller, faster
        self.mm.load_model(draft_name, model_type="router")
        
        # Target model: larger, more accurate
        self.mm.load_model(target_name, model_type="reasoner")
    
    def generate_with_speculation(
        self,
        prompt: str,
        max_tokens: int = 512,
        speculation_steps: int = 4
    ) -> str:
        """Generate with speculative decoding"""
        
        # Phase 1: Draft tokens
        draft_tokens = self._draft_tokens(prompt, speculation_steps)
        
        # Phase 2: Verify and correct with target model
        verified_tokens = self._verify_tokens(
            prompt,
            draft_tokens,
            max_tokens
        )
        
        return verified_tokens
    
    def _draft_tokens(self, prompt: str, num_tokens: int) -> List[int]:
        """Generate draft tokens with small model"""
        
        # Use router model for drafting
        response = self.mm.generate(
            prompt,
            model_type="router",
            max_tokens=num_tokens,
            temperature=0.5  # Lower temperature for drafts
        )
        
        # Tokenize
        tokens = self.mm.loaded_models["router"]['tokenizer'](
            response,
            return_tensors="pt"
        )['input_ids'][0]
        
        return tokens.tolist()[:num_tokens]
    
    def _verify_tokens(
        self,
        prompt: str,
        draft_tokens: List[int],
        max_tokens: int
    ) -> str:
        """Verify draft tokens with target model"""
        
        # Combine prompt with draft tokens
        # In real implementation, process through target model
        # and accept/reject each token based on probability
        
        target_model = self.mm.loaded_models.get("reasoner")
        if not target_model:
            return self.mm.generate(prompt, model_type="router", max_tokens=max_tokens)
        
        # Simplified: just use target model
        response = self.mm.generate(
            prompt,
            model_type="reasoner",
            max_tokens=max_tokens
        )
        
        return response
    
    def get_speedup_estimate(self) -> float:
        """Estimate speedup from speculative decoding"""
        if self.npu_available:
            return 3.81  # From research
        return 1.0
```

---

### 5.2 Battery Management

**Files:**
- Create: `src/optimization/battery_manager.py`
- Create: `src/optimization/power_profiles.py`

**Step 2: Implement battery management**

```python
# src/optimization/battery_manager.py
import os
import threading
from typing import Dict, Optional
import time

class BatteryManager:
    """Manages power consumption based on battery state"""
    
    def __init__(self):
        self.battery_level = 100
        self.is_charging = False
        self.power_profile = "balanced"
        self._monitor_thread = None
        self._running = False
        
        # Thresholds
        self.critical_threshold = 15
        self.low_threshold = 30
        self.high_threshold = 80
    
    def start_monitoring(self):
        """Start battery monitoring"""
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop battery monitoring"""
        self._running = False
        if self._monitor_thread:
            self._monitor_thread.join()
    
    def _monitor_loop(self):
        """Monitor battery in background"""
        while self._running:
            self._update_battery_status()
            self._adjust_power_profile()
            time.sleep(60)  # Check every minute
    
    def _update_battery_status(self):
        """Update battery status"""
        try:
            # Try Termux API
            if os.path.exists("/data/data/com.termux"):
                import subprocess
                result = subprocess.run(
                    ["termux-battery-status"],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    import json
                    status = json.loads(result.stdout)
                    self.battery_level = status.get("percentage", 100)
                    self.is_charging = status.get("plugged", "") != ""
        except:
            # Fallback: assume always charging for desktop
            self.battery_level = 100
            self.is_charging = True
    
    def _adjust_power_profile(self):
        """Adjust power profile based on battery"""
        
        if self.is_charging:
            self.power_profile = "performance"
            return
        
        if self.battery_level <= self.critical_threshold:
            self.power_profile = "ultra_low_power"
        elif self.battery_level <= self.low_threshold:
            self.power_profile = "low_power"
        elif self.battery_level <= self.high_threshold:
            self.power_profile = "balanced"
        else:
            self.power_profile = "performance"
    
    def get_model_config(self) -> Dict:
        """Get model configuration based on power profile"""
        
        configs = {
            "ultra_low_power": {
                "default_model": "router",
                "max_tokens": 256,
                "use_cache": True,
                "batch_size": 1
            },
            "low_power": {
                "default_model": "router",
                "max_tokens": 512,
                "use_cache": True,
                "batch_size": 1
            },
            "balanced": {
                "default_model": "router",
                "max_tokens": 1024,
                "use_cache": True,
                "batch_size": 2
            },
            "performance": {
                "default_model": "reasoner",
                "max_tokens": 2048,
                "use_cache": True,
                "batch_size": 4
            }
        }
        
        return configs.get(self.power_profile, configs["balanced"])
    
    def should_preload_models(self) -> bool:
        """Check if models should be preloaded"""
        return self.is_charging or self.battery_level > self.low_threshold
    
    def should_run_background_tasks(self) -> bool:
        """Check if background tasks should run"""
        return self.is_charging or self.battery_level > self.critical_threshold
```

---

### 5.3 ExPO Training Pipeline

**Files:**
- Create: `src/training/expo_trainer.py`
- Create: `config/training_config.yaml`

**Step 3: Implement ExPO training**

```python
# src/training/expo_trainer.py
import json
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from trl import DPOTrainer, SFTTrainer

class EXPOTrainer:
    """Self-Explanation Policy Optimization trainer"""
    
    def __init__(
        self,
        model_path: str,
        output_path: str = "models/trained"
    ):
        self.model_path = model_path
        self.output_path = output_path
        self.model = None
        self.tokenizer = None
    
    def load_model(self):
        """Load base model for training"""
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
    
    def prepare_expo_data(
        self,
        tasks: List[Dict],
        ground_truths: List[str]
    ) -> List[Dict]:
        """Prepare data for ExPO training"""
        
        expo_data = []
        
        for task, gt in zip(tasks, ground_truths):
            # Generate self-explanation
            prompt = f"""Task: {task}
Ground truth answer: {gt}

Generate a step-by-step self-explanation showing your reasoning:
1. Initial understanding of the task
2. Reasoning steps taken
3. How you arrived at the answer"""
            
            # In production, generate this from the model itself
            # This is a placeholder
            explanation = f"Step 1: Analyze task -> {task}\nStep 2: Reason -> {gt}\nStep 3: Conclude"
            
            expo_data.append({
                "prompt": task,
                "ground_truth": gt,
                "self_explanation": explanation
            })
        
        return expo_data
    
    def train_with_expo(
        self,
        train_data: List[Dict],
        epochs: int = 3,
        learning_rate: float = 1e-5
    ):
        """Train model using ExPO"""
        
        # Setup LoRA
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"],
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(self.model, lora_config)
        
        # Setup trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_data,
            tokenizer=self.tokenizer,
            max_seq_length=512,
            packing=True
        )
        
        # Train
        trainer.train()
        
        # Save
        model.save_pretrained(self.output_path)
        self.tokenizer.save_pretrained(self.output_path)
    
    def evaluate_reasoning(self, test_data: List[Dict]) -> Dict:
        """Evaluate reasoning performance"""
        
        results = {
            "total": len(test_data),
            "correct": 0,
            "reasoning_quality": []
        }
        
        for item in test_data:
            prompt = item["prompt"]
            expected = item["ground_truth"]
            
            response = self.model.generate(
                **self.tokenizer(prompt, return_tensors="pt").to("cuda"),
                max_new_tokens=256
            )
            
            generated = self.tokenizer.decode(response[0], skip_special_tokens=True)
            
            # Check if correct
            if expected.lower() in generated.lower():
                results["correct"] += 1
            
            # Evaluate reasoning (simplified)
            results["reasoning_quality"].append({
                "prompt": prompt,
                "generated": generated,
                "expected": expected
            })
        
        results["accuracy"] = results["correct"] / results["total"]
        return results
```

---

### 5.4 Phase 5 Resource Estimate

| Component | Effort | Dependencies |
|-----------|--------|--------------|
| Speculative Decoding | 2 weeks | Phase 1 |
| Battery Management | 2 weeks | Phase 1 |
| ExPO Training | 3 weeks | Phase 1-2 |
| Performance Testing | 2 weeks | All |
| **Total** | **9 weeks** | |

---

## Phase 6: Enterprise & Compliance (Months 16-18)

### Overview
Address enterprise readiness gaps from simulation: SOC2 prep, compliance documentation, support infrastructure.

### 6.1 Compliance Framework

**Files:**
- Create: `src/security/compliance_manager.py`
- Create: `docs/compliance/SOC2_READINESS.md`
- Create: `docs/compliance/PRIVACY.md`

**Step 1: Implement compliance manager**

```python
# src/security/compliance_manager.py
import hashlib
import json
import os
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class ComplianceManager:
    """Manages compliance and audit requirements"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.audit_log_path = os.path.join(data_dir, "audit.log")
        self.data_inventory_path = os.path.join(data_dir, "data_inventory.json")
        
        self.data_inventory = self._load_inventory()
    
    def _load_inventory(self) -> Dict:
        """Load data inventory"""
        if os.path.exists(self.data_inventory_path):
            with open(self.data_inventory_path) as f:
                return json.load(f)
        return {"data_categories": {}}
    
    def _save_inventory(self):
        """Save data inventory"""
        with open(self.data_inventory_path, 'w') as f:
            json.dump(self.data_inventory, f, indent=2)
    
    def register_data_category(
        self,
        category: str,
        pii: bool,
        retention_days: int,
        encryption: bool = True
    ):
        """Register data category"""
        self.data_inventory["data_categories"][category] = {
            "pii": pii,
            "retention_days": retention_days,
            "encryption_required": encryption,
            "registered_at": datetime.utcnow().isoformat()
        }
        self._save_inventory()
    
    def log_data_access(
        self,
        user_id: str,
        data_category: str,
        action: str,
        ip_address: Optional[str] = None
    ):
        """Log data access for audit"""
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_id": hashlib.sha256(user_id.encode()).hexdigest()[:16],  # Hash for privacy
            "data_category": data_category,
            "action": action,
            "ip_address": ip_address or "local"
        }
        
        with open(self.audit_log_path, 'a') as f:
            f.write(json.dumps(entry) + "\n")
    
    def generate_compliance_report(self) -> Dict:
        """Generate compliance report"""
        
        report = {
            "generated_at": datetime.utcnow().isoformat(),
            "data_categories": self.data_inventory["data_categories"],
            "total_accesses": 0,
            "pii_data_registered": []
        }
        
        # Count accesses
        if os.path.exists(self.audit_log_path):
            with open(self.audit_log_path) as f:
                report["total_accesses"] = sum(1 for _ in f)
        
        # List PII categories
        for cat, info in self.data_inventory["data_categories"].items():
            if info.get("pii"):
                report["pii_data_registered"].append(cat)
        
        return report
    
    def check_data_retention(self) -> List[Dict]:
        """Check and enforce data retention policies"""
        
        violations = []
        
        # In production, check actual data against retention policies
        # This is a placeholder
        
        return violations
```

---

### 6.2 Security Hardening

**Files:**
- Create: `src/security/secure_storage.py`
- Create: `src/security/privacy_verifier.py`

**Step 2: Implement secure storage**

```python
# src/security/secure_storage.py
import os
import hashlib
from cryptography.fernet import Fernet
from typing import Any, Optional
import json

class SecureStorage:
    """Encrypted local storage for sensitive data"""
    
    def __init__(self, key_path: str = "data/.storage.key"):
        self.key_path = key_path
        self.cipher = self._get_cipher()
    
    def _get_cipher(self) -> Optional[Fernet]:
        """Get or create encryption cipher"""
        if os.path.exists(self.key_path):
            with open(self.key_path, 'rb') as f:
                key = f.read()
            return Fernet(key)
        
        # Generate new key
        key = Fernet.generate_key()
        os.makedirs(os.path.dirname(self.key_path), exist_ok=True)
        
        # Store key securely
        with open(self.key_path, 'wb') as f:
            f.write(key)
        
        # Set restrictive permissions
        os.chmod(self.key_path, 0o600)
        
        return Fernet(key)
    
    def store(self, key: str, value: Any, encrypt: bool = True):
        """Store value securely"""
        data = json.dumps(value).encode()
        
        if encrypt and self.cipher:
            data = self.cipher.encrypt(data)
        
        path = self._get_path(key)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        with open(path, 'wb') as f:
            f.write(data)
    
    def retrieve(self, key: str, decrypt: bool = True) -> Optional[Any]:
        """Retrieve stored value"""
        path = self._get_path(key)
        
        if not os.path.exists(path):
            return None
        
        with open(path, 'rb') as f:
            data = f.read()
        
        if decrypt and self.cipher:
            data = self.cipher.decrypt(data)
        
        return json.loads(data.decode())
    
    def delete(self, key: str):
        """Securely delete value"""
        path = self._get_path(key)
        
        if os.path.exists(path):
            # Overwrite with zeros before delete
            with open(path, 'wb') as f:
                f.write(b'\x00' * os.path.getsize(path))
            
            os.remove(path)
    
    def _get_path(self, key: str) -> str:
        """Get storage path for key"""
        # Hash key for filesystem safety
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        return os.path.join("data", "secure", f"{key_hash}.dat")
```

---

### 6.3 Installation & Onboarding

**Files:**
- Create: `installation/one_click_install.sh`
- Create: `installation/setup_wizard.py`
- Create: `docs/GETTING_STARTED.md`

**Step 3: Implement one-click installer**

```bash
#!/bin/bash
# installation/one_click_install.sh

set -e

echo "========================================="
echo "AURA v3 One-Click Installer"
echo "========================================="

# Check if Termux
if [ ! -d "/data/data/com.termux" ]; then
    echo "Error: AURA v3 requires Termux on Android"
    echo "Visit: https://termux.netdroid.com/"
    exit 1
fi

# Update package list
echo "[1/6] Updating packages..."
apt update && apt upgrade -y

# Install required packages
echo "[2/6] Installing dependencies..."
apt install -y python python-pip git curl wget

# Clone or update AURA
echo "[3/6] Setting up AURA..."
if [ -d "$PREFIX/lib/aura" ]; then
    cd $PREFIX/lib/aura
    git pull
else
    git clone https://github.com/aura-ai/aura-v3.git $PREFIX/lib/aura
fi

# Install Python dependencies
echo "[4/6] Installing Python packages..."
cd $PREFIX/lib/aura
pip install -r requirements.txt

# Initialize
echo "[5/6] Initializing AURA..."
python installation/init.py

# Create shortcuts
echo "[6/6] Creating shortcuts..."
echo 'alias aura="python $PREFIX/lib/aura/src/main.py"' >> $PREFIX/etc/profile

echo ""
echo "========================================="
echo "Installation complete!"
echo "Run 'aura' to start AURA"
echo "========================================="
```

---

### 6.4 Phase 6 Resource Estimate

| Component | Effort | Dependencies |
|-----------|--------|--------------|
| Compliance Manager | 2 weeks | Phase 1-2 |
| Security Hardening | 2 weeks | All |
| One-Click Installer | 2 weeks | Phase 1 |
| Documentation | 3 weeks | All |
| **Total** | **9 weeks** | |

---

## Dependencies Diagram

```
Phase 1 (Foundation)
├── 1.1 Mobile Build System
├── 1.2 SLM Pipeline
│   └── Depends on: 1.1
├── 1.3 SQLite State Machine
└── 1.4 Basic Orchestrator
    └── Depends on: 1.2, 1.3

Phase 2 (Memory)
├── 2.1 Zvec Integration
│   └── Depends on: 1.3
├── 2.2 RAG Pipeline
│   └── Depends on: 2.1
├── 2.3 Graphiti Knowledge Graph
│   └── Depends on: 1.3
└── 2.4 User Profiling
    └── Depends on: 2.3

Phase 3 (App Discovery)
├── 3.1 App Discovery
├── 3.2 GraphPilot
│   └── Depends on: 3.1, 1.3
├── 3.3 OmniParser
└── 3.4 Tool Binding
    └── Depends on: 3.2

Phase 4 (Sub-Agents)
├── 4.1 Base Agent Class
│   └── Depends on: Phase 1-2
├── 4.2 Healthcare Agent
│   └── Depends on: 4.1
├── 4.3 Social-Life Agent
│   └── Depends on: 4.1
└── 4.4 Agent Manager
    └── Depends on: 4.2, 4.3

Phase 5 (Optimization)
├── 5.1 Speculative Decoding
│   └── Depends on: 1.2
├── 5.2 Battery Management
├── 5.3 ExPO Training
│   └── Depends on: Phase 1-2
└── 5.4 Performance Testing
    └── Depends on: All

Phase 6 (Enterprise)
├── 6.1 Compliance
├── 6.2 Security
│   └── Depends on: All
├── 6.3 Installer
│   └── Depends on: 1.1
└── 6.4 Documentation
    └── Depends on: All
```

---

## Risk Assessment & Mitigation

### High Priority Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| Model compatibility with Termux | High | Medium | Test each model early; maintain fallback models |
| Memory constraints on 4GB devices | High | High | Aggressive quantization; model swapping; streaming |
| NPU availability variance | Medium | High | CPU fallback; detect hardware; dynamic optimization |
| Battery drain complaints | High | High | Aggressive power management; user controls |
| Installation complexity (TOP complaint) | High | Very High | One-click installer; extensive testing |

### Medium Priority Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| GraphPilot graph construction time | Medium | Medium | Background processing; incremental builds |
| OmniParser model size | Medium | Low | Use smallest variant; NPU-only when available |
| Privacy claims verification | Medium | High | Open source components; independent audits |
| Enterprise compliance timeline | Medium | Medium | Start SOC2 prep early; modular compliance |

### Risk Response Strategy

1. **Prototype Early**: Test memory and model loading in Phase 1
2. **User Testing**: Gather feedback on battery and installation
3. **Fallback Architecture**: Always maintain simpler fallback modes
4. **Modular Design**: Allow disabling features for resource-constrained devices

---

## Summary Timeline

| Phase | Duration | Cumulative | Key Deliverables |
|-------|----------|------------|------------------|
| Phase 1: Foundation | 3 months | 3 months | Mobile build, SLM pipeline, state machine, orchestrator |
| Phase 2: Memory | 3 months | 6 months | Zvec RAG, Graphiti KG, user profiling |
| Phase 3: App Discovery | 3 months | 9 months | GraphPilot, OmniParser, dynamic tools |
| Phase 4: Sub-Agents | 3 months | 12 months | Healthcare agent, social-life agent, manager |
| Phase 5: Optimization | 3 months | 15 months | Speculative decoding, battery management, ExPO |
| Phase 6: Enterprise | 3 months | 18 months | Compliance, security, installer, docs |

**Total Timeline: 18 months**

---

## Success Metrics

### Technical Metrics
- [ ] Response time < 5s for simple queries on 4GB device
- [ ] Battery drain < 5% per hour of active use
- [ ] Memory usage < 3GB during operation
- [ ] Model swap time < 3 seconds

### User Experience Metrics
- [ ] Installation success rate > 95% on first try
- [ ] User satisfaction score > 4/5
- [ ] Privacy trust score > 4.5/5
- [ ] Feature adoption > 70% for core features

### Enterprise Metrics
- [ ] SOC2 readiness checklist complete
- [ ] Documentation coverage > 90%
- [ ] Security audit passed

---

## Conclusion

This implementation plan transforms AURA v3 from concept to production using 2026 edge AI research. The 6-phase, 18-month approach addresses:

1. **Top complaints from simulation**: Installation, battery, documentation
2. **Research-backed architecture**: Tool-first, GraphPilot, Zvec, Graphiti
3. **Mobile-first constraints**: 4GB RAM, NPU optimization, battery management
4. **Unique differentiator**: Adaptive personality, sub-agents, proactive assistance

The modular design allows parallel development and provides fallbacks for each major component. Risk mitigation strategies ensure the project stays on track despite hardware variability and technical complexity.

---

*Plan created: February 2026*
*Version: 1.0*
*Classification: Implementation Roadmap*
