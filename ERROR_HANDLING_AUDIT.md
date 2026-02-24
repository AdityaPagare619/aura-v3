# AURA Error Handling Audit

## Executive Summary

This audit identifies critical error handling, error recovery, and input validation issues across the AURA codebase. The review covered core modules including model management, task execution, memory systems, security, and communication channels.

---

## 1. Missing Error Handling

### Critical Issues

#### 1.1 model_manager.py - Silent Failures

| Location | Issue |
|----------|-------|
| `LazyModelProxy.generate` (line 668-683) | No try/except around inference call - complete failure with no graceful handling |
| `_do_load_model` (line 403-449) | No exception handling around `llama_cpp` initialization - crashes app if model file is corrupted |
| `get_inference` (line 571-573) | Uses `asyncio.run()` inside executor - event loop conflicts |
| `_ensure_initialized` (line 663-666) | No error handling if initialization fails |

#### 1.2 adb_bridge.py - No Error Handling

| Location | Issue |
|----------|-------|
| `shell()` (line 43-52) | No try/except around subprocess.run - crashes on ADB failure |
| `take_screenshot()` (line 106-114) | No error handling for pull command failure |
| `tap()`, `swipe()`, `text()` (lines 54-68) | Weak error detection - only checks for "error" in lowercase result |

#### 1.3 episodic_memory.py - Silent JSON Failures

| Location | Issue |
|----------|-------|
| `_search_similar` (line 270-272) | `json.loads()` without try/except - crashes on corrupted DB |
| `_get_recent` (line 326-328) | Same issue |
| `_get_all_embeddings` (line 350-355) | Same issue |

#### 1.4 telegram_bot.py - Hidden Errors

| Location | Issue |
|----------|-------|
| `_load_state` (line 163) | Bare `except:` silently swallows all errors |
| `_check_connectivity` (line 454) | Bare `except:` returns False without logging |
| `test_command` (line 1021-1084) | Tests fail silently without proper error context |

#### 1.5 memory_retrieval.py - AttributeErrors

| Location | Issue |
|----------|-------|
| `_search_skill` (line 231) | Calls `self.skill.list_skills()` without checking if skill is None |
| `_search_ancestor_async` (line 264) | `query.memory_types[0]` crashes if memory_types is None |

#### 1.6 task_engine.py - Division by Zero

| Location | Issue |
|----------|-------|
| `_execute_sequential` (line 635) | `i / len(tasks)` crashes if tasks list is empty |

---

## 2. Error Recovery

### Missing Recovery Mechanisms

#### 2.1 model_manager.py
- **Issue**: No circuit breaker when model fails to load repeatedly
- **Impact**: App crashes repeatedly on bad model files
- **Recommendation**: Add retry counter with exponential backoff

#### 2.2 adb_bridge.py
- **Issue**: No reconnection logic when ADB connection drops
- **Issue**: No retry on failed shell commands
- **Impact**: Single failure cascades to complete system failure

#### 2.3 tts.py
- **Issue**: `_check_installation` only logs warning - continues with missing Piper
- **Impact**: Later calls fail with unclear errors

#### 2.4 telegram_bot.py
- **Issue**: Inline query calculator (line 1446) uses `eval()` - security risk
- **Good**: Has fallback to rule-based parsing in task_engine.py

### Positive Patterns Found

- **task_engine.py**: Has fallback from LLM parser to rule-based parsing (line 249-251)
- **model_manager.py**: Has auto-unload timeout mechanism for memory management
- **tts.py**: Has queue size limits to prevent memory exhaustion

---

## 3. Input Validation

### Critical Validation Gaps

#### 3.1 model_manager.py

| Parameter | Issue |
|-----------|-------|
| `select_model` - available_mb | No validation for negative values |
| `load_model` - max_context | No range validation |
| `load_model` - n_gpu_layers | No bounds checking |

#### 3.2 adb_bridge.py

| Parameter | Issue |
|-----------|-------|
| `tap(x, y)` | No bounds validation against screen size |
| `swipe(x1, y1, x2, y2)` | No coordinate validation |
| `text(text)` | No length limit - potential buffer issues |
| `package_name` | No format validation |

#### 3.3 task_engine.py

| Parameter | Issue |
|-----------|-------|
| `execute_plan` tasks | No null check |
| Tool executor parameters | No schema validation |

#### 3.4 memory_retrieval.py

| Parameter | Issue |
|-----------|-------|
| `query.limit` | No bounds checking (could be negative) |
| `query.embedding` | No type/length validation |

#### 3.5 tts.py

| Parameter | Issue |
|-----------|-------|
| `speed` | No range validation (line 28 default=1.0) |
| `pitch` | No range validation |
| `noise_scale` | No bounds checking |

#### 3.6 telegram_bot.py - Code Injection Risk

```python
# Line 1446 - CRITICAL SECURITY ISSUE
result = eval(expression, {"__builtins__": {}}, {})
```
Despite attempts to restrict builtins, this pattern is dangerous and should be replaced with a proper math expression parser.

---

## 4. Priority Recommendations

### P0 - Critical (Fix Immediately)

1. **telegram_bot.py line 1446**: Replace `eval()` with safe expression parser
2. **adb_bridge.py shell()**: Add try/except around subprocess calls
3. **episodic_memory.py**: Wrap all `json.loads()` in try/except
4. **task_engine.py line 635**: Add zero-division protection

### P1 - High (Fix Soon)

1. **model_manager.py**: Add circuit breaker for repeated load failures
2. **adb_bridge.py**: Add reconnection logic for dropped connections
3. **memory_retrieval.py**: Add null checks for query parameters
4. **tts.py**: Add validation for speed/pitch parameters

### P2 - Medium (Plan for Next Sprint)

1. Add input validation across all public APIs
2. Replace bare `except:` with specific exception handling
3. Add logging to error recovery paths
4. Document expected parameter types in function signatures

---

## 5. File-by-File Summary

| File | Issues | Severity |
|------|--------|----------|
| model_manager.py | 6 | High |
| task_engine.py | 3 | Medium |
| memory_retrieval.py | 3 | High |
| adb_bridge.py | 8 | Critical |
| tts.py | 4 | Medium |
| episodic_memory.py | 4 | High |
| prompt_guard.py | 1 | Low (good validation) |
| telegram_bot.py | 4 | Critical |

---

*Audit completed: 2026-02-21*
