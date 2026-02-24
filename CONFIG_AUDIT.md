# AURA Configuration Management Audit

**Date:** 2026-02-21  
**Scope:** `C:\Users\Lenovo\AppData\Local\Temp\aura-v3\`

---

## 1. Hardcoded Values

### 1.1 Environment Variables

| Location | Variable | Status |
|----------|----------|--------|
| `src/channels/telegram_bot.py:1708` | `TELEGRAM_TOKEN` | **Required** - No default, fails if missing |

**Finding:** Only ONE environment variable is used. The system has minimal external configuration through env vars.

### 1.2 Configuration Files

| File | Purpose | Status |
|------|---------|--------|
| `config/config.yaml` | Optional YAML config | Referenced but **not present** in repo |
| `data/context/patterns.json` | Learned patterns | Auto-generated |
| `data/context/history.json` | Context history | Auto-generated |
| `data/security/passwords.db` | Encrypted password store | Auto-created |
| `data/telegram_state.json` | Telegram bot state | Auto-created |

**Finding:** No central configuration file exists. The system expects optional `config/config.yaml` but falls back to empty dict.

### 1.3 Hardcoded Paths

All paths are hardcoded with no centralized override:

| Component | Default Path |
|-----------|-------------|
| Logs | `logs/aura.log` |
| Security logs | `logs/security.log` |
| Memories | `data/memories/aura.db` |
| Self model | `data/memories/self_model.db` |
| Patterns | `data/patterns` |
| Sessions | `data/sessions` |
| Context | `data/context` |
| Addons | `data/addons` |
| Model | `models/qwen2.5-1b-q5_k_m.gguf` |

---

## 2. Config Structure Analysis

### 2.1 Centralization

**Rating: POOR**

- No central config file or module
- Each component accepts config via constructor parameters
- Config is passed as optional `Dict` to constructors
- Scattered default values throughout codebase

### 2.2 Environment-Based Configuration

**Rating: MINIMAL**

- Only `TELEGRAM_TOKEN` uses environment
- Optional `config/config.yaml` loading in `telegram_bot.py:1701-1705`
- Most configuration is code-based (constructor params)

### 2.3 Documentation

**Rating: NONE FOUND**

- No dedicated config documentation
- No config schema or validation
- Defaults embedded in code comments only

---

## 3. Default Values Assessment

### 3.1 Memory Budgets (src/memory/memory_budget.py:38-46)

```python
DEFAULT_BUDGETS = {
    "system": 1434,   # 35% of 4GB
    "llm": 1024,      # 25% of 4GB
    "working": 204,   # 5% of 4GB
    "episodic": 410,  # 10% of 4GB
    "semantic": 328,  # 8% of 4GB
    "cache": 286,     # 7% of 4GB
    "buffer": 410,    # 10% of 4GB
}
```

**Safe?** YES - Sensible defaults for 4GB device, easily overridden

### 3.2 Timeouts (scattered throughout)

| Component | Default | Overrideable? |
|-----------|---------|---------------|
| Health checks | 5-30s | Via config dict |
| ADB operations | 10-300s | Yes |
| Voice TTS | 30s | Yes |
| Tool execution | 30s | Yes |

**Safe?** YES - Reasonable defaults, configurable

### 3.3 Privacy Settings (src/context/proactive_engine.py:138-144)

```python
_privacy_settings = {
    "track_location": PrivacyLevel.STANDARD,
    "track_calendar": PrivacyLevel.STANDARD,
    "track_app_usage": PrivacyLevel.STANDARD,
    "track_messages": PrivacyLevel.MINIMAL,
    "track_activity": PrivacyLevel.STANDARD,
}
```

**Safe?** YES - Sensible defaults with MINIMAL for sensitive data (messages)

### 3.4 Model Configuration (src/llm/model_manager.py)

- RAM detection with fallback to 4096MB
- Auto-detects Android/Termux
- Graceful degradation on detection failure

**Safe?** YES - Good fallback defaults

---

## 4. Secrets Management

| Secret | Storage | Assessment |
|--------|---------|------------|
| Telegram Token | Env var `TELEGRAM_TOKEN` | Good - not in code |
| Passwords | AES-256 encrypted SQLite | Good - proper encryption |
| API Keys | Not found in code | N/A |

**Finding:** Password manager uses AES-256-GCM with PBKDF2 key derivation - security is adequate.

---

## 5. Issues & Recommendations

### Critical Issues

1. **No central config file** - All paths and many values hardcoded
2. **No config validation** - Invalid config could cause runtime errors
3. **No environment-based secrets** - Only 1 env var supported

### Recommendations

1. Create `config/defaults.yaml` with all configurable paths
2. Add `.env.example` template for required variables
3. Implement config validation on startup
4. Document all configurable parameters
5. Add environment variable prefix (e.g., `AURA_*`) to avoid conflicts

### Good Practices Found

- Constructor-based config injection (dependency inversion)
- Graceful fallbacks when detection fails
- Sensible privacy defaults
- Proper encryption for sensitive data

---

## 6. Summary

| Aspect | Rating | Notes |
|--------|--------|-------|
| Centralization | 1/5 | Fragmented across modules |
| Environment-based | 1/5 | Only 1 env var |
| Documentation | 0/5 | None |
| Security | 4/5 | Good password encryption |
| Defaults Safety | 4/5 | Generally safe & overridable |

**Overall:** Configuration management needs significant improvement. The system relies on hardcoded defaults with minimal external configuration capability.
