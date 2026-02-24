# AURA Hardcoded Values Audit Report

## Overview
This document catalogs ALL hardcoded values found in the AURA codebase that should be made adaptive.

---

## 1. HARDCODED WORD LISTS

### 1.1 Sentiment/Category Keywords

| File | Line | Hardcoded Value | Why Hardcoded | Suggested Adaptive Replacement |
|------|------|-----------------|---------------|-------------------------------|
| `src/services/communication_manager.py` | 118 | `["urgent", "important", "asap", "deadline"]` | User preferences should be learned | Learn from user behavior or allow config |
| `src/services/communication_manager.py` | 156 | `["respectfully", "kindly", "please", "regards", "sincerely"]` | Formality detection | Make configurable per user |
| `src/services/communication_manager.py` | 157 | `["hey", "thanks", "cool", "awesome", "!"]` | Formality detection | Make configurable per user |
| `src/services/communication_manager.py` | 175 | `["sale", "offer", "discount", "deal", "shop", "buy"]` | Email categorization | Learn from user labeling behavior |
| `src/services/communication_manager.py` | 176-182 | `["facebook", "instagram", "linkedin", "twitter", "notification"]` | Social email detection | User should configure or learn |
| `src/services/communication_manager.py` | 183 | `["meeting", "project", "report", "team", "client", "deadline"]` | Work email detection | User should configure or learn |
| `src/services/communication_manager.py` | 196-202 | Urgent patterns list | Important chat detection | User should customize |
| `src/services/health_analyzer.py` | 349 | `["home", "work", "outdoor", "gym", "park", "transit"]` | Location types | User should define custom locations |

### 1.2 Response Templates

| File | Line | Hardcoded Value | Why Hardcoded | Suggested Adaptive Replacement |
|------|------|-----------------|---------------|-------------------------------|
| `src/voice/pipeline.py` | 110 | `"I couldn't understand that. Please try again."` | Fixed error message | Template with params or i18n |
| `src/voice/pipeline.py` | 117 | `"Voice received but no intent handler configured."` | Fixed response | Template system |
| `src/voice/pipeline.py` | 123 | `"Sorry, I had trouble processing that."` | Fixed error | Template with params |
| `src/voice/integration.py` | 151 | `"I'm here, but my core systems aren't fully initialized yet."` | Fixed message | Template system |
| `src/agent/loop.py` | 294 | `"I'm having trouble completing this task. Let me know if you'd like to try a different approach."` | Error fallback | Template with params |
| `src/agent/loop.py` | 747 | `"Alright, I won't proceed with that action."` | Security rejection | Template system |
| `src/agent/loop.py` | 782 | `"I'm not sure what action was pending."` | State confusion | Template system |
| `src/services/communication_manager.py` | 406 | `"Hi"` / `"Hello"` | Greeting | Based on learned formality |
| `src/services/communication_manager.py` | 414 | `"Dear Sir/Madam,"` | Formal greeting | User preference |
| `src/services/communication_manager.py` | 416 | `"Best regards"` / `"Thanks"` | Closing | Based on formality |
| `src/services/communication_manager.py` | 884 | `"You're welcome!"` / `"No problem!"` | Quick reply | Learn from user style |
| `src/services/communication_manager.py` | 888 | `"Great!"` / `"Cool!"` | Quick reply | Learn from user style |

---

## 2. HARDCODED PATHS

| File | Line | Hardcoded Value | Why Hardcoded | Suggested Adaptive Replacement |
|------|------|-----------------|---------------|-------------------------------|
| `main.py` | 24 | `"logs", "data/memories", "data/sessions", "data/patterns", "models"` | Directory structure | Config file |
| `main.py` | 51 | `"logs/aura.log"` | Log file path | Configurable |
| `main.py` | 83 | `"models/qwen2.5-1b-q5_k_m.gguf"` | Model path | Config |
| `main.py` | 103 | `"data/memories/aura.db"` | Database path | Config |
| `main.py` | 105 | `"data/memories/self_model.db"` | Self model path | Config |
| `main.py` | 116 | `"logs/security.log"` | Security log | Config |
| `main.py` | 129 | `"data/patterns"` | Patterns path | Config |
| `main.py` | 147 | `"data/sessions"` | Session storage | Config |
| `main.py` | 479-522 | All paths in default config | Hardcoded defaults | Config file |
| `src/context/proactive_engine.py` | 124 | `"data/context"` | Data path | Config |
| `src/security/password_manager.py` | 72 | `"data/security/passwords.db"` | Password DB | Config |
| `src/addons/framework.py` | 414 | `"data/addons"` | Addons directory | Config |
| `src/channels/telegram_bot.py` | 153 | `"data/telegram_state.json"` | State file | Config |

---

## 3. HARDCODED SETTINGS

### 3.1 Timeouts

| File | Line | Value | Suggested Adaptive Replacement |
|------|------|-------|------------------------------|
| `src/context/proactive_engine.py` | 732 | `delay = 60` | Configurable based on urgency |
| `src/context/proactive_engine.py` | 736 | `delay = 1800` | Learn from user patterns |
| `src/context/proactive_engine.py` | 738 | `delay = 600` | Context-aware |
| `src/context/proactive_engine.py` | 740 | `delay = 300` | Learn from behavior |
| `src/services/health_analyzer.py` | 217 | `timeout=5` | Config |
| `src/services/health_analyzer.py` | 238 | `timeout=10` | Config |
| `src/services/health_analyzer.py` | 270 | `timeout=30` | Config |
| `src/services/health_analyzer.py` | 332 | `timeout=30` | Config |
| `src/voice/tts.py` | 81 | `timeout=30` | Config |
| `src/voice/tts.py` | 364 | `timeout=30` | Config |
| `src/voice/optimization.py` | 239 | `timeout=30` | Config |
| `src/tools/android_control/file_manager.py` | 31, 38 | `timeout=300` | Large file config |
| `src/tools/android_control/adb_bridge.py` | 35 | `timeout=10` | Config |
| `src/tools/android_control/adb_bridge.py` | 112 | `timeout=30` | Config |
| `src/tools/android_control/adb_bridge.py` | 122 | `timeout=120` | Long-running ops |
| `src/tools/android_control/adb_bridge.py` | 197 | `timeout=10` | Config |
| `src/tools/android_offline.py` | 130 | `timeout=10` | Config |
| `src/tools/android_offline.py` | 241 | `timeout=10` | Config |
| `src/tools/android_offline.py` | 338 | `timeout=10` | Config |
| `src/tools/android_offline.py` | 474 | `timeout=30` | Config |
| `src/tools/android_offline.py` | 591 | `timeout=10` | Config |
| `src/tools/android_offline.py` | 727 | `timeout=5` | Config |
| `src/tools/android_v3.py` | 951 | `timeout=2` | Config |
| `src/tools/android_v3.py` | 966 | `timeout=10` | Config |
| `src/agent/loop.py` | 503 | `timeout=30.0` | Config |
| `src/tools/registry.py` | 145 | `timeout=30.0` | Config |

### 3.2 Limits/Thresholds

| File | Line | Value | Why Hardcoded | Suggested Adaptive Replacement |
|------|------|-------|---------------|------------------------------|
| `main.py` | 88 | `max_context=4096` | Model limit | Config |
| `main.py` | 89 | `n_gpu_layers=0` | GPU config | Auto-detect |
| `main.py` | 90 | `temperature=0.7` | Model param | User preference |
| `main.py` | 91 | `max_tokens=512` | Output limit | Config |
| `main.py` | 101 | `working_size=10` | Memory size | Config |
| `main.py` | 102 | `short_term_size=100` | Memory size | Config |
| `main.py` | 130 | `min_confidence=0.6` | Learning threshold | User preference |
| `main.py` | 131 | `max_patterns=1000` | Pattern limit | Config |
| `main.py` | 138-141 | `work_start_hour=9`, `work_end_hour=18`, `sleep_start_hour=23`, `sleep_end_hour=7` | Context hours | User schedule |
| `main.py` | 149 | `max_history=100` | Session limit | Config |
| `main.py` | 169 | `max_react_iterations=10` | Agent limit | Config |
| `main.py` | 170 | `tool_timeout=30` | Tool timeout | Config |
| `main.py` | 155-157 | `tts_engine="sarvam"`, `language="en"`, `speed=1.0` | Voice settings | User preference |
| `src/context/proactive_engine.py` | 129 | `_cache_duration = 60` | Cache duration | Config |
| `src/context/proactive_engine.py` | 525 | `location_anomaly_threshold` | Detection sensitivity | User preference |
| `src/context/proactive_engine.py` | 551 | `time_pattern_threshold` | Pattern sensitivity | User preference |
| `src/context/proactive_engine.py` | 601 | `missed_event_hours` | Event tracking | User preference |
| `src/services/communication_manager.py` | 213 | `chat.unread_count > 10` | Importance threshold | User configurable |
| `src/services/communication_manager.py` | 236 | `time.sleep(3)` | Wait times | Adaptive |
| `src/services/communication_manager.py` | 338 | `for _ in range(10)` | Spam delete limit | Config |
| `src/services/communication_manager.py` | 344 | `300` (swipe duration) | UI timing | Device-specific |
| `src/services/health_analyzer.py` | 854 | `quality_score = (sleep_analysis["average_quality_score"] / 5.0) * 100` | Fixed scale | Configurable |

### 3.3 Screen Coordinates (Hardcoded UI Positions)

| File | Line | Hardcoded Value | Why Hardcoded | Suggested Adaptive Replacement |
|------|------|-----------------|---------------|------------------------------|
| `src/services/communication_manager.py` | 308 | `width * 0.95`, `height * 0.08` | Menu tap position | Percentage-based |
| `src/services/communication_manager.py` | 311 | `width * 0.5`, `height * 0.45` | Delete confirm | Percentage-based |
| `src/services/communication_manager.py` | 334 | `width * 0.15`, `height * 0.08` | Spam button | Percentage-based |
| `src/services/communication_manager.py` | 340-344 | Various `width * X` percentages | Swipe actions | Percentage-based |
| `src/services/communication_manager.py` | 358-369 | Category tap positions | Menu positions | Percentage-based |
| `src/services/communication_manager.py` | 394 | `width * 0.9`, `height * 0.9` | Send button | Percentage-based |
| `src/services/communication_manager.py` | 475 | `width * 0.9`, `height * 0.12` | Search tap | Percentage-based |
| `src/services/communication_manager.py` | 484 | `width * 0.5`, `height * 0.25` | Chat open | Percentage-based |
| `src/services/communication_manager.py` | 540 | `width * 0.9`, `height * 0.85` | Send tap | Percentage-based |
| `src/services/communication_manager.py` | 559 | `width * 0.9`, `height * 0.85` | Send in chat | Percentage-based |
| `src/services/communication_manager.py` | 573-578 | Swipe positions | Archive action | Percentage-based |

---

## 4. HARDCODED MESSAGES

### 4.1 Welcome/Greeting Messages

| File | Line | Hardcoded Value | Suggested Adaptive Replacement |
|------|------|-----------------|-------------------------------|
| `main.py` | 228-233 | CLI welcome banner | Template or i18n |
| `main.py` | 266-270 | Voice mode banner | Template |
| `main.py` | 245 | `"Goodbye!"` | Template |
| `main.py` | 384-386 | Telegram start message | Template system |
| `src/channels/telegram_bot.py` | 966 | Welcome message block | Template system |

### 4.2 Error Messages

| File | Line | Hardcoded Value | Suggested Adaptive Replacement |
|------|------|-----------------|-------------------------------|
| `main.py` | 223 | `"I encountered an error: {str(e)}. Please try again."` | Template system |
| `src/voice/pipeline.py` | Various | Error responses | Template with error codes |

---

## 5. HARDCODED REGEX PATTERNS

| File | Line | Hardcoded Value | Why Hardcoded | Suggested Adaptive Replacement |
|------|------|-----------------|---------------|------------------------------|
| `src/services/communication_manager.py` | 612-616 | Date patterns | Common formats only | User can add patterns |
| `src/services/communication_manager.py` | 621 | Time pattern | Fixed format | Config |
| `src/services/communication_manager.py` | 624 | Phone pattern | Fixed format | International support |
| `src/services/communication_manager.py` | 627 | Email pattern | Standard pattern | OK as-is |
| `src/services/communication_manager.py` | 630 | URL pattern | Standard pattern | OK as-is |
| `src/services/communication_manager.py` | 633-634 | Money patterns | US-centric | Configurable |
| `src/services/communication_manager.py` | 638 | Address pattern | US-centric | Region-specific |
| `src/security/prompt_guard.py` | 32-129 | All INJECTION_PATTERNS | Security patterns | Should be updatable |

---

## 6. HARDCODED CONFIGURATION ENUMS

| File | Line | Hardcoded Value | Why Hardcoded | Suggested Adaptive Replacement |
|------|------|-----------------|---------------|------------------------------|
| `main.py` | 114 | `PermissionLevel["L2"]` | Security default | Config |
| `main.py` | 117 | `banking_protection=True` | Security default | Config |
| `src/context/proactive_engine.py` | 138-144 | PrivacyLevel defaults | Privacy settings | User preference |

---

## 7. HARDCODED MODEL/PROVIDER NAMES

| File | Line | Hardcoded Value | Suggested Adaptive Replacement |
|------|------|-----------------|------------------------------|
| `main.py` | 155 | `tts_engine="sarvam"` | Config |
| `src/services/communication_manager.py` | 100-101 | `GMAIL_PACKAGE`, `WHATSAPP_PACKAGE` | Config |

---

## Summary Statistics

| Category | Count |
|----------|-------|
| Hardcoded word lists | ~15 |
| Hardcoded paths | ~15 |
| Hardcoded timeouts | ~25 |
| Hardcoded limits/thresholds | ~20 |
| Hardcoded screen coordinates | ~20 |
| Hardcoded messages | ~20 |
| Hardcoded regex patterns | ~10 |
| Hardcoded config enums | ~5 |

**Total: ~130+ hardcoded values identified**

---

## Recommended Actions

1. **Create central configuration file** (`config/adaptive.yaml` or similar)
2. **Implement template system** for all user-facing messages
3. **Add user preference learning** for word lists and thresholds
4. **Use percentage-based UI coordinates** instead of hardcoded positions
5. **Make security patterns updatable** via config or API
6. **Implement i18n system** for all messages
7. **Add adaptive timeout** based on operation type and device capabilities
