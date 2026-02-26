# AURA Self-Discovery Capabilities

> **What AURA learns about you and your device to become your truly personal AI assistant**

## Overview

AURA's self-discovery system consists of four interconnected components:

| Component | Purpose | Key File |
|-----------|---------|----------|
| **Self-Discovery Engine** | Learns AURA's own capabilities and personality | `src/services/self_discovery_engine.py` |
| **App Discovery** | Detects and categorizes installed apps | `src/addons/discovery.py` |
| **Context Detector** | Understands time, location, activity | `src/context/detector.py` |
| **Termux Bridge** | Android device control interface | `src/addons/termux_bridge.py` |

---

## 1. Device Discovery

### What AURA Discovers About Your Device

AURA gathers device information through the Termux Bridge to understand the environment:

```python
# Device information gathered
{
    "model": "ro.product.model",           # Device model (Samsung, Pixel, etc.)
    "android_version": "ro.build.version.release",  # Android OS version
    "battery": {
        "level": "percentage",              # Current battery level
        "state": "charging/discharging"     # Charging status
    },
    "storage": "available space"            # Storage capacity
}
```

### System State Detection

| State | Detection Method | Data Source |
|-------|------------------|-------------|
| Battery Level | `/sys/class/power_supply/battery/capacity` | System file |
| Charging Status | `/sys/class/power_supply/battery/status` | System file |
| Screen State | `/sys/class/backlight/panel/brightness` | System file |
| WiFi Status | Connection check | System state |
| Airplane Mode | Radio state check | System state |

### File System Access

```python
# Capabilities
- List directories: ls -la <path>
- Find files: find <directory> -name <pattern>
- Read files: cat <path> (with 1MB size limit for mobile)
- Search content: grep -r <query> <path>
```

---

## 2. App Detection & Categorization

### Automatic App Discovery

AURA scans multiple paths for apps:

```python
discovery_paths = [
    "src/addons/bundled",           # Built-in apps
    "aura_apps/",                   # Workspace apps
    "~/.aura/apps/",                # User-installed apps
    "~/.aura/termux_apps/"          # Termux-specific apps
]
```

### App Categories

| Category | Description | Examples |
|----------|-------------|----------|
| `COMMUNICATION` | Messaging and calling | WhatsApp, Telegram, SMS |
| `PRODUCTIVITY` | Work and task tools | Calendar, Notes, Email |
| `MEDIA` | Audio and video | Spotify, YouTube, Gallery |
| `SOCIAL` | Social networks | Twitter, Instagram, LinkedIn |
| `UTILITY` | System utilities | File managers, Settings |
| `SYSTEM` | Core system apps | Phone, Contacts |
| `CUSTOM` | User-defined apps | Custom scripts |

### App Capability Detection

AURA tracks what capabilities each app provides:

```python
class AppCapability(Enum):
    IMAGE_ANALYSIS = "image_analysis"
    VOICE_INPUT = "voice_input"
    VOICE_OUTPUT = "voice_output"
    TEXT_TO_SPEECH = "text_to_speech"
    SPEECH_TO_TEXT = "speech_to_text"
    VIDEO_ANALYSIS = "video_analysis"
    FILE_ACCESS = "file_access"
    CAMERA_ACCESS = "camera_access"
    LOCATION_ACCESS = "location_access"
    NOTIFICATIONS = "notifications"
    API_CALLS = "api_calls"
    WEB_SEARCH = "web_search"
    CODE_EXECUTION = "code_execution"
```

### App Manifest Format (APP.yaml)

Apps are defined via YAML manifests:

```yaml
---
name: WhatsApp Sender
description: Send WhatsApp messages via Termux
invoke_command: am start -a android.intent.action.SEND

metadata:
  icon: "\U0001F4AC"
  platforms: ["android"]
  category: communication
  capabilities:
    - notifications
    - api_calls
  trigger_patterns:
    - "send whatsapp"
    - "message on whatsapp"
    - "whatsapp message"
  fallback_apps:
    - telegram
    - sms
---
```

---

## 3. Context Detection

### Time-Based Context

```python
class TimeOfDay(Enum):
    MORNING = "morning"     # 5:00 - 12:00
    AFTERNOON = "afternoon" # 12:00 - 17:00
    EVENING = "evening"     # 17:00 - 21:00
    NIGHT = "night"         # 21:00 - 5:00

class DayType(Enum):
    WEEKDAY = "weekday"     # Monday - Friday
    WEEKEND = "weekend"     # Saturday - Sunday
```

### Location Context

AURA detects location type based on GPS and known places:

| Location Type | Detection Method |
|---------------|------------------|
| `HOME` | GPS matches saved home coordinates (< 100m) |
| `WORK` | GPS matches saved work coordinates (< 100m) |
| `COMMUTING` | Speed > 5 m/s (configurable threshold) |
| `UNKNOWN` | No match to known locations |

### Activity Detection

Activity is inferred from speed and device state:

| Speed | Activity | Confidence |
|-------|----------|------------|
| < 1 m/s | Still | 80% |
| 1-3 m/s | Walking | 70% |
| 3-8 m/s | Running | 70% |
| 8-15 m/s | Cycling | 60% |
| > 15 m/s | Driving | 80% |

Special detection:
- **Sleeping**: Still + Night hours + Screen off
- **In Call**: Active phone call detected
- **Exercising**: Walking/Running/Cycling activities

---

## 4. Pattern Learning from Behavior

### Interaction Pattern Tracking

AURA learns from your usage patterns:

```python
@dataclass
class UserInteractionPattern:
    pattern_type: str           # Action type
    trigger: str                # What triggered it
    frequency: int              # How often used
    last_observed: datetime     # Last time observed
    satisfaction_score: float   # User satisfaction (0-1)
```

### Capability Usage Learning

```python
# Tracked for each capability
{
    "usage_count": int,         # Times used
    "user_awareness": float,    # 0-1, how aware user is
    "first_used": datetime,     # First usage time
    "last_used": datetime,      # Most recent usage
    "discovery_potential": float # Recommendation priority
}
```

### Context History Analysis

```python
# Pattern detection from history
{
    "locations": ["home", "work", "gym"],
    "common_times": [9, 12, 18, 22],      # Hours of activity
    "activities": ["still", "driving", "walking"],
    "samples": 100                         # Data points collected
}
```

---

## 5. Privacy-Preserving Discovery

### Encrypted Storage

All discovery data is encrypted by default:

```python
self._storage = SecureStorage(storage_path, encrypt_by_default=True)
```

### Data Stored Locally

| Data Type | Storage Location | Encrypted |
|-----------|------------------|-----------|
| Capability usage | `capability_discovery.json` | Yes |
| Interaction patterns | `suggestion_engine.json` | Yes |
| Personality traits | `personality_tracker.json` | Yes |
| Context history | In-memory (100 samples) | N/A |

### Privacy Principles

1. **No Cloud Upload**: All data stays on device
2. **Local Learning**: Patterns learned locally only
3. **User Control**: Can delete/reset discovery data
4. **Minimal Collection**: Only collects what's needed
5. **Transparent**: User can query what's known

### Honest Limitations Disclosure

AURA proactively admits its limitations:

```python
LIMITATIONS = {
    "no_physical": "Cannot physically interact with world",
    "no_perms": "Limited by device permissions",
    "no_internet": "Need internet for advanced features",
    "no_memory": "Limited memory across sessions",
    "no_emotions": "No genuine emotions",
    "no_sensing": "Need sensors for physical world",
    "no_perfect": "Can make mistakes"
}
```

---

## 6. Termux-Specific Capabilities

### Available Termux Commands

| Command | Purpose | AURA Tool |
|---------|---------|-----------|
| `termux-location` | GPS location | `get_location` |
| `termux-camera-photo` | Take photos | `take_photo` |
| `termux-notification` | Send notifications | `send_notification` |
| `termux-vibrate` | Vibrate device | `vibrate` |
| `termux-battery-status` | Battery info | Device info |
| `termux-bluetooth-scan` | Nearby devices | Social context |
| `pm list packages` | List apps | `list_apps` |
| `am start` | Open apps | `open_app` |
| `dumpsys` | App info | `get_app_info` |

### Tool Execution Flow

```
User Intent
    |
    v
[App Discovery] --> Match intent to app
    |
    v
[Termux Bridge] --> Execute command (30s timeout)
    |
    v
[Result Handler] --> Parse and return
```

### Command Security

All commands are executed securely:

```python
# SECURED: Uses list args, NOT shell=True
cmd_list = shlex.split(command)
process = await asyncio.create_subprocess_exec(
    *cmd_list,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE,
)
```

---

## 7. Proactive Suggestions

### Time-Based Suggestions

```python
# Morning (6-10 AM)
suggestion = {
    "type": "productivity",
    "title": "Good morning",
    "suggestion": "I can summarize notifications. What matters to you?",
    "capability": "briefing",
    "asking_user": True  # Always asks permission
}

# Evening (6-10 PM)
suggestion = {
    "type": "planning",
    "title": "Plan tomorrow",
    "suggestion": "Prepare schedule? What are priorities?",
    "capability": "schedule",
    "asking_user": True
}
```

### Context-Aware Behavior Modes

| Mode | Trigger | AURA Behavior |
|------|---------|---------------|
| **Professional** | Work hours + At work | Formal, task-focused |
| **Personal** | Evening/Morning + At home | Casual, flexible |
| **Quiet** | Sleep hours (11 PM - 7 AM) | Minimal interruptions |
| **Driving** | Speed > threshold | Voice-only, safety first |

### Capability Gap Detection

When a capability is missing, AURA finds alternatives:

```python
fallback_strategies = {
    "IMAGE_ANALYSIS": [
        "Check gallery app for image metadata",
        "Use file system to find images by date/location",
        "Check cloud backups for image references"
    ],
    "VOICE_INPUT": [
        "Use text input as fallback",
        "Check for voice message files in chat apps"
    ],
    "TEXT_TO_SPEECH": [
        "Use notification with vibration pattern",
        "Display text on screen prominently"
    ]
}
```

---

## 8. Configuration

### Setting Known Locations

```python
detector = get_context_detector()

# Set home location
detector.set_home_location(lat=37.7749, lng=-122.4194)

# Set work location
detector.set_work_location(lat=37.7849, lng=-122.4094)

# Set work hours
detector.set_work_hours(start=9, end=18)
```

### Config File Structure

```json
// config/context_config.json
{
    "work_hours": {"start": 9, "end": 18},
    "sleep_hours": {"start": 23, "end": 7},
    "home_location": {"lat": 37.7749, "lng": -122.4194},
    "work_location": {"lat": 37.7849, "lng": -122.4094},
    "driving_speed_threshold": 5.0,
    "idle_time_threshold": 300
}
```

---

## 9. API Reference

### Get Current Context

```python
from src.context.detector import get_context_detector

detector = get_context_detector()
context = await detector.detect()

# Returns
{
    "timestamp": "2024-01-15T14:30:00",
    "time": {
        "time_of_day": "afternoon",
        "day_type": "weekday",
        "is_work_hours": True,
        "is_sleep_hours": False
    },
    "location": {
        "type": "work",
        "details": {"lat": 37.7849, "lng": -122.4094}
    },
    "activity": {
        "type": "active",
        "details": {"activity": "walking", "speed": 1.5}
    },
    "behavior_hints": {
        "professional_mode": True,
        "personal_mode": False,
        "quiet_mode": False,
        "driving_mode": False
    }
}
```

### Discover Untapped Capabilities

```python
from src.services.self_discovery_engine import get_self_discovery_engine

engine = await get_self_discovery_engine()
capabilities = await engine.discover_untapped_capabilities()

# Returns list of capabilities user hasn't explored
for cap in capabilities:
    print(f"{cap.name}: {cap.description}")
    print(f"  Discovery potential: {cap.discovery_potential}")
```

### Get Proactive Suggestions

```python
suggestions = await engine.proactive_suggestions(user_context)

for s in suggestions:
    print(f"{s['title']}: {s['suggestion']}")
```

---

## Architecture Diagram

```
+------------------+     +-------------------+
|  User Interaction|     |   Termux Bridge   |
+--------+---------+     +---------+---------+
         |                         |
         v                         v
+--------+---------+     +---------+---------+
| Self-Discovery   |<--->|   App Discovery   |
|     Engine       |     |                   |
+--------+---------+     +---------+---------+
         |                         |
         v                         v
+--------+---------+     +---------+---------+
| Personality      |     |  Context Provider |
|   Tracker        |     |   (Sensors/GPS)   |
+--------+---------+     +---------+---------+
         |                         |
         v                         v
+--------+---------+     +---------+---------+
| Suggestion       |<--->| Context Detector  |
|    Engine        |     |  (Behavior Modes) |
+------------------+     +-------------------+
         |
         v
+------------------+
| SecureStorage    |
|  (Encrypted)     |
+------------------+
```

---

## Summary

AURA's self-discovery system enables:

1. **Device Understanding**: Hardware, OS, battery, storage
2. **App Awareness**: Installed apps, categories, capabilities
3. **Context Intelligence**: Time, location, activity, behavior modes
4. **Pattern Learning**: Usage patterns, preferences, habits
5. **Privacy First**: Local-only, encrypted, transparent
6. **Proactive Help**: Time-based suggestions, capability gap bridging
7. **Honest Limits**: Always transparent about what it cannot do

The system continuously learns and adapts while keeping all data private and on-device.
