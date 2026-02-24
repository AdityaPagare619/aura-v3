# AURA v3 Privacy Tier System Design Document

## Overview
A granular, user-controlled privacy system that gives users precise control over AURA's proactivity based on data sensitivity.

## Core Concepts

### Privacy Tiers

| Tier | Description | AURA Behavior |
|------|-------------|---------------|
| **PRIVATE** | Gallery, messages, personal docs | Asks MORE before acting - requires explicit permission |
| **SENSITIVE** | Financial apps, health data | Asks BEFORE acting - confirm for writes, warn for reads |
| **PUBLIC** | News apps, general files | Acts proactively with suggestions |

## Architecture

### Components

1. **PrivacyTier Enum** - Defines the three tiers
2. **UserPermissionManager** - Core class managing permissions
3. **CategoryRegistry** - Maps data categories to default tiers
4. **PermissionChecker** - Evaluates if action should proceed
5. **ConfigLoader** - Loads/saves privacy.yaml

### Default Tier Assignments

```
CATEGORIES:
├── media (PRIVATE)
│   ├── gallery_photos
│   ├── gallery_videos
│   └── screenshots
│
├── messages (PRIVATE)
│   ├── sms
│   ├── whatsapp
│   ├── telegram_chats
│   └── email
│
├── personal_docs (PRIVATE)
│   ├── notes
│   ├── calendar
│   └── contacts
│
├── financial (SENSITIVE)
│   ├── banking_apps
│   ├── payment_apps
│   └── crypto
│
├── health (SENSITIVE)
│   ├── fitness_data
│   ├── medical_apps
│   └── sleep_data
│
├── work (SENSITIVE)
│   ├── slack
│   ├── teams
│   └── documents
│
├── public (PUBLIC)
│   ├── news_apps
│   ├── weather
│   ├── social_media
│   └── general_files
```

## Adaptive Proactivity Rules

### PRIVATE Tier
- **Always confirm** before reading any content
- **Always confirm** before analyzing content
- **Always confirm** before summarizing
- **Never proactively** suggest based on private data
- Log all access attempts

### SENSITIVE Tier
- **Confirm** before writes/modifications
- **Warn** before reading sensitive content
- **Caution** when making proactive suggestions
- May store audit logs

### PUBLIC Tier
- **Proactive suggestions** allowed
- **No confirmation** required for reads
- **Lightweight logging** only

## User Override Methods

### 1. Telegram Commands
- `/privacy` - Show current privacy settings
- `/privacy <tier>` - Set default tier
- `/permissions` - List all category permissions
- `/allow <category>` - Allow without confirmation
- `/restrict <category>` - Require confirmation

### 2. Config File (config/privacy.yaml)
```yaml
privacy:
  default_tier: SENSITIVE
  
  categories:
    gallery_photos: PRIVATE
    whatsapp: PRIVATE
    banking_apps: SENSITIVE
    news_apps: PUBLIC
    
  overrides:
    # User-specific overrides
    "trusted_device": true
    
  # Confirmation behavior
  confirm:
    private_always: true
    sensitive_writes: true
    public_proactive: false
```

### 3. Natural Language
- "You can access my photos"
- "Don't read my messages without asking"
- "You can be more helpful with my calendar"
- "Keep my financial apps private"

## Integration Points

### Tool Registry Integration
- Each tool declares its category and default tier
- Tool execution checks permission before running
- Returns permission request if needed

### Proactivity Controller
- Checks tier before making suggestions
- PRIVATE: Never proactively access
- SENSITIVE: Prompt before accessing
- PUBLIC: Full proactivity allowed

### Session Manager
- Tracks current context tier
- Stores user override preferences
- Persists across sessions

## Permission Flow

```
User Action → Tool/Feature → Check Category Tier
    │
    ├─→ PRIVATE ──→ Always prompt for permission
    │
    ├─→ SENSITIVE ─┬─→ Write: Confirm first
    │              └─→ Read: Warn but proceed
    │
    └─→ PUBLIC ──→ Execute with suggestion
```

## Data Structures

### PermissionGrant
- category: str
- tier: PrivacyTier
- granted_at: datetime
- expires_at: Optional[datetime]
- scope: "once" | "session" | "always"

### PermissionRequest
- tool_name: str
- category: str
- action: "read" | "write" | "analyze"
- tier: PrivacyTier
- user_response: Optional[bool]

## Security Considerations

1. **Audit Logging** - Track all permission requests and grants
2. **Timeout** - Session-only permissions expire on restart
3. **Encryption** - Sensitive permissions stored encrypted
4. **No Bypass** - System-level always requires confirmation

## Implementation Files

1. `src/core/privacy_tiers.py` - Core implementation
2. `config/privacy.yaml` - Default configuration
3. `src/channels/telegram_bot.py` - Add /privacy commands
4. `src/tools/registry.py` - Add tier info to tools
5. `src/core/tool_orchestrator.py` - Check permissions before execution
