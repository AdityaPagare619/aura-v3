# AURA v3 - Your Personal AI Assistant

<p align="center">
  <strong>100% Offline | Privacy-First | Mobile-Optimized</strong>
</p>

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Features](#features)
4. [Architecture](#architecture)
5. [Configuration](#configuration)
6. [Troubleshooting](#troubleshooting)
7. [Security & Privacy](#security--privacy)
8. [Enterprise](#enterprise)
9. [Support](#support)

---

## Quick Start

### Prerequisites

- **Python 3.8+** (3.10+ recommended)
- **Android device** with Termux app installed
- **4GB RAM minimum** (8GB recommended)
- **2GB storage** for models and data

### Step 1: Install Termux

Download Termux from F-Droid (recommended) or Google Play Store:
- **F-Droid**: https://f-droid.org/packages/com.termux/
- **Google Play**: Search "Termux" in Play Store

> **Note**: If you have the old Play Store version, consider switching to F-Droid as it's actively maintained.

### Step 2: Install AURA (Automated)

```bash
# Open Termux and run:
curl -sSL https://raw.githubusercontent.com/aura-ai/aura-v3/main/installation/quickstart.sh | bash
```

### Step 3: Verify Installation

```bash
# Test AURA runs
aura --version

# Or run directly
python -m src.main --test
```

---

## Installation

### Option 1: One-Click Install (Recommended)

```bash
curl -sSL https://aura-ai.github.io/install.sh | bash
```

This script will:
1. Check prerequisites
2. Install required Python packages
3. Download required models
4. Configure AURA
5. Start the service

### Option 2: Manual Installation

If you prefer manual control:

```bash
# 1. Update packages
pkg update && pkg upgrade

# 2. Install Python and dependencies
pkg install python python-dev git curl

# 3. Clone repository
git clone https://github.com/aura-ai/aura-v3.git
cd aura-v3

# 4. Install Python packages
pip install -r requirements.txt

# 5. Configure
cp config.example.yaml config.yaml
# Edit config.yaml with your settings

# 6. Run
python -m src.main
```

### Option 3: Development Installation

```bash
# Clone and install in development mode
git clone https://github.com/aura-ai/aura-v3.git
cd aura-v3
pip install -e .

# Run tests
pytest tests/

# Run with debug logging
python -m src.main --log-level=DEBUG
```

### Voice Input Setup (Optional)

To enable voice commands:

```bash
# Install voice dependencies
pkg install ffmpeg sox

# Download language model
# (Automatic on first run, or manual download)
```

---

## Features

### Core Capabilities

| Feature | Description |
|---------|-------------|
| **Offline AI** | 100% local processing - no data leaves your device |
| **Adaptive Personality** | Learns your communication style over time |
| **Multi-Agent System** | Parallel processing for efficiency |
| **Proactive Assistance** | Anticipates your needs |
| **Life Tracking** | Remembers important details |
| **Emotional Context** | Responds appropriately to your mood |

### Technical Features

- **Event-Driven Architecture**: Only processes when needed (battery efficient)
- **Thermal Management**: Automatically throttles when device heats up
- **Memory Hierarchy**: Optimizes for mobile RAM constraints
- **Privacy-First**: No telemetry, no cloud, no tracking

---

## Architecture

```
AURA v3 Architecture
===================

┌─────────────────────────────────────────────────────────┐
│                    USER INTERFACE                        │
│  (CLI, Telegram Bot, Dashboard, Voice)                 │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                 CORE PROCESSING                          │
│  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ Neuromorphic   │  │  Multi-Agent Orchestrator    │  │
│  │ Engine         │  │  • Attention Agent           │  │
│  │ (Event-driven) │  │  • Memory Agent              │  │
│  └─────────────────┘  │  • Action Agent              │  │
│                      │  • Monitor Agent             │  │
│                      │  • Communication Agent       │  │
│                      └─────────────────────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  INTELLIGENCE LAYER                      │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────┐   │
│  │ LLM Manager  │ │ Neural       │ │ Context       │   │
│  │ (Local LLM)  │ │ Memory       │ │ Provider      │   │
│  └──────────────┘ └──────────────┘ └────────────────┘   │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  MOBILE OPTIMIZATION                     │
│  ┌──────────────┐ ┌──────────────┐ ┌────────────────┐   │
│  │ Power Manager│ │ Thermal      │ │ Background     │   │
│  │              │ │ Manager      │ │ Manager        │   │
│  └──────────────┘ └──────────────┘ └────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

### Key Components

#### NeuromorphicEngine
- Event-driven processing (SNN-inspired)
- Sparse activation - only processes what's needed
- Hardware-aware resource management

#### MultiAgentOrchestrator
- Parallel sub-agents for different tasks
- Adaptive tick rates based on thermal state
- F.R.I.D.A.Y.-style coordination

#### MobilePowerManager
- Battery-aware processing
- Screen state detection
- Doze mode handling

#### AdaptivePersonalityEngine
- Learns from interactions
- Adapts to user preferences
- Maintains consistent core identity

---

## Configuration

### Configuration File

Create `config.yaml` in the project root:

```yaml
# AURA v3 Configuration

# Core Settings
core:
  # LLM Configuration
  llm:
    provider: "llama.cpp"  # or "ollama", "transformers"
    model_path: "./models/llama-7b-chat.gguf"
    max_tokens: 1024
    temperature: 0.7
    
  # Memory Configuration
  memory:
    episodic_limit: 1000
    semantic_limit: 5000
    consolidation_interval: 3600

# Power Management
power:
  full_power_threshold: 50    # % battery for full power
  balanced_threshold: 20     # % battery for balanced
  power_save_threshold: 10  # % battery for power save
  critical_threshold: 5     # % battery for critical
  
  # Processing limits per mode
  max_tokens_full: 2048
  max_tokens_balanced: 1024
  max_tokens_save: 512
  max_tokens_ultra: 256

# Privacy Settings
privacy:
  log_level: "WARNING"  # DEBUG, INFO, WARNING, ERROR
  store_conversations: true
  anonymize_data: true
  allow_telemetry: false  # NEVER sends data out

# Features
features:
  voice_enabled: false
  proactive_mode: true
  background_tasks: true
  notifications: true

# Security
security:
  require_auth: false
  session_timeout: 3600
  max_failed_attempts: 5
```

### Environment Variables

You can also configure via environment variables:

```bash
export AURA_CONFIG_PATH=/path/to/config.yaml
export AURA_DATA_PATH=/path/to/data
export AURA_LOG_LEVEL=INFO
export AURA_VOICE_ENABLED=true
```

---

## Troubleshooting

### Common Issues

#### 1. Installation Fails

**Problem**: Installation script fails on dependencies

**Solution**:
```bash
# Update package lists
pkg update && pkg upgrade

# Install manually
pip install -r requirements.txt --no-cache-dir

# If numpy fails, try:
pip install --upgrade pip setuptools wheel
pip install numpy
pip install -r requirements.txt
```

#### 2. Out of Memory

**Problem**: AURA crashes with memory errors

**Solution**:
- Reduce model size (use 7B instead of 13B)
- Adjust `max_tokens` in config
- Enable power save mode
- Close other apps

```yaml
# In config.yaml
power:
  max_tokens_balanced: 512  # Reduce from 1024
```

#### 3. Slow Response

**Problem**: First response takes 15+ seconds

**Solution**:
- Preload the model at startup
- Use a smaller model
- Check thermal state (may be throttling)

```yaml
# In config.yaml
core:
  llm:
    preload: true
    n_gpu_layers: 99  # Maximize GPU usage on mobile
```

#### 4. Voice Not Working

**Problem**: Voice input not recognized

**Solution**:
```bash
# Install audio dependencies
pkg install ffmpeg sox

# Check microphone permissions in Termux
termux-setup-storage

# Test microphone
rec test.wav
# Speak, then Ctrl+C
# Play back: play test.wav
```

#### 5. Battery Drain

**Problem**: Device gets warm / battery drains fast

**Solution**:
AURA v3 has built-in power management. Ensure it's enabled:

```yaml
# In config.yaml
power:
  full_power_threshold: 50
  balanced_threshold: 30  # More conservative
```

Also:
- Use power save mode when mobile
- Disable proactive features when not needed
- Reduce background task frequency

#### 6. Thermal Throttling

**Problem**: AURA slows down after extended use

**Solution**:
This is intentional! AURA monitors thermal state and automatically throttles to protect your device. The system will return to full speed once the device cools down.

---

## Security & Privacy

### Privacy Commitment

AURA is designed with privacy as a core principle:

| Aspect | AURA Implementation |
|--------|---------------------|
| **Data Storage** | All data stored locally on device |
| **Network Access** | None required for core functionality |
| **Telemetry** | Zero telemetry - we can't see your data |
| **Updates** | Optional, verifiable updates only |
| **Authentication** | Optional local PIN/biometric |

### Security Features

- Local-only processing
- Optional authentication
- Session timeout
- Failed attempt lockout
- Encrypted local storage option

### Verification

You can verify AURA's privacy:

```bash
# Monitor network traffic while AURA runs
# On another device or with tcpdump
# AURA should make ZERO network requests

# Check logs for any network calls
grep -i "http\|https\|requests" logs/aura.log
```

---

## Enterprise

### Enterprise Features

AURA v3 offers enterprise-ready capabilities:

| Feature | Availability |
|---------|---------------|
| SOC2 Compliance | In Progress |
| GDPR Compliance | Full |
| HIPAA Compliance | Roadmap |
| SSO Integration | Planned |
| Central Management | Planned |

### Deployment Options

#### On-Device (Current)
- Individual Android devices
- Full local control
- Zero infrastructure

#### Enterprise (Roadmap)
- Centralized management
- Policy enforcement
- Audit logging
- Support packages

### Compliance Documentation

See the `compliance/` directory for:
- [SECURITY.md](compliance/SECURITY.md) - Security practices
- [PRIVACY.md](compliance/PRIVACY.md) - Privacy policy
- [COMPLIANCE.md](compliance/COMPLIANCE.md) - Compliance details

---

## Support

### Documentation

- [API Documentation](docs/API.md)
- [Troubleshooting Guide](docs/TROUBLESHOOTING.md)
- [Architecture Guide](docs/ARCHITECTURE.md)

### Community

- GitHub Issues: https://github.com/aura-ai/aura-v3/issues
- Discussions: https://github.com/aura-ai/aura-v3/discussions

### Getting Help

```bash
# Run diagnostics
python -m src.main --diagnostics

# Get verbose logs
python -m src.main --log-level=DEBUG

# Check status
python -m src.main --status
```

---

## License

AURA v3 - Copyright 2024 AURA AI

See LICENSE file for details.

---

## Contributing

We welcome contributions! See CONTRIBUTING.md for guidelines.

---

*Built with privacy-first principles. Your data never leaves your device.*
