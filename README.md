# AURA v3 - Personal Mobile AGI Assistant

<p align="center">
  <img src="https://img.shields.io/badge/Version-3.0-blue" alt="Version">
  <img src="https://img.shields.io/badge/Platform-Termux-green" alt="Platform">
  <img src="https://img.shields.io/badge/License-MIT-orange" alt="License">
  <img src="https://img.shields.io/badge/Status-100%25%20Offline-purple" alt="Status">
</p>

Next-generation personal AI assistant that runs 100% offline on your Android device. Built with ReAct loop, hierarchical memory, self-improvement, and privacy-first design.

---

## Table of Contents

1. [What is AURA v3?](#what-is-aura-v3)
2. [Features](#features)
3. [Requirements](#requirements)
4. [Quick Start (5 Minutes)](#quick-start-5-minutes)
5. [Detailed Setup Guide](#detailed-setup-guide)
   - [Option 1: Auto Setup (Recommended)](#option-1-auto-setup-recommended)
   - [Option 2: Manual Setup](#option-2-manual-setup)
6. [How to Use](#how-to-use)
   - [Running AURA](#running-aura)
   - [Telegram Commands](#telegram-commands)
   - [CLI Mode](#cli-mode)
7. [Advanced Features](#advanced-features)
   - [Security Levels](#security-levels)
   - [Voice TTS](#voice-tts)
   - [LLM Models](#llm-models)
8. [Troubleshooting](#troubleshooting)
9. [Architecture](#architecture)
10. [Updating AURA](#updating-aura)

---

## What is AURA v3?

AURA v3 is a **next-generation personal mobile AGI assistant** that:

- 🤖 **Truly Agentic**: Uses ReAct (Reasoning + Acting) loop with tool schemas
- 📱 **Mobile-First**: Optimized for Android with Termux (4GB RAM)
- 🔒 **100% Private**: No cloud, no data sharing, everything stays on device
- 🧠 **Self-Improving**: Learns from every interaction (RISE-style reflection)
- 🎯 **Context-Aware**: Knows time, location, and activity
- 🛡️ **Secure**: L1-L4 permission levels with banking protection

### Why It's Different from v1/v2:

| Feature | v1/v2 | v3 |
|---------|-------|-----|
| Tool Schemas | ❌ None | ✅ JSON schemas passed to LLM |
| ReAct Loop | ❌ One-shot | ✅ Iterative reasoning |
| Tool Results | ❌ Discarded | ✅ Fed back to LLM |
| Model Loading | ❌ Every message | ✅ Persistent (stays in RAM) |
| Learning | ❌ None | ✅ Intent/strategy patterns |

---

## Features

- **ReAct Agent Loop**: Think → Act → Observe → Think → ... → Respond
- **Hierarchical Memory**: 5 layers (Immediate → Working → Short-term → Long-term → Self-model)
- **Learning Engine**: Extracts patterns from conversations
- **Context Detector**: Time, location, activity awareness
- **Security Layer**: L1-L4 permissions, banking protection
- **Android Tools**: SMS, calls, contacts, app launching with exploration memory
- **Telegram Control**: Full control via Telegram bot commands
- **Offline Voice**: eSpeak/pyttsx3 TTS (no internet needed)

---

## Requirements

### For Mobile (Android + Termux):

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| RAM | 4GB | 6GB+ |
| Storage | 2GB free | 5GB+ |
| Python | 3.9+ | 3.11+ |
| Termux | Latest from F-Droid | - |

### For Desktop Testing:

- Python 3.9+
- 4GB RAM
- Linux/macOS/Windows (WSL)

---

## Quick Start (5 Minutes)

### Step 1: Get Termux (Android)

```
1. Download Termux from F-Droid (NOT Google Play - outdated!)
2. Open Termux
```

### Step 2: Run Auto Setup

```bash
# Clone and run auto setup
git clone https://github.com/AdityaPagare619/aura-v3.git
cd aura-v3
python scripts/auto_setup.py
```

The script will:
- ✅ Install all dependencies
- ✅ Create required directories
- ✅ Configure Telegram
- ✅ Test imports
- ✅ Create run/stop scripts

### Step 3: Start AURA

```bash
# Start the bot
bash run_aura.sh

# OR manually
python main.py --mode telegram
```

### Step 4: Connect Telegram

```
1. Open Telegram
2. Search for your bot (check .env for token)
3. Send /start
```

**That's it!** 🎉

---

## Detailed Setup Guide

### Option 1: Auto Setup (Recommended)

```bash
# Clone repository
git clone https://github.com/AdityaPagare619/aura-v3.git
cd aura-v3

# Run auto setup
python scripts/auto_setup.py
```

This script handles everything automatically.

### Option 2: Manual Setup

If you prefer manual installation:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Create directories
mkdir -p logs data/memories data/sessions data/patterns models

# 3. Set Telegram token
export TELEGRAM_TOKEN="your_token_here"

# 4. Run
python main.py --mode telegram
```

---

## How to Use

### Running AURA

```bash
# Start in Telegram mode (recommended)
python main.py --mode telegram

# Or use convenience scripts
bash run_aura.sh    # Start in background
bash stop_aura.sh   # Stop
bash status_aura.sh # Check status
```

### Telegram Commands

| Command | Description |
|---------|-------------|
| `/start` | Start AURA |
| `/stop` | Stop AURA |
| `/restart` | Restart AURA |
| `/status` | Check system status |
| `/help` | Show all commands |
| `/test` | Run functionality tests |
| `/setup` | Interactive setup wizard |
| `/config` | Show current configuration |
| `/setlevel L1-L4` | Set security level |
| `/setvoice <engine>` | Set TTS engine |
| `/models` | List available LLM models |
| `/voices` | List TTS engines |
| `/memory` | Show memory statistics |
| `/clear` | Clear conversation |

### Just Chat!

Once started, you can just send messages to the bot and AURA will respond!

---

## Advanced Features

### Security Levels

AURA has 4 permission levels:

| Level | Name | Description |
|-------|------|-------------|
| L1 | Full Trust | No confirmations needed |
| L2 | Normal | Confirm calls/messages |
| L3 | Restricted | Confirm most actions |
| L4 | Maximum | Confirm everything |

Set via: `/setlevel L2`

### Voice TTS

Available engines:

| Engine | Type | Quality | Internet Needed |
|--------|------|---------|-----------------|
| espeak | Offline | Basic | No |
| pyttsx3 | Offline | Good | No |
| sarvam | Online | Best | Yes |

Set via: `/setvoice espeak`

### LLM Models

Recommended for mobile (4GB RAM):

| Model | Size | Quality |
|-------|------|---------|
| Qwen2.5-1B Q5_K_M | ~625MB | Good |
| Phi-3-mini | ~2.5GB | Better |
| Mistral-7B | ~4GB | Best |

Download from: https://huggingface.co/TheBloke

---

## Troubleshooting

### Common Issues

#### 1. "llama-cpp-python not found"

```bash
# Install with pre-built wheel
pip install llama-cpp-python --only-binary :all:
```

#### 2. "espeak not found"

```bash
# In Termux
apt install espeak-ng
```

#### 3. "Telegram token invalid"

Get new token from @BotFather, then:
```bash
# Update .env file
nano .env
# OR
export TELEGRAM_TOKEN="new_token"
```

#### 4. "Port already in use"

```bash
# Kill existing process
bash stop_aura.sh

# OR manually
pkill -f "python.*main.py"
```

#### 5. "Import errors"

```bash
# Reinstall dependencies
pip install -r requirements.txt
```

### Check Logs

```bash
# View live logs
tail -f aura.log

# Or check last 50 lines
tail -n 50 aura.log
```

### Test Components

```bash
# Test individual modules
python -c "from src.agent.loop import ReActAgent; print('Agent OK')"
python -c "from src.llm import LLMRunner; print('LLM OK')"
python -c "from src.memory import HierarchicalMemory; print('Memory OK')"
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        AURA v3                              │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────┐    ┌────────────┐    ┌──────────────────┐   │
│  │ Telegram │───▶│   Agent    │───▶│   Tool Executor  │   │
│  │   Bot    │    │  (ReAct)   │    │                  │   │
│  └──────────┘    └────────────┘    └──────────────────┘   │
│        │                 │                     │             │
│        ▼                 ▼                     ▼             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              Hierarchical Memory                    │    │
│  │  Immediate → Working → Short → Long → Self-Model   │    │
│  └─────────────────────────────────────────────────────┘    │
│        │                 │                     │             │
│        ▼                 ▼                     ▼             │
│  ┌──────────┐    ┌────────────┐    ┌──────────────────┐     │
│  │ Learning │    │  Context   │    │     Security    │     │
│  │ Engine   │    │  Detector  │    │      Layer      │     │
│  └──────────┘    └────────────┘    └──────────────────┘     │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐     │
│  │              LLM (llama.cpp / Mock)                 │     │
│  └─────────────────────────────────────────────────────┘     │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Key Components:

1. **Agent Loop**: ReAct pattern with JSON tool schemas
2. **Memory**: 5-layer hierarchical (immediate → working → short → long → self)
3. **Learning**: Intent patterns, contact priorities, strategy improvement
4. **Context**: Time, location, activity detection
5. **Security**: L1-L4 permissions, banking protection, audit logs
6. **Tools**: Android actions with exploration memory

---

## Updating AURA

```bash
# Navigate to directory
cd ~/aura-v3

# Pull latest
git pull origin main

# Restart
bash stop_aura.sh
bash run_aura.sh
```

---

## Support

- Issues: https://github.com/AdityaPagare619/aura-v3/issues
- Telegram: Message your bot and use /help

---

## License

MIT License - See LICENSE file

---

<p align="center">
  Made with ❤️ for privacy-first AI
</p>
