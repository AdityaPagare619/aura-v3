# AURA v3 - Complete Termux Setup Guide

A beginner-friendly guide to running AURA on your Android device using Termux.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installing Termux](#installing-termux)
3. [Initial Termux Setup](#initial-termux-setup)
4. [Installing Required Packages](#installing-required-packages)
5. [Installing AURA](#installing-aura)
6. [Storage Permissions](#storage-permissions)
7. [Running AURA](#running-aura)
8. [Running in Background](#running-in-background)
9. [RAM Optimization for 8GB Devices](#ram-optimization-for-8gb-devices)
10. [Troubleshooting](#troubleshooting)
11. [Tips & Best Practices](#tips--best-practices)

---

## Prerequisites

### Device Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **Android Version** | 7.0 (Nougat) | 10.0+ |
| **RAM** | 4GB | 8GB+ |
| **Free Storage** | 3GB | 5GB+ |
| **CPU** | 64-bit ARM | Snapdragon 6xx+ |

### What You'll Need

- Android device meeting the above requirements
- Stable WiFi connection (for initial setup only)
- ~30 minutes for complete setup
- Basic familiarity with typing commands (copy-paste works!)

### Storage Breakdown

| Component | Size |
|-----------|------|
| Termux + packages | ~500MB |
| Python + dependencies | ~300MB |
| AURA code | ~50MB |
| LLM model (Qwen2.5-1B) | ~625MB |
| Data & logs | ~100MB |
| **Total** | **~1.6GB** |

---

## Installing Termux

### Important: Get Termux from F-Droid

> **WARNING**: Do NOT install Termux from Google Play Store - it's outdated and broken!

### Step 1: Install F-Droid

1. Open your browser on Android
2. Go to: `https://f-droid.org`
3. Download and install the F-Droid app
4. You may need to allow "Install from unknown sources" in Settings

### Step 2: Install Termux from F-Droid

1. Open F-Droid
2. Search for "Termux"
3. Install **Termux** (the main app)
4. Also install **Termux:API** (needed for Android features)

### Step 3: Grant Permissions

When you first open Termux:
- Allow notifications (optional but recommended)
- Battery optimization: Set to "Don't optimize" for best background performance

---

## Initial Termux Setup

### First Launch

Open Termux. You'll see a terminal. Run these commands one by one:

```bash
# Update package lists
pkg update

# Upgrade all packages (press Y when asked)
pkg upgrade -y
```

### Configure Storage

```bash
# Grant storage access (IMPORTANT!)
termux-setup-storage
```

A popup will appear - tap **Allow**. This creates folders linking to your device storage.

### Set Up Keyboard Shortcuts

For easier typing, add extra keys:

```bash
# Create Termux config
mkdir -p ~/.termux

# Add extra keyboard row
echo 'extra-keys = [["ESC","TAB","CTRL","ALT","-","UP","DOWN"]]' >> ~/.termux/termux.properties

# Apply changes
termux-reload-settings
```

---

## Installing Required Packages

### Core Packages

```bash
# Install essential packages
pkg install -y python git wget curl

# Verify Python installation
python --version  # Should show 3.11+
```

### Build Tools (for compiling dependencies)

```bash
# Install build essentials
pkg install -y build-essential cmake ninja

# Install development libraries
pkg install -y libffi openssl
```

### Voice Support (Optional)

```bash
# Install eSpeak for offline TTS
pkg install -y espeak

# Test it
espeak "Hello from AURA"
```

### Termux API Tools (Optional but recommended)

```bash
# Install API tools for Android features
pkg install -y termux-api

# Test (should show your battery status)
termux-battery-status
```

---

## Installing AURA

### Step 1: Clone the Repository

```bash
# Go to home directory
cd ~

# Clone AURA
git clone https://github.com/AdityaPagare619/aura-v3.git

# Enter the directory
cd aura-v3
```

### Step 2: Install Python Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install AURA dependencies
pip install -r requirements.txt
```

**Note**: This may take 5-10 minutes on mobile.

### Step 3: Install LLM Support (Optional)

For local AI inference:

```bash
# Install llama-cpp-python (pre-built wheel)
pip install llama-cpp-python --only-binary :all:
```

If that fails, try:

```bash
# Build from source (takes longer)
CMAKE_ARGS="-DLLAMA_BLAS=ON" pip install llama-cpp-python
```

### Step 4: Create Required Directories

```bash
# Create data directories
mkdir -p logs data/memories data/sessions data/patterns models
```

### Step 5: Configure AURA

```bash
# Copy example config
cp config.example.yaml config.yaml

# Set up environment
cp .env.example .env

# Edit .env to add your Telegram token
nano .env
```

In the editor:
1. Add your Telegram bot token (get from @BotFather)
2. Press `Ctrl+O` to save, `Enter` to confirm, `Ctrl+X` to exit

### Step 6: Run Auto Setup (Alternative)

Instead of steps 2-5, you can use the auto setup script:

```bash
python scripts/auto_setup.py
```

This handles everything automatically!

---

## Storage Permissions

### Understanding Termux Storage

After running `termux-setup-storage`, you get these folders:

```
~/storage/
├── dcim/        -> Camera photos
├── downloads/   -> Downloads folder
├── movies/      -> Videos
├── music/       -> Music files
├── pictures/    -> Pictures
├── shared/      -> Internal storage root
└── external-1/  -> SD card (if present)
```

### Moving AURA Data to External Storage

If you have an SD card and need more space:

```bash
# Create AURA folder on SD card
mkdir -p ~/storage/external-1/aura-data

# Link data directory
rm -rf ~/aura-v3/data
ln -s ~/storage/external-1/aura-data ~/aura-v3/data

# Link models directory (for LLM files)
rm -rf ~/aura-v3/models
ln -s ~/storage/external-1/aura-models ~/aura-v3/models
```

### Backup Your Data

```bash
# Backup to Downloads folder
cp -r ~/aura-v3/data ~/storage/downloads/aura-backup
```

---

## Running AURA

### Basic Start

```bash
# Navigate to AURA directory
cd ~/aura-v3

# Start in Telegram mode
python main.py --mode telegram
```

### Using Convenience Scripts

```bash
# Start AURA
bash run_aura.sh

# Check status
bash status_aura.sh

# Stop AURA
bash stop_aura.sh
```

### CLI Mode (No Telegram)

```bash
# Interactive CLI mode
python main.py --mode cli
```

---

## Running in Background

### Method 1: Using nohup

```bash
# Start AURA and detach
nohup python main.py --mode telegram > aura.log 2>&1 &

# Check if running
ps aux | grep python

# View logs
tail -f aura.log
```

### Method 2: Using tmux (Recommended)

```bash
# Install tmux
pkg install -y tmux

# Start a new session
tmux new -s aura

# Inside tmux, start AURA
cd ~/aura-v3 && python main.py --mode telegram

# Detach from tmux: Press Ctrl+B, then D

# Reattach later
tmux attach -t aura

# List sessions
tmux ls
```

### Method 3: Termux Boot (Auto-start)

Install Termux:Boot from F-Droid, then:

```bash
# Create boot script directory
mkdir -p ~/.termux/boot

# Create startup script
cat > ~/.termux/boot/start-aura.sh << 'EOF'
#!/data/data/com.termux/files/usr/bin/bash
cd ~/aura-v3
python main.py --mode telegram >> ~/aura-v3/aura.log 2>&1 &
EOF

# Make it executable
chmod +x ~/.termux/boot/start-aura.sh
```

Now AURA will start automatically when your device boots!

### Preventing Android from Killing Termux

1. **Disable battery optimization**:
   - Settings > Apps > Termux > Battery > Don't optimize

2. **Lock Termux in recent apps**:
   - Open Termux
   - Go to Recent Apps
   - Tap the lock icon on Termux card

3. **Acquire wake lock** (in Termux):
   ```bash
   termux-wake-lock
   ```

---

## RAM Optimization for 8GB Devices

### Understanding Memory Usage

With 8GB RAM, typical allocation:
- Android OS: 2-3GB
- Background apps: 1-2GB
- Available for AURA: 3-4GB

### Optimal Configuration

Create an optimized config for 8GB devices:

```bash
# Edit config.yaml
nano ~/aura-v3/config.yaml
```

Add these settings:

```yaml
# Optimized for 8GB RAM
core:
  llm:
    provider: "llama.cpp"
    n_ctx: 2048        # Context window
    n_batch: 256       # Batch size
    n_threads: 4       # Use 4 threads (half of typical 8-core)
    n_gpu_layers: 0    # No GPU (saves RAM)
    preload: true      # Keep model in RAM
    
memory:
  immediate_limit: 5
  working_limit: 10
  short_term_limit: 50
  long_term_limit: 200
  
power:
  balanced_threshold: 50
  power_save_threshold: 20
  tick_rate_balanced: 3.0
```

### Model Selection for 8GB

| Model | RAM Usage | Quality | Recommended |
|-------|-----------|---------|-------------|
| Qwen2.5-0.5B Q4_K_M | ~350MB | Basic | Low RAM |
| Qwen2.5-1B Q4_K_M | ~625MB | Good | **8GB devices** |
| Qwen2.5-3B Q4_K_M | ~1.8GB | Better | High RAM |
| Phi-3-mini Q4_K_M | ~2.2GB | Best | 12GB+ only |

### Before Running AURA

Free up RAM by:

```bash
# 1. Close unnecessary apps on Android

# 2. Clear Termux cache
pkg clean

# 3. Check available memory
free -h
```

### Runtime Optimization

```bash
# Set Python to optimize memory
export MALLOC_TRIM_THRESHOLD_=100000
export PYTHONMALLOC=malloc

# Run AURA
python main.py --mode telegram
```

### Monitor Memory Usage

```bash
# Check memory in real-time
watch -n 2 free -h

# Or inside Python
python -c "import psutil; print(f'RAM: {psutil.virtual_memory().percent}%')"
```

---

## Troubleshooting

### Installation Issues

#### "pkg: command not found"
```bash
# Reinstall Termux or run:
apt update && apt upgrade
```

#### "error: externally-managed-environment"
```bash
# Use --break-system-packages flag
pip install --break-system-packages -r requirements.txt
```

#### "Building wheel failed" for llama-cpp-python
```bash
# Try pre-built wheel
pip install llama-cpp-python --only-binary :all: --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
```

### Runtime Issues

#### "Killed" or Process Stops Unexpectedly
This usually means Android killed the process. Solutions:

1. Acquire wake lock:
   ```bash
   termux-wake-lock
   ```

2. Reduce memory usage (use smaller model)

3. Disable battery optimization for Termux

#### "No module named 'src'"
```bash
# Run from correct directory
cd ~/aura-v3
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python main.py --mode telegram
```

#### Telegram Bot Not Responding
```bash
# Check if token is set
cat .env | grep TELEGRAM

# Test token
python -c "import os; print(os.getenv('TELEGRAM_TOKEN', 'NOT SET'))"

# Restart with verbose logging
python main.py --mode telegram --log-level DEBUG
```

#### "Storage permission denied"
```bash
# Re-grant storage permissions
termux-setup-storage

# Check if storage folder exists
ls -la ~/storage/
```

### Performance Issues

#### Slow Responses (>30 seconds)
1. Use smaller model (0.5B instead of 1B)
2. Reduce context size in config
3. Check thermal throttling:
   ```bash
   termux-battery-status  # Check temperature
   ```

#### Device Gets Hot
1. Reduce `n_threads` in config (try 2 instead of 4)
2. Enable power save mode
3. Let device cool between heavy usage

### Network Issues

#### Can't Clone Repository
```bash
# Try with different protocol
git clone git://github.com/AdityaPagare619/aura-v3.git

# Or download ZIP and extract
wget https://github.com/AdityaPagare619/aura-v3/archive/main.zip
unzip main.zip
mv aura-v3-main aura-v3
```

---

## Tips & Best Practices

### Daily Usage

1. **Start your day**: `tmux attach -t aura` to reconnect
2. **Check status**: Send `/status` to your bot
3. **End your day**: Leave AURA running in tmux

### Maintenance

```bash
# Weekly: Update packages
pkg update && pkg upgrade -y

# Monthly: Update AURA
cd ~/aura-v3 && git pull

# Clear old logs
find ~/aura-v3/logs -mtime +7 -delete
```

### Battery Tips

1. Use WiFi instead of mobile data
2. Reduce screen brightness when using Termux
3. Enable power save mode during idle:
   ```bash
   # In config.yaml
   power:
     power_save_threshold: 30
   ```

### Keyboard Shortcuts in Termux

| Shortcut | Action |
|----------|--------|
| `Ctrl+C` | Cancel current command |
| `Ctrl+Z` | Suspend process |
| `Ctrl+D` | Exit/logout |
| `Ctrl+L` | Clear screen |
| `Tab` | Auto-complete |
| `Vol Down + C` | Ctrl+C (alternative) |
| `Vol Down + Z` | Ctrl+Z (alternative) |

### Useful Aliases

Add to `~/.bashrc`:

```bash
# Quick AURA commands
alias aura='cd ~/aura-v3 && python main.py --mode telegram'
alias aura-logs='tail -f ~/aura-v3/aura.log'
alias aura-status='cd ~/aura-v3 && bash status_aura.sh'

# System info
alias mem='free -h'
alias disk='df -h'
alias temp='termux-battery-status'
```

Apply with: `source ~/.bashrc`

---

## Quick Reference

### Essential Commands

```bash
# Start AURA
cd ~/aura-v3 && python main.py --mode telegram

# Background with tmux
tmux new -s aura
# (start AURA, then Ctrl+B, D to detach)

# Reconnect
tmux attach -t aura

# Check logs
tail -f ~/aura-v3/aura.log

# Update AURA
cd ~/aura-v3 && git pull && pip install -r requirements.txt
```

### File Locations

| Item | Location |
|------|----------|
| AURA code | `~/aura-v3/` |
| Config | `~/aura-v3/config.yaml` |
| Environment | `~/aura-v3/.env` |
| Logs | `~/aura-v3/logs/` |
| Data | `~/aura-v3/data/` |
| Models | `~/aura-v3/models/` |

---

## Getting Help

- **In AURA**: Send `/help` to your Telegram bot
- **Logs**: Check `~/aura-v3/aura.log`
- **Issues**: https://github.com/AdityaPagare619/aura-v3/issues
- **Termux Wiki**: https://wiki.termux.com

---

<p align="center">
Happy hacking with AURA on Termux!
</p>
