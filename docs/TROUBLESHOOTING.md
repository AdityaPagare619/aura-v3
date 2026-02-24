# AURA v3 Troubleshooting Guide

This guide covers common issues and their solutions.

---

## Table of Contents

1. [Installation Issues](#installation-issues)
2. [Runtime Errors](#runtime-errors)
3. [Performance Problems](#performance-problems)
4. [Power & Battery](#power--battery)
5. [Voice Issues](#voice-issues)
6. [Network Issues](#network-issues)
7. [Data & Storage](#data--storage)
8. [Debugging](#debugging)

---

## Installation Issues

### "python: command not found"

**Problem**: Python not installed or not in PATH

**Solution**:
```bash
# Termux
pkg install python

# Linux
sudo apt-get install python3 python3-pip

# macOS
brew install python3
```

### "pip: command not found"

**Problem**: pip not installed

**Solution**:
```bash
# Install pip
python3 -m ensurepip --upgrade

# Or
curl -sSL https://bootstrap.pypa.io/get-pip.py | python3
```

### "No module named 'yaml'"

**Problem**: PyYAML not installed

**Solution**:
```bash
pip install pyyaml
```

### "No module named 'numpy'"

**Problem**: NumPy not installed

**Solution**:
```bash
# Install numpy (may take a while)
pip install numpy scipy

# On Termux, you may need:
pkg install python numpy
```

### "cannot import 'src'"

**Problem**: Wrong directory or Python path

**Solution**:
```bash
cd /path/to/aura-v3
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
python -m src.main
```

### "Permission denied" during install

**Problem**: Permission issues

**Solution**:
```bash
# Check file permissions
ls -la

# Fix permissions
chmod +x installation/*.sh

# Install with user flag
pip install --user -r requirements.txt
```

---

## Runtime Errors

### "AURA already initialized"

**Problem**: Trying to initialize twice

**Solution**: This is a warning, not an error. AURA will continue running.

### "ModuleNotFoundError: No module named 'src.XXX'"

**Problem**: Missing dependency module

**Solution**:
1. Check if module exists in source
2. Reinstall dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### "Config file not found"

**Problem**: Missing config.yaml

**Solution**:
```bash
# Create from example
cp config.example.yaml config.yaml

# Or create minimal config
cat > config.yaml << 'EOF'
core:
  llm:
    provider: "llama.cpp"
EOF
```

### "SSL certificate verification failed"

**Problem**: SSL certificate issues (rare)

**Solution**:
```bash
# Update certificates
# Termux
pkg install ca-certificates

# Linux
sudo apt-get update && sudo apt-get install ca-certificates
```

---

## Performance Problems

### Slow First Response (Cold Start)

**Problem**: First response takes 15+ seconds

**Causes**:
- Model not loaded
- Model too large
- Device thermal throttling

**Solutions**:

1. Enable model preload:
   ```yaml
   # config.yaml
   core:
     llm:
       preload: true
   ```

2. Use smaller model (7B instead of 13B)

3. Reduce context size:
   ```yaml
   core:
     llm:
       n_ctx: 1024  # Reduce from 2048
   ```

4. Check thermal state:
   ```bash
   # In Termux
   dumpsys battery
   ```

### High Memory Usage

**Problem**: Out of memory errors

**Solutions**:

1. Reduce model size
2. Reduce batch size
3. Enable swap (Termux):
   ```bash
   swapenable 512M
   ```

4. Lower processing limits:
   ```yaml
   power:
     max_tokens_balanced: 512
   ```

### Laggy Response

**Problem**: General slowness

**Solutions**:

1. Check CPU usage:
   ```bash
   top
   ```

2. Enable GPU acceleration:
   ```yaml
   core:
     llm:
       n_gpu_layers: 99
   ```

3. Check for background processes:
   ```bash
   ps aux
   ```

---

## Power & Battery

### Battery Drains Quickly

**Problem**: AURA uses too much power

**Solutions**:

1. Enable power save mode in config:
   ```yaml
   power:
     power_save_threshold: 30
     balanced_threshold: 50
   ```

2. Disable proactive features:
   ```yaml
   features:
     proactive_mode: false
     background_tasks: false
   ```

3. Reduce tick rates:
   ```yaml
   power:
     tick_rate_balanced: 5.0
     tick_rate_save: 10.0
   ```

4. Use smaller model

### Device Gets Warm

**Problem**: Thermal throttling

**Solutions**:

1. AURA automatically throttles - this is expected
2. Reduce processing load:
   ```yaml
   core:
     llm:
       max_tokens: 512
   ```
3. Use power save mode more aggressively
4. Let device cool between uses

### "Thermal throttle" in logs

**Problem**: Device too hot

**Explanation**: This is AURA's built-in protection. It automatically reduces processing when the device gets too warm.

**Action**: Let the device cool down. AURA will return to full speed automatically.

---

## Voice Issues

### Voice Not Recognized

**Problem**: Voice commands don't work

**Solutions**:

1. Install audio dependencies:
   ```bash
   # Termux
   pkg install ffmpeg sox portaudio
   
   pip install pyaudio sounddevice
   ```

2. Grant microphone permission:
   ```bash
   termux-setup-storage
   # Then go to Android Settings > Apps > Termux > Permissions
   ```

3. Check microphone:
   ```bash
   # Test recording
   rec test.wav
   # Speak, then Ctrl+C
   play test.wav
   ```

### "No audio device found"

**Problem**: Audio device issues

**Solutions**:

1. Check if audio device exists:
   ```bash
   # Linux
   aplay -l
   arecord -l
   ```

2. Install audio drivers:
   ```bash
   # Linux
   sudo apt-get install alsa-utils
   ```

### Voice Response Broken/Static

**Problem**: Audio output issues

**Solutions**:

1. Check speakers:
   ```bash
   speaker-test
   ```

2. Update audio libraries:
   ```bash
   pip install --upgrade sounddevice pyaudio
   ```

---

## Network Issues

### "Connection refused" errors

**Problem**: Network connectivity

**Explanation**: AURA should work 100% offline. These errors indicate unexpected network calls.

**Solutions**:

1. Check network settings:
   ```yaml
   # config.yaml
   network:
     allow_outgoing: false
   ```

2. Block network for AURA:
   ```bash
   # iptables (root required)
   iptables -A OUTPUT -d $(hostname -I) -j ACCEPT
   iptables -A OUTPUT -j DROP
   ```

### Model Download Fails

**Problem**: Can't download LLM model

**Solutions**:

1. Check internet connection
2. Try alternate download method
3. Download manually:
   ```bash
   # Find model URL and download
   wget <model_url>
   mv model.gguf ./models/
   ```

---

## Data & Storage

### "No space left on device"

**Problem**: Storage full

**Solutions**:

1. Check storage:
   ```bash
   df -h
   ```

2. Clear cache:
   ```bash
   rm -rf cache/
   rm -rf logs/*.log
   ```

3. Delete old models (if not needed)

4. Use SD card (Android):
   ```bash
   # Link data to SD card
   ln -s /sdcard/aura-data ./data
   ```

### Corrupted Data

**Problem**: Data files corrupted

**Solution**:
```bash
# Backup and reset
cp -r data data_backup
rm -rf data/*
# Restart AURA - will recreate
```

### Memory Full

**Problem**: Too much conversation history

**Solution**:
```yaml
# config.yaml - limit memory
core:
  memory:
    episodic_limit: 100
    semantic_limit: 500
```

---

## Debugging

### Enable Debug Logging

```bash
# Via command line
python -m src.main --log-level=DEBUG

# Or in config.yaml
privacy:
  log_level: "DEBUG"
```

### Check System Status

```bash
python -m src.main --status
```

### Run Diagnostics

```bash
python -m src.main --diagnostics
```

### View Logs

```bash
# Tail logs in real-time
tail -f logs/aura.log

# Search for errors
grep ERROR logs/aura.log

# Search for specific patterns
grep "battery\|power\|thermal" logs/aura.log
```

### Profile Performance

```yaml
# config.yaml
debug:
  profile_performance: true
  verbose_logging: true
```

### Test Individual Components

```python
# Test power manager
from src.core import get_power_manager
pm = get_power_manager()
print(pm.get_power_mode())

# Test LLM
from src.llm import get_llm_manager
llm = get_llm_manager()
print(llm.get_status())
```

---

## Getting Help

### Run System Info

```bash
# Collect system information
python -m src.main --sysinfo

# Output:
# Python version: 3.x.x
# Platform: Linux/Android
# CPU: xxx
# RAM: xxx
# Disk: xxx
```

### Create Bug Report

When reporting issues, include:

1. Output of `--sysinfo`
2. Relevant log excerpts
3. Config (sanitized)
4. Steps to reproduce

```bash
# Collect all debug info
python -m src.main --diagnostics > aura_diagnostics.txt
```

---

## Common Error Messages

| Error | Cause | Solution |
|-------|-------|----------|
| `OOM` | Out of memory | Reduce model size, close apps |
| `Timeout` | Processing too slow | Use smaller model, enable GPU |
| `ImportError` | Missing package | `pip install <package>` |
| `ConfigError` | Invalid config | Check YAML syntax |
| `ThermalThrottle` | Device too hot | Let cool, reduce load |

---

## Recovery

### Full Reset

```bash
# Backup data
cp -r data data_backup
cp config.yaml config_backup.yaml

# Clean install
rm -rf data/* cache/* logs/*

# Reinstall dependencies
pip install -r requirements.txt

# Restore config
cp config_backup.yaml config.yaml

# Restart
python -m src.main
```

### Factory Reset

```bash
# Complete reset (deletes all data)
rm -rf data/ cache/ logs/
# Keep config - edit as needed
python -m src.main
```
