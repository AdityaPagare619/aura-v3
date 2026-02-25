# A Guide

## Overview

AURA v3 supports offline voice input/output through multiple backends:

### STT (Speech-to-Text)
| Backend | Quality | Speed | Offline | Mobile |
|---------|---------|-------|---------|--------|
| Vosk | Good | Fast | Yes | Excellent |
| Faster Whisper | Excellent | Medium | Yes | Good |
| Whisper.cpp | Good | Fast | Yes | Good |
| Coqui | Good | Slow | Yes | Poor |

### TTS (Text-to-Speech)
| Backend | Quality | Speed | Offline | Mobile |
|---------|---------|-------|---------|--------|
| PyTTSx3 | Fair | Fast | Yes | Good |
| Espeak | Basic | Fast | Yes | Excellent |
| Edge TTS | Excellent | Fast | No | N/A |
| Coqui | Excellent | Slow | Yes | Poor |

## Installation

### For Mobile (Termux) - Recommended

```bash
# Install dependencies
pkg install python numpy

# STT: Vosk (lightweight, fast)
pip install vosk

# TTS: PyTTSx3 (offline, works on mobile)
pip install pyttsx3

# Download Vosk model (choose one):
# Tiny (fastest, least accurate)
wget https://alphacephei.com/vosk/models/vosk-model-tiny-en-us-0.15.zip
# Small (recommended balance)
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip

# Extract to models directory
mkdir -p ~/.cache/aura/vosk
unzip vosk-model-small-en-us-0.15.zip -d ~/.cache/aura/vosk/
```

### For Desktop (High Quality)

```bash
# STT: Faster Whisper (best quality/speed)
pip install faster-whisper

# TTS: Coqui TTS (highest quality)
pip install TTS
```

## Configuration

### (Auto Default Config-detected)

The system automatically selects the best available backend:

```python
from src.voice import stt, tts

# STs best available
T - auto-selectconfig = stt.STTConfig()
processor = stt.STTProcessor(config)

# TTS - defaults to PyTTSx3 (offline)
config = tts.TTSConfig()
engine = tts.TTSEngine(config)
```

### Explicit Configuration

```python
# Force Vosk for STT
config = stt.STTConfig(
    backend=stt.STTBackend.VOSK,
    model_name="small",
    model_path="/path/to/vosk-model"
)

# Force PyTTSx3 for TTS
config = tts.TTSConfig(
    backend=tts.TTSBackend.PYTTSX3
)
```

## Usage

### Basic STT

```python
import asyncio
from src.voice import stt

async def transcribe_audio():
    config = stt.STTConfig()
    processor = stt.STTProcessor(config)
    
    await processor.initialize()
    
    result = await processor.process_voice_message("audio.wav")
    print(result["text"])

asyncio.run(transcribe_audio())
```

### Basic TTS

```python
import asyncio
from src.voice import tts

async def speak_text():
    config = tts.TTSConfig()
    engine = tts.TTSEngine(config)
    
    await engine.initialize()
    
    result = await engine.speak("Hello, I am AURA!")
    print(f"Spoke {result.duration:.2f} seconds of audio")

asyncio.run(speak_text())
```

### Full Pipeline

```python
from src.voice.pipeline import VoicePipeline, VoicePipelineConfig
from src.voice.stt import STTConfig
from src.voice.tts import TTSConfig
from src.voice.hotword import HotWordConfig

# Create configs
stt_config = STTConfig()
tts_config = TTSConfig()
hotword_config = HotWordConfig()

# Create pipeline
pipeline_config = VoicePipelineConfig(
    stt_config=stt_config,
    tts_config=tts_config,
    hotword_config=hotword_config,
    enable_hotword=True
)

pipeline = VoicePipeline(pipeline_config)
await pipeline.initialize()

# Process voice
response = await pipeline.process_telegram_voice("voice.ogg")
```

## Troubleshooting

### Vosk Model Not Found
```
Error: Vosk model not found
```
Solution: Download and extract model to:
- `models/vosk/`
- `~/.cache/aura/vosk/`
- `/data/data/com.termux/files/home/.cache/aura/vosk/`

### PyTTSx3 Not Working on Windows
```
Error: pyttsx3 not installed
```
Solution:
```bash
pip install pyttsx3
# On Windows, may need:
pip install pypiwin32
```

### Audio Format Issues
Ensure audio is 16kHz mono PCM for STT:
- Vosk requires 16-bit PCM
- Faster Whisper accepts float32

## Model Downloads

### Vosk Models
- https://alphacephei.com/vosk/models

### Faster Whisper Models
Auto-downloaded on first use (tiny.en, base, small, etc.)

### Piper TTS Models
- https://github.com/rhasspy/piper/releases

## Performance Notes

| Model | Size | RAM | CPU Usage |
|-------|------|-----|-----------|
| Vosk tiny | 40MB | ~500MB | Low |
| Vosk small | 150MB | ~1GB | Medium |
| Faster Whisper tiny | 75MB | ~1GB | Medium |
| Faster Whisper base | 150MB | ~2GB | High |

For mobile, use Vosk small or smaller.
EOF
echo "Documentation created: VOICE_SETUP.md"
