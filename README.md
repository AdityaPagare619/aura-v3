# AURA v3 - Personal Mobile AGI Assistant

Next-generation personal AI assistant that runs 100% offline on your Android device.

## Features

- **Self-Aware**: Knows its capabilities and limitations
- **Learning**: Improves from every interaction
- **Privacy-First**: All data stays on device
- **Context-Aware**: Adapts behavior based on time, location, activity
- **ReAct Loop**: Reasoning + Acting pattern for complex tasks
- **Tool Schemas**: LLM knows exactly what tools it has

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run CLI mode
python main.py --mode cli

# Run voice mode  
python main.py --mode voice
```

## Architecture

- **Agent**: ReAct loop with JSON tool schemas
- **Memory**: Hierarchical (Immediate→Working→Short→Long→Self-model)
- **Learning**: Intent patterns, contact priorities, smart delays
- **Security**: L1-L4 permission levels, banking protection
- **Tools**: Android actions with exploration memory

## Requirements

- Python 3.9+
- 4GB RAM minimum
- Android with Termux (for mobile)
