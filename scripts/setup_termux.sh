#!/bin/bash
# AURA v3 Termux Setup Script
# Run on Android with Termux

echo "🤖 Setting up AURA v3..."

# Update
apt update && apt upgrade -y

# Install Python
apt install python python3 -y

# Install required packages
apt install git clang -y

# Create directory
mkdir -p ~/aura
cd ~/aura

# Clone repo (or copy files)
# git clone <repo_url> .

# Install Python deps
pip install -r requirements.txt

# Install espeak (TTS)
apt install espeak-ng -y

# Download model (example)
echo "📦 Download LLM model..."
mkdir -p models
# Download qwen2.5-1b (example)
# wget <model_url>

echo "✅ Setup complete!"
echo ""
echo "To run:"
echo "  python main.py --mode cli"
echo ""
echo "For Telegram:"
echo "  Set TELEGRAM_TOKEN environment variable"
