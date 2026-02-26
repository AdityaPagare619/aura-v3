#!/usr/bin/env python3
"""
AURA v3 - Main Entry Point
===========================
Run: python main.py [--mode cli|telegram] [--debug]

This is a convenience wrapper that delegates to src/main.py
"""

import sys
import os
import asyncio

# Add src to path so imports work correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Import the async main function
from main import main as aura_main

if __name__ == "__main__":
    asyncio.run(aura_main())
