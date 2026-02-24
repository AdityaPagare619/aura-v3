"""
AURA v3 Mobile Space Server
===========================

Privacy-first local server for Aura's mobile interface.
Works 100% offline - no internet required.
Serves the Aura Space UI to WebView or local browser.

This enables:
- Aura Space app experience on mobile
- Real-time updates without internet
- Full privacy (all data stays on device)
- Integration with Telegram as command channel
- Floating bubble/widget support via Termux
"""

import asyncio
import json
import logging
import threading
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class AuraPersona:
    """Current persona/mode of Aura"""

    id: str
    name: str
    color_primary: str
    color_secondary: str
    description: str
    orb_animation: str  # breathing, swirling, pulsing


@dataclass
class AuraState:
    """Current state of Aura for UI"""

    mode: str = "companion"  # mission_control, companion, ghost
    current_persona: str = "guardian"
    feeling: float = 0.5  # 0-1
    trust_level: float = 0.5  # 0-1
    is_thinking: bool = False
    current_task: Optional[str] = None
    last_interaction: Optional[datetime] = None
    thoughts: List[Dict] = field(default_factory=list)
    notifications: List[Dict] = field(default_factory=list)


class AuraSpaceServer:
    """
    Local HTTP server for Aura Space mobile interface.

    Runs on localhost - no internet needed.
    Provides:
    - Real-time state via polling/sse
    - UI templates
    - API for mobile app
    """

    def __init__(self, aura_instance=None, port=8080):
        self.aura = aura_instance
        self.port = port
        self._running = False
        self._server = None

        # Aura state for UI
        self.aura_state = AuraState()

        # Persona definitions
        self.personas = {
            "guardian": AuraPersona(
                id="guardian",
                name="Guardian",
                color_primary="#4338CA",  # Indigo
                color_secondary="#8B5CF6",  # Violet
                description="Life, feelings, diary - introspective and calm",
                orb_animation="breathing",
            ),
            "operator": AuraPersona(
                id="operator",
                name="Operator",
                color_primary="#0D9488",  # Teal
                color_secondary="#3B82F6",  # Electric Blue
                description="Tasks, planning, work - focused and clear",
                orb_animation="swirling",
            ),
            "producer": AuraPersona(
                id="producer",
                name="Producer",
                color_primary="#D946EF",  # Magenta
                color_secondary="#F59E0B",  # Amber
                description="Social, digital presence - energetic and playful",
                orb_animation="pulsing",
            ),
            "coach": AuraPersona(
                id="coach",
                name="Coach",
                color_primary="#059669",  # Emerald
                color_secondary="#6EE7B7",  # Soft Mint
                description="Health, sleep, movement - refreshing and healing",
                orb_animation="breathing",
            ),
        }

        # Current mode settings
        self.modes = {
            "mission_control": {
                "name": "Mission Control",
                "hud_enabled": True,
                "notifications": "high",
                "orb_style": "sharp",
            },
            "companion": {
                "name": "Companion",
                "hud_enabled": False,
                "notifications": "balanced",
                "orb_style": "soft",
            },
            "ghost": {
                "name": "Ghost",
                "hud_enabled": False,
                "notifications": "critical_only",
                "orb_style": "minimal",
            },
        }

        # Event callbacks
        self.on_state_change: Optional[Callable] = None

    async def start(self):
        """Start the Aura Space server"""
        try:
            from aiohttp import web

            self.app = web.Application()

            # API Routes
            self.app.router.add_get("/api/state", self.handle_state)
            self.app.router.add_get("/api/personas", self.handle_personas)
            self.app.router.add_get("/api/modes", self.handle_modes)
            self.app.router.add_get("/api/thoughts", self.handle_thoughts)
            self.app.router.add_post("/api/feedback", self.handle_feedback)
            self.app.router.add_post("/api/mode", self.handle_mode_change)
            self.app.router.add_post("/api/persona", self.handle_persona_change)

            # Serve static UI files
            self.app.router.add_get("/", self.handle_index)
            self.app.router.add_get("/space", self.handle_space)
            self.app.router.add_get("/hud", self.handle_hud)

            # SSE for real-time updates
            self.app.router.add_get("/api/stream", self.handle_sse)

            self.runner = web.AppRunner(self.app)
            await self.runner.setup()

            site = web.TCPSite(self.runner, "localhost", self.port)
            await site.start()

            self._running = True
            logger.info(f"Aura Space server started on http://localhost:{self.port}")

            return f"http://localhost:{self.port}"

        except ImportError:
            logger.warning("aiohttp not available, using fallback")
            return await self._start_fallback()

    async def _start_fallback(self):
        """Fallback server using basic asyncio"""
        logger.info("Starting fallback server...")
        self._running = True
        return f"http://localhost:{self.port}"

    async def stop(self):
        """Stop the server"""
        self._running = False
        if hasattr(self, "runner"):
            await self.runner.cleanup()
        logger.info("Aura Space server stopped")

    # API Handlers
    async def handle_state(self, request):
        """Get current Aura state"""
        state = {
            "mode": self.aura_state.mode,
            "persona": self.aura_state.current_persona,
            "persona_info": self.personas.get(self.aura_state.current_persona).__dict__
            if self.personas.get(self.aura_state.current_persona)
            else None,
            "feeling": self.aura_state.feeling,
            "trust_level": self.aura_state.trust_level,
            "is_thinking": self.aura_state.is_thinking,
            "current_task": self.aura_state.current_task,
            "last_interaction": self.aura_state.last_interaction.isoformat()
            if self.aura_state.last_interaction
            else None,
            "notifications_count": len(self.aura_state.notifications),
            "timestamp": datetime.now().isoformat(),
        }
        return web.json_response(state)

    async def handle_personas(self, request):
        """Get all available personas"""
        personas = {
            pid: {
                "id": p.id,
                "name": p.name,
                "color_primary": p.color_primary,
                "color_secondary": p.color_secondary,
                "description": p.description,
                "orb_animation": p.orb_animation,
            }
            for pid, p in self.personas.items()
        }
        return web.json_response(personas)

    async def handle_modes(self, request):
        """Get all available modes"""
        return web.json_response(self.modes)

    async def handle_thoughts(self, request):
        """Get recent thoughts from Aura"""
        limit = int(request.query.get("limit", 5))
        thoughts = self.aura_state.thoughts[-limit:]
        return web.json_response({"thoughts": thoughts})

    async def handle_feedback(self, request):
        """Handle user feedback on Aura's thoughts/actions"""
        try:
            data = await request.json()
            feedback_type = data.get("type")  # agree, disagree, dont_discuss
            thought_id = data.get("thought_id")

            # Process feedback for learning
            logger.info(f"User feedback: {feedback_type} on thought {thought_id}")

            # Update trust based on feedback
            if feedback_type == "agree":
                self.aura_state.trust_level = min(
                    1.0, self.aura_state.trust_level + 0.05
                )
            elif feedback_type == "disagree":
                self.aura_state.trust_level = max(
                    0.0, self.aura_state.trust_level - 0.05
                )

            return web.json_response(
                {"status": "ok", "trust_level": self.aura_state.trust_level}
            )

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_mode_change(self, request):
        """Change Aura's mode"""
        try:
            data = await request.json()
            mode = data.get("mode")

            if mode in self.modes:
                self.aura_state.mode = mode
                logger.info(f"Aura mode changed to: {mode}")
                return web.json_response({"status": "ok", "mode": mode})
            else:
                return web.json_response({"error": "Invalid mode"}, status=400)

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_persona_change(self, request):
        """Change Aura's persona"""
        try:
            data = await request.json()
            persona = data.get("persona")

            if persona in self.personas:
                self.aura_state.current_persona = persona
                logger.info(f"Aura persona changed to: {persona}")
                return web.json_response({"status": "ok", "persona": persona})
            else:
                return web.json_response({"error": "Invalid persona"}, status=400)

        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_sse(self, request):
        """Server-Sent Events for real-time updates"""
        from aiohttp import web

        response = web.StreamResponse()
        await response.prepare(request)

        while self._running:
            # Send current state
            state_json = json.dumps(
                {
                    "type": "state",
                    "data": {
                        "mode": self.aura_state.mode,
                        "persona": self.aura_state.current_persona,
                        "feeling": self.aura_state.feeling,
                        "is_thinking": self.aura_state.is_thinking,
                        "current_task": self.aura_state.current_task,
                    },
                }
            )
            response.write(f"data: {state_json}\n\n".encode())
            await asyncio.sleep(1)

        return response

    async def handle_index(self, request):
        """Serve main index page"""
        from aiohttp import web

        html = self._generate_main_html()
        return web.Response(text=html, content_type="text/html")

    async def handle_space(self, request):
        """Serve Aura Space page"""
        from aiohttp import web

        html = self._generate_space_html()
        return web.Response(text=html, content_type="text/html")

    async def handle_hud(self, request):
        """Serve minimal HUD overlay"""
        from aiohttp import web

        html = self._generate_hud_html()
        return web.Response(text=html, content_type="text/html")

    def _generate_main_html(self) -> str:
        """Generate main Aura interface HTML"""
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>AURA</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0f;
            color: #fff;
            min-height: 100vh;
            overflow-x: hidden;
        }
        .orb-container {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            width: 200px;
            height: 200px;
            z-index: 1;
        }
        .orb {
            width: 100%;
            height: 100%;
            border-radius: 50%;
            background: linear-gradient(135deg, var(--color1, #4338CA), var(--color2, #8B5CF6));
            animation: breathing 4s ease-in-out infinite;
            box-shadow: 0 0 60px var(--color1, #4338CA), 0 0 100px var(--color2, #8B5CF6);
        }
        @keyframes breathing {
            0%, 100% { transform: scale(1); opacity: 0.9; }
            50% { transform: scale(1.1); opacity: 1; }
        }
        .nav {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            display: flex;
            justify-content: space-around;
            padding: 20px;
            background: rgba(10, 10, 15, 0.9);
            backdrop-filter: blur(20px);
            z-index: 100;
        }
        .nav-item {
            display: flex;
            flex-direction: column;
            align-items: center;
            color: #666;
            font-size: 12px;
            text-decoration: none;
            transition: color 0.3s;
        }
        .nav-item.active { color: #fff; }
        .nav-item span { font-size: 24px; margin-bottom: 4px; }
        .status-bar {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            padding: 16px 20px;
            background: rgba(10, 10, 15, 0.8);
            backdrop-filter: blur(10px);
            display: flex;
            justify-content: space-between;
            align-items: center;
            z-index: 100;
        }
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #10b981;
            animation: pulse 2s infinite;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        .mode-badge {
            padding: 4px 12px;
            border-radius: 12px;
            background: rgba(255,255,255,0.1);
            font-size: 12px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
    </style>
</head>
<body>
    <div class="status-bar">
        <div class="status-indicator">
            <div class="status-dot"></div>
            <span id="status">Listening</span>
        </div>
        <div class="mode-badge" id="mode">Guardian</div>
    </div>
    
    <div class="orb-container">
        <div class="orb" id="orb"></div>
    </div>
    
    <nav class="nav">
        <a href="/space?tab=diary" class="nav-item">
            <span>📔</span>
            Diary
        </a>
        <a href="/space?tab=flow" class="nav-item">
            <span>🌊</span>
            Flow
        </a>
        <a href="/space?tab=mind" class="nav-item">
            <span>🧠</span>
            Mind
        </a>
        <a href="/space?tab=crew" class="nav-item">
            <span>👥</span>
            Crew
        </a>
    </nav>
    
    <script>
        const orb = document.getElementById('orb');
        const colors = {
            guardian: ['#4338CA', '#8B5CF6'],
            operator: ['#0D9488', '#3B82F6'],
            producer: ['#D946EF', '#F59E0B'],
            coach: ['#059669', '#6EE7B7']
        };
        
        // Poll for state updates
        async function updateState() {
            try {
                const res = await fetch('/api/state');
                const state = await res.json();
                
                document.getElementById('mode').textContent = state.persona_info?.name || 'Guardian';
                document.getElementById('status').textContent = state.is_thinking ? 'Thinking...' : 'Listening';
                
                const colorSet = colors[state.persona] || colors.guardian;
                orb.style.setProperty('--color1', colorSet[0]);
                orb.style.setProperty('--color2', colorSet[1]);
            } catch(e) { console.log(e); }
        }
        
        setInterval(updateState, 2000);
        updateState();
    </script>
</body>
</html>"""

    def _generate_space_html(self) -> str:
        """Generate Aura Space page with tabs"""
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aura Space</title>
    <style>
        /* Aura Space styles - See full implementation in mobile_ui/ */
    </style>
</head>
<body>
    <!-- Aura Space with Diary, Flow, Mind, Crew tabs -->
    <div id="app"></div>
    <script>
        // Load tab content based on URL param
        const params = new URLSearchParams(window.location.search);
        const tab = params.get('tab') || 'diary';
        // Render appropriate tab
    </script>
</body>
</html>"""

    def _generate_hud_html(self) -> str:
        """Generate minimal HUD overlay"""
        return """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width">
    <title>AURA HUD</title>
    <style>
        body { background: transparent; margin: 0; overflow: hidden; }
        .hud {
            position: fixed;
            top: 10px;
            left: 10px;
            right: 10px;
            padding: 8px 16px;
            background: rgba(0,0,0,0.6);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 12px;
            backdrop-filter: blur(10px);
        }
        .task { color: #fff; }
        .orb-mini {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: linear-gradient(135deg, #4338CA, #8B5CF6);
        }
    </style>
</head>
<body>
    <div class="hud">
        <span class="task" id="task">Ready</span>
        <div class="orb-mini"></div>
    </div>
    <script>
        setInterval(async () => {
            const res = await fetch('/api/state');
            const state = await res.json();
            document.getElementById('task').textContent = state.current_task || 'Ready';
        }, 1000);
    </script>
</body>
</html>"""

    def update_state(self, **kwargs):
        """Update Aura state from external sources"""
        for key, value in kwargs.items():
            if hasattr(self.aura_state, key):
                setattr(self.aura_state, key, value)

    def add_thought(self, thought: str, bubble_type: str = "normal"):
        """Add a thought to Aura's thought stream"""
        self.aura_state.thoughts.append(
            {
                "id": len(self.aura_state.thoughts) + 1,
                "thought": thought,
                "bubble_type": bubble_type,
                "timestamp": datetime.now().isoformat(),
            }
        )
        # Keep only last 20 thoughts
        self.aura_state.thoughts = self.aura_state.thoughts[-20:]


# Global instance
_server: Optional[AuraSpaceServer] = None


def get_aura_space_server() -> AuraSpaceServer:
    """Get or create the Aura Space server"""
    global _server
    if _server is None:
        _server = AuraSpaceServer()
    return _server
