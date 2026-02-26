"""
AURA v3 API Server
==================

Simple API server to connect frontend (React) to backend (AURA).
Works on mobile via Termux - lightweight and efficient.

This enables:
- Frontend to send messages to AURA
- Frontend to receive responses from AURA
- Real-time status updates
- Inner voice visualization
"""

import asyncio
import json
import logging
import os
from typing import Optional
from datetime import datetime
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from dataclasses import dataclass, field
import hashlib
import hmac
import secrets
import time

try:
    import yaml
except ImportError:
    yaml = None
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# =============================================================================
# SECURITY CONFIGURATION
# =============================================================================


@dataclass
class SecurityConfig:
    require_auth: bool = True
    api_token: str = ""
    localhost_only: bool = True
    rate_limit_requests: int = 30
    rate_limit_window: int = 60

    @classmethod
    def load_from_file(cls, config_path="config/security.yaml"):
        try:
            with open(config_path) as f:
                data = yaml.safe_load(f) or {}
                sec = data.get("security", {})
                # Priority: 1. Environment variable, 2. YAML config, 3. Generate new
                token = os.environ.get("API_TOKEN")
                if not token:
                    token = sec.get("api_token", "")
                if not token:
                    token = secrets.token_urlsafe(32)
                return cls(
                    require_auth=sec.get("require_auth", True),
                    api_token=token,
                    localhost_only=sec.get("localhost_only", True),
                    rate_limit_requests=sec.get("rate_limit_requests", 30),
                    rate_limit_window=sec.get("rate_limit_window", 60),
                )
        except:
            pass
        return cls(
            require_auth=True,
            api_token=os.environ.get("API_TOKEN") or secrets.token_urlsafe(32),
        )


class RateLimiter:
    def __init__(self, max_requests=30, window_seconds=60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests = {}

    def is_allowed(self, request):
        now = time.time()
        client = request.remote
        if client not in self._requests:
            self._requests[client] = []
        self._requests[client] = [
            t for t in self._requests[client] if now - t < self.window_seconds
        ]
        if len(self._requests[client]) >= self.max_requests:
            return False, {"error": "Rate limit exceeded"}
        self._requests[client].append(now)
        return True, {}


class AuthMiddleware:
    def __init__(self, config):
        self.config = config
        self._hash = (
            hashlib.sha256(config.api_token.encode()).hexdigest()
            if config.api_token
            else None
        )

    async def authenticate(self, request):
        if not self.config.require_auth:
            return True, None
        header = request.headers.get("Authorization", "")
        if not header:
            return False, "Missing Authorization header"
        parts = header.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            return False, "Use: Authorization: Bearer <token>"
        token_hash = hashlib.sha256(parts[1].encode()).hexdigest()
        if not hmac.compare_digest(token_hash, self._hash):
            return False, "Invalid token"
        return True, None


@dataclass
class AuraResponse:
    """Response from AURA"""

    message: str
    state: str
    inner_voice: Optional[list] = None
    feeling: Optional[float] = None
    trust_level: Optional[float] = None
    timestamp: str = ""


class AuraAPIServer:
    """
    Simple API server for AURA frontend-backend communication.

    SECURED ENDPOINTS (require auth): POST /api/chat, POST /api/feedback
    PUBLIC ENDPOINTS: GET /api/status, /api/inner-voice, /api/feeling, /api/trust

    Rate limiting applied to all endpoints.
    """

    PROTECTED_ENDPOINTS = ["/api/chat", "/api/feedback"]

    def __init__(self, aura_instance=None, security_config=None):
        self.aura = aura_instance
        self._running = False
        self._config = security_config or SecurityConfig.load_from_file()
        self._rate_limiter = RateLimiter(
            self._config.rate_limit_requests, self._config.rate_limit_window
        )
        self._auth = AuthMiddleware(self._config)

    def _requires_auth(self, path):
        return path in self.PROTECTED_ENDPOINTS

    async def start(self, host=None, port=5000):
        """Start the API server with security middleware"""
        if host is None:
            host = "127.0.0.1" if self._config.localhost_only else "0.0.0.0"

        if not self._config.require_auth:
            logger.warning("SECURITY: Authentication is DISABLED!")

        logger.info(
            f"API Token: {self._config.api_token[:8]}... (set in config/security.yaml)"
        )

        try:
            from aiohttp import web

            self.app = web.Application(middlewares=[self._security_middleware])

            # Auth status endpoint
            self.app.router.add_get("/api/auth/status", self.handle_auth_status)
            self.app.router.add_post("/api/chat", self.handle_chat)
            self.app.router.add_get("/api/status", self.handle_status)
            self.app.router.add_get("/api/inner-voice", self.handle_inner_voice)
            self.app.router.add_get("/api/inner-voice/logs", self.handle_logs)
            self.app.router.add_get("/api/feeling", self.handle_feeling)
            self.app.router.add_get("/api/trust", self.handle_trust)
            self.app.router.add_post("/api/feedback", self.handle_feedback)
            self.app.router.add_get("/api/auth/status", self.handle_auth_status)

            self.runner = web.AppRunner(self.app)
            await self.runner.setup()

            site = web.TCPSite(self.runner, host, port)
            await site.start()

            self._running = True
            logger.info(f"AURA API server started on http://{host}:{port}")

        except ImportError:
            logger.warning("aiohttp not available, using fallback server")
            await self._start_fallback(port)

    async def _start_fallback(self, port=5000):
        """Fallback to basic asyncio server"""
        logger.info("Starting basic asyncio server...")
        # Basic implementation without aiohttp
        self._running = True

    async def _security_middleware(self, app, handler):
        async def middleware(request):
            allowed, info = self._rate_limiter.is_allowed(request)
            if not allowed:
                return web.json_response(info, status=429)

            if self._requires_auth(request.path):
                auth, msg = await self._auth.authenticate(request)
                if not auth:
                    return web.json_response({"error": msg}, status=401)

            return await handler(request)

        return middleware

    async def handle_auth_status(self, request):
        return web.json_response({"auth_required": self._config.require_auth})

    async def handle_chat(self, request):  # REQUIRES AUTH
        """Handle chat requests from frontend"""
        try:
            data = await request.json()
            user_message = data.get("message", "")

            if not self.aura:
                return web.json_response({"error": "AURA not initialized"}, status=500)

            # Process through AURA
            response = await self.aura.process(user_message)

            # Get additional context
            inner_voice = await self._get_inner_voice()
            feeling = await self._get_feeling()
            trust = await self._get_trust()

            return web.json_response(
                {
                    "message": response,
                    "timestamp": datetime.now().isoformat(),
                    "inner_voice": inner_voice,
                    "feeling": feeling,
                    "trust_level": trust,
                }
            )

        except Exception as e:
            logger.error(f"Chat error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_status(self, request):
        """Get AURA status"""
        if not self.aura:
            return web.json_response(
                {"running": False, "error": "AURA not initialized"}
            )

        status = self.aura.get_status()
        return web.json_response(status)

    async def handle_inner_voice(self, request):
        """Get inner voice state"""
        inner_voice = await self._get_inner_voice()
        return web.json_response({"thoughts": inner_voice})

    async def handle_logs(self, request):
        """Get transparent logs"""
        try:
            from src.core.transparent_logger import get_transparent_logger, LogLevel

            # Get query params
            level = request.query.get("level", "")
            limit = int(request.query.get("limit", 10))

            # Get logger
            logger = get_transparent_logger()

            # Filter by level if specified
            levels = None
            if level:
                try:
                    levels = [LogLevel(level)]
                except ValueError:
                    pass

            # Get logs
            logs = logger.get_logs(levels=levels, limit=limit)

            # Format for JSON
            entries = []
            for log in logs:
                entries.append(
                    {
                        "id": log.id,
                        "level": log.level.value,
                        "content": log.display_content or log.content,
                        "category": log.category,
                        "timestamp": log.timestamp.isoformat(),
                        "status": log.status,
                        "duration_ms": log.duration_ms,
                        "data_access": log.data_categories,
                    }
                )

            # Get processing status
            status = logger.get_status()

            return web.json_response(
                {
                    "logs": entries,
                    "processing": {
                        "is_processing": status.is_processing,
                        "phase": status.current_phase,
                        "message": status.get_display_message(),
                        "data_access": status.data_access,
                    },
                    "stats": logger.get_stats(),
                }
            )

        except ImportError:
            return web.json_response(
                {"error": "Transparent logger not available"}, status=500
            )
        except Exception as e:
            logger.error(f"Logs error: {e}")
            return web.json_response({"error": str(e)}, status=500)

    async def handle_feeling(self, request):
        """Get current feeling"""
        feeling = await self._get_feeling()
        return web.json_response({"feeling": feeling})

    async def handle_trust(self, request):
        """Get trust level"""
        trust = await self._get_trust()
        return web.json_response({"trust_level": trust})

    async def handle_feedback(self, request):
        """Handle user feedback"""
        try:
            data = await request.json()
            # Process feedback for learning
            return web.json_response({"status": "received"})
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def _get_inner_voice(self):
        """Get inner voice thoughts"""
        if self.aura and self.aura._inner_voice:
            try:
                thoughts = await self.aura._inner_voice.get_recent_thoughts(limit=3)
                return [
                    {
                        "thought": t.thought,
                        "bubble_type": t.bubble_type,
                        "timestamp": t.timestamp.isoformat() if t.timestamp else None,
                    }
                    for t in thoughts
                ]
            except:
                pass
        return []

    async def _get_feeling(self):
        """Get current feeling level"""
        if self.aura and self.aura._inner_voice:
            try:
                return await self.aura._inner_voice.get_current_feeling()
            except:
                pass
        return 0.5

    async def _get_trust(self):
        """Get current trust level"""
        if self.aura and self.aura._inner_voice:
            try:
                return await self.aura._inner_voice.get_trust_level()
            except:
                pass
        return 0.5

    async def stop(self):
        """Stop the server"""
        self._running = False
        if hasattr(self, "runner"):
            await self.runner.cleanup()
        logger.info("AURA API server stopped")


# Global instance
_server: Optional[AuraAPIServer] = None


async def start_aura_server(
    aura_instance=None, host=None, port=5000, security_config=None
):
    """Start the AURA API server with security defaults"""
    global _server
    _server = AuraAPIServer(aura_instance, security_config)
    await _server.start(host, port)
    return _server


def get_security_config(config_path="config/security.yaml"):
    """Get the security configuration"""
    return SecurityConfig.load_from_file(config_path)


async def stop_aura_server():
    """Stop the AURA API server"""
    global _server
    if _server:
        await _server.stop()
        _server = None
