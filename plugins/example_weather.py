"""
Example Plugin: Weather
Demonstrates a plugin that fetches weather data with offline fallback.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional

from src.extensibility.plugin_system import PluginBase, PluginMetadata

logger = logging.getLogger(__name__)


class WeatherPlugin(PluginBase):
    """
    Weather plugin with online API and offline fallback.

    Features:
    - Fetches weather from public API (wttr.in)
    - Offline fallback with manual conditions
    - Registers custom /weather command
    - Hooks into message processing for weather queries
    """

    def __init__(self):
        super().__init__()
        self._location = "auto"
        self._use_celsius = True

    def get_metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="weather",
            version="1.0.0",
            author="AURA Team",
            description="Weather information plugin with offline fallback",
            dependencies=[],
            tags=["utility", "information", "weather"],
        )

    def on_load(self, api) -> bool:
        """Initialize plugin and register commands"""
        if not super().on_load(api):
            return False

        self.register_command("weather", self.handle_weather_command)
        self.register_hook("message", self.on_message)

        logger.info("Weather plugin loaded successfully")
        return True

    def on_message(self, message: Dict[str, Any]) -> Optional[Dict]:
        """Intercept messages for weather queries"""
        content = message.get("content", "").lower()

        if any(word in content for word in ["weather", "temperature", "forecast"]):
            if "?" in content or "what" in content:
                message["_weather_triggered"] = True

        return None

    def handle_weather_command(self, args: str = "") -> Dict[str, Any]:
        """Handle /weather command"""
        location = args.strip() if args else self._location

        weather_data = self.fetch_weather(location)

        if weather_data:
            return {
                "success": True,
                "location": weather_data.get("location", location),
                "temperature": weather_data.get("temperature"),
                "condition": weather_data.get("condition"),
                "humidity": weather_data.get("humidity"),
                "wind": weather_data.get("wind"),
                "formatted": self.format_weather(weather_data),
            }
        else:
            return {
                "success": False,
                "error": "Unable to fetch weather data",
                "fallback": self.get_offline_weather(),
            }

    def fetch_weather(self, location: str) -> Optional[Dict[str, Any]]:
        """Fetch weather from online API or return None for offline fallback"""
        try:
            import urllib.request
            import json

            url = f"https://wttr.in/{location}?format=j1"

            req = urllib.request.Request(url, headers={"User-Agent": "AURA/1.0"})

            with urllib.request.urlopen(req, timeout=5) as response:
                data = json.loads(response.read().decode())

                current = data.get("current_condition", [{}])[0]

                return {
                    "location": location.title(),
                    "temperature": current.get("temp_C")
                    if self._use_celsius
                    else current.get("temp_F"),
                    "feels_like": current.get("FeelsLikeC")
                    if self._use_celsius
                    else current.get("FeelsLikeF"),
                    "condition": current.get("weatherDesc", [{}])[0].get(
                        "value", "Unknown"
                    ),
                    "humidity": current.get("humidity"),
                    "wind": current.get("windspeedKmph"),
                    "source": "online",
                }

        except Exception as e:
            logger.warning(f"Weather API unavailable: {e}")
            return None

    def get_offline_weather(self) -> Dict[str, Any]:
        """Provide offline fallback weather data"""
        return {
            "location": "Offline Mode",
            "temperature": "--",
            "condition": "Data unavailable (offline)",
            "humidity": "--",
            "wind": "--",
            "source": "offline",
            "note": "Connect to internet for live weather data",
        }

    def format_weather(self, data: Dict) -> str:
        """Format weather data as readable string"""
        temp_unit = "°C" if self._use_celsius else "°F"

        return (
            f"📍 **{data.get('location', 'Unknown')}**\n"
            f"🌡️ Temperature: {data.get('temperature')}{temp_unit}\n"
            f"🤔 Feels like: {data.get('feels_like')}{temp_unit}\n"
            f"☁️ Condition: {data.get('condition')}\n"
            f"💧 Humidity: {data.get('humidity')}%\n"
            f"💨 Wind: {data.get('wind')} km/h"
        )

    def on_tool_call(self, tool_name: str, parameters: Dict) -> Optional[Dict]:
        """Hook into tool calls"""
        if tool_name == "get_weather":
            return self.handle_weather_command(parameters.get("location", ""))
        return None

    def register_tool(self):
        """Register weather tool with the system"""
        from src.tools.registry import ToolDefinition

        tool_def = ToolDefinition(
            name="get_weather",
            description="Get current weather for a location",
            parameters={
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "Location to get weather for (city name)",
                    }
                },
                "required": ["location"],
            },
            category="information",
            risk_level="low",
        )

        self.register_tool(
            "get_weather",
            {
                "name": tool_def.name,
                "description": tool_def.description,
                "parameters": tool_def.parameters,
            },
            lambda **p: self.handle_weather_command(p.get("location", "")),
        )


def get_plugin():
    """Factory function to create plugin instance"""
    return WeatherPlugin()
