"""
AURA v3 Communication Channels
Telegram and WhatsApp bot integration for user interaction
"""

import asyncio
import logging
import os
import json
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class ChannelType(Enum):
    """Types of communication channels"""

    TELEGRAM = "telegram"
    WHATSAPP = "whatsapp"
    TERMUX_NOTIFICATION = "termux"
    ANDROID_NOTIFICATION = "android"


class ChannelStatus(Enum):
    """Status of channel connection"""

    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class MessageType(Enum):
    """Types of messages"""

    TEXT = "text"
    IMAGE = "image"
    VOICE = "voice"
    VIDEO = "video"
    DOCUMENT = "document"
    LOCATION = "location"


@dataclass
class IncomingMessage:
    """Incoming message from any channel"""

    id: str
    channel: ChannelType
    message_type: MessageType
    sender: str
    sender_name: str
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutgoingMessage:
    """Message to send"""

    channel: ChannelType
    recipient: str
    message_type: MessageType
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ChannelConfig:
    """Configuration for a communication channel"""

    channel_type: ChannelType
    enabled: bool = False
    config: Dict[str, Any] = field(default_factory=dict)


class CommunicationChannel:
    """Base class for communication channels"""

    def __init__(self, channel_type: ChannelType):
        self.channel_type = channel_type
        self.status = ChannelStatus.DISCONNECTED
        self._message_callback: Optional[Callable] = None

    async def connect(self) -> bool:
        """Connect to the channel"""
        raise NotImplementedError

    async def disconnect(self):
        """Disconnect from the channel"""
        raise NotImplementedError

    async def send_message(self, message: OutgoingMessage) -> bool:
        """Send a message"""
        raise NotImplementedError

    def set_message_callback(self, callback: Callable):
        """Set callback for incoming messages"""
        self._message_callback = callback

    async def _handle_incoming(self, message: IncomingMessage):
        """Handle incoming message"""
        if self._message_callback:
            try:
                if asyncio.iscoroutinefunction(self._message_callback):
                    await self._message_callback(message)
                else:
                    self._message_callback(message)
            except Exception as e:
                logger.error(f"Message callback error: {e}")


class TelegramChannel(CommunicationChannel):
    """
    Telegram Bot Channel

    Uses Telegram Bot API for communication.
    Note: Requires bot token from @BotFather
    """

    def __init__(self, bot_token: str = ""):
        super().__init__(ChannelType.TELEGRAM)
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN", "")
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}"
        self._long_poll_task: Optional[asyncio.Task] = None
        self._offset = 0

    async def connect(self) -> bool:
        """Connect to Telegram"""
        if not self.bot_token:
            logger.warning("Telegram bot token not set")
            return False

        try:
            self.status = ChannelStatus.CONNECTING
            # Test connection
            # In real implementation: await self._make_request("getMe")
            self.status = ChannelStatus.CONNECTED
            logger.info("Telegram channel connected")
            return True
        except Exception as e:
            self.status = ChannelStatus.ERROR
            logger.error(f"Telegram connection error: {e}")
            return False

    async def disconnect(self):
        """Disconnect from Telegram"""
        if self._long_poll_task:
            self._long_poll_task.cancel()
            self._long_poll_task = None
        self.status = ChannelStatus.DISCONNECTED
        logger.info("Telegram channel disconnected")

    async def send_message(self, message: OutgoingMessage) -> bool:
        """Send message via Telegram"""
        if self.status != ChannelStatus.CONNECTED:
            logger.warning("Telegram not connected")
            return False

        try:
            # Placeholder - actual implementation would call Telegram API
            logger.info(
                f"Would send Telegram message to {message.recipient}: {message.content[:50]}"
            )
            return True
        except Exception as e:
            logger.error(f"Telegram send error: {e}")
            return False

    async def _make_request(self, method: str, params: Dict = None) -> Dict:
        """Make API request to Telegram"""
        import aiohttp

        url = f"{self.api_url}/{method}"
        async with aiohttp.ClientSession() as session:
            async with session.post(url, json=params) as response:
                return await response.json()


class WhatsAppChannel(CommunicationChannel):
    """
    WhatsApp Channel

    Note: WhatsApp Business API requires approval and setup.
    For personal use, can use WhatsApp Web with automation (against ToS).
    This is a placeholder for official API integration.
    """

    def __init__(self, phone_id: str = "", access_token: str = ""):
        super().__init__(ChannelType.WHATSAPP)
        self.phone_id = phone_id or os.getenv("WHATSAPP_PHONE_ID", "")
        self.access_token = access_token or os.getenv("WHATSAPP_ACCESS_TOKEN", "")
        self.api_url = "https://graph.facebook.com/v18.0"

    async def connect(self) -> bool:
        """Connect to WhatsApp"""
        if not self.phone_id or not self.access_token:
            logger.warning("WhatsApp credentials not set")
            return False

        try:
            self.status = ChannelStatus.CONNECTING
            # Test connection
            self.status = ChannelStatus.CONNECTED
            logger.info("WhatsApp channel connected")
            return True
        except Exception as e:
            self.status = ChannelStatus.ERROR
            logger.error(f"WhatsApp connection error: {e}")
            return False

    async def disconnect(self):
        """Disconnect from WhatsApp"""
        self.status = ChannelStatus.DISCONNECTED
        logger.info("WhatsApp channel disconnected")

    async def send_message(self, message: OutgoingMessage) -> bool:
        """Send message via WhatsApp"""
        if self.status != ChannelStatus.CONNECTED:
            logger.warning("WhatsApp not connected")
            return False

        try:
            # Placeholder - actual implementation would call WhatsApp API
            logger.info(
                f"Would send WhatsApp message to {message.recipient}: {message.content[:50]}"
            )
            return True
        except Exception as e:
            logger.error(f"WhatsApp send error: {e}")
            return False


class TermuxChannel(CommunicationChannel):
    """
    Termux Channel - Local notifications

    Uses Termux API for local notifications and communication.
    Works 100% offline on Android with Termux.
    """

    def __init__(self):
        super().__init__(ChannelType.TERMUX_NOTIFICATION)
        self._notification_callback: Optional[Callable] = None

    async def connect(self) -> bool:
        """Connect to Termux (always available in Termux)"""
        self.status = ChannelStatus.CONNECTED
        logger.info("Termux channel connected")
        return True

    async def disconnect(self):
        """Disconnect from Termux"""
        self.status = ChannelStatus.DISCONNECTED

    async def send_message(self, message: OutgoingMessage) -> bool:
        """Send notification via Termux"""
        try:
            # Use termux-notification
            cmd = [
                "termux-notification",
                "-t",
                "AURA",
                "-c",
                message.content[:100],
            ]

            # Placeholder - would execute in Termux
            logger.info(f"Would send Termux notification: {message.content[:50]}")
            return True
        except Exception as e:
            logger.error(f"Termux notification error: {e}")
            return False


class CommunicationManager:
    """
    Manages all communication channels

    Handles routing messages between channels and provides
    unified interface for all communication.
    """

    def __init__(self, data_dir: str = "data/communications"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Channels
        self._channels: Dict[ChannelType, CommunicationChannel] = {}
        self._configs: Dict[ChannelType, ChannelConfig] = {}

        # Message routing
        self._routing_rules: List[Dict] = []

        # History
        self._message_history: List[Dict] = []

    def register_channel(self, channel: CommunicationChannel):
        """Register a communication channel"""
        self._channels[channel.channel_type] = channel
        logger.info(f"Registered channel: {channel.channel_type.value}")

    async def connect_all(self) -> Dict[ChannelType, bool]:
        """Connect all enabled channels"""
        results = {}

        for channel_type, channel in self._channels.items():
            config = self._configs.get(channel_type, ChannelConfig(channel_type, False))
            if config.enabled:
                results[channel_type] = await channel.connect()
            else:
                results[channel_type] = False

        return results

    async def disconnect_all(self):
        """Disconnect all channels"""
        for channel in self._channels.values():
            await channel.disconnect()

    async def send_message(self, message: OutgoingMessage) -> bool:
        """Send message via appropriate channel"""
        channel = self._channels.get(message.channel)

        if not channel:
            logger.error(f"Channel not found: {message.channel}")
            return False

        # Log to history
        self._message_history.append(
            {
                "direction": "outgoing",
                "channel": message.channel.value,
                "recipient": message.recipient,
                "content": message.content[:100],
                "timestamp": datetime.now().isoformat(),
            }
        )

        return await channel.send_message(message)

    async def broadcast(
        self, content: str, channels: List[ChannelType] = None
    ) -> Dict[ChannelType, bool]:
        """Broadcast message to multiple channels"""
        if channels is None:
            channels = list(self._channels.keys())

        results = {}
        for channel_type in channels:
            message = OutgoingMessage(
                channel=channel_type,
                recipient="broadcast",
                message_type=MessageType.TEXT,
                content=content,
            )
            results[channel_type] = await self.send_message(message)

        return results

    def set_channel_config(self, config: ChannelConfig):
        """Set configuration for a channel"""
        self._configs[config.channel_type] = config

    def add_routing_rule(self, rule: Dict):
        """Add message routing rule"""
        self._routing_rules.append(rule)

    def route_message(self, message: IncomingMessage) -> ChannelType:
        """Route incoming message based on rules"""
        for rule in self._routing_rules:
            if self._matches_rule(message, rule):
                return rule.get("target_channel", message.channel)

        return message.channel

    def _matches_rule(self, message: IncomingMessage, rule: Dict) -> bool:
        """Check if message matches routing rule"""
        # Simple rule matching - can be extended
        if "sender_contains" in rule:
            if rule["sender_contains"] not in message.sender:
                return False
        return True

    async def get_message_history(
        self, channel: ChannelType = None, limit: int = 50
    ) -> List[Dict]:
        """Get message history"""
        history = self._message_history

        if channel:
            history = [m for m in history if m["channel"] == channel.value]

        return history[-limit:]

    def get_status(self) -> Dict[str, Any]:
        """Get status of all channels"""
        status = {}

        for channel_type, channel in self._channels.items():
            config = self._configs.get(channel_type)
            status[channel_type.value] = {
                "enabled": config.enabled if config else False,
                "status": channel.status.value,
            }

        return status


# ==============================================================================
# FACTORY
# ==============================================================================


def create_communication_manager(
    telegram_token: str = "",
    whatsapp_phone_id: str = "",
    whatsapp_token: str = "",
) -> CommunicationManager:
    """Create and configure communication manager"""

    manager = CommunicationManager()

    # Register Telegram
    if telegram_token:
        telegram = TelegramChannel(telegram_token)
        manager.register_channel(telegram)
        manager.set_channel_config(
            ChannelConfig(ChannelType.TELEGRAM, True, {"token": telegram_token})
        )

    # Register WhatsApp
    if whatsapp_phone_id and whatsapp_token:
        whatsapp = WhatsAppChannel(whatsapp_phone_id, whatsapp_token)
        manager.register_channel(whatsapp)
        manager.set_channel_config(
            ChannelConfig(ChannelType.WHATSAPP, True, {"phone_id": whatsapp_phone_id})
        )

    # Register Termux (always available in Termux)
    termux = TermuxChannel()
    manager.register_channel(termux)
    manager.set_channel_config(ChannelConfig(ChannelType.TERMUX_NOTIFICATION, True))

    return manager
