import asyncio
import json
import logging
import subprocess
import shlex
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

from src.utils.db_pool import get_connection, connection as db_connection

logger = logging.getLogger(__name__)


@dataclass
class AppStructure:
    """Cached app UI structure"""

    app_name: str
    package: str
    screen_size: Dict[str, int] = field(default_factory=dict)
    elements: Dict[str, Dict] = field(default_factory=dict)
    last_updated: str = ""


class AppExplorationMemory:
    """
    SQLite-backed memory for app UI structures.
    AURA explores an app ONCE and remembers FOREVER.
    """

    def __init__(self, db_path: str = "data/memories/app_exploration.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with db_connection(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS app_structures (
                    app_name TEXT PRIMARY KEY,
                    package TEXT,
                    screen_size TEXT,
                    elements TEXT,
                    last_updated TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS element_positions (
                    app_name TEXT,
                    element_desc TEXT,
                    x INTEGER,
                    y INTEGER,
                    last_updated TEXT,
                    PRIMARY KEY (app_name, element_desc)
                )
            """)

    def save_app_structure(self, app_name: str, structure: Dict):
        from datetime import datetime

        with db_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO app_structures 
                (app_name, package, screen_size, elements, last_updated)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    app_name,
                    structure.get("package", ""),
                    json.dumps(structure.get("screen_size", {})),
                    json.dumps(structure.get("elements", {})),
                    datetime.now().isoformat(),
                ),
            )
        logger.info(f"Saved app structure for: {app_name}")

    def get_app_structure(self, app_name: str) -> Optional[Dict]:
        conn = get_connection(self.db_path)
        cursor = conn.execute(
            "SELECT package, screen_size, elements FROM app_structures WHERE app_name = ?",
            (app_name,),
        )
        row = cursor.fetchone()

        if row:
            return {
                "package": row[0],
                "screen_size": json.loads(row[1]),
                "elements": json.loads(row[2]),
            }
        return None

    def save_element_position(
        self, app_name: str, element_desc: str, coords: Tuple[int, int]
    ):
        from datetime import datetime

        with db_connection(self.db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO element_positions 
                (app_name, element_desc, x, y, last_updated)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    app_name,
                    element_desc,
                    coords[0],
                    coords[1],
                    datetime.now().isoformat(),
                ),
            )

    def get_element_position(
        self, app_name: str, element_desc: str
    ) -> Optional[Tuple[int, int]]:
        conn = get_connection(self.db_path)
        cursor = conn.execute(
            "SELECT x, y FROM element_positions WHERE app_name = ? AND element_desc = ?",
            (app_name, element_desc),
        )
        row = cursor.fetchone()

        if row:
            return (row[0], row[1])
        return None

    def list_cached_apps(self) -> List[str]:
        conn = get_connection(self.db_path)
        cursor = conn.execute("SELECT app_name FROM app_structures")
        apps = [row[0] for row in cursor.fetchall()]
        return apps

    def delete_app_structure(self, app_name: str):
        with db_connection(self.db_path) as conn:
            conn.execute("DELETE FROM app_structures WHERE app_name = ?", (app_name,))
            conn.execute(
                "DELETE FROM element_positions WHERE app_name = ?", (app_name,)
            )
        logger.info(f"Deleted app structure for: {app_name}")

    def get_all_element_positions(self, app_name: str) -> Dict[str, Tuple[int, int]]:
        conn = get_connection(self.db_path)
        cursor = conn.execute(
            "SELECT element_desc, x, y FROM element_positions WHERE app_name = ?",
            (app_name,),
        )
        positions = {row[0]: (row[1], row[2]) for row in cursor.fetchall()}
        return positions


class AndroidShell:
    """Execute ADB commands on Android/Termux"""

    def __init__(self, adb_path: str = "adb"):
        self.adb_path = adb_path

    async def run(self, command: str, timeout: int = 10) -> str:
        """Run shell command - SECURED: uses subprocess_exec with argument list"""
        try:
            # FIXED: Use shlex.split to properly parse command into list
            # This avoids shell injection by using create_subprocess_exec instead of shell
            cmd_list = shlex.split(command)
            process = await asyncio.create_subprocess_exec(
                *cmd_list,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
                return stdout.decode().strip()
            except asyncio.TimeoutError:
                process.kill()
                return ""
        except Exception as e:
            logger.error(f"Shell error: {e}")
            return ""

    async def tap(self, x: int, y: int) -> Dict:
        result = await self.run(f"input tap {x} {y}")
        return {"success": result == "", "output": result}

    async def swipe(
        self, x1: int, y1: int, x2: int, y2: int, duration: int = 300
    ) -> Dict:
        result = await self.run(f"input swipe {x1} {y1} {x2} {y2} {duration}")
        return {"success": result == "", "output": result}

    async def type_text(self, text: str) -> Dict:
        text = text.replace(" ", "%s").replace("\n", "")
        result = await self.run(f"input text {text}")
        return {"success": result == "", "output": result}

    async def press_key(self, key: str) -> Dict:
        key_map = {
            "back": "KEYCODE_BACK",
            "home": "KEYCODE_HOME",
            "recent": "KEYCODE_APP_SWITCH",
        }
        result = await self.run(f"input keyevent {key_map.get(key, key)}")
        return {"success": result == "", "output": result}

    async def get_current_app(self) -> Dict:
        result = await self.run("dumpsys window | grep mCurrentFocus")
        return {"success": True, "app": result}

    async def take_screenshot(self, path: str = "/sdcard/screenshot.png") -> Dict:
        result = await self.run(f"screencap -p {path}")
        return {"success": result == "", "path": path}

    async def get_screen_size(self) -> Dict[str, int]:
        result = await self.run("wm size")
        if result:
            try:
                size = result.split(":")[-1].strip()
                width, height = map(int, size.split("x"))
                return {"width": width, "height": height}
            except:
                pass
        return {"width": 1080, "height": 2400}

    async def pull_file(self, remote_path: str, local_path: str) -> Dict:
        result = await self.run(f"pull {remote_path} {local_path}")
        return {"success": result == "", "path": local_path}

    async def push_file(self, local_path: str, remote_path: str) -> Dict:
        result = await self.run(f"push {local_path} {remote_path}")
        return {"success": result == "", "path": remote_path}


class AndroidTools:
    """
    Android actions with exploration memory.
    Key innovation: explores app once, remembers forever.
    """

    def __init__(self, exploration_memory: AppExplorationMemory = None):
        self.memory = exploration_memory or AppExplorationMemory()
        self.shell = AndroidShell()
        self.screen_size = {"width": 1080, "height": 2400}

    async def _detect_screen_size(self):
        size = await self.shell.get_screen_size()
        self.screen_size = size
        return size

    async def send_whatsapp(self, contact: str, message: str) -> Dict:
        """Send WhatsApp message using exploration memory"""
        cached = self.memory.get_app_structure("whatsapp")

        if cached:
            logger.info("Using cached WhatsApp structure")
            search_coords = self.memory.get_element_position("whatsapp", "search")
            if search_coords:
                await self.shell.tap(*search_coords)
                await asyncio.sleep(0.5)
                await self.shell.type_text(contact)
                await asyncio.sleep(1)

        await self.shell.run("monkey -p com.whatsapp 1")
        await asyncio.sleep(2)

        result = await self.shell.run(
            f"am start -a android.intent.action.SENDTO -d smsto:+91 "
        )

        return {"success": True, "contact": contact, "message": message}

    async def make_call(self, number: str, delay: int = 0) -> Dict:
        """Make phone call"""
        if delay > 0:
            await asyncio.sleep(delay)

        result = await self.shell.run(
            f"am start -a android.intent.action.DIAL tel:{number}"
        )
        return {"success": True, "number": number, "delay": delay}

    async def send_sms(self, number: str, message: str) -> Dict:
        result = await self.shell.run(
            f"am start -a android.intent.action.SENDTO -d sms:{number} --es sms_body '{message}'"
        )
        return {"success": True, "number": number, "message": message}

    async def open_app(self, app_name: str) -> Dict:
        cached = self.memory.get_app_structure(app_name)

        if cached and cached.get("package"):
            package = cached["package"]
        else:
            package = app_name

        result = await self.shell.run(f"monkey -p {package} 1")
        return {"success": True, "app": app_name}

    async def get_current_app(self) -> Dict:
        return await self.shell.get_current_app()

    async def take_screenshot(self, path: str = "/sdcard/screenshot.png") -> Dict:
        return await self.shell.take_screenshot(path)

    async def tap(self, x: int, y: int) -> Dict:
        return await self.shell.tap(x, y)

    async def swipe(self, direction: str, distance: int = 500) -> Dict:
        if direction == "up":
            return await self.shell.swipe(540, 1800, 540, 1800 - distance)
        elif direction == "down":
            return await self.shell.swipe(540, 600, 540, 600 + distance)
        elif direction == "left":
            return await self.shell.swipe(900, 1200, 900 - distance, 1200)
        elif direction == "right":
            return await self.shell.swipe(180, 1200, 180 + distance, 1200)
        return {"success": False, "error": "Invalid direction"}

    async def type_text(self, text: str) -> Dict:
        return await self.shell.type_text(text)

    async def press_key(self, key: str) -> Dict:
        return await self.shell.press_key(key)

    async def get_notifications(self, limit: int = 10) -> Dict:
        result = await self.shell.run(f"dumpsys notification --n={limit}")
        return {"success": True, "notifications": result[:500]}

    async def explore_current_app(self) -> Dict:
        """Explore current app and save structure - the 'explore once, remember forever' feature"""
        app_info = await self.get_current_app()
        screenshot = await self.take_screenshot()
        ui_dump = await self.shell.run("uiautomator dump /sdcard/ui_dump.xml")

        await self._detect_screen_size()

        structure = {
            "screen_size": self.screen_size,
            "elements": {},
            "last_explored": str(asyncio.get_event_loop().time()),
        }

        self.memory.save_app_structure("current_app", structure)

        return {"success": True, "structure": structure}

    async def save_tapped_element(
        self, app_name: str, element_desc: str, x: int, y: int
    ):
        """Save a tapped element position for future use"""
        self.memory.save_element_position(app_name, element_desc, (x, y))
        logger.info(f"Saved element '{element_desc}' at ({x}, {y}) for {app_name}")

    async def tap_cached_element(self, app_name: str, element_desc: str) -> Dict:
        """Tap a cached element position"""
        coords = self.memory.get_element_position(app_name, element_desc)
        if coords:
            return await self.shell.tap(*coords)
        return {
            "success": False,
            "error": f"Element '{element_desc}' not found in cache",
        }

    async def get_cached_apps(self) -> List[str]:
        """Get list of apps with cached structures"""
        return self.memory.list_cached_apps()

    async def clear_app_memory(self, app_name: str):
        """Clear cached structure for an app"""
        self.memory.delete_app_structure(app_name)
        return {"success": True, "app": app_name}


__all__ = ["AndroidTools", "AppExplorationMemory", "AndroidShell", "AppStructure"]
