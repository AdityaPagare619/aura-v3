"""
AURA v3 CLI Resource Dashboard
Text-based dashboard for monitoring system resources
Designed for terminal/Termux display
"""

import asyncio
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Import resource monitor
try:
    from src.utils.resource_monitor import (
        get_resource_monitor,
        ResourceMonitor,
        MetricType,
    )
except ImportError:
    # Handle relative import for testing
    from .resource_monitor import get_resource_monitor, ResourceMonitor, MetricType


class CLIDashboard:
    """
    CLI-based Resource Dashboard

    Displays:
    - RAM usage with bar graph
    - CPU usage with bar graph
    - Database sizes
    - LLM inference stats
    - Active background tasks
    - Alerts and warnings

    Supports:
    - One-shot display
    - Continuous refresh mode
    - Compact and detailed views
    """

    # ANSI colors
    RESET = "\033[0m"
    BOLD = "\033[1m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    DIM = "\033[2m"

    def __init__(self, resource_monitor: Optional[ResourceMonitor] = None):
        self.monitor = resource_monitor or get_resource_monitor()
        self._running = False
        self._refresh_task: Optional[asyncio.Task] = None
        self.refresh_interval = 2  # seconds
        self.use_colors = True
        self.compact_mode = False

    def _color(self, text: str, color: str) -> str:
        """Apply color if enabled"""
        if self.use_colors:
            return f"{color}{text}{self.RESET}"
        return text

    def _progress_bar(
        self,
        value: float,
        max_value: float = 100,
        width: int = 30,
        warn_threshold: float = 70,
        critical_threshold: float = 90,
    ) -> str:
        """Create a text progress bar"""
        percent = min(value / max_value * 100, 100) if max_value > 0 else 0
        filled = int(width * percent / 100)
        empty = width - filled

        # Choose color based on value
        if percent >= critical_threshold:
            color = self.RED
            fill_char = "█"
        elif percent >= warn_threshold:
            color = self.YELLOW
            fill_char = "█"
        else:
            color = self.GREEN
            fill_char = "█"

        bar = fill_char * filled + "░" * empty

        if self.use_colors:
            return f"{color}{bar}{self.RESET}"
        return bar

    def _format_bytes(self, mb: float) -> str:
        """Format MB to human readable"""
        if mb >= 1024:
            return f"{mb / 1024:.1f} GB"
        return f"{mb:.1f} MB"

    def _get_status_icon(self, status: str) -> str:
        """Get status icon"""
        icons = {
            "healthy": "●" if not self.use_colors else self._color("●", self.GREEN),
            "warning": "●" if not self.use_colors else self._color("●", self.YELLOW),
            "critical": "●" if not self.use_colors else self._color("●", self.RED),
            "unknown": "○" if not self.use_colors else self._color("○", self.DIM),
        }
        return icons.get(status, "○")

    def render_header(self) -> str:
        """Render dashboard header"""
        now = datetime.now().strftime("%H:%M:%S")
        title = "AURA RESOURCE DASHBOARD"

        lines = [
            self._color("╔" + "═" * 60 + "╗", self.CYAN),
            self._color("║", self.CYAN)
            + f" {self._color(title, self.BOLD):^68} "
            + self._color("║", self.CYAN),
            self._color("║", self.CYAN)
            + f" Last Update: {now:<47} "
            + self._color("║", self.CYAN),
            self._color("╠" + "═" * 60 + "╣", self.CYAN),
        ]
        return "\n".join(lines)

    def render_ram_section(self, metrics: Dict[str, Any]) -> str:
        """Render RAM usage section"""
        ram = metrics.get("current", {}).get("ram", {})
        peaks = metrics.get("peaks", {})

        if not ram:
            return (
                self._color("║", self.CYAN)
                + " RAM: No data available"
                + " " * 38
                + self._color("║", self.CYAN)
            )

        used = ram.get("used_mb", 0)
        avail = ram.get("available_mb", 0)
        percent = ram.get("percent", 0)
        peak = peaks.get("ram_mb", 0)
        process = ram.get("process_mb", 0)

        bar = self._progress_bar(percent)

        lines = [
            self._color("║", self.CYAN)
            + f" {self._color('RAM USAGE', self.BOLD):<35}"
            + " " * 26
            + self._color("║", self.CYAN),
            self._color("║", self.CYAN)
            + f"  {bar} {percent:5.1f}%"
            + " " * 14
            + self._color("║", self.CYAN),
            self._color("║", self.CYAN)
            + f"  Used: {self._format_bytes(used):<12} Available: {self._format_bytes(avail):<12}"
            + " " * 6
            + self._color("║", self.CYAN),
            self._color("║", self.CYAN)
            + f"  Peak: {self._format_bytes(peak):<12} Process: {self._format_bytes(process):<12}"
            + " " * 8
            + self._color("║", self.CYAN),
        ]

        # Component breakdown if available
        components = metrics.get("components", {})
        if components and not self.compact_mode:
            lines.append(
                self._color("║", self.CYAN)
                + f"  {self._color('Components:', self.DIM):<50}"
                + " " * 9
                + self._color("║", self.CYAN)
            )
            for name, mem in list(components.items())[:3]:
                lines.append(
                    self._color("║", self.CYAN)
                    + f"    {name}: {self._format_bytes(mem):<40}"
                    + " " * 8
                    + self._color("║", self.CYAN)
                )

        return "\n".join(lines)

    def render_cpu_section(self, metrics: Dict[str, Any]) -> str:
        """Render CPU usage section"""
        cpu = metrics.get("current", {}).get("cpu", {})
        peaks = metrics.get("peaks", {})

        if not cpu:
            return (
                self._color("║", self.CYAN)
                + " CPU: No data available"
                + " " * 38
                + self._color("║", self.CYAN)
            )

        percent = cpu.get("percent", 0)
        process_cpu = cpu.get("process_percent", 0)
        peak = peaks.get("cpu_percent", 0)

        bar = self._progress_bar(percent)

        lines = [
            self._color("╠" + "─" * 60 + "╣", self.CYAN),
            self._color("║", self.CYAN)
            + f" {self._color('CPU USAGE', self.BOLD):<35}"
            + " " * 26
            + self._color("║", self.CYAN),
            self._color("║", self.CYAN)
            + f"  {bar} {percent:5.1f}%"
            + " " * 14
            + self._color("║", self.CYAN),
            self._color("║", self.CYAN)
            + f"  Peak: {peak:5.1f}%         Process: {process_cpu:5.1f}%         "
            + " " * 5
            + self._color("║", self.CYAN),
        ]
        return "\n".join(lines)

    def render_database_section(self, metrics: Dict[str, Any]) -> str:
        """Render database section"""
        dbs = metrics.get("databases", {})
        db_list = metrics.get("current", {}).get("databases", [])

        total_size = dbs.get("total_size_mb", 0)
        count = dbs.get("count", 0)

        lines = [
            self._color("╠" + "─" * 60 + "╣", self.CYAN),
            self._color("║", self.CYAN)
            + f" {self._color('DATABASES', self.BOLD):<35}"
            + " " * 26
            + self._color("║", self.CYAN),
            self._color("║", self.CYAN)
            + f"  Total: {self._format_bytes(total_size):<12} Count: {count:<20}"
            + " " * 6
            + self._color("║", self.CYAN),
        ]

        # List individual databases
        if db_list and not self.compact_mode:
            for db in db_list[:5]:  # Show top 5
                name = db.get("name", "unknown")[:20]
                size = db.get("size_mb", 0)
                lines.append(
                    self._color("║", self.CYAN)
                    + f"    {name:<20} {self._format_bytes(size):<20}"
                    + " " * 8
                    + self._color("║", self.CYAN)
                )

        return "\n".join(lines)

    def render_llm_section(self, metrics: Dict[str, Any]) -> str:
        """Render LLM inference section"""
        llm = metrics.get("llm", {})

        total = llm.get("total_inferences", 0)
        avg_time = llm.get("avg_inference_time_ms", 0)
        recent_avg = llm.get("recent_avg_time_ms", 0)
        tokens_per_sec = llm.get("avg_tokens_per_second", 0)

        lines = [
            self._color("╠" + "─" * 60 + "╣", self.CYAN),
            self._color("║", self.CYAN)
            + f" {self._color('LLM INFERENCE', self.BOLD):<35}"
            + " " * 26
            + self._color("║", self.CYAN),
            self._color("║", self.CYAN)
            + f"  Total Inferences: {total:<40}"
            + " " * 0
            + self._color("║", self.CYAN),
            self._color("║", self.CYAN)
            + f"  Avg Time: {avg_time:.1f}ms     Recent Avg: {recent_avg:.1f}ms      "
            + " " * 5
            + self._color("║", self.CYAN),
            self._color("║", self.CYAN)
            + f"  Tokens/sec: {tokens_per_sec:.1f}                             "
            + " " * 7
            + self._color("║", self.CYAN),
        ]
        return "\n".join(lines)

    def render_tasks_section(self, metrics: Dict[str, Any]) -> str:
        """Render background tasks section"""
        tasks = metrics.get("current", {}).get("tasks", {})

        active = tasks.get("active", 0)
        paused = tasks.get("paused", 0)

        status_icon = self._get_status_icon(
            "healthy" if active == 0 else "warning" if active > 3 else "healthy"
        )

        lines = [
            self._color("╠" + "─" * 60 + "╣", self.CYAN),
            self._color("║", self.CYAN)
            + f" {self._color('BACKGROUND TASKS', self.BOLD):<35}"
            + " " * 26
            + self._color("║", self.CYAN),
            self._color("║", self.CYAN)
            + f"  {status_icon} Active: {active:<10} Paused: {paused:<20}"
            + " " * 6
            + self._color("║", self.CYAN),
        ]
        return "\n".join(lines)

    def render_footer(self) -> str:
        """Render dashboard footer"""
        lines = [
            self._color("╠" + "═" * 60 + "╣", self.CYAN),
            self._color("║", self.CYAN)
            + " Press Ctrl+C to exit | 'r' to reset peaks          "
            + " " * 6
            + self._color("║", self.CYAN),
            self._color("╚" + "═" * 60 + "╝", self.CYAN),
        ]
        return "\n".join(lines)

    def render_compact(self, metrics: Dict[str, Any]) -> str:
        """Render compact single-line status"""
        current = metrics.get("current", {})
        ram = current.get("ram", {})
        cpu = current.get("cpu", {})
        tasks = current.get("tasks", {})

        ram_pct = ram.get("percent", 0)
        cpu_pct = cpu.get("percent", 0)
        active_tasks = tasks.get("active", 0)

        # Color-coded status
        ram_color = (
            self.RED if ram_pct > 90 else self.YELLOW if ram_pct > 70 else self.GREEN
        )
        cpu_color = (
            self.RED if cpu_pct > 85 else self.YELLOW if cpu_pct > 70 else self.GREEN
        )

        status = f"RAM: {self._color(f'{ram_pct:.1f}%', ram_color)} | CPU: {self._color(f'{cpu_pct:.1f}%', cpu_color)} | Tasks: {active_tasks}"

        return status

    def render(self) -> str:
        """Render the full dashboard"""
        metrics = self.monitor.get_summary()

        if self.compact_mode:
            return self.render_compact(metrics)

        sections = [
            self.render_header(),
            self.render_ram_section(metrics),
            self.render_cpu_section(metrics),
            self.render_database_section(metrics),
            self.render_llm_section(metrics),
            self.render_tasks_section(metrics),
            self.render_footer(),
        ]

        return "\n".join(sections)

    def display(self):
        """Display the dashboard once"""
        print(self.render())

    def clear_screen(self):
        """Clear terminal screen"""
        os.system("cls" if os.name == "nt" else "clear")

    async def run_continuous(self, refresh_interval: int = 2):
        """Run dashboard in continuous refresh mode"""
        self.refresh_interval = refresh_interval
        self._running = True

        try:
            while self._running:
                self.clear_screen()
                self.display()
                await asyncio.sleep(self.refresh_interval)
        except KeyboardInterrupt:
            self._running = False
            print("\nDashboard stopped.")

    def stop(self):
        """Stop continuous mode"""
        self._running = False


def print_dashboard():
    """Print dashboard once (convenience function)"""
    dashboard = CLIDashboard()
    dashboard.display()


async def run_dashboard(refresh_interval: int = 2, compact: bool = False):
    """Run the dashboard (convenience function)"""
    monitor = get_resource_monitor()

    # Start monitor if not running
    if not monitor._running:
        await monitor.start()

    dashboard = CLIDashboard(monitor)
    dashboard.compact_mode = compact

    await dashboard.run_continuous(refresh_interval)


# ==============================================================================
# CLI ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AURA Resource Dashboard")
    parser.add_argument("--once", action="store_true", help="Display once and exit")
    parser.add_argument("--compact", action="store_true", help="Compact mode")
    parser.add_argument(
        "--refresh", type=int, default=2, help="Refresh interval in seconds"
    )
    parser.add_argument("--no-color", action="store_true", help="Disable colors")

    args = parser.parse_args()

    monitor = get_resource_monitor()
    dashboard = CLIDashboard(monitor)
    dashboard.use_colors = not args.no_color
    dashboard.compact_mode = args.compact

    if args.once:
        # One-shot mode
        asyncio.run(monitor._collect_snapshot())
        dashboard.display()
    else:
        # Continuous mode
        async def main():
            await monitor.start()
            await dashboard.run_continuous(args.refresh)

        asyncio.run(main())
