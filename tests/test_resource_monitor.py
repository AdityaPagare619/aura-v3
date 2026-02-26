"""
Tests for AURA v3 Resource Monitor and CLI Dashboard
"""

import pytest
import asyncio
import tempfile
import sqlite3
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.resource_monitor import (
    ResourceMonitor,
    get_resource_monitor,
    RAMMetrics,
    CPUMetrics,
    LLMMetrics,
    DatabaseMetrics,
    TaskMetrics,
    ResourceSnapshot,
    MetricType,
)
from src.utils.cli_dashboard import CLIDashboard


class TestResourceMonitor:
    """Tests for ResourceMonitor class"""

    @pytest.fixture
    def monitor(self, tmp_path):
        """Create a resource monitor for testing"""
        return ResourceMonitor(
            sample_interval=1,
            history_size=10,
            data_dir=str(tmp_path),
        )

    @pytest.fixture
    def temp_db(self, tmp_path):
        """Create a temporary SQLite database"""
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test_table (id INTEGER PRIMARY KEY, name TEXT)")
        cursor.execute("INSERT INTO test_table (name) VALUES ('test1')")
        cursor.execute("INSERT INTO test_table (name) VALUES ('test2')")
        conn.commit()
        conn.close()
        return db_path

    def test_monitor_creation(self, monitor):
        """Test resource monitor can be created"""
        assert monitor is not None
        assert monitor.sample_interval == 1
        assert monitor.history_size == 10
        assert not monitor._running

    @pytest.mark.asyncio
    async def test_start_stop(self, monitor):
        """Test start and stop functionality"""
        await monitor.start()
        assert monitor._running
        assert monitor._monitor_task is not None

        await monitor.stop()
        assert not monitor._running

    @pytest.mark.asyncio
    async def test_collect_ram_metrics(self, monitor):
        """Test RAM metrics collection"""
        ram = await monitor._collect_ram_metrics()

        assert isinstance(ram, RAMMetrics)
        assert ram.total_mb > 0
        assert ram.available_mb > 0
        assert ram.used_mb > 0
        assert 0 <= ram.percent <= 100
        assert ram.process_mb >= 0

    @pytest.mark.asyncio
    async def test_collect_cpu_metrics(self, monitor):
        """Test CPU metrics collection"""
        cpu = await monitor._collect_cpu_metrics()

        assert isinstance(cpu, CPUMetrics)
        assert 0 <= cpu.percent <= 100
        assert isinstance(cpu.per_cpu, list)
        assert cpu.process_percent >= 0

    @pytest.mark.asyncio
    async def test_collect_database_metrics(self, monitor, temp_db, tmp_path):
        """Test database metrics collection"""
        # Set data dir to tmp_path with the test db
        monitor.data_dir = tmp_path

        databases = await monitor._collect_database_metrics()

        assert len(databases) >= 1
        db_metric = databases[0]
        assert isinstance(db_metric, DatabaseMetrics)
        assert db_metric.name == "test"
        assert db_metric.size_mb > 0
        assert db_metric.tables_count >= 1

    @pytest.mark.asyncio
    async def test_collect_snapshot(self, monitor):
        """Test full snapshot collection"""
        snapshot = await monitor._collect_snapshot()

        assert isinstance(snapshot, ResourceSnapshot)
        assert snapshot.ram is not None
        assert snapshot.cpu is not None
        assert isinstance(snapshot.databases, list)

    def test_record_llm_inference(self, monitor):
        """Test LLM inference recording"""
        monitor.record_llm_inference(
            model_name="test-model",
            inference_time_ms=1500.0,
            tokens_generated=100,
            prompt_tokens=50,
            memory_used_mb=256.0,
        )

        assert monitor._total_inferences == 1
        assert len(monitor._llm_history) == 1

        metrics = monitor._llm_history[0]
        assert metrics.model_name == "test-model"
        assert metrics.inference_time_ms == 1500.0
        assert metrics.tokens_generated == 100
        assert metrics.tokens_per_second == pytest.approx(66.67, rel=0.1)

    def test_llm_stats(self, monitor):
        """Test LLM statistics calculation"""
        # Record multiple inferences
        for i in range(5):
            monitor.record_llm_inference(
                model_name="test-model",
                inference_time_ms=1000.0 + i * 100,
                tokens_generated=50 + i * 10,
            )

        stats = monitor.get_llm_stats()

        assert stats["total_inferences"] == 5
        assert stats["avg_inference_time_ms"] > 0
        assert stats["total_tokens_generated"] > 0

    def test_component_memory_tracking(self, monitor):
        """Test component memory registration"""
        monitor.register_component_memory("memory_system", 128.5)
        monitor.register_component_memory("llm", 512.0)

        breakdown = monitor.get_component_breakdown()

        assert breakdown["memory_system"] == 128.5
        assert breakdown["llm"] == 512.0

        monitor.unregister_component("llm")
        breakdown = monitor.get_component_breakdown()
        assert "llm" not in breakdown

    def test_peak_tracking(self, monitor):
        """Test peak values tracking"""
        # Initial peaks should be 0
        peaks = monitor.get_peaks()
        assert peaks["ram_mb"] == 0.0
        assert peaks["cpu_percent"] == 0.0

        # Collect metrics to set peaks
        asyncio.run(monitor._collect_ram_metrics())
        asyncio.run(monitor._collect_cpu_metrics())

        peaks = monitor.get_peaks()
        assert peaks["ram_mb"] > 0

        # Test reset
        monitor.reset_peaks()
        peaks = monitor.get_peaks()
        assert peaks["ram_mb"] == 0.0

    @pytest.mark.asyncio
    async def test_alerts(self, monitor):
        """Test alert system"""
        alerts_received = []

        def alert_callback(alert):
            alerts_received.append(alert)

        monitor.register_alert_callback(alert_callback)

        # Set very low thresholds to trigger alerts
        monitor.ram_warning_threshold = 1.0  # Will always trigger
        monitor.cpu_warning_threshold = 1.0

        snapshot = await monitor._collect_snapshot()
        await monitor._check_alerts(snapshot)

        # Should have received alerts
        assert len(alerts_received) >= 1

    def test_get_history(self, monitor):
        """Test history retrieval"""
        # Record some data
        for i in range(5):
            monitor.record_llm_inference(
                model_name="test",
                inference_time_ms=100 * i,
                tokens_generated=10,
            )

        history = monitor.get_history(MetricType.LLM, limit=3)

        assert len(history) == 3
        assert all("timestamp" in h for h in history)
        assert all("model" in h for h in history)

    @pytest.mark.asyncio
    async def test_get_current_metrics(self, monitor):
        """Test getting current metrics"""
        # Collect a snapshot first and ensure it's stored
        snapshot = await monitor._collect_snapshot()
        monitor._snapshot_history.append(snapshot)  # Ensure it's in history

        metrics = monitor.get_current_metrics()

        assert "timestamp" in metrics
        assert "ram" in metrics
        assert "cpu" in metrics
        assert "databases" in metrics

    @pytest.mark.asyncio
    async def test_get_summary(self, monitor):
        """Test getting full summary"""
        # Record some data
        monitor.record_llm_inference("test", 1000, 50)
        monitor.register_component_memory("test_comp", 64.0)
        await monitor._collect_snapshot()

        summary = monitor.get_summary()

        assert "current" in summary
        assert "peaks" in summary
        assert "llm" in summary
        assert "databases" in summary
        assert "components" in summary

    def test_singleton_instance(self):
        """Test singleton pattern"""
        monitor1 = get_resource_monitor()
        monitor2 = get_resource_monitor()

        assert monitor1 is monitor2


class TestCLIDashboard:
    """Tests for CLI Dashboard"""

    @pytest.fixture
    def mock_monitor(self):
        """Create a mock resource monitor"""
        monitor = Mock(spec=ResourceMonitor)
        monitor.get_summary.return_value = {
            "current": {
                "ram": {
                    "used_mb": 4096.0,
                    "available_mb": 8192.0,
                    "percent": 50.0,
                    "peak_mb": 5000.0,
                    "process_mb": 256.0,
                },
                "cpu": {
                    "percent": 30.0,
                    "process_percent": 5.0,
                    "peak_percent": 60.0,
                },
                "databases": [
                    {"name": "test_db", "size_mb": 10.5},
                    {"name": "memory_db", "size_mb": 5.2},
                ],
                "tasks": {
                    "active": 3,
                    "paused": 1,
                },
            },
            "peaks": {
                "ram_mb": 5000.0,
                "cpu_percent": 60.0,
                "process_ram_mb": 300.0,
            },
            "llm": {
                "total_inferences": 10,
                "avg_inference_time_ms": 1500.0,
                "recent_avg_time_ms": 1200.0,
                "avg_tokens_per_second": 40.0,
            },
            "databases": {
                "total_size_mb": 15.7,
                "count": 2,
            },
            "components": {
                "memory": 128.0,
                "llm": 512.0,
            },
            "history_size": 50,
        }
        return monitor

    @pytest.fixture
    def dashboard(self, mock_monitor):
        """Create a dashboard with mock monitor"""
        return CLIDashboard(mock_monitor)

    def test_dashboard_creation(self, dashboard):
        """Test dashboard can be created"""
        assert dashboard is not None
        assert dashboard.refresh_interval == 2
        assert dashboard.use_colors is True

    def test_progress_bar(self, dashboard):
        """Test progress bar generation"""
        # Test different percentage levels
        bar_low = dashboard._progress_bar(30, 100, width=10)
        bar_med = dashboard._progress_bar(75, 100, width=10)
        bar_high = dashboard._progress_bar(95, 100, width=10)

        assert (
            len(
                bar_low.replace("\033[91m", "")
                .replace("\033[92m", "")
                .replace("\033[93m", "")
                .replace("\033[0m", "")
            )
            == 10
        )

    def test_format_bytes(self, dashboard):
        """Test byte formatting"""
        assert dashboard._format_bytes(500) == "500.0 MB"
        assert dashboard._format_bytes(1500) == "1.5 GB"
        assert dashboard._format_bytes(2048) == "2.0 GB"

    def test_render_header(self, dashboard):
        """Test header rendering"""
        header = dashboard.render_header()

        assert "AURA RESOURCE DASHBOARD" in header
        assert "╔" in header
        assert "╗" in header

    def test_render_ram_section(self, dashboard, mock_monitor):
        """Test RAM section rendering"""
        metrics = mock_monitor.get_summary()
        ram_section = dashboard.render_ram_section(metrics)

        assert "RAM USAGE" in ram_section
        assert "50.0%" in ram_section or "50%" in ram_section

    def test_render_cpu_section(self, dashboard, mock_monitor):
        """Test CPU section rendering"""
        metrics = mock_monitor.get_summary()
        cpu_section = dashboard.render_cpu_section(metrics)

        assert "CPU USAGE" in cpu_section
        assert "30.0%" in cpu_section or "30%" in cpu_section

    def test_render_database_section(self, dashboard, mock_monitor):
        """Test database section rendering"""
        metrics = mock_monitor.get_summary()
        db_section = dashboard.render_database_section(metrics)

        assert "DATABASES" in db_section
        assert "15.7" in db_section  # Total size

    def test_render_llm_section(self, dashboard, mock_monitor):
        """Test LLM section rendering"""
        metrics = mock_monitor.get_summary()
        llm_section = dashboard.render_llm_section(metrics)

        assert "LLM INFERENCE" in llm_section
        assert "10" in llm_section  # Total inferences

    def test_render_tasks_section(self, dashboard, mock_monitor):
        """Test tasks section rendering"""
        metrics = mock_monitor.get_summary()
        tasks_section = dashboard.render_tasks_section(metrics)

        assert "BACKGROUND TASKS" in tasks_section
        assert "Active: 3" in tasks_section

    def test_render_full(self, dashboard):
        """Test full dashboard rendering"""
        output = dashboard.render()

        # Check all sections are present
        assert "AURA RESOURCE DASHBOARD" in output
        assert "RAM USAGE" in output
        assert "CPU USAGE" in output
        assert "DATABASES" in output
        assert "LLM INFERENCE" in output
        assert "BACKGROUND TASKS" in output

    def test_compact_mode(self, dashboard, mock_monitor):
        """Test compact mode rendering"""
        dashboard.compact_mode = True

        output = dashboard.render()

        # Compact mode should be a single line
        assert "RAM:" in output
        assert "CPU:" in output
        assert "Tasks:" in output

    def test_no_color_mode(self, dashboard):
        """Test rendering without colors"""
        dashboard.use_colors = False

        output = dashboard.render()

        # Should not contain ANSI escape codes
        assert "\033[" not in output


class TestResourceMonitorIntegration:
    """Integration tests for resource monitor with other components"""

    @pytest.fixture
    def monitor(self, tmp_path):
        """Create a resource monitor for testing"""
        return ResourceMonitor(
            sample_interval=1,
            history_size=10,
            data_dir=str(tmp_path),
        )

    @pytest.mark.asyncio
    async def test_background_manager_integration(self, monitor):
        """Test integration with background manager"""
        # Create a mock background manager
        mock_manager = Mock()
        mock_manager.get_status.return_value = {
            "running": True,
            "tasks": {
                "total": 5,
                "running": 3,
                "paused": 1,
            },
        }
        mock_manager._background_tasks = {
            "task1": Mock(
                id="task1",
                name="Test Task 1",
                is_running=True,
                is_paused=False,
                progress=0.5,
            ),
            "task2": Mock(
                id="task2",
                name="Test Task 2",
                is_running=True,
                is_paused=False,
                progress=0.3,
            ),
        }

        monitor.set_background_manager(mock_manager)

        task_metrics = await monitor._collect_task_metrics()

        assert task_metrics is not None
        assert task_metrics.active_count == 3
        assert task_metrics.paused_count == 1
        assert len(task_metrics.task_details) == 2

    @pytest.mark.asyncio
    async def test_full_monitoring_cycle(self, monitor, tmp_path):
        """Test a complete monitoring cycle"""
        # Create test database
        db_path = tmp_path / "test_cycle.db"
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE data (id INTEGER)")
        conn.commit()
        conn.close()

        # Record some LLM inferences
        monitor.record_llm_inference("model1", 1000, 50)
        monitor.record_llm_inference("model1", 1500, 75)

        # Register component memory
        monitor.register_component_memory("test_component", 128.0)

        # Start monitoring
        await monitor.start()

        # Wait for one collection cycle
        await asyncio.sleep(1.5)

        # Stop monitoring
        await monitor.stop()

        # Verify data was collected
        summary = monitor.get_summary()

        assert summary["llm"]["total_inferences"] == 2
        assert "test_component" in summary["components"]
        assert summary["history_size"] >= 1


class TestHealthMonitorIntegration:
    """Test health monitor integration with resource monitor"""

    @pytest.mark.asyncio
    async def test_get_health_and_resources(self):
        """Test combined health and resources function"""
        from src.utils.health_monitor import get_health_and_resources

        result = get_health_and_resources()

        assert "status" in result
        assert "resources" in result
        assert result["status"] in ["healthy", "degraded", "critical", "unknown"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
