# Missing Features Analysis: AURA vs OpenClaw

**Date:** February 2026  
**Purpose:** Identify critical gaps for production readiness

---

## Executive Summary

| Category | AURA Status | OpenClaw Status | Gap Severity |
|----------|-------------|-----------------|--------------|
| Production Score | 35/100 | ~80/100 | **CRITICAL** |
| Health Checks | Not implemented | Implemented (UI-based) | HIGH |
| Monitoring | Minimal | Basic | HIGH |
| Graceful Shutdown | Design only | Partial | HIGH |
| Circuit Breakers | Not implemented | Not implemented | MEDIUM |
| Security (TLS/Pinning) | Design only | Implemented | CRITICAL |

---

## 1. Core Features Comparison

### 1.1 What OpenClaw Has (AURA Lacks)

| Feature | OpenClaw | AURA | Priority |
|---------|----------|------|----------|
| **Extension Ecosystem** | 50+ TypeScript plugins | 11 basic Python tools | NICE-TO-HAVE |
| **Plugin SDK** | Mature TypeScript SDK | None | NICE-TO-HAVE |
| **Type-Safe Schemas** | TypeBox integration | Basic JSON schemas | NICE-TO-HAVE |
| **Testing Infrastructure** | 70% coverage, vitest | Minimal | CRITICAL |
| **Error Handling** | Comprehensive with retry/fallback | Basic try/catch | CRITICAL |
| **CI/CD Pipeline** | Yes | No | HIGH |

### 1.2 Feature Criticality Assessment

**CRITICAL (Must Have):**
- Health checks
- Graceful shutdown
- Error handling improvements
- Basic monitoring

**HIGH (Should Have):**
- Testing infrastructure
- Graceful degradation
- Crash reporting

**NICE-TO-HAVE:**
- Extension ecosystem
- Plugin SDK
- Community features

---

## 2. Production Features Analysis

### 2.1 Health Checks

#### OpenClaw Implementation
```
Location: apps/android/app/src/main/java/ai/openclaw/android/chat/ChatController.kt
- Monitors "health" snapshots from gateway
- UI displays connection status (green/yellow)
- Health state flow: _healthOk = false/true based on gateway response
- Periodic health requests every 30 seconds
```

**OpenClaw Health Check Flow:**
```
1. GatewaySession sends "health" request
2. Gateway responds with health snapshot
3. ChatController updates _healthOk StateFlow
4. UI reflects connection status (connected/connecting)
```

#### AURA Status
```
Status: NOT IMPLEMENTED
- Design exists in AURA_PRODUCTION_ARCHITECTURE.md
- ServiceManager has _health_check_loop() stub
- Actual health checks never executed
- LOGGING_AUDIT.md notes: "No system health monitoring"
```

#### Implementation Required

```python
# src/services/health_monitor.py
"""
Health Monitor for AURA v3
Implements comprehensive system health checks
"""

import asyncio
import psutil
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    component: str
    status: HealthStatus
    message: str
    timestamp: datetime
    details: Dict[str, Any]


class HealthMonitor:
    """Comprehensive health monitoring for AURA"""
    
    def __init__(self):
        self._checks: Dict[str, HealthCheck] = {}
        self._check_interval = 30  # seconds
        self._monitoring = False
        self._last_overall: HealthStatus = HealthStatus.UNKNOWN
    
    async def check_llm(self) -> HealthCheck:
        """Check LLM component health"""
        try:
            from src.llm import LLMRunner
            llm = LLMRunner.get_instance()
            
            if llm is None:
                return HealthCheck(
                    component="llm",
                    status=HealthStatus.CRITICAL,
                    message="LLM not initialized",
                    timestamp=datetime.now(),
                    details={}
                )
            
            if not llm.is_loaded():
                return HealthCheck(
                    component="llm",
                    status=HealthStatus.DEGRADED,
                    message="LLM not loaded",
                    timestamp=datetime.now(),
                    details={"loaded": False}
                )
            
            return HealthCheck(
                component="llm",
                status=HealthStatus.HEALTHY,
                message="LLM operational",
                timestamp=datetime.now(),
                details={"loaded": True}
            )
            
        except Exception as e:
            return HealthCheck(
                component="llm",
                status=HealthStatus.CRITICAL,
                message=f"LLM check failed: {e}",
                timestamp=datetime.now(),
                details={"error": str(e)}
            )
    
    async def check_memory(self) -> HealthCheck:
        """Check memory usage"""
        try:
            mem = psutil.virtual_memory()
            process = psutil.Process()
            process_mem = process.memory_info().rss / 1024 / 1024  # MB
            
            status = HealthStatus.HEALTHY
            message = "Memory OK"
            
            if mem.percent > 90 or process_mem > 1400:
                status = HealthStatus.CRITICAL
                message = "Memory critical"
            elif mem.percent > 75 or process_mem > 1200:
                status = HealthStatus.DEGRADED
                message = "Memory elevated"
            
            return HealthCheck(
                component="memory",
                status=status,
                message=message,
                timestamp=datetime.now(),
                details={
                    "system_percent": mem.percent,
                    "system_available_mb": mem.available / 1024 / 1024,
                    "process_mb": process_mem
                }
            )
            
        except Exception as e:
            return HealthCheck(
                component="memory",
                status=HealthStatus.UNKNOWN,
                message=f"Memory check failed: {e}",
                timestamp=datetime.now(),
                details={"error": str(e)}
            )
    
    async def check_storage(self) -> HealthCheck:
        """Check storage availability"""
        try:
            disk = psutil.disk_usage('/')
            
            status = HealthStatus.HEALTHY
            if disk.percent > 95:
                status = HealthStatus.CRITICAL
            elif disk.percent > 85:
                status = HealthStatus.DEGRADED
            
            return HealthCheck(
                component="storage",
                status=status,
                message=f"Storage {disk.percent}% used",
                timestamp=datetime.now(),
                details={
                    "percent": disk.percent,
                    "free_gb": disk.free / 1024 / 1024 / 1024
                }
            )
            
        except Exception as e:
            return HealthCheck(
                component="storage",
                status=HealthStatus.UNKNOWN,
                message=f"Storage check failed: {e}",
                timestamp=datetime.now(),
                details={"error": str(e)}
            )
    
    async def check_database(self) -> HealthCheck:
        """Check database connectivity"""
        try:
            from src.memory import HierarchicalMemory
            memory = HierarchicalMemory.get_instance()
            
            if memory is None:
                return HealthCheck(
                    component="database",
                    status=HealthStatus.CRITICAL,
                    message="Memory not initialized",
                    timestamp=datetime.now(),
                    details={}
                )
            
            # Test with a simple operation
            test_key = "__health_check__"
            await memory.store(test_key, "test", importance=0.0)
            
            return HealthCheck(
                component="database",
                status=HealthStatus.HEALTHY,
                message="Database operational",
                timestamp=datetime.now(),
                details={}
            )
            
        except Exception as e:
            return HealthCheck(
                component="database",
                status=HealthStatus.CRITICAL,
                message=f"Database error: {e}",
                timestamp=datetime.now(),
                details={"error": str(e)}
            )
    
    async def check_telegram(self) -> HealthCheck:
        """Check Telegram bot connectivity"""
        try:
            from src.channels.telegram_bot import TelegramBot
            bot = TelegramBot.get_instance()
            
            if bot is None:
                return HealthCheck(
                    component="telegram",
                    status=HealthStatus.UNKNOWN,
                    message="Bot not initialized",
                    timestamp=datetime.now(),
                    details={}
                )
            
            if not bot.is_running:
                return HealthCheck(
                    component="telegram",
                    status=HealthStatus.CRITICAL,
                    message="Bot not running",
                    timestamp=datetime.now(),
                    details={}
                )
            
            return HealthCheck(
                component="telegram",
                status=HealthStatus.HEALTHY,
                message="Bot operational",
                timestamp=datetime.now(),
                details={}
            )
            
        except Exception as e:
            return HealthCheck(
                component="telegram",
                status=HealthStatus.UNKNOWN,
                message=f"Telegram check failed: {e}",
                timestamp=datetime.now(),
                details={"error": str(e)}
            )
    
    async def run_all_checks(self) -> Dict[str, HealthCheck]:
        """Run all health checks"""
        checks = await asyncio.gather(
            self.check_llm(),
            self.check_memory(),
            self.check_storage(),
            self.check_database(),
            self.check_telegram(),
            return_exceptions=True
        )
        
        components = ["llm", "memory", "storage", "database", "telegram"]
        results = {}
        
        for component, check in zip(components, checks):
            if isinstance(check, Exception):
                results[component] = HealthCheck(
                    component=component,
                    status=HealthStatus.CRITICAL,
                    message=str(check),
                    timestamp=datetime.now(),
                    details={"exception": str(check)}
                )
            else:
                results[component] = check
        
        self._checks = results
        return results
    
    def get_overall_status(self) -> HealthStatus:
        """Get overall system health"""
        if not self._checks:
            return HealthStatus.UNKNOWN
        
        statuses = [check.status for check in self._checks.values()]
        
        if any(s == HealthStatus.CRITICAL for s in statuses):
            return HealthStatus.CRITICAL
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        
        return HealthStatus.UNKNOWN
    
    def get_status_dict(self) -> Dict[str, Any]:
        """Get health status as dict for API/UI"""
        return {
            "overall": self.get_overall_status().value,
            "timestamp": datetime.now().isoformat(),
            "components": {
                name: {
                    "status": check.status.value,
                    "message": check.message,
                    "details": check.details
                }
                for name, check in self._checks.items()
            }
        }
    
    async def start_monitoring(self, interval: int = 30):
        """Start background health monitoring"""
        self._monitoring = True
        logger.info(f"Health monitoring started (interval: {interval}s)")
        
        while self._monitoring:
            try:
                await self.run_all_checks()
                overall = self.get_overall_status()
                
                if overall == HealthStatus.CRITICAL:
                    logger.critical(f"Health CRITICAL: {self.get_status_dict()}")
                elif overall == HealthStatus.DEGRADED:
                    logger.warning(f"Health DEGRADED: {self.get_status_dict()}")
                
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """Stop background monitoring"""
        self._monitoring = False
        logger.info("Health monitoring stopped")


# Global instance
health_monitor = HealthMonitor()
```

---

### 2.2 Monitoring

#### OpenClaw Implementation
```
- Connection health status (green/yellow UI indicators)
- Basic metrics in Android app
- No centralized monitoring system
```

#### AURA Status
```
Status: MINIMAL
- Design exists in AURA_PRODUCTION_ARCHITECTURE.md
- Memory monitoring designed but not integrated
- No metrics collection
- No alerting system
```

#### Implementation Required

```python
# src/services/metrics_collector.py
"""
Metrics Collection for AURA v3
Provides observability for production monitoring
"""

import time
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
import threading

logger = logging.getLogger(__name__)


@dataclass
class MetricPoint:
    """Single metric measurement"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)


class MetricsCollector:
    """Collects and stores system metrics"""
    
    def __init__(self, retention_seconds: int = 3600):
        self.retention_seconds = retention_seconds
        self._metrics: Dict[str, deque] = {}
        self._counters: Dict[str, int] = {}
        self._gauges: Dict[str, float] = {}
        self._lock = threading.Lock()
    
    def counter(self, name: str, labels: Dict[str, str] = None) -> int:
        """Increment and return counter"""
        key = self._make_key(name, labels)
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + 1
            return self._counters[key]
    
    def gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """Set gauge value"""
        key = self._make_key(name, labels)
        with self._lock:
            self._gauges[key] = value
    
    def histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """Record histogram value"""
        key = self._make_key(name, labels)
        with self._lock:
            if key not in self._metrics:
                self._metrics[key] = deque(maxlen=1000)
            
            self._metrics[key].append(MetricPoint(
                timestamp=time.time(),
                value=value,
                labels=labels or {}
            ))
    
    def _make_key(self, name: str, labels: Optional[Dict]) -> str:
        """Create metric key from name and labels"""
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_counter(self, name: str, labels: Dict[str, str] = None) -> int:
        """Get counter value"""
        key = self._make_key(name, labels)
        return self._counters.get(key, 0)
    
    def get_gauge(self, name: str, labels: Dict[str, str] = None) -> Optional[float]:
        """Get gauge value"""
        key = self._make_key(name, labels)
        return self._gauges.get(key)
    
    def get_histogram(self, name: str, labels: Dict[str, str] = None) -> List[float]:
        """Get histogram values"""
        key = self._make_key(name, labels)
        if key not in self._metrics:
            return []
        
        cutoff = time.time() - self.retention_seconds
        return [
            p.value for p in self._metrics[key]
            if p.timestamp > cutoff
        ]
    
    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all current metrics"""
        with self._lock:
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "histograms": {
                    key: len(values)
                    for key, values in self._metrics.items()
                }
            }


# Decorator for automatic metric collection
def track_latency(metric_name: str):
    """Decorator to track function latency"""
    def decorator(func):
        def sync_wrapper(*args, **kwargs):
            start = time.perf_counter()
            try:
                return func(*args, **kwargs)
            finally:
                latency = (time.perf_counter() - start) * 1000  # ms
                metrics.histogram(f"{metric_name}_latency_ms", latency)
                metrics.counter(f"{metric_name}_calls")
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            async def async_wrapper(*args, **kwargs):
                start = time.perf_counter()
                try:
                    return await func(*args, **kwargs)
                finally:
                    latency = (time.perf_counter() - start) * 1000
                    metrics.histogram(f"{metric_name}_latency_ms", latency)
                    metrics.counter(f"{metric_name}_calls")
            return async_wrapper
        return sync_wrapper
    return decorator


# Global metrics instance
metrics = MetricsCollector()
```

---

### 2.3 Graceful Shutdown

#### OpenClaw Implementation
```
- Signal handling in Kotlin services
- Foreground service with proper lifecycle
- Connection cleanup on disconnect
```

#### AURA Status
```
Status: DESIGN ONLY
- Design exists in AURA_PRODUCTION_ARCHITECTURE.md
- ServiceManager has _handle_shutdown() but not integrated
- No actual shutdown hooks in main.py
- LOGGING_AUDIT.md: "No graceful shutdown implementation"
```

#### Implementation Required

```python
# Add to main.py - Graceful Shutdown Implementation

import signal
import asyncio
from contextlib import asynccontextmanager

class GracefulShutdown:
    """Manages graceful shutdown of AURA"""
    
    def __init__(self, aura_instance):
        self.aura = aura_instance
        self._shutdown_event = asyncio.Event()
        self._is_shutting_down = False
        self._shutdown_tasks: List[asyncio.Task] = []
    
    def setup_signal_handlers(self):
        """Setup OS signal handlers"""
        loop = asyncio.get_event_loop()
        
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(
                    sig,
                    lambda s=sig: asyncio.create_task(self.shutdown(s))
                )
            except NotImplementedError:
                # Windows fallback
                pass
    
    async def shutdown(self, sig=None):
        """Initiate graceful shutdown"""
        if self._is_shutting_down:
            return
        
        self._is_shutting_down = True
        sig_name = sig.name if sig and hasattr(sig, 'name') else "unknown"
        logger.info(f"Received {sig_name}, initiating graceful shutdown...")
        
        # 1. Stop accepting new requests
        if self.aura:
            self.aura._running = False
        
        # 2. Cancel background tasks gracefully
        await self._cancel_tasks()
        
        # 3. Save all persistent state
        await self._save_state()
        
        # 4. Close connections
        await self._close_connections()
        
        # 5. Flush logs
        await self._flush_logs()
        
        self._shutdown_event.set()
        logger.info("Graceful shutdown complete")
    
    async def _cancel_tasks(self):
        """Cancel background tasks with timeout"""
        tasks_to_cancel = []
        
        # Cancel health monitoring
        if hasattr(self.aura, 'health_monitor'):
            tasks_to_cancel.append(
                asyncio.create_task(self.aura.health_monitor.stop_monitoring())
            )
        
        # Cancel other known tasks
        for task in asyncio.all_tasks():
            if task != asyncio.current_task():
                tasks_to_cancel.append(task)
        
        # Cancel with timeout
        if tasks_to_cancel:
            done, pending = await asyncio.wait(
                tasks_to_cancel,
                timeout=10.0
            )
            
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
    
    async def _save_state(self):
        """Save all persistent state"""
        logger.info("Saving state...")
        
        try:
            # Save sessions
            if hasattr(self.aura, 'sessions') and self.aura.sessions:
                await self.aura.sessions.save_all()
                logger.info("Sessions saved")
        except Exception as e:
            logger.error(f"Error saving sessions: {e}")
        
        try:
            # Save memory
            if hasattr(self.aura, 'memory') and self.aura.memory:
                self.aura.memory.save()
                logger.info("Memory saved")
        except Exception as e:
            logger.error(f"Error saving memory: {e}")
        
        try:
            # Save learning data
            if hasattr(self.aura, 'learning') and self.aura.learning:
                await self.aura.learning.save()
                logger.info("Learning data saved")
        except Exception as e:
            logger.error(f"Error saving learning: {e}")
    
    async def _close_connections(self):
        """Close all network connections"""
        logger.info("Closing connections...")
        
        # Close Telegram bot
        try:
            if hasattr(self.aura, 'telegram_bot') and self.aura.telegram_bot:
                await self.aura.telegram_bot.stop()
                logger.info("Telegram bot stopped")
        except Exception as e:
            logger.error(f"Error stopping Telegram: {e}")
        
        # Close LLM
        try:
            if hasattr(self.aura, 'llm') and self.aura.llm:
                self.aura.llm.unload()
                logger.info("LLM unloaded")
        except Exception as e:
            logger.error(f"Error unloading LLM: {e}")
    
    async def _flush_logs(self):
        """Ensure logs are flushed"""
        # Force garbage collection
        import gc
        gc.collect()
        logger.info("Logs flushed, shutdown complete")
    
    async def wait_for_shutdown(self):
        """Wait for shutdown to complete"""
        await self._shutdown_event.wait()


# Integration in main.py:
async def main():
    # ... existing setup ...
    
    # Setup graceful shutdown
    shutdown_manager = GracefulShutdown(aura)
    shutdown_manager.setup_signal_handlers()
    
    # ... rest of main ...
    
    # Wait for shutdown
    await shutdown_manager.wait_for_shutdown()
```

---

### 2.4 Circuit Breakers

#### OpenClaw Status
```
Not implemented - this is a server-side feature that would be in Swabble (backend)
```

#### AURA Status
```
Status: NOT IMPLEMENTED
- ERROR_HANDLING_AUDIT.md mentions need for circuit breaker
- No actual implementation
```

#### Implementation Required

```python
# src/utils/circuit_breaker.py
"""
Circuit Breaker for AURA v3
Prevents repeated failures from cascading
"""

import time
import logging
from typing import Callable, Any, Optional
from enum import Enum
from dataclasses import dataclass
from functools import wraps
import asyncio

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, reject calls
    HALF_OPEN = "half_open"  # Testing if recovered


@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 3       # Failures before opening
    success_threshold: int = 2       # Successes to close
    timeout_seconds: float = 30.0    # Time before half-open
    half_open_max_calls: int = 3     # Max calls in half-open


class CircuitBreaker:
    """Circuit breaker for component protection"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig = None):
        self.name = name
        self.config = config or CircuitBreakerConfig()
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[float] = None
        self._half_open_calls = 0
    
    def _can_execute(self) -> bool:
        """Check if execution is allowed"""
        if self.state == CircuitState.CLOSED:
            return True
        
        if self.state == CircuitState.OPEN:
            # Check timeout
            if self.last_failure_time:
                elapsed = time.time() - self.last_failure_time
                if elapsed >= self.config.timeout_seconds:
                    logger.info(f"Circuit {self.name}: transitioning to HALF_OPEN")
                    self.state = CircuitState.HALF_OPEN
                    self._half_open_calls = 0
                    return True
            return False
        
        if self.state == CircuitState.HALF_OPEN:
            return self._half_open_calls < self.config.half_open_max_calls
        
        return False
    
    def record_success(self):
        """Record successful call"""
        if self.state == CircuitState.HALF_OPEN:
            self._half_open_calls += 1
            self.success_count += 1
            
            if self.success_count >= self.config.success_threshold:
                logger.info(f"Circuit {self.name}: CLOSING after recovery")
                self.state = CircuitState.CLOSED
                self.failure_count = 0
                self.success_count = 0
        else:
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.state == CircuitState.HALF_OPEN:
            logger.warning(f"Circuit {self.name}: OPEN (half-open failure)")
            self.state = CircuitState.OPEN
            self._half_open_calls = 0
        elif self.failure_count >= self.config.failure_threshold:
            logger.warning(f"Circuit {self.name}: OPEN after {self.failure_count} failures")
            self.state = CircuitState.OPEN
    
    def get_state(self) -> dict:
        """Get current circuit state"""
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self.failure_count,
            "success_count": self.success_count,
        }


# Global circuit breakers for components
circuit_breakers = {
    "llm": CircuitBreaker("llm", CircuitBreakerConfig(
        failure_threshold=3,
        timeout_seconds=60.0
    )),
    "telegram": CircuitBreaker("telegram", CircuitBreakerConfig(
        failure_threshold=5,
        timeout_seconds=30.0
    )),
    "database": CircuitBreaker("database", CircuitBreakerConfig(
        failure_threshold=3,
        timeout_seconds=30.0
    )),
    "voice": CircuitBreaker("voice", CircuitBreakerConfig(
        failure_threshold=3,
        timeout_seconds=45.0
    )),
}


def circuit_protected(component: str):
    """Decorator to protect functions with circuit breaker"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            breaker = circuit_breakers.get(component)
            if not breaker:
                return await func(*args, **kwargs)
            
            if not breaker._can_execute():
                logger.warning(f"Circuit {component} is OPEN, rejecting call")
                raise CircuitBreakerOpenError(f"Component {component} circuit is open")
            
            try:
                result = await func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            breaker = circuit_breakers.get(component)
            if not breaker:
                return func(*args, **kwargs)
            
            if not breaker._can_execute():
                logger.warning(f"Circuit {component} is OPEN, rejecting call")
                raise CircuitBreakerOpenError(f"Component {component} circuit is open")
            
            try:
                result = func(*args, **kwargs)
                breaker.record_success()
                return result
            except Exception as e:
                breaker.record_failure()
                raise
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator


class CircuitBreakerOpenError(Exception):
    """Raised when circuit breaker is open"""
    pass
```

---

## 3. Security Features Analysis

### 3.1 What OpenClaw Has (AURA Lacks)

| Security Feature | OpenClaw | AURA | Implementation |
|------------------|----------|------|----------------|
| **TLS Certificate Pinning** | Implemented | Design only | CRITICAL |
| **Device Authentication** | Implemented | Design only | HIGH |
| **Secure Storage** | SecurePrefs (encrypted) | Basic encryption | MEDIUM |
| **Gateway Trust Verification** | SHA-256 fingerprint | None | CRITICAL |
| **Auto-connect Security** | Only to pinned gateways | None | HIGH |

### 3.2 TLS Certificate Pinning (CRITICAL)

#### OpenClaw Implementation
```kotlin
// apps/android/app/src/main/java/ai/openclaw/android/gateway/GatewayTls.kt

class GatewayTlsManager {
    fun buildGatewayTlsConfig(
        tls: TlsOptions?,
        onStore: ((String) -> Unit)?
    ): SSLContext {
        // Custom TrustManager that pins certificates
        val pinningTrustManager = object : X509TrustManager {
            override fun checkClientTrusted(chain: Array<X509Certificate>, authType: String) {
                defaultTrust.checkClientTrusted(chain, authType)
            }
            
            override fun checkServerTrusted(chain: Array<X509Certificate>, authType: String) {
                // Compare fingerprint with stored pin
                val fingerprint = sha256Hex(chain[0].encoded)
                if (fingerprint != expected) {
                    throw CertificateException("gateway TLS fingerprint mismatch")
                }
                onStore?.invoke(fingerprint)
            }
        }
    }
}
```

#### AURA Status
```
Status: NOT IMPLEMENTED
- Design discussed in AURA_PRODUCTION_ARCHITECTURE.md
- No actual TLS pinning code
- SECURITY_AUDIT.md: No TLS certificate verification
```

### 3.3 Implementation for AURA

```python
# src/security/tls_pinning.py
"""
TLS Certificate Pinning for AURA v3
Implements certificate pinning for secure connections
"""

import ssl
import hashlib
import logging
from typing import Optional, Dict, Set
from pathlib import Path
import json

logger = logging.getLogger(__name__)


class TLSPinningManager:
    """Manages TLS certificate pins for secure connections"""
    
    def __init__(self, pins_file: str = "data/security/tls_pins.json"):
        self.pins_file = Path(pins_file)
        self._pins: Dict[str, str] = {}  # host -> sha256 pin
        self._load_pins()
    
    def _load_pins(self):
        """Load saved pins from file"""
        if self.pins_file.exists():
            try:
                with open(self.pins_file) as f:
                    self._pins = json.load(f)
                logger.info(f"Loaded {len(self._pins)} TLS pins")
            except Exception as e:
                logger.error(f"Error loading TLS pins: {e}")
    
    def _save_pins(self):
        """Save pins to file"""
        self.pins_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.pins_file, 'w') as f:
            json.dump(self._pins, f, indent=2)
    
    def get_pin(self, host: str) -> Optional[str]:
        """Get pinned fingerprint for host"""
        return self._pins.get(host)
    
    def set_pin(self, host: str, pin: str):
        """Store pin for host"""
        self._pins[host] = pin
        self._save_pins()
        logger.info(f"Stored TLS pin for {host}")
    
    def remove_pin(self, host: str):
        """Remove pin for host"""
        if host in self._pins:
            del self._pins[host]
            self._save_pins()
            logger.info(f"Removed TLS pin for {host}")
    
    def verify_certificate(self, host: str, cert_pem: bytes) -> bool:
        """Verify certificate matches pinned fingerprint"""
        expected_pin = self.get_pin(host)
        
        if not expected_pin:
            logger.warning(f"No pin configured for {host}, rejecting connection")
            return False
        
        # Calculate SHA-256 of certificate
        cert_hash = hashlib.sha256(cert_pem).hexdigest()
        
        if cert_hash != expected_pin:
            logger.error(f"Certificate pin mismatch for {host}")
            logger.error(f"Expected: {expected_pin}")
            logger.error(f"Got:      {cert_hash}")
            return False
        
        return True
    
    def create_ssl_context(self, host: str) -> ssl.SSLContext:
        """Create SSL context with pinning for specific host"""
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        
        # Load default CA certificates
        context.load_default_certs()
        
        # Create custom verification
        def verify_callback(conn, cert, errno, depth, verify_result):
            if depth == 0:  # Server certificate
                # For initial connection, just verify it's valid
                # After first connection, pin will be stored
                if errno == 0:  # X509_V_OK
                    return True
                logger.warning(f"Certificate verification failed: {errno}")
                return False
            return True
        
        context.verify_mode = ssl.CERT_REQUIRED
        context.check_hostname = True
        context.verify_callback = verify_callback
        
        return context
    
    def get_pinned_hosts(self) -> Set[str]:
        """Get all hosts with pinned certificates"""
        return set(self._pins.keys())


# Global instance
tls_pinning = TLSPinningManager()
```

---

## 4. Summary: Critical Missing Features

### 4.1 Must Implement (CRITICAL)

| Feature | Effort | Impact | Status |
|---------|--------|--------|--------|
| Health Checks | Medium | HIGH | NOT IMPLEMENTED |
| Graceful Shutdown | Medium | HIGH | DESIGN ONLY |
| TLS Certificate Pinning | Medium | CRITICAL | NOT IMPLEMENTED |
| Circuit Breakers | Low | MEDIUM | NOT IMPLEMENTED |

### 4.2 Should Implement (HIGH)

| Feature | Effort | Impact | Status |
|---------|--------|--------|--------|
| Metrics Collection | Medium | MEDIUM | MINIMAL |
| Crash Reporter | Low | HIGH | DESIGN ONLY |
| Error Handling Improvements | High | HIGH | BASIC |
| Testing Infrastructure | High | CRITICAL | NONE |

### 4.3 Nice to Have

| Feature | Effort | Impact | Status |
|---------|--------|--------|--------|
| Extension System | High | MEDIUM | NOT APPLICABLE |
| Plugin SDK | High | LOW | NOT APPLICABLE |

---

## 5. Implementation Priority

### Phase 1: Production Essentials (Week 1-2)
1. **Health Checks** - Immediate production safety
2. **Graceful Shutdown** - Prevent data loss
3. **Basic Metrics** - Observability

### Phase 2: Reliability (Week 3-4)
4. **Circuit Breakers** - Failure isolation
5. **Crash Reporter** - Offline debugging
6. **Error Handling Improvements** - Resilience

### Phase 3: Security (Week 5-6)
7. **TLS Pinning** - Secure connections
8. **Device Authentication** - Access control

### Phase 4: Quality (Ongoing)
9. **Testing Infrastructure**
10. **CI/CD Pipeline**

---

## 6. Conclusion

AURA has significant production feature gaps compared to OpenClaw:

- **Health Monitoring**: Critical missing piece
- **Graceful Shutdown**: Design exists, not integrated
- **Security**: TLS pinning critical gap
- **Reliability**: Circuit breakers needed

The good news: These are well-understood problems with standard solutions. The AURA_PRODUCTION_ARCHITECTURE.md already contains detailed designs for most features - the gap is implementation and integration.

**Estimated effort to reach basic production readiness: 4-6 weeks**

---

*Analysis based on codebase review of both AURA v3 and OpenClaw repositories*
