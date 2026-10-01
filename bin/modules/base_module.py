"""Shared base class for body, face, and hand inference modules."""

import time
from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseModule(ABC):
    """Common lifecycle, streaming switches, and lightweight scheduling."""

    def __init__(self, name: str, config: Dict, streaming_config: Dict):
        self.name = name
        self._config = config
        self._streaming_config = streaming_config
        self._enabled = config.get("enabled", True)
        self._streaming_enabled = streaming_config.get("streaming_enabled", True)
        self._initialized = False

        self._detect_count = 0
        self._total_count = 0
        self._last_process_time = 0.0
        self._skip_count = 0
        self._last_result = None
        self._last_result_fresh = False
        self._last_process_started_at = 0.0
        self._next_process_at = 0.0
        self._max_fps = 0.0
        self._min_process_interval = 0.0
        self._reuse_last_result = True
        self._configure_scheduler(config)

    @property
    def enabled(self) -> bool:
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        old = self._enabled
        self._enabled = value
        if old and not value:
            self._last_result_fresh = False
            self.on_disabled()
        elif not old and value:
            self._last_process_started_at = 0.0
            self._next_process_at = 0.0
            self._last_result_fresh = False
            self.on_enabled()

    @property
    def streaming_enabled(self) -> bool:
        return self._streaming_enabled

    @streaming_enabled.setter
    def streaming_enabled(self, value: bool):
        self._streaming_enabled = value

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def detect_rate(self) -> float:
        if self._total_count == 0:
            return 0.0
        return (self._detect_count / self._total_count) * 100

    @property
    def last_process_time_ms(self) -> float:
        return self._last_process_time * 1000

    @property
    def last_result(self):
        return self._last_result

    @property
    def last_result_fresh(self) -> bool:
        return self._last_result_fresh

    @property
    def max_fps(self) -> float:
        return self._max_fps

    def _configure_scheduler(self, config: Dict):
        processing = config.get("processing", {})
        if not isinstance(processing, dict):
            processing = {}
        rate_limit_enabled = config.get(
            "module_rate_limit_enabled",
            processing.get("module_rate_limit_enabled", True),
        )
        raw_max_fps = config.get(
            "max_fps",
            processing.get(f"{self.name}_max_fps", processing.get("max_fps", 0)),
        )
        try:
            max_fps = float(raw_max_fps or 0)
        except (TypeError, ValueError):
            max_fps = 0.0
        if not rate_limit_enabled:
            max_fps = 0.0
        self._max_fps = max(0.0, max_fps)
        self._min_process_interval = 1.0 / self._max_fps if self._max_fps > 0 else 0.0
        self._next_process_at = 0.0
        self._reuse_last_result = bool(config.get(
            "reuse_last_result",
            processing.get("reuse_last_result", True),
        ))

    def should_process(self, now: float = None) -> bool:
        if self._max_fps <= 0 or self._min_process_interval <= 0:
            return True
        now = time.perf_counter() if now is None else now
        return self._next_process_at <= 0.0 or now >= self._next_process_at

    def _advance_scheduler(self, now: float):
        if self._max_fps <= 0 or self._min_process_interval <= 0:
            self._next_process_at = 0.0
            return

        interval = self._min_process_interval
        if self._next_process_at <= 0.0:
            self._next_process_at = now + interval
            return

        self._next_process_at += interval
        if self._next_process_at <= now:
            missed = int((now - self._next_process_at) // interval) + 1
            self._next_process_at += missed * interval

    def initialize(self) -> bool:
        if self._initialized:
            return True
        if not self._enabled:
            print(f"[{self.name}] module disabled, skip initialization")
            return False
        try:
            if self._do_initialize():
                self._initialized = True
                self._last_process_started_at = 0.0
                self._next_process_at = 0.0
                self._last_result_fresh = False
                print(f"[{self.name}] initialized")
                return True
            print(f"[{self.name}] initialization failed")
            return False
        except Exception as e:
            print(f"[{self.name}] initialization exception: {e}")
            return False

    def process(self, frame) -> Optional[Dict]:
        if not self._enabled or not self._initialized:
            self._last_result_fresh = False
            return None

        now = time.perf_counter()
        if not self.should_process(now):
            self._skip_count += 1
            self._last_result_fresh = False
            return self._last_result if self._reuse_last_result else None

        self._last_process_started_at = now
        self._advance_scheduler(now)
        self._total_count += 1
        start = now

        try:
            result = self._do_process(frame)
        except Exception as e:
            if self._total_count <= 3:
                print(f"[{self.name}] process exception: {e}")
            self._last_process_time = time.perf_counter() - start
            self._last_result = None
            self._last_result_fresh = False
            return None

        self._last_process_time = time.perf_counter() - start
        if result is not None:
            self._detect_count += 1
        self._last_result = result
        self._last_result_fresh = result is not None
        return result

    def cleanup(self):
        if self._initialized:
            self._do_cleanup()
            self._initialized = False
            self._last_result = None
            self._last_result_fresh = False
            print(f"[{self.name}] cleaned up")

    def on_enabled(self):
        pass

    def on_disabled(self):
        pass

    def update_config(self, config: Dict, streaming_config: Dict):
        new_enabled = config.get("enabled", True)
        new_streaming_enabled = streaming_config.get("streaming_enabled", True)
        self._config = config
        self._streaming_config = streaming_config
        self._configure_scheduler(config)
        self.streaming_enabled = new_streaming_enabled
        self.enabled = new_enabled

    def get_status(self) -> Dict:
        return {
            "name": self.name,
            "enabled": self._enabled,
            "streaming_enabled": self._streaming_enabled,
            "initialized": self._initialized,
            "detect_rate": self.detect_rate,
            "process_time_ms": self.last_process_time_ms,
            "max_fps": self._max_fps,
            "skipped_frames": self._skip_count,
            "last_result_fresh": self._last_result_fresh,
        }

    @abstractmethod
    def _do_initialize(self) -> bool:
        ...

    @abstractmethod
    def _do_process(self, frame) -> Optional[Dict]:
        ...

    @abstractmethod
    def _do_cleanup(self):
        ...
