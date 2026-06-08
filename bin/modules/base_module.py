"""
追踪模块基类

每个模块独立管理: 推理器初始化、推理执行、推流发送、开关控制。
主循环只需将摄像头帧分发给各活跃模块。
"""

import time
from abc import ABC, abstractmethod
from typing import Dict, Optional


class BaseModule(ABC):
    """追踪模块基类

    子类需实现:
    - _do_initialize() -> bool: 初始化推理器和推流器
    - _do_process(frame) -> Optional[Dict]: 执行推理，返回结果数据
    - _do_cleanup(): 释放资源

    属性:
    - enabled: 模块是否启用 (推理+推流总开关)
    - streaming_enabled: 推流是否启用
    - name: 模块名称
    """

    def __init__(self, name: str, config: Dict, streaming_config: Dict):
        self.name = name
        self._config = config
        self._streaming_config = streaming_config
        self._enabled = config.get("enabled", True)
        self._streaming_enabled = streaming_config.get("streaming_enabled", True)
        self._initialized = False

        # 统计
        self._detect_count = 0
        self._total_count = 0
        self._last_process_time = 0.0

    @property
    def enabled(self) -> bool:
        """模块是否启用 (推理总开关)"""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        old = self._enabled
        self._enabled = value
        if old and not value:
            self.on_disabled()
        elif not old and value:
            self.on_enabled()

    @property
    def streaming_enabled(self) -> bool:
        """推流是否启用"""
        return self._streaming_enabled

    @streaming_enabled.setter
    def streaming_enabled(self, value: bool):
        self._streaming_enabled = value

    @property
    def initialized(self) -> bool:
        return self._initialized

    @property
    def detect_rate(self) -> float:
        """检测率 (0~100)"""
        if self._total_count == 0:
            return 0.0
        return (self._detect_count / self._total_count) * 100

    @property
    def last_process_time_ms(self) -> float:
        """最近一次推理耗时 (毫秒)"""
        return self._last_process_time * 1000

    def initialize(self) -> bool:
        """初始化模块 (推理器 + 推流器)"""
        if self._initialized:
            return True
        if not self._enabled:
            print(f"[{self.name}] 模块未启用，跳过初始化")
            return False
        try:
            if self._do_initialize():
                self._initialized = True
                print(f"[{self.name}] 初始化成功")
                return True
            else:
                print(f"[{self.name}] 初始化失败")
                return False
        except Exception as e:
            print(f"[{self.name}] 初始化异常: {e}")
            return False

    def process(self, frame) -> Optional[Dict]:
        """执行推理并推流

        Args:
            frame: 摄像头帧 (numpy array)

        Returns:
            推理结果数据，或 None
        """
        if not self._enabled or not self._initialized:
            return None

        self._total_count += 1
        start = time.perf_counter()

        try:
            result = self._do_process(frame)
        except Exception as e:
            if self._total_count <= 3:
                print(f"[{self.name}] 推理异常: {e}")
            return None

        self._last_process_time = time.perf_counter() - start

        if result is not None:
            self._detect_count += 1

        return result

    def cleanup(self):
        """释放资源"""
        if self._initialized:
            self._do_cleanup()
            self._initialized = False
            print(f"[{self.name}] 已释放资源")

    def on_enabled(self):
        """模块启用时的回调"""
        pass

    def on_disabled(self):
        """模块禁用时的回调"""
        pass

    def update_config(self, config: Dict, streaming_config: Dict):
        """热更新配置"""
        self._config = config
        self._streaming_config = streaming_config
        self._enabled = config.get("enabled", True)
        self._streaming_enabled = streaming_config.get("streaming_enabled", True)

    def get_status(self) -> Dict:
        """获取模块状态"""
        return {
            "name": self.name,
            "enabled": self._enabled,
            "streaming_enabled": self._streaming_enabled,
            "initialized": self._initialized,
            "detect_rate": self.detect_rate,
            "process_time_ms": self.last_process_time_ms,
        }

    @abstractmethod
    def _do_initialize(self) -> bool:
        """子类实现: 初始化推理器和推流器"""
        ...

    @abstractmethod
    def _do_process(self, frame) -> Optional[Dict]:
        """子类实现: 执行推理，返回结果数据"""
        ...

    @abstractmethod
    def _do_cleanup(self):
        """子类实现: 释放资源"""
        ...
