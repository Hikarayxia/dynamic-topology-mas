"""只依赖本机位置与感知半径的离散覆盖记忆，不读取目标或队友轨迹。"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


class LocalCoverageMemory:
    """记录历史感知圆到达过的网格中心，并提供固定长度的局部摘要。

    参数 width、height 是已知地图边界；v1 固定使用 4 米网格。这里的覆盖是
    网格中心采样近似，不是精确面积，更不是未知目标的真值地图。实例不共享
    可变状态，节点增多不会改变单个观测的维度。
    """

    CELL_SIZE = 4.0
    FEATURE_DIM = 18
    PROFILE = "local_coverage_v1"

    def __init__(self, width: float, height: float) -> None:
        if any(not math.isfinite(value) or value <= 0 for value in (width, height)):
            raise ValueError("覆盖记忆地图边界必须为正有限数值")
        self.width, self.height = float(width), float(height)
        self.columns = math.ceil(width / self.CELL_SIZE)
        self.rows = math.ceil(height / self.CELL_SIZE)
        x_left = np.arange(self.columns) * self.CELL_SIZE
        y_bottom = np.arange(self.rows) * self.CELL_SIZE
        # 末端不足整格时使用实际边界内的中心，避免生成地图之外的虚拟覆盖点。
        self._x_centers = (x_left + np.minimum(x_left + self.CELL_SIZE, width)) / 2
        self._y_centers = (y_bottom + np.minimum(y_bottom + self.CELL_SIZE, height)) / 2
        self._covered = np.zeros((self.rows, self.columns), dtype=bool)

    def _position(self, position: Sequence[float]) -> np.ndarray:
        """验证本机二维位置，返回数组；越界不能被静默裁剪为另一处覆盖记录。"""

        value = np.asarray(position, dtype=float)
        if (value.shape != (2,) or not np.isfinite(value).all()
                or not 0 <= value[0] <= self.width or not 0 <= value[1] <= self.height):
            raise ValueError("覆盖记忆位置必须是地图内的二维有限向量")
        return value

    def observe(self, position: Sequence[float], sensor_range: float) -> None:
        """累积本机本次感知范围；参数为位置与半径，无返回值，不融合队友地图。"""

        value = self._position(position)
        if not math.isfinite(sensor_range) or sensor_range < 0:
            raise ValueError("覆盖记忆感知半径必须为非负有限数值")
        squared = ((self._x_centers[None, :] - value[0]) ** 2
                   + (self._y_centers[:, None] - value[1]) ** 2)
        self._covered |= squared <= sensor_range ** 2 + 1e-12

    def features(self, position: Sequence[float]) -> np.ndarray:
        """返回 18 维新数组：附近 3×3 已覆盖标志，再接 3×3 地图有效掩码。

        两块均按 dy=-1..1、dx=-1..1 排列，中心在索引 4。独立掩码将地图外区域
        与地图内未搜索区域区分；查询不会更新记忆，避免重复生成观测产生副作用。
        """

        value = self._position(position)
        column = min(int(value[0] // self.CELL_SIZE), self.columns - 1)
        row = min(int(value[1] // self.CELL_SIZE), self.rows - 1)
        covered, valid = np.zeros(9), np.zeros(9)
        for index, (dy, dx) in enumerate((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1)):
            if 0 <= row + dy < self.rows and 0 <= column + dx < self.columns:
                valid[index] = 1
                covered[index] = self._covered[row + dy, column + dx]
        return np.concatenate((covered, valid))
