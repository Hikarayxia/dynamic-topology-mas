"""只读记录指令、位移和边界状态，辅助区分运动错误与策略本身的退化。"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np

from core import ActionType


def motion_snapshot(environment, actions, positions_before, info) -> dict:
    """返回一个时间步的新字典，不向策略提供额外信息、不修改环境。

    参数是执行后的环境、实际动作、执行前位置副本及本步 info。分别保留执行器
    指令速度、位移平均速度和步末状态速度；步末刚离线的节点仍可在本步发生位移。
    """

    states, commands, active_positions = [], [], []
    upper = np.array([environment.config.width, environment.config.height])
    dt = environment.config.dt
    for agent_id, agent in environment.agents.items():
        before = positions_before[agent_id]
        after = agent.state.position
        action = actions.get(agent_id)
        command_velocity = None
        raw_action = None
        if action is not None:
            raw_action = (np.zeros(2) if action.action_type is ActionType.NOOP
                          else np.asarray(action.value, dtype=float))
            command_velocity = raw_action / max(1.0, np.linalg.norm(raw_action)) * agent.attributes.max_speed
            commands.append(command_velocity)
        if agent.state.active:
            active_positions.append(after)
        states.append(dict(
            agent_id=str(agent_id), position_before=before.tolist(), position_after=after.tolist(),
            action=raw_action.tolist() if raw_action is not None else None,
            command_velocity=command_velocity.tolist() if command_velocity is not None else None,
            displacement_velocity=((after - before) / dt).tolist(),
            state_velocity=agent.state.velocity.tolist(), had_action=action is not None,
            active=bool(agent.state.active), yaw=float(agent.state.yaw), energy=float(agent.state.energy),
            on_boundary=bool(np.any(np.isclose(after, 0, atol=1e-9, rtol=0)
                                       | np.isclose(after, upper, atol=1e-9, rtol=0))),
            boundary_clipped=bool(command_velocity is not None and not np.allclose(
                after, before + command_velocity * dt, atol=1e-9, rtol=0)),
        ))
    command_matrix = np.asarray(commands)
    spread = (float(np.sqrt(np.mean(np.sum((command_matrix - command_matrix.mean(axis=0)) ** 2, axis=1))))
              if commands else None)
    distances = [float(np.linalg.norm(a - b)) for a, b in combinations(active_positions, 2)]
    return dict(step=int(info["step"]), timestamp=float(info["time"]), dt=dt,
                map_size=upper.tolist(), collision_count=int(info["collision_count"]),
                team_reward=float(info["team_reward"]), detection_ratio=float(info["detection_ratio"]),
                topology_version=environment.topology.version,
                min_active_pair_distance=min(distances) if distances else None,
                command_spread_rms=spread, agents=states)


def summarize_motion(rows: list[dict]) -> dict:
    """汇总实际轨迹；无可用样本的比例或事件时间保持 None。

    比较所有步末在线节点的状态速度与本步位移，包括本步尚未收到动作的恢复
    节点。离线节点不作为可用物理状态样本；不假设故障会自动清空其历史速度。
    """

    if not rows:
        raise ValueError("运动轨迹不能为空")
    moved = [state for row in rows for state in row["agents"] if state["had_action"]]
    checked = [state for row in rows for state in row["agents"] if state["active"]]
    mismatches = sum(not np.allclose(state["state_velocity"], state["displacement_velocity"],
                                    atol=1e-8, rtol=0) for state in checked)
    collision_times = [row["timestamp"] for row in rows if row["collision_count"] > 0]
    boundary_times = [row["timestamp"] for row in rows if any(state["had_action"] and state["on_boundary"]
                                                           for state in row["agents"])]
    spreads = [row["command_spread_rms"] for row in rows if row["command_spread_rms"] is not None]
    return dict(
        steps=len(rows), action_samples=len(moved), velocity_checked_samples=len(checked),
        velocity_mismatch_samples=int(mismatches),
        velocity_mismatch_fraction=mismatches / len(checked) if checked else None,
        boundary_sample_fraction=sum(state["on_boundary"] for state in moved) / len(moved) if moved else None,
        clipped_sample_fraction=sum(state["boundary_clipped"] for state in moved) / len(moved) if moved else None,
        stationary_boundary_fraction=sum(state["on_boundary"] and np.linalg.norm(state["displacement_velocity"]) < 1e-9
                                         for state in moved) / len(moved) if moved else None,
        first_collision_time=min(collision_times) if collision_times else None,
        first_boundary_time=min(boundary_times) if boundary_times else None,
        mean_command_spread_rms=float(np.mean(spreads)) if spreads else None,
    )


def plot_motion(rows: list[dict], output_path: Path) -> Path:
    """把已记录的单回合轨迹画为四幅图，返回文件路径；不平滑或补造轨迹。"""

    from .plotting import plt

    if not rows:
        raise ValueError("绘图需要非空运动轨迹")
    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    trajectories: dict[str, list] = {}
    for row in rows:
        for state in row["agents"]:
            path = trajectories.setdefault(state["agent_id"], [state["position_before"]])
            path.append(state["position_after"])
    for agent_id, points in trajectories.items():
        points = np.asarray(points)
        axes[0, 0].plot(points[:, 0], points[:, 1], label=agent_id)
        axes[0, 0].scatter(*points[0], marker="o", s=20)
        axes[0, 0].scatter(*points[-1], marker="x", s=30)
    axes[0, 0].set(xlim=(0, rows[-1]["map_size"][0]), ylim=(0, rows[-1]["map_size"][1]),
                   title="Paths (circle=start, cross=end)", xlabel="x (m)", ylabel="y (m)")
    axes[0, 0].legend(fontsize=7)
    times = [row["timestamp"] for row in rows]
    distances = [row["min_active_pair_distance"] if row["min_active_pair_distance"] is not None else np.nan for row in rows]
    axes[0, 1].plot(times, distances)
    collisions = [index for index, row in enumerate(rows) if row["collision_count"] > 0]
    axes[0, 1].scatter([times[index] for index in collisions], [distances[index] for index in collisions],
                       color="red", s=12, label="collision step")
    axes[0, 1].set(title="Minimum online pair distance", xlabel="time (s)", ylabel="distance (m)")
    axes[0, 1].legend(fontsize=7)
    axes[1, 0].plot(times, [sum(state["on_boundary"] for state in row["agents"] if state["had_action"])
                          / max(1, sum(state["had_action"] for state in row["agents"])) for row in rows])
    axes[1, 0].set(title="Boundary fraction of applied-action agents", xlabel="time (s)", ylim=(-0.02, 1.02))
    axes[1, 1].plot(times, [row["command_spread_rms"] for row in rows])
    axes[1, 1].set(title="Within-step command velocity spread", xlabel="time (s)", ylabel="RMS (m/s)")
    for axis in axes.flat:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
    return output_path
