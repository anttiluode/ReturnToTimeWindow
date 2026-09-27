from __future__ import annotations
import numpy as np


def peak_order(activity: np.ndarray, threshold: float = 0.2) -> list[int]:
    activity = np.asarray(activity)
    active = [i for i in range(activity.shape[0]) if activity[i].max(initial=0.0) > threshold]
    return sorted(active, key=lambda i: int(np.argmax(activity[i])))


def in_order_reach(order: list[int], expected: list[int]) -> int:
    if not expected or not order:
        return 0
    pos = 0
    for unit in order:
        if pos < len(expected) and unit == expected[pos]:
            pos += 1
        elif pos > 0 and unit in expected[:pos]:
            continue
        elif pos > 0:
            break
    return pos


def front_position(activity: np.ndarray, expected: list[int], threshold: float = 0.2) -> np.ndarray:
    arr = np.asarray(activity)
    front = np.full(arr.shape[1], -1, dtype=int)
    index = {u: i for i, u in enumerate(expected)}
    for t in range(arr.shape[1]):
        active = np.flatnonzero(arr[:, t] > threshold)
        if len(active):
            locations = [index[u] for u in active if u in index]
            if locations:
                front[t] = max(locations)
    return front


def secondary_wave_events(
    activity: np.ndarray,
    expected: list[int],
    threshold: float = 0.2,
    behind_by: int = 5,
) -> int:
    """Count threshold crossings that ignite well behind the furthest front reached so far."""
    arr = np.asarray(activity)
    pos = {u: i for i, u in enumerate(expected)}
    prev = np.zeros(arr.shape[0], dtype=bool)
    furthest = -1
    count = 0
    for t in range(arr.shape[1]):
        now = arr[:, t] > threshold
        rising = np.flatnonzero(now & ~prev)
        for unit in rising:
            p = pos.get(int(unit))
            if p is not None and furthest >= 0 and p <= furthest - behind_by:
                count += 1
        active_positions = [pos[int(u)] for u in np.flatnonzero(now) if int(u) in pos]
        if active_positions:
            furthest = max(furthest, max(active_positions))
        prev = now
    return count
