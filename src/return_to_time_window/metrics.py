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
