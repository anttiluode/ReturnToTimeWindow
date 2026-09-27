from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Repertoire:
    tokens: tuple[str, ...]
    token_to_id: dict[str, int]
    sequences: tuple[tuple[int, ...], ...]


def encode_repertoire(sequences: list[list[str]]) -> Repertoire:
    token_to_id: dict[str, int] = {}
    tokens: list[str] = []
    coded: list[tuple[int, ...]] = []
    for seq in sequences:
        ids = []
        for token in seq:
            if token not in token_to_id:
                token_to_id[token] = len(tokens)
                tokens.append(token)
            ids.append(token_to_id[token])
        coded.append(tuple(ids))
    return Repertoire(tuple(tokens), token_to_id, tuple(coded))


def compressed_window_spikes(
    repertoire: Repertoire,
    sequence_index: int,
    rng: np.random.Generator,
    *,
    windows: int = 20,
    period_s: float = 0.125,
    item_dt_s: float = 0.008,
    jitter_sd_s: float = 0.001,
    shuffle_within_window: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    seq = np.asarray(repertoire.sequences[sequence_index], dtype=int)
    times: list[np.ndarray] = []
    units: list[np.ndarray] = []
    eps = np.finfo(float).eps
    for w in range(windows):
        base = w * period_s
        offsets = 0.010 + np.arange(len(seq), dtype=float) * item_dt_s
        offsets += rng.normal(0.0, jitter_sd_s, len(seq))
        offsets = np.clip(offsets, eps, period_s - eps)
        t = base + offsets
        if shuffle_within_window:
            t = t[rng.permutation(len(t))]
        times.append(t)
        units.append(seq.copy())
    return np.concatenate(times), np.concatenate(units)


def stdp(
    spike_times: np.ndarray,
    unit_ids: np.ndarray,
    n_units: int,
    *,
    tau_s: float = 0.020,
    W: np.ndarray | None = None,
) -> np.ndarray:
    """Balanced pairwise exponential STDP, adapted from Rytmi's rytmi_core.stdp."""
    st = np.asarray(spike_times, dtype=float)
    sid = np.asarray(unit_ids, dtype=int)
    order = np.argsort(st)
    st, sid = st[order], sid[order]
    out = np.zeros((n_units, n_units), dtype=float) if W is None else np.array(W, copy=True, dtype=float)
    for k in range(len(st)):
        lo = np.searchsorted(st, st[k] - 5 * tau_s)
        hi = np.searchsorted(st, st[k] + 5 * tau_s)
        for m in range(lo, hi):
            if m == k or sid[m] == sid[k]:
                continue
            d = st[k] - st[m]
            out[sid[k], sid[m]] += np.exp(-d / tau_s) if d > 0 else -np.exp(d / tau_s)
    return out


def learn_repertoire(
    repertoire: Repertoire,
    rng: np.random.Generator,
    *,
    windows_per_sequence: int = 20,
    shuffle_within_window: bool = False,
) -> np.ndarray:
    W = np.zeros((len(repertoire.tokens), len(repertoire.tokens)), dtype=float)
    for i in range(len(repertoire.sequences)):
        st, sid = compressed_window_spikes(
            repertoire,
            i,
            rng,
            windows=windows_per_sequence,
            shuffle_within_window=shuffle_within_window,
        )
        W = stdp(st, sid, len(repertoire.tokens), W=W)
    W = np.maximum(W, 0.0)
    mx = float(W.max()) if W.size else 0.0
    if mx > 0.0:
        W /= mx
    return W
