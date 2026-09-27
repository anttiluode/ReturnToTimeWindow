#!/usr/bin/env python3
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from return_to_time_window.experiments import run_g5

tr = run_g5(100)["trace"]
t = np.asarray(tr["time"], dtype=float)
internal = np.asarray(tr["internal"], dtype=float)
public = np.asarray(tr["public"], dtype=float)
inhibition = np.asarray(tr["inhibition"], dtype=float)
context_gain = np.asarray(tr["context_gain"], dtype=float)
veto_gain = np.asarray(tr["veto_gain"], dtype=float)
publication_mask = np.asarray(tr["publication_mask"], dtype=float)
names = tr["token_names"]
D, H = names.index("D"), names.index("H")

fig, axes = plt.subplots(4, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
dt = t[1] - t[0] if len(t) > 1 else 0.005

axes[0].imshow(internal, aspect="auto", origin="lower",
               extent=[t[0], t[-1] + dt, -0.5, len(names) - 0.5])
axes[0].set_yticks(range(len(names)))
axes[0].set_yticklabels(names)
axes[0].set_ylabel("internal")
axes[0].set_title("ReturnToTimeWindow v0 — representative combined trial (seed 100)")

axes[1].plot(t, inhibition, label="inhibition")
axes[1].plot(t, context_gain[:, D] - 1.0, label="context D")
axes[1].plot(t, context_gain[:, H] - 1.0, label="context H")
axes[1].plot(t, 1.0 - veto_gain[:, D], label="veto D")
axes[1].set_ylabel("control")
axes[1].legend(loc="upper right", ncol=4, fontsize=8)

axes[2].plot(t, publication_mask, label="publication mask")
axes[2].plot(t, public.max(axis=0), label="public max")
axes[2].plot(t, internal.max(axis=0), label="internal max")
axes[2].set_ylabel("publish")
axes[2].legend(loc="upper right", ncol=3, fontsize=8)

axes[3].plot(t, internal[D], label="D branch")
axes[3].plot(t, internal[H], label="H branch")
axes[3].set_ylabel("branch")
axes[3].set_xlabel("time (s)")
axes[3].legend(loc="upper right", fontsize=8)

out = ROOT / "results" / "time_window.png"
fig.savefig(out, dpi=150)
plt.close(fig)
print(out)
