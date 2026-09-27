# ReturnToTimeWindow

**A return to the neural-time-window question.** This repository asks whether a learned, self-running sequence can be controlled by several time-local mechanisms that are *causally different*: when recurrent activity may advance, which continuation is favored, whether a continuation is vetoed, and whether an already-computed internal state is published.

This is deliberately **not** a GRU/RNN benchmark. The sequence substrate is the small travelling-wave replay mechanism inherited from the TATWATASW/Rytmi line: compressed within-window order is written by pairwise STDP, then a recurrent rate network with adaptation replays the learned chain after a short cue.

The frozen v0 result is **mixed**. Three things survived cleanly, and three stronger mechanistic distinctions did not.

![time-window mechanism](results/time_window.png)

## The machine

The v0 circuit keeps these signals separate in the simulator and logs every one of them:

```text
learned recurrent sequence  -> internal state
rhythmic inhibition         -> when recurrence is admitted
context gain                -> which branch is more susceptible
veto gain                   -> whether selected recurrent drive is allowed now
publication mask            -> whether internal activity becomes public
```

The important separation is between `internal` and `public`. Publication can be zero while recurrent dynamics continue. Nothing in the publication mask feeds back into the internal state.

The biological names are hypotheses and analogies, not identifications. In particular, v0 does **not** claim that SST/Martinotti cells are literal sequence-stoppers, that chandelier cells are proven publication gates, or that this is a model of thought or consciousness.

## Frozen gates

Pilot seeds 0–3 were run first. Numerical thresholds were then derived mechanically from the formulas in the approved implementation plan and committed in `results/frozen_thresholds.json` **before** canonical seeds 100–111 were run. The thresholds were not changed after seeing canonical results.

| gate | question | canonical result |
|---|---|---:|
| **G0** | can a learned sequence continue after the external cue ends, while zero-recurrence and timing-shuffled controls fail? | **PASS 12/12** |
| **G1** | does rhythmic dead time beat matched tonic inhibition on the frozen full-replay criterion, without worse secondary-wave behavior? | **FAIL 0/12** |
| **G2** | does multiplicative context select a branch without supplying content in a regime the matched additive replacement cannot also achieve? | **FAIL 0/12** |
| **G3** | does phase-local veto redirect the sequence in a way an equal-integral tonic suppression control does not? | **FAIL 0/12** |
| **G4** | can internal replay keep advancing while publication is blocked, unlike actually shunting the internal computation? | **PASS 12/12** |
| **G5** | do one-at-a-time interventions leave the other control schedules/invariants intact? | **PASS 12/12** |

Machine-readable receipt: `results/receipt.json`. Frozen rules: `GATES.md`.

## What survived

### G0 — a genuinely self-running sequence substrate

All 12 canonical seeds replayed the full 20-item learned sequence after only the initial cue. Zero recurrence and within-window timing-shuffled learning stayed below the frozen control ceiling in every seed.

This is the substrate the rest of the repository needed: after the world stops supplying sequence content, the learned recurrent dynamics can continue a trajectory internally.

### G4 — not publishing is not the same as not computing

This is the cleanest new result in v0.

A publication block from 40–100 ms changed only the public trace. The complete deterministic internal trace was bit-identical to the unblocked baseline. During the silent interval the internal front advanced by **7 items** in every canonical seed. At release, it was already at the later internal state. The matched-duration internal-shunt replacement lagged by **20 items**.

So this machine cleanly distinguishes:

```text
continue computing silently
```

from

```text
stop the computation.
```

That is the narrow computational point behind the AIS/publication analogy. It is not evidence that biological chandelier cells implement this exact operation.

### G5 — the control channels are technically separable

All 12 seeds passed the frozen selective-intervention invariants: removing publication left the internal trace unchanged; flipping context left the pre-context state and rhythm schedule unchanged; moving the veto left the context/publication schedules unchanged; replacing rhythmic inhibition with matched tonic inhibition left the other schedules unchanged.

This is an **architectural isolation check**, not proof that every control has a unique useful computational role. G1–G3 are the stronger functional tests, and they failed.

## What failed — and why that matters

### G1 — rhythm did not earn the stronger claim here

The rhythmic arm sometimes propagated farther than the tonic arm (canonical median reach 2 items versus 1.5; mean 6.58 versus 1.67), but **neither arm achieved full 40-item replay in any canonical seed**, so the preregistered full-replay improvement was zero. Secondary-wave behavior was also inconsistent: median counts were 11 for rhythmic versus 9 for tonic.

This does **not** overturn Rytmi's earlier dead-time result. It says that this new 25 ms local admission implementation did not reproduce that advantage under this assay. The stronger claim that "rhythmic dead time is the special controller of replay here" is not supported by v0.

### G2 — multiplicative context worked, but so did a tiny additive push

Multiplicative context selected D versus H correctly in the pilot and canonical runs while producing zero activity in context-only trials. But the replacement control exposed the problem: an additive input of only **0.05** also selected the requested branch in **12/12 canonical seeds** while still producing zero context-only activity.

So the useful behavior is real—context can disambiguate a shared prefix—but v0 did **not** show that multiplicative/apical-style susceptibility is needed to do it. The mechanism collapsed to a simpler input bias in this task.

### G3 — the veto redirected, but phase specificity did not survive the matched control

The relevant-phase veto plus context switch redirected D toward H without a new external cue, preserved nonzero internal state, and left the learned weights unchanged. However, spreading the same integrated suppression over a longer tonic interval also redirected to H in every pilot and canonical seed.

Therefore v0 does **not** support the stronger claim that the success depended specifically on a brief phase-local veto. Again, the behavior exists; the special mechanism was not isolated.

## Reading the result

The original question was whether time could be turned into a **factorized computational control surface**:

```text
WHEN may the sequence advance?
WHICH continuation is susceptible?
MAY that continuation proceed now?
DOES the internal computation become public?
```

v0 gives a useful partial answer.

- A learned internal sequence can run after external input stops.
- Internal computation and public emission can be cleanly separated.
- The simulator can intervene on rhythm, context, veto, and publication independently.
- But in this first task, **rhythm, multiplicative context, and phase-local veto did not prove unique functional advantages over simpler replacement controls**.

That is exactly why this repository returned to replacement controls instead of another end-to-end AI benchmark. The negative results narrow the mechanism rather than being hidden by a successful aggregate score.

## Lineage

The immediate line is:

```text
KolmeOvea
  -> TATWATASW
     plateau -> asymmetric predictive field
     rhythm + dead time -> writable temporal order
  -> Rytmi
     emergent phase precession -> STDP -> travelling-wave replay
  -> ReturnToTimeWindow
     put a self-running replay under independently manipulable time/context/veto/publication controls
```

The numerical core here deliberately reuses the small ideas from Rytmi/TATWATASW rather than importing a generic recurrent learner. `learning.stdp` adapts the pairwise exponential rule from `Rytmi/rytmi_core.py`; the replay state uses the same style of recurrent rate dynamics with adaptation.

## What is *not* in v0

The September 2026 Martin-Burgos et al. action-potential waveform result motivated a later question—whether what leaves a time window can retain a small state-dependent trace—but **state-shaped emitted events are not part of this experiment**. They should only be added after the present timing/control architecture is understood.

Likewise, inverse models, self/other mirroring, culture, language, inheritance, multi-agent communication, and large-model benchmarks are intentionally outside v0.

## Post-v0 G6 — phase can act as an address in the constructed assay

G6 was preregistered and committed **after** the frozen G0–G5 receipt, without altering any v0 threshold. It asks a narrower question than G1: not whether rhythmic dead time generally improves replay, but whether the **relative phase of a global gate can carry branch-selecting information when mean excitation and mean inhibition are held fixed**.

The shared prefix is identical in every arm. Learned prefix-to-branch edges are lesioned identically, then two equal-energy upstream test volleys arrive at `D` and `H` in opposite halves of one 20 ms cycle. There is no unit-specific context vector. Swapping only the phase of the global inhibitory window swaps which volley is admitted.

Frozen canonical result, seeds `100–111`:

| G6 measure | Result |
|---|---:|
| phase-D selects D | **12/12** |
| phase-H selects H | **12/12** |
| all invariants hold | **12/12** |
| matched tonic encodes both requests | **no** |
| pooled random-phase accuracy | **99/192 = 0.515625** |
| frozen G6 gate | **PASS** |

So the stronger G1 claim remains unsupported, but a different temporal claim survives: in a task where candidate events are deliberately separated in time, **the same global inhibitory energy arranged at a different phase produces a different learned continuation**.

```text
same learned downstream network
same shared present
same D/H candidate energy
same mean inhibition
        +
different temporal alignment
        ->
different continuation
```

This is an existence proof inside this small simulator, not evidence that cortical rhythms literally implement this exact two-bin gate. The assay is deliberately constructed: D and H candidate volleys occupy opposite half-cycles, and shared-prefix-to-branch edges are removed so the test isolates temporal admission rather than ordinary recurrent branch bias. The useful next question is therefore not whether phase *can* route a sequence—it can here—but whether a less hand-separated, conductance/delay-based circuit can **generate and exploit those temporal addresses endogenously**.

The frozen design is in `G6.md`; the canonical receipt is `results/g6_phase_address.json`.

## Reproduce

Python 3.11+, NumPy, Matplotlib and pytest are sufficient.

```bash
pip install -e '.[test]'
pytest -q

# exploratory/frozen scientific order
python scripts/run_pilot.py
python scripts/freeze_gates.py
python scripts/run_gates.py
python scripts/make_figure.py

# post-v0 phase-addressing extension (preregistered separately)
python scripts/run_g6.py

# verify the stored receipt without changing thresholds
python scripts/run_gates.py --verify-existing
```

The committed scientific order is visible in Git: implementation → pilot → frozen thresholds → canonical seeds → interpretation.
