# ReturnToTimeWindow

![time-window trace](results/time_window.png)

**Return to the question before the abstraction.**  
This repository asks what happens when a learned, self-running sequence is placed inside a locally constructed time window with separable controls over **when it advances, which continuation is susceptible, whether a continuation is suppressed, and whether the internal computation is published**.

It deliberately does **not** begin with a generic RNN, communication task, cultural learner, or language model.

The lineage is:

```text
KolmeOvea
  -> TATWATASW
  -> Rytmi
  -> ReturnToTimeWindow
```

The inherited substrate is the small Rytmi/TATWATASW idea: temporally compressed ordered spikes write directional recurrent weights with pairwise STDP, and a short cue can then launch a travelling internal replay wave.

## The machine

The v0 machine keeps four control surfaces separate:

```text
recurrent sequence                         internal trajectory
        |
        +-- rhythmic admission             WHEN can it advance?
        +-- contextual susceptibility      WHICH continuation is favoured?
        +-- veto                           MAY that continuation proceed now?
        +-- publication mask               DOES the internal result become public?
```

The controls are explicitly logged. Publication never feeds back into internal state.

This is an engineering hypothesis inspired by cortical compartmentalization, inhibition, rhythm and AIS placement. It is **not** a claim that the simulated variables are literal SST/Martinotti, basket, or chandelier cells.

## Frozen v0 result

Pilot seeds `0–3` were used only to resolve the numerical thresholds using the formulas frozen in the implementation plan. Those thresholds were committed before the canonical seeds `100–111` were run.

| gate | question | canonical result |
|---|---|---:|
| **G0 PASS** | Can a short cue launch a learned self-running sequence? | **12/12** |
| **G1 FAIL** | Does rhythmic dead time beat matched tonic inhibition? | **0/12** |
| **G2 FAIL** | Does multiplicative context select without content in a way matched additive context cannot? | **0/12** |
| **G3 FAIL** | Is phase-specific veto/redirection better than matched tonic suppression? | **0/12** |
| **G4 PASS** | Can computation continue internally while publication is silent? | **12/12** |
| **G5 PASS** | Do the four authored controls remain selectively separable under intervention? | **12/12** |

Machine-readable results are in [`results/receipt.json`](results/receipt.json). Frozen thresholds are in [`results/frozen_thresholds.json`](results/frozen_thresholds.json).

### G0 — a real self-running substrate

A 20-item learned chain replayed completely from a short cue on every canonical seed.

- learned sequence: full replay **12/12**
- no recurrence: full replay **0/12**
- within-window timing shuffled before STDP: full replay **0/12**

So v0 has the object this project needed first: after external sequence content stops, recurrent state can continue the learned trajectory.

### G1 — dead time did **not** earn the proposed role

The stronger time-window prediction failed before the canonical run: the pilot did not establish the preregistered separation between rhythmic and matched tonic inhibition, so the frozen prerequisite is false.

On the canonical panel neither arm completed the 40-item noisy sequence. Median in-order reach was **2 items for rhythmic** versus **1.5 for tonic**; median detected secondary-wave events were **11 versus 9**.

That is not evidence that local dead time is useless. It is evidence that **this particular 25 ms open/dead gate did not reproduce the protection seen in Rytmi's larger re-cue/dead-time construction**.

### G2 — context works, but multiplication is not special here

Multiplicative context selected the requested branch on **12/12 seeds** and produced exactly zero activity when there was no recurrent/basal evidence.

But the matched additive replacement found an easy loophole: at additive strength `0.05`, it also selected the requested branch on **12/12 seeds** while producing **zero context-only activity**.

So the attractive sentence

> context changes susceptibility without supplying content

is implemented by the multiplicative arm, but **this task does not require that implementation**. A weak additive bias does the job too.

That is the frozen negative result, not something to tune away.

### G3 — veto redirects, but phase specificity did not matter enough

The relevant-phase intervention redirected the running branch toward `H`, preserved non-zero internal state, left the learned weights unchanged, and required no new external cue.

But matched tonic suppression redirected too. The pilot therefore failed the tonic-separation prerequisite, and G3 was frozen as a failed mechanism before canonical seeds were seen.

So v0 does **not** support the strong claim that a special phase-local veto is required for redirection.

### G4 — silent computation is genuinely different from stopped computation

This is the clean positive result.

During a publication block, the internal sequence continued while public output was exactly zero. Across all canonical seeds:

- internal traces with and without publication block were exactly equal;
- the internal front advanced **7 items** during the silent interval;
- when publication reopened it exposed the progressed state;
- the matched internal-shunt arm lagged by **20 items** at release.

In this machine,

```text
not emitting != not computing
```

This is the clearest surviving computational distinction from the original "three doors" picture.

### G5 — factorization survives as architecture, not yet as necessity

Changing only publication leaves the internal trace unchanged. Flipping context changes the selected branch without changing the pre-context state or rhythm schedule. Moving the veto leaves context/publication schedules unchanged. Replacing rhythmic inhibition with tonic inhibition leaves the other authored controls unchanged.

Those selective invariants held on **12/12 canonical seeds**.

This means the software really does expose four independently intervenable control surfaces. It does **not** mean all four are functionally necessary: G1–G3 show that several proposed special roles were not distinguished from simpler replacements in v0.

## What this changes

The result is narrower, and more useful, than the design hope.

The project did recover the right experimental object:

```text
external cue
    -> internally running sequence
    -> interventions during the trajectory
    -> separate internal and public state
```

But three attractive mechanistic stories did not survive their first replacements:

```text
rhythmic gap > tonic inhibition          not shown
multiplicative context > additive bias   not shown
phase veto > tonic suppression           not shown
```

The one strong distinction is:

```text
publication gate != computation gate
```

That gives the next version a better target. Instead of adding more biological labels, it should make the task one where **phase, susceptibility, or veto timing is mathematically necessary**—for example competing trajectories whose inputs are identical in mean drive and differ only in relative arrival time or in a late counterfactual that cannot be solved by a static bias.

## The figure

The figure at the top is a representative canonical G5 trace (seed 100), not a proof by itself.

- top: internal activity over the shared/branching sequence;
- second: rhythmic inhibition, D/H context, and D veto;
- third: internal activity continues while the publication mask hides output;
- bottom: competing D and H branch amplitudes.

It is meant to make the intervention geometry visible before any aggregate score.

## Claim boundary

v0 supports only the following:

1. temporally ordered STDP weights can support self-running replay in this small constructed system;
2. internal replay and publication can be separated causally;
3. four control arrays can be intervened on independently in the implementation.

v0 does **not** establish:

- that cortical thought is a replay wave;
- that SST/Martinotti cells stop thoughts or sequences;
- that basket-cell rhythms provide the exact gate simulated here;
- that chandelier cells are literal "publication gates";
- that multiplicative/apical context is computationally superior to additive context;
- that state-dependent action-potential waveform carries sequence state;
- consciousness, inner speech, or reasoning mechanisms.

The Martin-Burgos waveform result is deliberately **not part of v0**. If added later, it gets its own rate/timing/scalar-gain falsifiers.

## Reproduce

Python 3.11+; NumPy, Matplotlib and pytest only.

```bash
python -m pip install -e '.[test]'
pytest -q

# The historical order matters:
python scripts/run_pilot.py
python scripts/freeze_gates.py      # thresholds are now frozen
python scripts/run_gates.py         # canonical seeds 100..111
python scripts/make_figure.py

# Re-evaluate an existing receipt without rerunning/tuning:
python scripts/run_gates.py --verify-existing
```

## Files

```text
src/return_to_time_window/
  learning.py       compressed sequence events + STDP
  sequence.py       recurrent internal state and full control logging
  rhythm.py         rhythmic admission and matched tonic replacement
  context.py        multiplicative context and additive replacement
  veto.py           sequence suppression windows
  publication.py    public mask and internal-shunt replacement
  metrics.py        order/front/secondary-wave measures
  experiments.py    G0-G5 trials
  gates.py          frozen-threshold derivation and evaluator

GATES.md             pre-canonical frozen rules
results/pilot.json   pilot seeds 0..3
results/frozen_thresholds.json
results/receipt.json canonical seeds 100..111
results/time_window.png
```

## Why the negative gates stay

The point of this repository is to return to physical-time hypotheses **without protecting them from ordinary explanations**.

If an additive bias can replace contextual gain, it stays in the result.  
If tonic suppression can replace a phase veto, it stays.  
If this rhythm does not buy dead-time protection, it stays.

The next mechanism has to earn its complexity.
