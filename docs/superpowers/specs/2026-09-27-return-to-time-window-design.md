# ReturnToTimeWindow — design

Date: 2026-09-27

## Purpose

This repository returns to the mechanistic question that was being lost when the recent work was flattened into generic hidden-state learners, messages, gates, and RNN-like abstractions.

The object here is not "better sequence memory" and not "a brain-inspired RNN".

The object is a **self-running sequence inside a locally constructed time window**, with several control surfaces that remain causally distinct:

1. **recurrent sequence dynamics** — the internal trajectory can continue after external input stops;
2. **rhythmic admission / dead time** — determines when recurrent activity is allowed to advance and suppresses competing waves;
3. **contextual/apical control** — changes which continuation is susceptible without directly supplying the continuation;
4. **sequence veto / dendritic inhibitory control** — can suppress, release, or redirect contextual continuation while leaving the underlying learned repertoire intact;
5. **AIS-like publication control** — determines whether an internally advancing computation is emitted without necessarily stopping the internal trajectory;
6. later, **state-shaped emitted events** — what leaves the circuit may retain a small trace of the state/window that produced it.

The central claim to test is narrower than a theory of cortex:

> **Time can be turned into a factorized computational control surface.** A running sequence may be independently controlled in terms of when it advances, which continuation becomes excitable, whether it is vetoed, whether learning occurs, and whether the result is published.

The biological mapping is a source of hypotheses, not a claim that the simulated variables are literal cortical cell classes.

---

## Why this repo exists

Recent projects such as LittleWorld, Perinto, CoupledSequenceLearners, Limitys, and the umbrella "learned susceptibility" framing were useful falsifiers and boundary markers. But they mostly tested what remained after the physical timing structure had been abstracted into generic state, messages, or gates.

The line to recover is the one exposed by KolmeOvea, TATWATASW, FrequencyAndNeurons, and Rytmi:

```text
plateau
    -> asymmetric predictive field
    -> rhythmic inhibition
    -> emergent phase precession
    -> STDP
    -> ordered replay
```

The important observation is that phase need not be supplied as a coordinate. It can emerge from excitation meeting rhythmic inhibition, and the quiet part of the rhythm can itself have computational work: preventing wrap-around during learning and suppressing spurious replay waves during recall.

The next question is therefore not "can another recurrent model solve a sequence task?"

It is:

> **What becomes possible when a learned, self-propagating sequence is placed under separable temporal, contextual, inhibitory, and publication controls?**

---

## Working picture

A local circuit is continuously active and powered. External input can entrain it, but the internal dynamics need not disappear when external forcing disappears.

A useful transition is:

```text
world-driven trajectory
    -> world + internal model co-drive
    -> internally generated trajectory
```

The internal trajectory is the primary object. The controller does not write every next item into it. It changes the conditions under which the learned trajectory continues.

A minimal mathematical sketch is:

```math
x_{t+1}
= F\!\left(
    x_t,
    G_{\mathrm{rhythm}}(\phi_t),
    G_{\mathrm{context}}(a_t),
    G_{\mathrm{veto}}(v_t),
    u_t
  \right)
```

with a separate publication operator

```math
y_t = P_{\mathrm{AIS}}(x_t, q_t).
```

The design deliberately keeps `x_t` and `y_t` separate. A sequence may continue internally while publication is silent.

A later emitted-event extension may replace a binary publication with

```math
e_t = E(x_t, y_t),
```

so that the downstream target receives an event whose fine structure depends weakly on the state in which it was produced.

---

## The first machine: The Running Sequence

The first version should be small enough to inspect completely.

### Learned repertoire

Use several trajectories with deliberately shared prefixes and branch points, for example:

```text
A B C D E F G
A B C H I J K
A B L M N O P
Q R C D U V W
```

The network must replay them as propagating internal activity rather than as direct table lookup.

The first cue is external. After the cue, normal replay is internal.

The exact learning machinery should be inherited from the Rytmi/TATWATASW line wherever possible rather than replaced by a generic trained RNN. The point is to preserve the physical-time mechanism already measured there.

### Four required control surfaces

#### 1. Rhythmic admission

A rhythmic inhibitory variable creates alternating excitable and quiet intervals.

Its jobs are hypotheses to test, not assumptions to bake into the score:

- keep a replay front temporally organized;
- prevent one local sequence window from wrapping into the next;
- suppress secondary/spurious replay waves;
- create moments at which a controller can interrupt or redirect a trajectory cleanly.

Matched tonic inhibition is the primary replacement control.

#### 2. Contextual/apical susceptibility

Context must not inject the next sequence item.

It changes which continuation is excitable after an ambiguous/shared state. In the shared-prefix example, the same internal state `A B C` can continue toward `D` or `H` depending on context.

The primary replacement control is an additive context signal with matched magnitude/parameter budget. If additive context can solve the task only by directly pushing sequence content or by firing in the absence of sequence drive, that is exactly the distinction the experiment should expose.

#### 3. Sequence veto / dendritic inhibitory control

This controller asks the user's key question:

> **Do I let this sequence keep going?**

The minimal operation is a phase-sensitive veto delivered during a running trajectory. It may:

- allow the present sequence to continue;
- suppress the next admissible continuation;
- release a different continuation at the next window;
- terminate the current replay without erasing the learned repertoire.

The biology-inspired labels here remain provisional. SST/Martinotti-like dendritic inhibition is a plausible source of control over apical/dendritic events, but the repository must not claim that real SST cells are "thought stoppers" or sequence gates.

A useful architectural distinction is:

```text
context says which continuation is currently favored
veto says whether that contextual continuation is allowed to take effect now
```

This makes context selection and sequence suppression separable.

#### 4. AIS-like publication

Publication is separate from internal continuation.

During an AIS-block interval, internal replay should continue. When publication is released, the output should reflect where the internal trajectory has reached, not restart from the point at which output was silenced.

The primary replacement control is to stop/shunt the internal dynamics themselves for the same interval.

That gives a causal distinction between:

```text
continue computing silently
```

and

```text
stop computing.
```

---

## The key experiment: recall by ignition, veto, and branch

The first scientifically interesting assay should be visible in a single raster/trajectory plot.

### Trial

1. Present an incomplete cue such as `A, B`.
2. Remove external sequence content.
3. Allow one learned internal continuation to ignite.
4. At a chosen phase, deliver contextual evidence that the current continuation is wrong.
5. Optionally apply a veto for one or more windows.
6. Release the veto while favoring the alternate branch.
7. Observe whether replay redirects cleanly without globally resetting the state.

Desired qualitative behavior:

```text
candidate internal trajectory:
A -> B -> C -> D -> E ...
               X
               |
context/veto --+
               |
alternate:      H -> I -> J ...
```

This is not yet called reasoning in the code or claims. It is a mechanistic test of **candidate-sequence ignition, interruption, and redirection**.

### Why it matters

If this works causally, the same primitive can later support:

```text
cue
 -> internally generated candidate sequence
 -> predicted consequence
 -> accept / inhibit / branch
```

That would connect the time-window mechanism to inverse/forward-model computation without hand-coding a symbolic planner.

---

## Frozen experimental gates

The exact numerical thresholds must be chosen from pilot scale checks and then frozen before canonical seed panels. The qualitative gates below define what the repository is allowed to claim.

### G0 — self-running replay

After a short external cue, the circuit replays a complete learned trajectory with no further sequence content supplied.

Required controls:

- no learned recurrent sequence weights;
- phase-shuffled learned timing or equivalent timing-destructive control;
- a direct lookup/control implementation may be shown only as a ceiling, not as evidence.

Earned statement: the circuit has a genuinely self-propagating internal sequence substrate.

### G1 — dead time is computational

Under matched noise and total inhibitory load:

- rhythmic inhibition with a quiet interval suppresses spurious secondary waves;
- matched tonic inhibition does not provide the same protection;
- removing the quiet interval while retaining similar mean inhibition increases wrap-around or competing-wave errors.

Earned statement: the temporal gap does useful work beyond average gain suppression.

### G2 — context selects without supplying content

At a shared branch state, context changes which learned continuation wins while context alone does not generate sequence output.

Required controls:

- no context;
- additive context with matched scale/parameter count;
- shuffled or wrong context;
- context-only trials with basal/recurrent sequence drive absent.

Earned statement: the contextual channel changes susceptibility/selection rather than merely injecting the answer.

### G3 — veto changes continuation policy, not memory

A brief veto during replay can prevent or delay continuation, and release can allow either the original or an alternate branch to resume depending on context.

Required controls:

- veto at irrelevant phase;
- tonic suppression with matched integrated inhibition;
- full state reset.

Measurements:

- whether internal state survives;
- whether branch choice changes;
- restart latency;
- whether learned weights/repertoire remain intact across trials.

Earned statement: the controller can act on the running trajectory without rewriting the sequence memory.

### G4 — silent computation is different from stopped computation

Suppress publication for a fixed interval while leaving internal recurrence intact.

At release:

- the internally running arm publishes the state appropriate to the elapsed sequence time;
- the state-stopped/shunted arm is delayed, stale, or must restart.

This is the clean AIS-like test.

Earned statement: "not emitting" and "not computing" are distinct operations in this machine.

### G5 — combined factorization

Run a trial in which all four controls are used independently:

- rhythm determines when the trajectory can advance;
- context determines which continuation is favored;
- veto determines whether that continuation is allowed now;
- publication determines whether the internally computed result becomes externally visible.

The decisive criterion is **selective intervention**: manipulating one control should primarily change its target variable while leaving the others measurable.

If all controls collapse into one scalar gain or threshold under the actual dynamic task, the central factorization claim fails.

---

## Optional second-stage gate: what leaves the window

Only after G0–G5 are established should the Martin-Burgos action-potential result be introduced.

The biological observation motivating this stage is deliberately limited:

- action-potential waveform varies systematically with recent input and network state;
- waveform features can carry information beyond simple firing interval/rate in the reported intracellular data;
- prior physiology cited in that work shows that presynaptic spike width/duration can affect downstream postsynaptic currents.

This does **not** establish a waveform-based sequence-memory code.

The computational question is:

> If two otherwise identical publication events occur at the same time, can a small state-dependent event shape change downstream trajectory selection after rate/timing controls are matched?

Controls must include:

- waveform clamped;
- waveform shuffled while preserving event times;
- firing-rate matched;
- integrated event strength/area matched where applicable;
- sender-history shuffled;
- a scalar-gain replacement control.

If the scalar-gain/rate-matched control ties the waveform arm, the state-shaped event adds no unique computation in this model.

---

## What the first plots should show

The primary figure should be a time plot, not an accuracy leaderboard.

A useful layout is:

```text
internal replay:    A  B  C  D  E  F  G
rhythmic window:    ███___███___███___███
context:                    branch-H -------->
veto:                       XXXX
publication:                    XXXXX
public output:      A  B  C            F  G
```

For branch trials, overlay the competing sequence-front amplitudes so that the reader can see one candidate die and another ignite.

Secondary plots:

- replay-front position versus time;
- number/location of spurious waves;
- branch amplitude before/after context;
- internal versus public state during AIS block;
- phase of intervention versus probability of successful veto/redirect.

The repository should make the mechanism legible before it makes it performant.

---

## Non-goals

Version 0 must not:

- benchmark against large language models;
- claim consciousness, thought selection, or subjective experience;
- claim that SST/Martinotti cells literally terminate thoughts;
- claim that chandelier cells are proven publication gates;
- claim that action-potential waveform is a biological message alphabet;
- replace the mechanism with a generic GRU/RNN and call successful behavior confirmation;
- optimize end-to-end task accuracy before the individual control surfaces have causal replacement controls;
- add culture, social learning, language, inheritance, or multi-agent communication.

Those may become later branches only if the time-window mechanism survives its own falsifiers.

---

## Biological claim boundary

The simulation deliberately draws hypotheses from several established or reported biological facts:

- pyramidal cells have compartmental basal and apical integration;
- distal/apical input can interact nonlinearly with somatic/basal drive;
- dendrite-targeting inhibition can regulate apical events and plasticity;
- perisomatic inhibition can impose strong temporal structure;
- axon-initial-segment control is physically downstream of dendritic/somatic integration;
- rhythmic activity and phase precession can compress behavioral sequences into shorter temporal windows;
- the 2026 Martin-Burgos et al. preprint reports state-dependent action-potential waveform variation.

The assembled control architecture is **our engineering hypothesis**. No one-to-one biological identification is assumed.

---

## Connection to the inverse-model line

This repo should not begin with social mirroring or an explicit inverse-model architecture. Those are downstream uses.

The bridge is simpler:

```text
current state
 + desired / imagined consequence
 -> ignite candidate learned trajectory
 -> run internally
 -> forward-predict consequence
 -> accept, veto, or branch
```

If the time-window machine can generate and regulate candidate trajectories internally, then an inverse model can later be expressed as **which sequence should be ignited under a desired future**, rather than as one static matrix mapping goal to action.

That is the point at which the mirror-neuron/self-other line becomes worth returning to.

---

## Structural analogy to DNA

The DNA comparison is retained only as an abstract systems analogy.

The shared problem is not "DNA uses neural rhythms". It is:

> A system contains many possible sequential programs, but only a context-dependent subset should be expressed at any one time.

Gene regulation solves this with control over accessibility, initiation, continuation, splicing, and termination of sequence expression.

The neural hypothesis explored here is similarly about **controlled sequence expression**:

```text
learned trajectory repertoire
+
mechanisms controlling which trajectory is allowed to unfold now
```

This analogy is conceptually useful but is not part of the empirical claim.

---

## Implementation philosophy

1. **Reuse measured mechanisms.** Prefer the existing Rytmi/TATWATASW replay and timing logic over inventing a new recurrent learner.
2. **One control surface at a time.** Add each mechanism only with its replacement control.
3. **Make interventions visible.** Internal sequence state, phase, inhibition, context, veto, and publication must be logged directly.
4. **Pre-register canonical gates.** Exploratory pilot runs may set sensible numerical thresholds; canonical seed panels run only after the gate file is frozen.
5. **Preserve negative results.** A collapse into generic gain/threshold control is a useful result and must remain in the repository.
6. **CPU scale.** The first machine should run in minutes, not hours, and should not require paid compute.

---

## Proposed repository shape

After this design is approved and an implementation plan exists, the likely minimal layout is:

```text
README.md
GATES.md
src/
  sequence.py        # learned propagating trajectory substrate
  rhythm.py          # admission window / dead time
  context.py         # branch susceptibility
  veto.py            # phase-sensitive sequence suppression/release
  publication.py     # AIS-like internal/public separation
  experiment.py      # canonical interventions
scripts/
  run_gates.py
  make_figure.py
results/
tests/
docs/
  superpowers/
    specs/
```

Names may change during planning; the conceptual separation should not.

---

## Success condition for ReturnToTimeWindow v0

The repository succeeds if it demonstrates, in one small self-running sequence machine, that the following interventions are **not interchangeable**:

```text
WHEN may the sequence advance?
WHICH continuation is currently susceptible?
MAY that continuation proceed now?
DOES the internal computation become public?
```

A later stage may add:

```text
WHAT trace of the producing state remains in the emitted event?
```

If those controls collapse to one generic gain knob under dynamic replay, that is the correct negative result.

If they remain causally separable, the project has recovered the question that motivated it:

> **computation manufactured out of physical time, rather than hidden state merely indexed by time.**
