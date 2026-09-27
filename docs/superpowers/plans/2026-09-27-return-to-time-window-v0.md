# ReturnToTimeWindow v0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build one CPU-scale learned, self-running sequence machine in which rhythmic admission, contextual branch susceptibility, sequence veto, and publication can be intervened on independently and tested by G0–G5.

**Architecture:** Reuse the measured Rytmi/TATWATASW ingredients that matter here—compressed temporal ordering, pairwise STDP, recurrent travelling-wave replay, adaptation, and destructive timing controls—rather than substituting a generic RNN. The replay core owns only continuous internal state and recurrent dynamics; rhythm, context, veto, and publication each generate separate control signals so their effects can be logged and replaced one at a time.

**Tech Stack:** Python 3.11+, NumPy, Matplotlib, pytest. No GPU, no paid compute, no deep-learning framework in v0.

**Spec:** `docs/superpowers/specs/2026-09-27-return-to-time-window-design.md`

## Global Constraints

- v0 stops at G0–G5; the Martin-Burgos/state-shaped-waveform extension is explicitly deferred.
- Reuse measured Rytmi/TATWATASW mechanisms rather than replacing the substrate with a GRU/RNN.
- Every control surface gets a matched/replacement control before it contributes to a scientific claim.
- Internal state, rhythm/inhibition, context gain, veto gain, publication mask, and public output are logged directly.
- Pilot seeds may set numerical canonical thresholds, but canonical seeds run only after those thresholds are frozen and committed.
- Preserve failures. If the controls collapse into one scalar gain/threshold under replay, record that as the result.
- The full v0 suite must run on CPU in minutes, not hours.
- Do not add culture, social learning, language, inheritance, multi-agent communication, LLM benchmarks, consciousness claims, or biological one-to-one claims.

## Review Focus

1. **Shared-token branch indexing:** a token such as `C` that occurs in multiple trajectories must be one recurrent unit, not duplicated by sequence. Task 1 pins this with an identity test.
2. **Context with zero recurrent evidence:** multiplicative context must produce no sequence activity when recurrent/basal drive is absent. Task 3 pins this with a context-only test.
3. **Veto in an irrelevant phase:** a veto that lies entirely inside a dead interval must not masquerade as a successful sequence intervention. Task 4 pins this explicitly.
4. **Publication blocks near boundaries:** masking output must never feed back into or reset internal state, including a block that spans the end of a sequence. Task 5 tests internal-trace invariance.
5. **Degenerate/empty learned weights:** normalization must not divide by zero and a no-recurrence control must remain silent after the cue. Task 1 tests both.

---

## File Structure

Create this minimal package; keep files split by the causal object they own.

```text
pyproject.toml
README.md
GATES.md                         # qualitative first; numeric table added after pilot
src/return_to_time_window/
  __init__.py
  learning.py                    # token repertoire, compressed-window spikes, STDP
  sequence.py                    # continuous recurrent replay and complete trace logging
  rhythm.py                      # rhythmic admission + matched tonic/no-dead control
  context.py                     # multiplicative branch susceptibility + additive replacement
  veto.py                        # phase-sensitive suppression windows
  publication.py                 # public-output mask + internal-shunt replacement control
  metrics.py                     # peak order, replay reach, branch choice, second-wave metrics
  experiments.py                 # G0–G5 trial constructors; no CLI/file I/O
  gates.py                       # pilot-threshold derivation and canonical pass/fail evaluator
scripts/
  run_pilot.py
  freeze_gates.py
  run_gates.py
  make_figure.py
results/                         # generated JSON/PNG receipts; committed after runs
  .gitkeep
tests/
  test_learning_and_sequence.py
  test_rhythm.py
  test_context.py
  test_veto.py
  test_publication.py
  test_factorization.py
  test_gates.py
```

The core numerical lineage should be acknowledged in comments/README: `learning.stdp` adapts Rytmi `rytmi_core.stdp`; the replay update adapts Rytmi/TATWATASW's rate network with adaptation. Do not import sibling repositories at runtime.

---

### Task 1: Learned propagating sequence substrate and G0

**Files:**
- Create: `pyproject.toml`
- Create: `src/return_to_time_window/__init__.py`
- Create: `src/return_to_time_window/learning.py`
- Create: `src/return_to_time_window/sequence.py`
- Create: `src/return_to_time_window/metrics.py`
- Create: `src/return_to_time_window/experiments.py`
- Create: `tests/test_learning_and_sequence.py`
- Create: `results/.gitkeep`

**Interfaces:**
- `learning.Repertoire`: token names, one token-to-id map, and integer-coded trajectories.
- `learning.encode_repertoire(sequences: list[list[str]]) -> Repertoire`
- `learning.compressed_window_spikes(repertoire: Repertoire, sequence_index: int, rng: np.random.Generator, *, windows: int = 20, period_s: float = 0.125, item_dt_s: float = 0.008, jitter_sd_s: float = 0.001, shuffle_within_window: bool = False) -> tuple[np.ndarray, np.ndarray]`
- `learning.stdp(spike_times: np.ndarray, unit_ids: np.ndarray, n_units: int, *, tau_s: float = 0.020, W: np.ndarray | None = None) -> np.ndarray`
- `learning.learn_repertoire(repertoire: Repertoire, rng: np.random.Generator, *, windows_per_sequence: int = 20, shuffle_within_window: bool = False) -> np.ndarray`
- `sequence.ReplayConfig(dt_s=0.005, tau_s=0.020, tau_adapt_s=0.250, adapt_gain=1.5, recurrent_gain=2.2, threshold=0.3, noise_sd=0.0)`
- `sequence.ControlSignals`: arrays `inhibition[T]`, `context_gain[T,N]`, `additive_context[T,N]`, `veto_gain[T,N]`, `publication_mask[T]`, `internal_shunt[T]`.
- `sequence.identity_controls(steps: int, n_units: int) -> ControlSignals`
- `sequence.cue_drive(steps: int, n_units: int, dt_s: float, cue: list[tuple[float, float, int, float]]) -> np.ndarray`
- `sequence.ReplayTrace`: `time`, `internal`, `public`, `recurrent_drive`, plus every control signal used.
- `sequence.simulate_replay(W: np.ndarray, duration_s: float, config: ReplayConfig, rng: np.random.Generator, *, external_drive: np.ndarray | None = None, controls: ControlSignals | None = None) -> ReplayTrace`
- `metrics.peak_order(activity: np.ndarray, threshold: float = 0.2) -> list[int]`
- `metrics.in_order_reach(order: list[int], expected: list[int]) -> int`
- `metrics.front_position(activity: np.ndarray, expected: list[int], threshold: float = 0.2) -> np.ndarray`
- `experiments.run_g0(seed: int) -> dict`

`compressed_window_spikes` is the deliberately small TATWATASW-style training schedule: each trajectory is replayed in short ordered packets inside 125 ms windows; adjacent items are 8 ms apart, so the ±20 ms STDP rule can write direction. The phase-shuffled control preserves unit identities, spike counts, and window membership while permuting within-window times.

- [ ] **Step 1: Write repertoire identity and timing-control tests**

```python
def test_shared_token_is_one_unit():
    rep = encode_repertoire([["A", "B", "C", "D"], ["A", "B", "C", "H"]])
    assert rep.token_to_id["C"] == rep.sequences[0][2] == rep.sequences[1][2]


def test_phase_shuffle_preserves_spike_multiset_but_changes_times():
    # same unit counts and same number of spikes; ordered and shuffled times differ
    ...
```

- [ ] **Step 2: Run the two tests and verify they fail before implementation**

Run: `pytest tests/test_learning_and_sequence.py -k 'shared_token or phase_shuffle' -v`
Expected: FAIL because the package/interfaces do not exist.

- [ ] **Step 3: Implement `Repertoire`, `encode_repertoire`, and `compressed_window_spikes`**

Token IDs follow first occurrence across the supplied trajectory list. The timing generator must be deterministic for a given RNG state and must keep shuffled spikes inside their original 125 ms window.

- [ ] **Step 4: Add STDP directionality tests**

```python
def test_stdp_writes_forward_adjacency():
    W = learn_repertoire(rep, np.random.default_rng(0))
    for seq in rep.sequences:
        for a, b in zip(seq, seq[1:]):
            assert W[b, a] > W[a, b]


def test_shuffling_reduces_directional_asymmetry():
    ...
```

- [ ] **Step 5: Run STDP tests and verify failure**

Run: `pytest tests/test_learning_and_sequence.py -k 'stdp or shuffling' -v`
Expected: FAIL because STDP/learning are not implemented.

- [ ] **Step 6: Implement `stdp` and `learn_repertoire`**

Port the balanced pairwise exponential rule from `anttiluode/Rytmi:rytmi_core.py` (`TAU_STDP = 0.02`), then rectify learned replay weights with `np.maximum(W, 0)`. Normalize only when `W.max() > 0`; otherwise return the all-zero matrix unchanged.

- [ ] **Step 7: Add replay tests**

```python
def test_replay_continues_after_external_cue_ends():
    # cue only the first item; a later item must peak after the cue interval
    ...


def test_zero_recurrence_does_not_continue_after_cue():
    ...


def test_zero_weight_matrix_is_safe_to_simulate():
    ...
```

- [ ] **Step 8: Run replay tests and verify failure**

Run: `pytest tests/test_learning_and_sequence.py -k 'replay or zero' -v`
Expected: FAIL before `sequence.py` exists.

- [ ] **Step 9: Implement `ReplayConfig`, `ControlSignals`, cue creation, `ReplayTrace`, and `simulate_replay`**

Use the Rytmi/TATWATASW rate update as the substrate: recurrent drive from `W @ r`, rectified rate state, and a slower adaptation state. Controls default to identity and are only multiplied/added at the explicitly named points. `publication_mask` affects `public` only; it must never feed back into `internal`.

- [ ] **Step 10: Implement minimal metrics and `run_g0(seed)`**

`run_g0` uses a 20-item linear trajectory, reports self-running full/in-order reach for learned weights, zero-recurrence, and within-window-shuffled learning, and returns raw scalar metrics only—no pass/fail yet.

- [ ] **Step 11: Run Task 1 tests**

Run: `pytest tests/test_learning_and_sequence.py -v`
Expected: PASS.

- [ ] **Step 12: Commit**

```bash
git add pyproject.toml src tests/test_learning_and_sequence.py results/.gitkeep
git commit -m "feat: add learned self-running sequence substrate"
```

---

### Task 2: Rhythmic admission and dead-time falsifier (G1)

**Files:**
- Create: `src/return_to_time_window/rhythm.py`
- Modify: `src/return_to_time_window/metrics.py`
- Modify: `src/return_to_time_window/experiments.py`
- Create: `tests/test_rhythm.py`

**Interfaces:**
- `rhythm.RhythmicAdmission(period_s: float = 0.025, open_fraction: float = 0.60, dead_inhibition: float = 0.8, phase_s: float = 0.0)`
- `RhythmicAdmission.inhibition(times: np.ndarray) -> np.ndarray`
- `rhythm.matched_tonic(admission: RhythmicAdmission, times: np.ndarray) -> np.ndarray`
- `rhythm.with_inhibition(base: ControlSignals, inhibition: np.ndarray) -> ControlSignals`
- `metrics.secondary_wave_events(activity: np.ndarray, expected: list[int], threshold: float = 0.2, behind_by: int = 5) -> int`
- `experiments.run_g1(seed: int, *, noise_sd: float = 0.30) -> dict`

A 25 ms local cycle is used here because the inherited replay wave advances on the order of 10–20 ms/item; 60% open / 40% dead creates a short temporal gap without the 50 ms full-silence/re-cue mechanism from Rytmi G0. The matched tonic arm has exactly the same mean inhibitory area over time.

- [ ] **Step 1: Write rhythm conservation tests**

```python
def test_rhythm_has_explicit_open_and_dead_intervals(): ...
def test_matched_tonic_has_same_mean_inhibitory_area(): ...
```

- [ ] **Step 2: Verify rhythm tests fail**

Run: `pytest tests/test_rhythm.py -v`
Expected: FAIL before `rhythm.py` exists.

- [ ] **Step 3: Implement rhythmic and tonic admission signals**

The rhythmic signal is piecewise constant: zero in the open fraction, `dead_inhibition` in the dead fraction. Do not hide phase state inside the replay core.

- [ ] **Step 4: Add a synthetic second-wave metric test**

Construct activity with a normal advancing front plus a late reactivation more than five items behind it; assert one secondary-wave event. A purely advancing trace must score zero.

- [ ] **Step 5: Implement `secondary_wave_events` and `run_g1`**

`run_g1` trains a 40-item line once per seed, uses identical weights/noise tape for rhythmic and tonic arms, and reports full-in-order replay, in-order reach, and second-wave counts. This is the direct G1 replacement comparison.

- [ ] **Step 6: Run Task 2 tests**

Run: `pytest tests/test_rhythm.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/return_to_time_window/rhythm.py src/return_to_time_window/metrics.py src/return_to_time_window/experiments.py tests/test_rhythm.py
git commit -m "feat: add rhythmic admission and dead-time control"
```

---

### Task 3: Context selects a branch without supplying content (G2)

**Files:**
- Create: `src/return_to_time_window/context.py`
- Modify: `src/return_to_time_window/experiments.py`
- Create: `tests/test_context.py`

**Interfaces:**
- `context.ContextWindow(start_s: float, end_s: float, target_ids: tuple[int, ...], strength: float)`
- `context.multiplicative_gain(times: np.ndarray, n_units: int, windows: list[ContextWindow]) -> np.ndarray`
- `context.additive_drive(times: np.ndarray, n_units: int, windows: list[ContextWindow]) -> np.ndarray`
- `context.with_multiplicative_context(base: ControlSignals, gain: np.ndarray) -> ControlSignals`
- `context.with_additive_context(base: ControlSignals, drive: np.ndarray) -> ControlSignals`
- `experiments.run_g2(seed: int) -> dict`

The canonical branch pair is `A B C D E F G` versus `A B C H I J K`. Context begins only after the shared state has formed. The multiplicative arm scales recurrent drive into the branch head (`D` or `H`); it does not create a drive vector on its own. The additive replacement uses the same target vector and swept strength but adds it directly.

- [ ] **Step 1: Write zero-evidence and same-present tests**

```python
def test_multiplicative_context_cannot_fire_with_zero_recurrent_drive(): ...
def test_two_context_arms_are_identical_before_context_onset(): ...
```

- [ ] **Step 2: Verify context tests fail**

Run: `pytest tests/test_context.py -v`
Expected: FAIL before `context.py` exists.

- [ ] **Step 3: Implement multiplicative and additive context signal builders**

`strength=1.0` means a multiplicative factor of `2.0` on targeted recurrent drive. Outside the window the multiplicative factor is exactly `1.0` and additive drive exactly `0.0`.

- [ ] **Step 4: Add branch-switch and additive-leak sweep tests**

Use the same cue/noise/weights for context-D and context-H. Assert opposite branch heads win while pre-context internal traces are equal. Add a context-only trial (`W=0`, no external cue) and verify the multiplicative arm remains silent. The additive sweep records branch selectivity versus context-only peak rather than assuming beforehand that every additive strength fails.

- [ ] **Step 5: Implement `run_g2`**

Return multiplicative branch accuracy/selectivity, context-only leakage, and the additive sweep's Pareto points. The later gate asks whether multiplicative control reaches useful selection with no leakage in a region where the matched additive replacement cannot do both simultaneously.

- [ ] **Step 6: Run Task 3 tests**

Run: `pytest tests/test_context.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/return_to_time_window/context.py src/return_to_time_window/experiments.py tests/test_context.py
git commit -m "feat: add content-free contextual branch control"
```

---

### Task 4: Phase-sensitive sequence veto and redirection (G3)

**Files:**
- Create: `src/return_to_time_window/veto.py`
- Modify: `src/return_to_time_window/context.py`
- Modify: `src/return_to_time_window/experiments.py`
- Create: `tests/test_veto.py`

**Interfaces:**
- `veto.VetoWindow(start_s: float, end_s: float, target_ids: tuple[int, ...] | None = None, suppression: float = 1.0)`
- `veto.veto_gain(times: np.ndarray, n_units: int, windows: list[VetoWindow]) -> np.ndarray`
- `veto.with_veto(base: ControlSignals, gain: np.ndarray) -> ControlSignals`
- `context.combine_multiplicative_contexts(times, n_units, windows) -> np.ndarray` supports a D-favoring interval followed by an H-favoring interval.
- `experiments.run_g3(seed: int) -> dict`

`suppression=1.0` makes the targeted recurrent susceptibility zero during the window. A global veto uses `target_ids=None`. The relevant-phase veto overlaps the branch decision's open interval; the irrelevant-phase control has identical duration/integral but lies wholly inside the rhythmic dead interval.

- [ ] **Step 1: Write veto signal tests**

Assert targeted units reach factor `0.0` only inside the veto window and untargeted units remain `1.0`.

- [ ] **Step 2: Verify signal tests fail**

Run: `pytest tests/test_veto.py -k signal -v`
Expected: FAIL before `veto.py` exists.

- [ ] **Step 3: Implement veto signal generation**

Keep veto as a distinct factor from context gain even though both multiply recurrent drive; separate arrays are required in `ReplayTrace` so the factorization can later fail or survive empirically.

- [ ] **Step 4: Add relevant-phase, irrelevant-phase, and memory-preservation tests**

The working trial starts D-favored, applies a veto at the branch decision, changes context to H, then releases. Assertions:
- relevant veto + context switch can redirect to H without a new external cue;
- same veto shifted entirely into dead time does not redirect the unchanged D-context trial;
- `W` is byte-identical before/after the intervention;
- internal activity does not globally reset to zero during the brief veto.

Also add a full-reset control and measure restart latency separately from veto.

- [ ] **Step 5: Implement `run_g3`**

Return redirect success, delay, state-survival norm, irrelevant-phase effect, tonic/matched suppression control, and full-reset restart behavior.

- [ ] **Step 6: Run Task 4 tests**

Run: `pytest tests/test_veto.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/return_to_time_window/veto.py src/return_to_time_window/context.py src/return_to_time_window/experiments.py tests/test_veto.py
git commit -m "feat: add phase-sensitive sequence veto"
```

---

### Task 5: Publication is not computation (G4)

**Files:**
- Create: `src/return_to_time_window/publication.py`
- Modify: `src/return_to_time_window/experiments.py`
- Create: `tests/test_publication.py`

**Interfaces:**
- `publication.PublicationBlock(start_s: float, end_s: float)`
- `publication.publication_mask(times: np.ndarray, blocks: list[PublicationBlock]) -> np.ndarray`
- `publication.InternalShunt(start_s: float, end_s: float)`
- `publication.internal_shunt(times: np.ndarray, blocks: list[InternalShunt]) -> np.ndarray`
- `publication.with_publication(base: ControlSignals, mask: np.ndarray) -> ControlSignals`
- `publication.with_internal_shunt(base: ControlSignals, shunt: np.ndarray) -> ControlSignals`
- `experiments.run_g4(seed: int) -> dict`

`internal_shunt=1` is a replacement control, not another claimed cortical mechanism: while active, recurrent/external drive is suppressed and the rate/adaptation state is allowed to decay normally. Publication blocking only multiplies `public`; it never changes `internal`, adaptation, noise, or recurrent drive.

- [ ] **Step 1: Write output-only invariance test**

Run the same deterministic trial with and without a publication block. Assert the complete `internal` arrays are exactly equal; only `public` differs inside the block.

- [ ] **Step 2: Verify invariance test fails**

Run: `pytest tests/test_publication.py -k invariance -v`
Expected: FAIL before `publication.py` exists.

- [ ] **Step 3: Implement publication mask and internal shunt**

Blocks are half-open `[start_s, end_s)`. Multiple blocks combine by minimum mask / maximum shunt.

- [ ] **Step 4: Add silent-advance versus stopped-computation tests**

During the publication block assert the public activity is zero while the internal front advances. After release, the first visible state must correspond to the progressed internal front, not the pre-block item. Under matched-duration internal shunt, the front must be stale/delayed or the sequence must need to die/restart.

Include a block that extends beyond normal sequence completion and assert internal completion is unaffected.

- [ ] **Step 5: Implement `run_g4`**

Return internal items advanced during silence, release-front position, baseline internal-trace equality, and the shunt arm's lag/stall.

- [ ] **Step 6: Run Task 5 tests**

Run: `pytest tests/test_publication.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/return_to_time_window/publication.py src/return_to_time_window/experiments.py tests/test_publication.py
git commit -m "feat: separate internal replay from publication"
```

---

### Task 6: Combined factorization assay (G5)

**Files:**
- Modify: `src/return_to_time_window/experiments.py`
- Create: `tests/test_factorization.py`

**Interfaces:**
- `experiments.run_g5(seed: int) -> dict`
- `experiments.run_seed(seed: int) -> dict[str, dict]` returns `g0` … `g5`.

The G5 trial uses all four controls on the same learned branched repertoire. It also reruns four selective interventions while reusing weights and random tapes:
- change rhythm only;
- flip context only;
- move veto from relevant to irrelevant phase only;
- remove publication block only.

- [ ] **Step 1: Write selective-invariance tests**

```python
def test_publication_change_leaves_internal_trace_identical(): ...
def test_context_flip_changes_branch_but_not_pre_context_state_or_rhythm_trace(): ...
def test_veto_phase_change_leaves_context_and_publication_schedules_identical(): ...
def test_rhythm_change_leaves_context_veto_and_publication_schedules_identical(): ...
```

- [ ] **Step 2: Verify the combined tests fail**

Run: `pytest tests/test_factorization.py -v`
Expected: FAIL before `run_g5` exists.

- [ ] **Step 3: Implement `run_g5` and `run_seed`**

Return raw traces only for one representative combined trial and compact metrics for the intervention variants. The scientific criterion is selective causality, not an accuracy leaderboard: if an intervention on one control changes every other control's measured state, G5 should expose that instead of hiding it in one score.

- [ ] **Step 4: Run Task 6 tests**

Run: `pytest tests/test_factorization.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/return_to_time_window/experiments.py tests/test_factorization.py
git commit -m "feat: add combined time-window factorization assay"
```

---

### Task 7: Pilot, freeze numerical gates, canonical receipt, figure, and README

**Files:**
- Create: `src/return_to_time_window/gates.py`
- Create: `tests/test_gates.py`
- Create: `scripts/run_pilot.py`
- Create: `scripts/freeze_gates.py`
- Create: `scripts/run_gates.py`
- Create: `scripts/make_figure.py`
- Create/Modify: `GATES.md`
- Create/Modify: `README.md`
- Generate: `results/pilot.json`
- Generate: `results/frozen_thresholds.json`
- Generate: `results/receipt.json`
- Generate: `results/time_window.png`

**Interfaces:**
- `gates.derive_thresholds(pilot: list[dict]) -> dict`
- `gates.evaluate(seed_result: dict, thresholds: dict) -> dict`
- Pilot seeds: `(0, 1, 2, 3)`.
- Canonical seeds: `(100, 101, ..., 111)`; scripts must reject overlap with pilot seeds.

Threshold derivation is mechanical so the post-pilot freeze is not a hidden tuning session:
- G0 main replay floor: `max(0.50, 0.80 * pilot_median_main_full_replay)`; timing/no-recurrence control ceiling: `min(0.25, max(0.05, 1.25 * pilot_max_control_full_replay))`.
- G1 rhythmic-minus-tonic full-replay improvement floor: `max(0.10, 0.60 * pilot_median_delta)`; require secondary-wave count not worse than tonic on at least 3/4 pilot seeds before G1 is allowed into the canonical table.
- G2 multiplicative branch-success floor: `max(0.75, 0.80 * pilot_median_branch_success)`; context-only peak ceiling: `min(0.10, max(0.02, 1.25 * pilot_max_context_only_peak))`; additive replacement fails the mechanism only if it cannot reach both of those criteria at any swept strength.
- G3 redirect-success floor: `max(0.60, 0.80 * pilot_median_redirect_success)`; irrelevant-phase change ceiling: `max(0.05, 1.25 * pilot_median_irrelevant_effect)` capped at `0.15`.
- G4 silent-advance floor: `max(1.0, 0.60 * pilot_median_items_advanced)` and publication/no-publication internal traces must remain equal to numerical precision (`atol=1e-12`) in deterministic arms.
- G5 is Boolean/selective rather than fitted: each of the four selective-intervention invariants must hold in at least 9/12 canonical seeds.

A gate is reported as failed, not tuned further, if the mechanically derived threshold is impossible or pilot behavior does not establish the prerequisite separation.

- [ ] **Step 1: Write threshold-derivation and seed-separation tests**

Use synthetic pilot dictionaries with known medians/maxima and assert exact derived thresholds. Assert canonical/pilot seed overlap raises `ValueError`.

- [ ] **Step 2: Verify gate tests fail**

Run: `pytest tests/test_gates.py -v`
Expected: FAIL before `gates.py` exists.

- [ ] **Step 3: Implement `derive_thresholds`, `evaluate`, and the three runner scripts**

`run_pilot.py` writes only `results/pilot.json`. `freeze_gates.py` reads that file, writes `results/frozen_thresholds.json`, and rewrites the numeric table in `GATES.md`. `run_gates.py` refuses to run without the frozen threshold file and writes all seed-level metrics plus summary pass/fail to `results/receipt.json`.

- [ ] **Step 4: Run the complete unit suite before scientific runs**

Run: `pytest -q`
Expected: all tests PASS.

- [ ] **Step 5: Run pilot seeds**

Run: `python scripts/run_pilot.py`
Expected: `results/pilot.json` with G0–G5 metrics for exactly seeds 0–3; no canonical pass/fail claims.

- [ ] **Step 6: Freeze the canonical thresholds and commit them before canonical seeds**

Run: `python scripts/freeze_gates.py`
Expected: updated `GATES.md` and `results/frozen_thresholds.json` containing the formulas, resolved numbers, and pilot seed list.

Commit boundary:

```bash
git add GATES.md results/pilot.json results/frozen_thresholds.json src/return_to_time_window/gates.py tests/test_gates.py scripts/run_pilot.py scripts/freeze_gates.py scripts/run_gates.py
git commit -m "test: freeze ReturnToTimeWindow v0 gates"
```

Do **not** alter thresholds after this commit.

- [ ] **Step 7: Run the canonical panel**

Run: `python scripts/run_gates.py`
Expected: `results/receipt.json` covering exactly seeds 100–111, with per-seed metrics and mechanical G0–G5 classifications.

- [ ] **Step 8: Implement and run the mechanism-first figure**

`scripts/make_figure.py` reads the representative G5 trace embedded in the receipt and renders `results/time_window.png` with four aligned panels: internal token activity/front; rhythm/context/veto signals; publication mask versus public activity; competing branch amplitudes. A reader must be able to see candidate continuation, veto/redirection, and silent internal advance without reading an accuracy table.

Run: `python scripts/make_figure.py`
Expected: non-empty `results/time_window.png`.

- [ ] **Step 9: Write README from the frozen receipt, including failures**

README structure:
1. the recovered question—computation manufactured out of local time windows;
2. the four causally separate controls;
3. one figure;
4. frozen G0–G5 table with exact receipt values;
5. what failed/collapsed;
6. lineage to KolmeOvea → TATWATASW → Rytmi;
7. biological claim boundary;
8. explicit statement that waveform/state-shaped events are not part of v0;
9. reproducible commands.

Do not call the machine a model of thought or consciousness. The introspective motivation may be described as motivation, not evidence.

- [ ] **Step 10: Final verification**

Run:

```bash
pytest -q
python scripts/run_gates.py --verify-existing
python scripts/make_figure.py
```

Expected: tests PASS; receipt re-evaluates without changing thresholds; figure regenerates successfully.

- [ ] **Step 11: Commit canonical results and documentation**

```bash
git add README.md results/receipt.json results/time_window.png scripts/make_figure.py
git commit -m "feat: complete ReturnToTimeWindow v0 gates"
```

---

## Self-Review Notes

- **Spec coverage:** G0–G5 are each owned by a task; internal/public separation, visible interventions, matched controls, preregistration boundary, CPU scale, negative-result preservation, inverse-model deferral, and waveform deferral are all represented. The DNA analogy needs no code and therefore stays in the spec/README discussion only.
- **Step scan:** implementation steps specify exact interfaces and tests without transcribing full function bodies. The one deliberate scientific decision that occurs after code—the numerical gate freeze—is made mechanical by fixed formulas and a hard commit boundary.
- **Type consistency:** all controls produce arrays consumed through one `ControlSignals` object; `ReplayTrace` is the sole source for downstream metrics/figures.
- **Review Focus:** all five high-risk conditions listed above now have an owning test task.
- **Proportion:** the plan is narrower than the spec's possible future directions; it stops at the first inspectable machine and deliberately excludes waveform, inverse-model, social, and cultural branches.
