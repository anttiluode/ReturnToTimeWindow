# Spike: The Wake Behind the Wave

Status: **exploratory falsifier, not a numbered stage**.

This spike asks one narrow question inspired by the current time-window thread:

> Can an early event in a propagating episode recruit a delayed, spatially local apical-inhibition wake that suppresses a later schema candidate **where episode evidence exists**, while leaving genuine episode gaps available for schema fill?

The biological mapping is deliberately weak. `wake` is an SST/Martinotti-like *analogy*, not a claim that cortical spiral waves are SST waves or that Martinotti cells implement this exact circuit.

## Frozen toy assay

The assay has 12 spatial positions. Three positions are episodic gaps, chosen from the seed. Each position has a 10-step local window:

- step 2: an early episodic event, if that position is occupied;
- step 6: a later schema candidate;
- an occupied episodic event recruits a wake after 2 steps;
- wake decays with `tau=4` steps and spreads to immediate neighbors with weight `0.22`;
- wake acts only by reducing the later apical/schema amplification;
- it never supplies or deletes content.

Numerical parameters were fixed during construction from threshold arithmetic and exploratory seeds 0–31, **before canonical seeds 100–111**:

- threshold `1.0`
- episode drive `1.25`
- schema basal `0.72`
- schema apical contribution `0.48`
- wake strength `0.95`
- delay `2` steps
- temporal-shift control: additional delay `5` steps

This toy deliberately puts the schema candidate near threshold: without apical inhibition it can fire (`0.72 + 0.48 > 1`); a sufficiently aligned wake can remove the apical advantage without touching the basal term.

## Replacement controls

1. **Aligned wake**: recruited by actual early episode events, correct location and delay.
2. **Matched tonic**: exactly the same total decision-time apical inhibition as the aligned wake, spread uniformly across all 12 decisions.
3. **Spatial shuffle**: same event count, delay, decay, spread and wake mass, but each source's wake is delivered through a seed-fixed random permutation of positions.
4. **Temporal shift**: same event count, location, spread and wake mass, but the wake arrives 5 steps later.

The tonic comparison is intentionally matched at the times when apical inhibition can matter, not merely over irrelevant simulation time.

## Metrics

- `occupied_schema_intrusion_rate`: fraction of occupied positions where the competing schema candidate also crosses threshold. Lower is better.
- `gap_fill_rate`: fraction of episodic gaps where the schema candidate crosses threshold. Higher is better.
- selectivity = `gap_fill_rate - occupied_schema_intrusion_rate`.

The desired behavior is not generic suppression. It is the conjunction:

```text
episode present -> schema intrusion suppressed
episode absent  -> schema remains able to fill
```

## Canonical gate, frozen before seeds 100–111

The spike passes only if all of the following hold:

- aligned wake achieves intrusion <= 0.10 and gap fill >= 0.90 in at least 10/12 canonical seeds;
- matched tonic achieves that same dual job in at most 2/12 seeds;
- tonic decision-time inhibition sum exactly matches the aligned wake in every seed;
- spatial-shuffle and temporal-shift controls preserve generated wake mass exactly;
- pooled aligned-wake selectivity exceeds both spatial-shuffle and temporal-shift selectivity by at least 0.50.

No parameter changes are allowed after canonical seeds are run. Failure remains a result.

## Canonical result

The frozen canonical panel **passes**:

| condition | occupied schema intrusion | gap fill | selectivity |
|---|---:|---:|---:|
| aligned wake | **0.000** | **1.000** | **1.000** |
| matched tonic | **0.000** | **0.000** | **0.000** |
| spatial shuffle | 0.907 | 1.000 | 0.093 |
| temporal shift | 1.000 | 1.000 | 0.000 |

Aligned wake meets the dual criterion in **12/12** canonical seeds. Matched tonic meets it in **0/12**. The decision-time inhibition integral is exactly matched between aligned wake and tonic in every seed, and the shuffled dynamic controls preserve generated wake mass.

The narrow result is therefore real *inside this toy*: **where and when inhibition arrives matters more than its total amount** for simultaneously protecting occupied episode positions and leaving gaps available to schema fill.

## Post-canonical controls: what did *not* earn its keep

Two sweeps were run only after the frozen canonical result.

**Delay.** With every other parameter fixed, delays 1–4 steps retain perfect selectivity; delay 0 and delays 5–8 lose it completely. The mechanism therefore has a finite timing band rather than being equivalent to arbitrary suppression.

**Lateral spread in the original one-lane assay.** Setting immediate-neighbor spread all the way to **0.0** leaves the result unchanged. Increasing it through 1.2 also leaves the result unchanged.

That is an important negative result. The original assay did **not** demonstrate a special role for Martinotti-like lateral topology. It demonstrated a role for **activity-recruited, correctly timed local apical suppression**.

## Harder spatial fork: make geometry necessary

A second, explicitly post-canonical toy removes that loophole. It uses two distinct nearby trajectories:

```text
episode lane   ---- E ---->
                      |
                      | generic spatial inhibitory kernel
                      v
schema lane    ---- S ---->
```

At an occupied position, the episode lane fires early. The competing schema lane presents its candidate later. The inhibitory rule receives **no competitor label**: its strength depends only on physical distance from the early episode event. At an episode gap there is no early event, so no wake is generated.

With all original threshold/timing parameters unchanged:

| competitor geometry | occupied schema intrusion | gap fill |
|---|---:|---:|
| adjacent, distance 1 | **0.000** | **1.000** |
| distance 2 | 1.000 | 1.000 |
| distance 3 | 1.000 | 1.000 |
| distance 4 | 1.000 | 1.000 |
| same-site-only inhibition | 1.000 | 1.000 |

So in this constructed fork, **lateral geometry is causally necessary**: the competing trajectory has to lie inside the generic local spatial kernel. Same-site inhibition cannot touch it, and moving it farther away removes the effect.

This still is not a biological finding or new inhibitory principle; it is classical local lateral inhibition embedded in a moving sequence problem. Its value is narrower: it shows that the vortex/Martinotti picture can be turned into a geometry-dependent computation without adding an explicit `suppress_competitor` variable.

Machine-readable receipt: `results/wake_spike.json`.

## Interpretation boundary

What the spike has earned:

```text
recent local activity
    -> delayed inhibition
    -> a finite susceptibility window
    -> different effect on a later competing trajectory
```

And, in the harder fork:

```text
spatial adjacency
    -> whether the competing trajectory falls inside that wake
```

What it has **not** earned:

- that Martinotti cells carry cortical vortices;
- that cortical spiral waves actually recruit this wake;
- that this is how episodic versus schematic memory is arbitrated;
- that SST/Martinotti circuitry is uniquely needed rather than one biological implementation of generic delayed lateral inhibition.

The next serious step should leave hand-authored threshold slots behind: couple the wake to an actual propagating population wave and ask whether the moving inhibitory field bends or blocks a competing self-running trajectory without being told where the fork is.
