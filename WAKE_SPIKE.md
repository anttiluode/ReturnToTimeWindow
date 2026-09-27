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

## Interpretation boundary

Even a pass would show only that **activity-recruited delayed local inhibition can implement a useful moving access mask in this constructed threshold model**. It would not show that Martinotti cells carry cortical vortices, that cortical waves arbitrate memory this way, or that the episode/schema decomposition is a biological mechanism.
