# Reverse inference: findings (2026-07-18)

**Context.** PrismFormer paper 1 (§4.2, limitation ii, §6) reports that *reverse inference* is **0% for both
models** at this scale: train "opposite of A = B", hold out "opposite of B = A", and neither the algebraic
model nor the transformer recovers the symmetric direction. This note asks whether that 0% is fundamental or
whether scratchpad reasoning / the codec algebra can beat it. Bench: `dotnet run --project bench -c Release -- --revinfer`.

## Setup

24 symmetric pairs (arms 1 and 3); a shared 40-token pool for arm 2. Three arms, each reporting held-out
**reverse** accuracy:

1. **Closed-book trained** (the paper's setup): AlgFormer trained forward only, reverse held out, queried closed-book.
2. **Open-book scratchpad**: the forward fact is provided in context (`a opp b | b opp = ?`) and the model learns
   to *select the other element*. Held out on **novel pairings of tokens it has already seen**, so this isolates
   whether the reversal generalises rather than testing out-of-vocabulary tokens.
3. **Algebra (HRR associative memory, no training)**: each pair is stored as the commutative bind `L ⊛ R` in a
   single bundle; the reverse is retrieved by unbinding the query (bind with its conjugate).

## Results

| arm | forward | **reverse (held-out)** |
|---|---|---|
| 1. closed-book trained | 100% | **0%** |
| 2. open-book scratchpad (novel pairings of seen tokens) | — | **100%** |
| 3. algebra, HRR memory — 24 pairs | 83% | **83%** |
| 3. algebra, HRR memory — 12 pairs | 100% | **100%** |

## What it means

- **The 0% is a directional-recall failure, not a reasoning failure.** A closed-book model trained on A→B stores
  a *directed* map; nothing in it says the relation runs backward. Reversing is not the hard part.
- **Scratchpad (arm 2) fixes it by removing the recall bottleneck.** Provide the forward fact and the reverse
  becomes "select the element that is not the query" — the same copy/select PrismFormer already does at 100% on
  the copy task. It generalises to unseen pairings (100%). Caveat: this requires the fact to be *available*
  (retrieval / open-book); it does not conjure a fact the model was only ever taught in one direction.
- **Algebra (arm 3) fixes it for free.** Store a symmetric relation as a *commutative* bind and forward = reverse
  **exactly** (`L ⊛ R ⊛ R⁻¹ = L`, and the bind commutes), with no training. The only shortfall from 100% is HRR
  crosstalk as the bundle fills (100% at 12 pairs, 83% at 24), which is a capacity property, not asymmetry. This
  is "arithmetic as algebra" extended to relations: the reverse falls out of the codec the way `+` and `×` do.

## Honest boundaries

- Arm 2's win is conditional on the fact being provided (open-book / retrieval). It reframes the limitation
  ("reverse is easy once you have the fact") rather than making a closed-book directional model reverse itself.
- Arm 3 is capacity-limited (HRR crosstalk grows with the number of bundled pairs); it demonstrates the algebra,
  not an unbounded memory.
- These are single-configuration runs on a toy relation set. They show the *mechanism*, consistent with the rest
  of paper 1, not a large-scale result.

## Bearing on the paper

Turns limitation (ii) from a flat "0% for both models" into "0% closed-book, recoverable by scratchpad
(retrieval) or by algebraic storage (commutative bind)". **Merged into paper 1 as §4.9 "Reverse inference,
recovered"** (the three-arm table + diagnosis); limitation (ii), the §4.2 caveat, the abstract, the setup table,
and the reproducibility list were updated to point at it. Bench: `--revinfer`.
