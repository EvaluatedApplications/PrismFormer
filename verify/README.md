# Verify PrismFormer's claims

Everything here lets you reproduce the core claims **yourself**, from source, in minutes. No trust
required. You need the **.NET 8 SDK** and nothing else (pure CPU, no GPU, no accounts).

## Quick check — seconds, no training

From the repository root:

```
dotnet run --project verify/Verify -c Release
```

It runs two decisive checks and prints `PASS`/`FAIL`:

**1. Arithmetic is a property of the representation, not a trained calculator.**
Two numbers are encoded as phasor faces and *bound* together with one per-component complex multiply.
The result decodes to their **sum** (linear-phase band) and their **product** (log-phase band) with no
training and no lookup table. The check runs every operand pair in a small range and reports exact-match
accuracy. You should see `bind(face(6), face(9)) -> sum = 15, product = 54`.

**2. Training is exactly parallelisable — the swarm property.**
A batch is split into shards and its gradients are computed both sequentially and in parallel, then
reduced in the same fixed order. The two results are **bit-for-bit identical** (an FNV checksum over
every gradient value), and re-running is identical too. Because the substrate is exact IEEE `double`,
a slow machine's contribution and a fast machine's combine into the same bits — which is exactly what
lets a swarm of mismatched CPUs train one coherent model.

## The headline result — minutes

The generalisation claim (PrismFormer vs a **parameter-matched** dense transformer on operand
combinations **neither model was trained on**) is the existing benchmark:

```
dotnet run --project bench -c Release             # compute + reasoning, train vs HELD-OUT
dotnet run --project bench -c Release -- --lm     # next-character language modelling
dotnet run --project bench -c Release -- --scale  # held-out accuracy vs model size
```

Both models are gradient-checked before the race, their parameter budgets are matched (the transformer's
shape is searched to fit the same budget), and the **held-out** split — operands and instances never
trained on — is what measures generalisation. Reported figures: ~**52.5%** (PrismFormer) vs ~**15.9%**
(transformer) held-out on compute; **42.3%** vs **32.7%** next-character on language.

## What is *not* claimed here

Kept honest on purpose. These checks verify the representation and the exact-merge property; the
benchmark verifies small-scale generalisation. **Large-scale convergence of the full colony is not yet
proven**, out-of-*range* numeric extrapolation is ~1% for both models, and the token economy in
`ECONOMY.md` is a design, not code. See the main [`README.md`](../README.md) for the full scope and the
[`PRIOR_ART.md`](../PRIOR_ART.md) disclosure for how this relates to prior work (Holographic Reduced
Representations / VSA and Random Fourier Features).
