# PrismFormer

**A self-replicating colony of tiny language models.** Every node runs the *whole* model on a plain CPU — it
trains it, serves it, and teaches its neighbours by passing them training examples. No datacenter, no GPU, no
sharding. The network is not one big model spread thin across machines; it is a colony of identical **cells**,
each a complete copy that keeps working if every other cell disappears.

Pure `double[]` CPU math — no TorchSharp, no GPU, no autograd — on `net8.0`, with one dependency
(`EvalApp.Consumer`, for tuned parallel training).

> **Status — honest.** The foundation is real and verified: the algebraic model, its bit-exact mergeable
> gradients, and the manifest that lets nodes reconstruct data by index. The colony *runs today at small scale*
> — Prism Studio nodes and a headless anchor discover each other over a relay mesh and bleed pairs, gossip,
> group-chat, and weight-average. What is **not** yet proven is large-scale convergence, and the token
> [economy](ECONOMY.md) is still vision, not code. This README says which is which.

---

## The one idea

Frontier distributed training is shaped by a constraint PrismFormer doesn't have: **the model doesn't fit on one
machine.** That forces systems like Petals or `torch.distributed` to *shard* one organism across matched GPUs — a
fragile pipeline where losing a peer severs the model.

PrismFormer is CPU-small, so we invert the whole design:

> **Every node holds the entire model.** A phone-class ARM chip and a desktop run the *same cell*. Kill any cell
> — every survivor still trains and serves the full model. And because gradients are **bit-exact mergeable**, a
> slow node's contribution and a fast node's combine into *identical* math: a ragtag swarm of mismatched machines
> stays mathematically coherent, which `torch.distributed` cannot say without matched hardware.

Smallness stops being a limitation and becomes the enabling property. But it only works because of a specific
piece of representation math — the part that lets a tiny model actually *compute*, and lets its gradients merge
to the bit. That foundation is next; the colony that stands on it follows.

---

## Why small works — the algebraic foundation (`AlgFormer`)

Most language models don't really *do* arithmetic. They pattern-match answers they've seen, because each number
is stored as a point along one line and those points all look alike to the output stage — so the model can recall
a memorised answer but can't work out a new one.

PrismFormer stores each number as a distinct **pattern of spinning dials** (clock hands turning at different
speeds). Two things follow:

- **It can actually compute.** To add two numbers it spins their dial-patterns together; the result is a new,
  *readable* pattern — so it adds and multiplies *combinations of numbers it never trained on* (within the range
  it has seen).
- **Its internals are readable.** Values inside the model are meaningful patterns, not opaque numbers, so what
  each part represents can be decoded.

Head-to-head against a conventional model of the *same size*, on maths whose numbers neither had trained on,
PrismFormer answered correctly **more than three times as often (53% vs 16%)**. (Full benchmark
[below](#benchmark).)

### The mechanism

Every concept is a bundle of **phasors** (unit complex numbers), and the single composition operator is
**complex multiply**:

| operation | meaning | effect on a number |
|-----------|---------|--------------------|
| **bind** `⊛` — per-component complex multiply | associate / superpose roles | **add** on the linear-phase band, **multiply** on the log-phase band |
| **unbind** — multiply by the conjugate | invert a binding | **subtract** / **divide** |
| **bundle** `+` — component-wise sum | form a set / superposition | — |

Because each value is its own **phase pattern** rather than a shared direction, a computed result is **decodable
by correlation**. `bind` computes; correlation names the result — algebra over faces, no calculator, no routing.
A word is a phasor face too (a signature hashed from its spelling), so words and numbers share one representation
and one readout; the *only* special-casing is that a token which parses as a number has its value written as
phase at encode time.

```csharp
// arithmetic with no training — a property of the codec alone
var bound = PhasorCodec.Bind(PhasorCodec.NumberFace(6), PhasorCodec.NumberFace(9));
PhasorCodec.DecodeSum(bound, 24);       // 15   (linear-phase band)
PhasorCodec.DecodeProduct(bound, 144);  // 54   (log-phase band)
```

### The model

A transformer shape (multi-head attention → per-token feed-forward → residuals, stacked, **causal**) in which
every dense map `W·x` is the algebraic **relation-bank**

```
y[i] = Σ_{k<S} bank[k][i] · x[(i+k) mod d]
```

— a bundle of `bind`+`permute` terms with `S·d` parameters instead of `d²`. At `S=d` it exactly reparametrises a
full matrix, so `S` is a lean↔full-rank control. Every parameter is itself a decodable **face**; the identity
components (linear+log phase) are **frozen** so numbers stay exact while only the orbital meaning learns; the
readout is tied to the face table. The backward pass is hand-derived and **gradchecked**, and the hot inner cell
is vectorised (`System.Numerics.Vector<double>`), bit-identical to the scalar form.

### The property that makes a colony possible

Because the substrate is exact and gradients accumulate into a **detached, mergeable buffer** (the model is
read-only during backprop), a minibatch splits into independent shards that reduce with one Adam step —
**bit-for-bit the serial result**, since the gradient of a sum is the sum of gradients. No computation graph to
synchronise, no parameter server. The *same* interface that fans a batch across CPU cores fans it across
machines. **This is the gold mine under the gold mine:** a CPU-small model with exactly-mergeable gradients is
the one case where a full-replication swarm — every node a whole cell that both trains and serves — actually
works. Big models can't (they must shard → fragile). PrismFormer can.

---

## The colony — every node is a whole cell

Identical software on every machine; each cell carries the complete works. There is exactly one production model
shape, the **PRISM-2** spec (frozen so any two cells can merge), and the only knob is how much you train it:

```
PRISM-2 · char-level, vocab 96 (95 printable ASCII + STOP) · context 256 · 8 layers · shifts S=64
        · dim 256 (128 phasor comps, 64-real frozen identity prefix) · signature v96/d256/f64/c256/L8/S64
```

| Organ | What it is | In repo |
|---|---|---|
| **Genome** | full model weights (`AlgFormer`) | `AlgFormer.Serialize/Deserialize` |
| **Program** | the manifest — a deterministic data recipe, so workers rebuild data by index and only *positions* cross the wire | `IJobSource.Manifest()` / `FromManifest()` |
| **Metabolism** | train loop: pull positions → gradient → gossip merged delta → merge inbound (bit-exact) | `PrismTrainer`, `SerializeGradient` |
| **Voice** | causal generation with a **KV cache** (O(T)/token serve) off an immutable weight snapshot, so serving never blocks training | `AlgFormer.Prime/Step`, `StudioModel.Serve` |
| **Senses** | mesh discovery, ping-ranked neighbours, gossip membership, auto-reconnect | `SwarmChatter` |
| **Reproduction** | on invite, hand a peer a paste-able join code — never self-installs | `MqttRelay` room codes |

### Bleed — share examples, not weights

When cells "bleed," they mostly **don't** average weights (a blunt dial). A cell sends a small sample of
**training pairs** — raw high-value examples, or *distilled* `(input → my prediction)` pairs — which the
neighbour writes into its `gossip/` folder and trains on. A training pair is a readable fragment of skill: the
receiver absorbs a bit of your behaviour without ever copying your weights, so it stays itself. Cells also
exchange small **weight slices** (elastic averaging) and run backprop-on-wrong on pairs they get wrong. All
channels are capacity-capped (oldest ejected) and deduped, with per-pair provenance.

### Mesh coherence — gate + upgrade-in-place

- **Compatibility gate.** A cell advertises its spec signature in every hello; only exact-spec peers join, so a
  node on an old or incompatible shape can't corrupt anyone's weight slices — it's blocked until it updates.
- **Upgrade-in-place (no retrain).** Growing capacity (more shifts, longer context) is identity-preserving: new
  shift rows are zeroed (they add nothing to the algebraic sum, so output is byte-identical the instant you
  extend, then training fills them in). A checkpoint upgrades to the larger shape without starting over.

### The anchor

A cheap always-on box next to the broker runs a **passive anchor**: it relays gossip, weight-averages toward the
swarm, carries the group chat, and *drives* the conversation by asking peers to continue it — but it does not
train or answer, so its single core stays free. It keeps the colony discoverable and the learning ticking during
idle time. See **[SWARM.md](SWARM.md)** for convergence, verifiable work, poisoning defenses, and the honest hard
problems (bandwidth, Byzantine robustness, NAT).

---

## Prism Studio & Gym — how you run it

**The one insight:** *everything is a training pair.* The built-in curriculum, your own text, and neighbour
gossip are all the same `(context → target)` stream over one shared char vocabulary — one model, many pair
sources, some arriving over the network. Folders are the interface: drop raw text or `prompt<TAB>target` lines
under `data/`, and `gossip/` fills from neighbours.

- **Prism Studio** — a WinForms GUI: train the model, chat with it (REPL), watch live training samples, and join
  the network (bleed + group chat) from one window. Data lives under `%LOCALAPPDATA%\Prism`.
- **Prism Gym** — the headless host/worker/anchor CLI:

```
prismgym                      # short train run, then a REPL
prismgym train [cycles]       # run the gym curriculum, save the checkpoint
prismgym host [dataDir]       # open a relay room; fan gradient batches out to workers that join
prismgym headless <code>      # join a room as a worker (co-train + bleed, no window)
prismgym anchor <room>        # always-on passive relay/driver for a box next to the broker
prismgym ask <code> <prompt>  # one-shot: ask the swarm from the command line
```

See **[STUDIO.md](STUDIO.md)** for the data-plane design and build sequence.

---

## The economy (vision — no code yet)

A phase-2 idea captured in **[ECONOMY.md](ECONOMY.md)**: a market of *personal currencies* over the swarm.
Everyone mints their own coin (no mining); it's worth something because people need it to query your model, so
its price tracks your model's quality. Coin is minted at countersigned query/training events (proof-of-useful-
interaction), and every would-be exploit is designed to *become* the useful behaviour (wash-trading is
self-training; sybils are real nodes; spamming sponsors the swarm). PrismFormer's edge here is the same one that
powers the colony: the manifest makes work **challenge-and-recompute verifiable to the bit**, which most
crypto-ML has to bolt on. It is a multi-year distributed-ledger build in its own right — captured now, built only
after the swarm itself is real and used.

---

## Using the library directly

`AlgFormer` is a standalone, EvalApp-free library; the swarm is one layer up.

```csharp
using PrismFormer;

var model = AlgFormer.Mini(vocab: 4096);           // phasor face, 4 layers, 16-token context
model.Seed(id, PhasorCodec.Encode("cat"));         // frozen identity comps + learnable orbital, per vocab id

var trainer = new PrismTrainer(model);             // tuned EvalApp fan-out by default (else in-box TPL)
trainer.Train(data, epochs: 4);                    // data: IReadOnlyList<(int[] Ctx, int Target)>

int next = model.Predict(context);                 // next-token id
int[] more = model.Generate(prompt, maxNewTokens: 8);
```

`PrismTrainer` splits each minibatch into shards that accumulate gradients independently (the model is read-only
during backprop, so shards are race-free), merges them, and applies one Adam step — bit-for-bit the serial
result. The fan-out runs through **EvalApp**'s resource-gated, adaptively-tuned `ForEach` (CPU width tuned to the
machine), and falls back to the in-box TPL when no license is present; both backends produce identical results. A
default key ships in `PrismTrainer.DefaultEvalAppKey` (EvalApp is the author's own product, embedded by
permission); override per instance or via `PRISM_EVALAPP_KEY`.

## Source layout

| file | contents |
|------|----------|
| `src/PhasorCodec.cs` | the uniform phasor face: `Encode`, `NumberFace`, `Bind`/`Unbind`/`Bundle`, correlation `DecodeSum`/`DecodeProduct`, `IsNumber` |
| `src/PhasorLayout.cs` | face layout — 16 linear-phase + 16 log-phase (frozen identity) + 96 orbital comps → 128 comps, **Dim 256**, 64 reals frozen |
| `src/AlgFormer.cs` | the model: causal forward, manual/gradchecked backward, vectorised cell, KV-cache serve (`Prime`/`Step`), `Save`/`Load`/`LoadUpgrade`, `Generate` |
| `src/PrismSpec.cs` | the frozen PRISM-2 spec + signature + mesh-compatibility / upgrade rules |
| `src/PairSource.cs` | the data plane — char vocab, `(ctx→target)` pairs from folders + chat + gossip, `GossipInbox` |
| `src/PrismTrainer.cs` | data-parallel trainer (EvalApp fan-out / in-box TPL) |
| `src/SwarmChatter.cs` | the mesh — discovery, gossip, pair/weight-slice bleed, group chat, spec gate |
| `studio/PrismGym/` | `StudioModel` (one model, many pair sources), `HeadlessNode` (host/worker/anchor), skill generators |
| `studio/PrismStudio/` | the WinForms GUI |
| `bench/` | the benchmark (below) |

---

## Benchmark

`bench/` compares `AlgFormer` with a parameter-matched dense transformer (its shape searched to match the
parameter budget) under a stable **train vs held-out** split — held-out operands and instances are never trained,
so held-out accuracy measures generalisation.

```
dotnet run --project bench -c Release            # the compute + reasoning suite (150 epochs)
dotnet run --project bench -c Release -- --lm     # next-token language modelling
dotnet run --project bench -c Release -- --scale  # held-out accuracy vs model size
dotnet run --project bench -c Release -- --extrap # out-of-range magnitude extrapolation
```

**Compute & reasoning** — 186,880 params/model, 150 epochs, train / **held-out** accuracy:

| task | transformer | PrismFormer |
|------|-------------|-------------|
| add | 9% / 0% | 100% / **44%** |
| sub | 6% / 8% | 99% / **25%** |
| mul | 9% / 0% | 100% / **42%** |
| div | 10% / 0% | 99% / **12%** |
| gt (a>b) | 38% / 29% | 100% / **94%** |
| max | 24% / 31% | 100% / **97%** |
| min | 19% / 25% | 100% / **97%** |
| copy | 27% / 29% | 100% / **100%** |

**Held-out compute generalisation: 15.9% (transformer) vs 52.5% (PrismFormer), +36.6 points** at matched
parameters — on unseen operands PrismFormer computes where the transformer stays near zero.

**Language modelling** (`--lm`, ≈94k params each, next-**char**): PrismFormer **42.3%** acc / **3.369** bits/char
vs transformer 32.7% / 3.676 — the algebraic architecture is competitive at language, not only arithmetic.

**Scaling** (`--scale`): because a relation-bank costs `S·d` per map instead of `d²`, PrismFormer reaches ≈80%
held-out at the smallest size (a level the transformer never reaches) and stays stable across depth.

**Honest limits.** Out-of-*range* extrapolation is ≈1% for both (the trained model learns the mapping
statistically, not via the codec's homomorphism); relational reverse-inference is 0% for both at this scale;
`seq`/`parity` are memorised, not computed. PrismFormer's edge is generalisation to unseen operand *combinations
within* the trained range.

---

## Requirements

`.NET 8`. Restore pulls `EvalApp.Consumer` from NuGet; `dotnet build` compiles the library, benchmark, gym, and
(on Windows) the Studio GUI.

## License

Copyright © 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.

**Source-available.** Anyone is welcome to read, build, run, and independently verify this code and its results.
Commercial or production use, redistribution, and derivative works require prior written permission — see
[LICENSE](LICENSE). A **defensive publication** of the methods is in [PRIOR_ART.md](PRIOR_ART.md).
