# A Self-Replicating Swarm of Tiny AIs
*Phasor-face transformers (PrismFormer): arithmetic as algebra, and bit-exact, mergeable gradients on plain CPUs.*

**Dongyang Stephen Chen**, Evaluated Applications

**DOI:** [10.5281/zenodo.21384525](https://doi.org/10.5281/zenodo.21384525)
**Cite this as:** Chen, D. S. (2026). *A Self-Replicating Swarm of Tiny AIs: Phasor-Face Transformers (PrismFormer) with Arithmetic as Algebra and Bit-Exact, Mergeable Gradients.* Zenodo. https://doi.org/10.5281/zenodo.21384525

*Draft, 2026. Correspondence: dongyang.stephen.chen@gmail.com. The exact source, benchmarks, and runnable verification kit that produce every number below are archived as a frozen snapshot alongside this document (repository tag `paper-v1`); every quantitative result is produced by a script in that snapshot and reproducible with the command given in its section.*

---

## Abstract

I present **PrismFormer**, a transformer-shaped sequence model in which every token, word or
number, is a **phasor face**: a bundle of unit complex numbers composed by one algebraic operation,
per-component complex multiplication ("binding"). Numbers are encoded so that binding performs
*arithmetic*: addition on a linear-phase band, multiplication on a log-phase band, with results read
back by correlation, so a computed value is *decodable* rather than merely reachable. This numeric
representation is a direct application of the frequency-domain (phasor) form of **Holographic Reduced
Representations** [Plate 1995; 2003] and of **Random Fourier Features** [Rahimi & Recht 2007]; my
contribution is to make it the trainable substrate of a transformer whose dense maps are replaced by
parameter-lean **relation-banks**. Because that substrate can be evaluated read-only during
backpropagation, gradients accumulate into a detached buffer: a minibatch split into shards and summed
is **bit-for-bit identical** to the serial result, which enables a **full-replication training swarm**,
in which every node holds the whole model on a plain CPU, that stays mathematically coherent across
mismatched hardware, unlike sharded systems that need matched accelerators.

I report results honestly and reproducibly, against a parameter-matched dense transformer at each
setting. Arithmetic emerges from the codec with **no training** (binding two number faces decodes to
their exact sum and product; 100/100 and 81/81 single-digit pairs exact). Given the same *worked,
column-by-column* problem, PrismFormer learns multi-digit addition, subtraction, and multiply/divide
by a digit and generalises to **unseen** operand pairs (add 54.9%, sub 70.0%, mul 27.7%, div 33.2%
held-out) where a matched transformer given the identical problem stays near zero; and its working is
legible one column at a time with no trained probe. On comparison and relational tasks it generalises
better than a transformer that *fits the same training data* (held-out 65.9% vs 49.1% on the fair
tasks; both reach 100% on copy). It is a competitive character language model at matched size (41.0%
vs 21.7% next-character accuracy). On pure memorisation of an arbitrary key-value store, with nothing to
generalise from, a matched transformer's reliable capacity runs out and it destabilises while PrismFormer
holds full recall to at least twice as many facts (a single exploratory sweep). I report a two-sided
ablation of my central design choice: freezing
the numeric identity is *not* what makes any single operation generalise (with one task in isolation, an
unfrozen identity does better), but under realistic **shared multi-task load** freezing wins the held-out
comparison, because the identity is a residual every task shares, and drifting it to fit one operation
corrupts the rest. So freezing buys both legibility and shared-corpus generalisation, and I say exactly
when each holds. Out-of-range extrapolation, relational reverse-inference, and large-scale convergence of
the full colony remain open.

---

## 1. Introduction

I should say up front what this is and where it came from, because the path is what shaped the design.
I am not a machine-learning researcher. I am a full-stack developer with a background in computer
networking, and this started as a stubborn question rather than a gap in a literature I did not yet
know: if you wanted a model to *reason* instead of recall, what would you actually have to represent,
and how?

I began from a few axioms and a picture. The picture was a kind of Platonic space: somewhere concepts
sit at definite addresses and the relationships between them are moves you can make, so that reasoning
is navigation under rules rather than lookup. The axioms were small. A concept should have a stable
address. Combining two concepts should be one well-defined operation, not a learned table. And the
result of that operation should be something you can *read back out*, not just something the network
happens to be able to reach. I wanted the structure to carry the reasoning, and the network only to
learn which moves to make.

Every time I worked out what such a representation would have to be, I landed on the same shape: a
*codec*. A fixed, deterministic way to write any concept, word or number, down as a vector, chosen so
that one operation composes two of them and correlation reads the answer back. As I refined the codec,
the encoding that fell out was phasors: each concept a bundle of unit complex numbers, composed by
complex multiplication. Write numbers as phases and they *add* on one band and *multiply* on another,
so arithmetic comes out of the very same bind that composes words. That was the moment it felt real. I
had not taught it to add; the representation just did.

Only afterwards did I find that the mathematics I had backed into is decades old. Representing concepts
as unit-magnitude phasors and binding them by complex multiplication is the frequency-domain form of
Plate's Holographic Reduced Representations [Plate 1994; 1995; 2003], later called FHRR in the Vector
Symbolic Architectures literature [Gayler 2003; Kanerva 2009]; writing a scalar as phases of random
frequencies and reading it back by correlation is a Random Fourier Feature map [Rahimi & Recht 2007]. I
cite that work in full in §2 and claim none of it as mine. What I did was reach it from the
reasoning-space side instead of the kernel or VSA side, and then take it somewhere it had not gone:
make the phasor codec the trainable substrate of a transformer, and build around it the training system
a networking person would build.

That is the part where my actual background earns its keep, and it turns out to be the other half of
the paper. Because the substrate is only read, never mutated, while gradients pile up in a separate
buffer, a batch split across shards and summed lands on a result that is *bit-for-bit identical* to
running it on one machine (§3.3, §4.6). Bit-exactness is a networking property before it is a learning
one: a slow node's contribution and a fast node's combine into exactly the same bits, so a swarm of
mismatched, unreliable CPUs can train one coherent model with no parameter server and no matched
hardware, every node holding the whole model the way every piece of a hologram still holds the whole
image. (That holographic picture of memory has a long history as a *metaphor* for the brain, in
Pribram's holonomic theory [Pribram 1971; 1991]; I use it only as a metaphor and lean on the
mathematics of HRR, not on any neuroscience claim.)

For context on why numbers are worth this trouble: language models handle them unevenly. Standard
embeddings capture magnitude only partly, and that fidelity falls off with sub-word tokenization and
past modest magnitudes [Wallace et al. 2019], while numbers rarely get any special representational
treatment [Thawani et al. 2021]. Embedded as points along shared directions, two different magnitudes
look nearly collinear to a dot-product readout, so the model recalls a memorised answer instead of
computing a new one. The codec is a direct answer to that: a computed value is decodable, not merely
reachable.

So the paper is two halves of one idea, and I make three contributions:

1. **A phasor-face transformer substrate (§3.1–3.2).** Every parameter and activation is a decodable
   face; one binding operation performs symbolic association and arithmetic at once; numeric embeddings
   are seeded from the phasor codec so arithmetic is decodable from the representation with no training;
   dense maps are replaced by relation-banks costing `S·d` parameters instead of `d²`.
2. **Exactly-mergeable gradients (§3.3).** The substrate is read-only during backprop, so a minibatch
   splits into independent shards whose summed gradients equal the serial result *bit for bit*. I
   verify this to the bit.
3. **A full-replication CPU swarm (§3.3).** Because every node holds the whole model and gradients
   merge exactly, a swarm of mismatched CPUs trains one coherent model. It is the opposite design to
   sharded systems such as Petals [Borzunov et al. 2023], which split one model across matched
   accelerators and lose it if a peer drops.

I try to be honest about scope, and I ablate my own design choices both ways: an ablation (§4.3) shows
that *freezing* the numeric identity does not help any single operation in isolation (unfreezing does
better there), yet it wins under shared multi-task load, so I present it as legibility *plus* a
shared-corpus generalisation benefit, with the isolated-task caveat stated. Large-scale convergence and
out-of-range extrapolation are open problems I do not claim to have solved.

---

## 2. Related work and attribution

I build directly on prior inventions and cite them plainly; the novelty here is in their combination
and in the systems consequence, not in the primitives. In one line, an AlgFormer is the transformer
block [Vaswani et al. 2017] with three parts swapped in: **FHRR phasor binding** [Plate 1994; 2003] as
the compose operator, a **Random Fourier Feature** scalar code [Rahimi & Recht 2007] as the number
embedding, and **banded circular operators** [Cheng et al. 2015; Sindhwani et al. 2015] in place of
every dense matrix, trained by data-parallel SGD [McMahan et al. 2017] made bit-exact. The rest of
this section credits each part; §3 shows how they compose into a whole model.

**Binding and holographic representations.** Tensor-product binding [Smolensky 1990] and Holographic
Reduced Representations [Plate 1995] introduced fixed-width distributed representations of structure
with a circular-convolution bind and a correlation unbind. The frequency-domain form (Plate's
"unitary vectors" [Plate 1994; 2003], later named FHRR in the VSA literature) uses unit-magnitude
**phasors** bound by element-wise complex multiplication, the exact bind I use. Vector Symbolic
Architectures [Gayler 2003] and Hyperdimensional Computing [Kanerva 2009], surveyed by Kleyko et al.
[2022; 2023], generalise this family. My phasor codec is an instance of this frequency-domain HRR; I
do not claim to have invented phasor binding.

**Encoding scalars as phases.** Writing a value `v` as `[cos(vθ_k), sin(vθ_k)]` over frequencies `θ_k`
and reading it back by correlation is a Random Fourier Feature map [Rahimi & Recht 2007]: the
correlation approximates a kernel peaked at the true value, which is why a computed result is
*decodable* rather than merely reachable. Prior work characterises how numeracy is only partially
captured by standard embeddings and where it breaks down [Wallace et al. 2019], and surveys the broader
need for better numeric representation [Thawani et al. 2021].

**Parameter-efficient linear maps.** Replacing dense `W·x` with structured operators is well studied:
circulant projections [Cheng et al. 2015] and structured transforms [Sindhwani et al. 2015]. My
relation-bank `y[i] = Σ_{k<S} bank[k][i]·x[(i+k) mod d]` is a sum of `S` diagonally-scaled cyclic
shifts (a "banded" family over the cyclic group ℤ/dℤ); at `S=d` it reparametrises a full matrix, so `S`
is a lean-to-full-rank control.

**Architecture and optimisation.** The block structure (attention, then per-token feed-forward, with
residuals) follows the transformer [Vaswani et al. 2017]; I train with Adam [Kingma & Ba 2015].

**Distributed and decentralised training.** Data-parallel synchronous SGD sums gradients across
workers; my exact-merge is that identity made bit-exact on a CPU substrate. Weight-slice averaging
between peers is a form of elastic/federated averaging [McMahan et al. 2017]. In contrast to
model-parallel collaborative inference [Borzunov et al. 2023], every PrismFormer node replicates the
whole model, so no peer is load-bearing.

---

## 3. Method

### 3.1 The phasor codec

A **face** is a vector of `C = 128` unit complex numbers, stored as `2C = 256` interleaved reals
`(cos φ₀, sin φ₀, cos φ₁, sin φ₁, …)`. Write it as `f = (e^{iφ_0}, …, e^{iφ_{C−1}})`; the meaning lives
in the phases `φ`. Three operations act on faces:

- **bind** `⊛`, per-component complex multiplication `(a ⊛ b)_c = a_c b_c`, so phases **add** per
  component, `φ^a_c + φ^b_c`. In the interleaved reals: `h[2c] = a[2c]b[2c] − a[2c+1]b[2c+1]`,
  `h[2c+1] = a[2c]b[2c+1] + a[2c+1]b[2c]`.
- **unbind**, bind with the conjugate, so phases **subtract**.
- **bundle**, component-wise real sum, superposing faces into a set.

Readout is **correlation**, the real inner product `⟨a, b⟩ = Σ_c (a[2c]b[2c] + a[2c+1]b[2c+1])`,
optionally restricted to a band of components. The `C` components split into three bands:

| band | comps | phase written for value `v` | what `bind` does | frozen |
|------|-------|------------------------------|------------------|--------|
| linear | 16 | `φ_k = v · θ_k` | **adds** values | yes (identity) |
| log | 16 | `φ_k = ln v · θ_k`  (`v > 0`) | **multiplies** values | yes (identity) |
| orbital | 96 | learned tail (symbol seed / trained) | — | no (learned) |

The frequencies `θ_k` are fixed and deterministic (drawn from a hash). The bind identities are then just
exponent arithmetic. For two number faces on the **linear** band,
`e^{i v θ_k} · e^{i w θ_k} = e^{i (v+w) θ_k}`, so the bound face **is** the number face of `v + w`; on
the **log** band, `e^{i \ln v · θ_k} · e^{i \ln w · θ_k} = e^{i \ln(vw) · θ_k}`, the number face of
`v·w`. One bind therefore computes a sum on one band and a product on the other at the same time. The
value is read back by correlation against the codebook: `DecodeSum(h) = argmax_c ⟨face(c), h⟩` over the
linear band, `DecodeProduct` the same over the log band. As a Random Fourier Feature map [Rahimi &
Recht 2007], that correlation is a kernel peaked at the true value, so a *computed* result is nameable,
not merely reachable: there is no table and no arithmetic unit, only a phase pattern named by
correlation.

A non-numeric token is handled by the same machinery with a different on-ramp: it hashes (FNV-1a) to a
deterministic phasor **signature** on the identity band and a **seed** on the orbital tail. Numbers and
symbols thus share one space and one algebra; the only difference is that a number's *value* becomes
phase while a symbol's *spelling* does. The identity band (value or signature) is a concept's ground
truth and is **frozen**; the orbital tail is what training moves. This is exactly the frequency-domain
HRR of Plate [1994; 2003] with an RFF scalar encoding [Rahimi & Recht 2007]; the contribution (§3.2) is
to make it a trainable transformer substrate.

*Verifiable now, no training:* `bind(face(6), face(9))` decodes to `15` (linear band) and `54` (log
band). Over all single-digit pairs the codec is exact: **100/100** sums (0–9) and **81/81** products
(1–9). Run `dotnet run --project verify/Verify -c Release`.

### 3.2 The relation-bank transformer (AlgFormer)

AlgFormer keeps the transformer block shape [Vaswani et al. 2017] and changes two things: every
activation is a face, and every dense linear map `W·x` (a `d×d` matrix, `d²` parameters) becomes a
**relation-bank** of `S` faces (`S·d` parameters):

```
y[i] = Σ_{k=0}^{S−1} bank[k][i] · x[(i+k) mod d]
```

Read it algebraically: `x[(i+k) mod d]` is `x` cyclically rotated by `k`, `bank[k]` scales it
component-wise, and the sum over `k` bundles `S` such shifted-and-scaled copies. So a bank is a sum of
`S` diagonally-scaled cyclic shifts, a *banded circular operator* over ℤ/dℤ. At `S = d` the shifts
span the group and the bank reparametrises a full `d×d` matrix; at `S ≪ d` it is lean. `S` is the dial
between the two, and I run it well below `d` (single digits to a few dozen shifts against `d = 256`).

A block takes the `T×d` stack `X` of input faces (row `X[t]` = the face at position `t`) and computes:

1. **Embed.** `X[t] = Emb[tok_t] + Pos[t]`. `Emb` is seeded from the codec (§3.1) with its identity
   band frozen; `Pos` is a learned position face.
2. **Project.** `Q = bank_Q(X)`, `K = bank_K(X)`, `V = bank_V(X)`: three separate relation-banks.
3. **Causal attention.** `a[t][j] = softmax_{j ≤ t}( ⟨Q[t], K[j]⟩ / √d )`, then
   `ctx[t] = Σ_{j ≤ t} a[t][j] · V[j]`. Position `t` attends only to `j ≤ t`, so the same forward pass
   serves training and KV-cached generation.
4. **Attention residual.** `O = bank_O(ctx)`, `M = X + O`.
5. **Gated feed-forward.** `A = bank_A(M)`, `G = bank_G(M)`, then `Z = A ⊙ σ(G)` (a GLU gate; a
   complex-bind variant `Z = A ⊛ G` is also gradient-checked), and `F = bank_F(Z)`.
6. **FFN residual.** `Y = M + F`, passed to the next block. Depth is `L` blocks.

There is **no layer normalization** anywhere: the residual stream and the bounded gate keep activations
in range, and a linear learning-rate decay keeps the multiplicative feed-forward stable. The readout is
**tied** to the embedding table (`logit_w = C_w + ⟨Emb[w], Y_last⟩`), so the model names its answer by
correlating the final face against every token's face, the *same* correlation that decodes the codec
(§3.1). Training minimises cross-entropy on that softmax. The whole backward pass is hand-derived and
finite-difference gradient-checked at production dimensions; the inner cell is SIMD-vectorised over the
output index (`System.Numerics.Vector<double>`), bit-identical to the scalar form.

Each face separates a fixed **identity** region (the numeric value / symbol signature, the first
`FrozenReals` reals) from a learnable **orbital** tail. My implementation *freezes* the identity
region during training. I treat this as a design choice and test it both ways (§4.3): freezing does
**not** improve generalisation on a single operation in isolation (an unfrozen, still codec-seeded
identity does better there), but it **wins** under shared multi-task load, where the identity is a
residual every task shares and drifting it to fit one operation hurts the others. On top of that it buys
legibility: the identity stays on the codec geometry, so activations remain decodable with no probe
(§4.4). The load-bearing ingredient is the codec-seeded initialisation; freezing then adds shared-corpus
robustness and readability.

### 3.3 Exactly-mergeable gradients and the swarm

The loss over a batch is a sum of per-example losses, `L = Σ_i ℓ_i`, so the gradient is the sum of
per-example gradients, `∇L = Σ_i ∇ℓ_i`. That much is ordinary. Two implementation facts make the sum
reproduce **bit for bit** regardless of how it is scheduled:

1. **The substrate is read-only during backprop.** The forward pass caches activations; the backward
   pass writes only into a *detached* gradient buffer and thread-local scratch, mutating no parameter or
   shared state. So `∇ℓ_i` is a pure, deterministic function of example `i` alone; shards share nothing.
2. **The reduction is fixed-order.** Gradient buffers are summed in a fixed parameter-index order, and
   the algebraic cell itself accumulates in ascending-shift order. Floating-point addition is not
   associative, so the *order* is what fixes the bits; fixing it makes the result independent of thread
   timing.

Put together: split a batch into shards, accumulate each shard's `∇ℓ_i` into its own buffer, sum the
buffers in the fixed order, take one optimiser step. Running the shards **in parallel gives a gradient
identical to the bit to running them serially**, verified by an FNV checksum over every gradient value
(§4.6). There is no computation graph to synchronise and no parameter server. (Freezing is one clearing
of the identity-band gradient before the step, so exact numbers never drift.)

This is what the swarm is built on. Every node holds the whole model (full replication), trains on its
own data, and contributes gradient buffers (or averaged weight slices [McMahan et al. 2017]) that
merge without a coordinator. Because a slow CPU's buffer and a fast CPU's buffer combine into the same
bits, a colony of mismatched, unreliable machines stays one coherent model, and a new node joins simply
by receiving the model and, from then on, replicating and spreading it. That is the sense in which the
swarm is *self-replicating*. Capacity grows in place: extending the shift count or context
zero-pads the new parameters, which contribute nothing to the algebraic sum until trained, so a
checkpoint upgrades without a retrain. What I show here is that the mechanism runs and merges exactly
(§4.6); convergence of a large live colony is left open (§5).

---

## 4. Experiments

### 4.1 Setup

I compare PrismFormer against a **parameter-matched** dense transformer (a real encoder-style
transformer with hand-derived, gradient-checked backprop) whose shape is auto-searched to fit the same
budget at a trainable depth. Both models share vocab, context, training order, LR schedule, and
parameter budget at each setting; both pass a startup gradient check. Every number below is a mean over
several random seeds with its standard deviation, produced by the listed command.

| experiment (command) | PrismFormer | transformer | seeds × budget | data / held-out |
|---|---|---|---|---|
| §4.2 comparison/relational (`bench`) | 295,936 | 297,792 | 5 × 150 ep | 1,112 train / 309 held (12 tasks, operands 0–12) |
| §4.3 worked arithmetic (`--columnar`) | 353,296 | 352,784 | 4 × 15k steps | operands 0–99, ~20% of pairs held |
| §4.4 mechanistic (`--inspect`) | 177,421 | 169,213 | 8 × 6k steps | single-digit, ~20% held |
| §4.5 language (`--lm`) | 249,920 | 259,776 | 5 × 30 ep | 1,670-char corpus, contiguous held tail |
| §4.6 exact-merge (`verify/Verify`) | `Mini` | — | deterministic | — |
| §4.7 atomic capacity (`--capacity`) | 67,210 | 67,306 | early-stop, 16 N | random key→value, memorise |

### 4.2 Comparison and relational generalisation

On tasks where the answer is **not** a trivial algebraic function of the input faces, so the codec
gives no shortcut, I test whether each model learns the relation and generalises to a held-out split
(20% of instances by a fixed hash, never trained). I report per-task **train and held-out** accuracy,
so the transformer's fit is visible: it is a competent baseline, not a strawman.

| task | transformer (train / held) | PrismFormer (train / held) |
|------|-----------------------------|-----------------------------|
| copy (first of three) | 100% / **100% ± 0%** | 100% / **100% ± 0%** |
| greater-than | 77% / 65% ± 9%  | 100% / **97% ± 3%** |
| max          | 84% / 66% ± 23% | 100% / **98% ± 3%** |
| min          | 75% / 61% ± 28% | 100% / **98% ± 2%** |
| parity       | 77% / 53% ± 14% | 100% / **68% ± 11%** |

The transformer **fits its training data** (85.8% mean train accuracy on these tasks) and **ties**
PrismFormer at 100% on copy, so the comparison is fair. On the relational tasks PrismFormer generalises
markedly better: **65.9% vs 49.1%** held-out, averaged over tasks and seeds. Two honest caveats.
*Reverse-inference* (training "opposite of A = B" or "capital of C = city" and holding out the
reverse direction) is **0% for both models** at this scale (neither infers the symmetric relation);
I keep it out of the summary and list it as a limitation (§5). And single-token arithmetic (add/sub/
mul/div where the whole answer is one vocab token) is *representation-favoured*: the codec makes the
answer face equal to `bind(inputs)`, so PrismFormer decodes it by construction (held-out 27–64% vs the
transformer's 3–21%). That is a property of the encoding, not evidence a transformer cannot add; the
fair arithmetic test is worked, multi-digit, and appears next. Run `dotnet run --project bench -c
Release`.

### 4.3 Worked multi-digit arithmetic: the algorithm, not the table

A sequence model cannot emit an exact multi-digit result in one token, so the fair test is to hand both
models the same *worked* problem and see who learns the algorithm. I write each answer digit by digit,
least significant first, so carries flow causally, and the model must generate those digits itself and
carry across columns; no outside loop does the work. Each of the four operations trains its own model at
full budget (all four, plus their transformer baselines, in parallel), on operands over 0–99, with
about 20% of pairs held out by a fixed hash. I use the atomic single-pass columnar form of each
operation: add/sub are two-operand (carry / borrow); multiply is a number times a single digit and
divide a number by a single digit, so every answer digit is one local column step: a digit-multiply on
the log-phase band, an add or subtract on the linear band. A seen-distribution control tells a memorised
table (held-out far below seen) from a learned algorithm (held-out about equal to seen).

| op | PrismFormer held-out [seen] | transformer held-out [seen] | face-decode |
|----|------------------------------|------------------------------|-------------|
| add | **54.9% ± 8.5%** [58%] | 0.4% ± 0.6% [1%] | 69% |
| sub | **70.0% ± 8.9%** [74%] | 1.3% ± 0.5% [1%] | 67% |
| mul | **27.7% ± 3.3%** [43%] | 1.7% ± 2.1% [1%] | 61% |
| div | **33.2% ± 12.9%** [43%] | 2.0% ± 2.5% [1%] | 50% |

PrismFormer computes all four; the transformer does none of them. Given the identical worked problem it
sits near zero even on pairs it *was* trained on (it cannot do exact multi-digit arithmetic at this
scale), while PrismFormer answers unseen problems far better than the transformer answers ones it has
already seen. For add/sub, held-out matches seen (55 against 58, 70 against 74): it learned the column
algorithm, not a lookup. Multiply and divide sit below their seen numbers (28 against 43, 33 against
43); with a single-digit second operand the pair space is smaller, so part of the score is memorisation,
but 28–33% on genuinely unseen pairs against a transformer at ~2% is still the column step generalising.
Full multi-digit × multi-digit multiplication is left to future work: producing the product digits
directly is a digit *convolution*, not a single-pass column, so it needs an explicit partial-product
scratchpad (single-digit multiplies on the log band, summed on the linear band), the same two
primitives, composed.

**Ablation: does freezing the numeric identity help?** I hypothesised that freezing the identity
region (keeping numbers on the exact codec geometry) would aid generalisation. The answer depends on
the training regime, and the two regimes tell a consistent story. Trained on **one operation in
isolation** (this section's setup), the identity made *learnable* (`frozenPrefix=0`, still codec-seeded)
generalises **better** on every op:

| op | frozen identity | learnable identity |
|----|-----------------|--------------------|
| add | 54.9% ± 8.5% | **70.9% ± 13.6%** |
| sub | 70.0% ± 8.9% | **73.4% ± 8.0%** |
| mul | 27.7% ± 3.3% | **36.7% ± 8.0%** |
| div | 33.2% ± 12.9% | **40.8% ± 3.2%** |

This is what you would expect: with a single task and nothing else to protect, an adaptable pass-through
is simply extra capacity the model folds that operation's transform into. But the identity is a *shared*
residual: the same component carries every token's own value through every task. Re-running the ablation
under **shared multi-task load** (§4.2's corpus: seven relational tasks plus all four arithmetic ops, one
model, 5 seeds) reverses the result. Held-out accuracy, mean over seeds:

| held-out group | frozen identity | learnable identity |
|----------------|-----------------|--------------------|
| relational tasks | **65.9%** | 65.2% |
| arithmetic | **51.5%** | 47.6% |
| all tasks | **60.7%** | 58.8% |

The reversal is sharpest exactly where the isolated gain was largest: **mul** goes from unfrozen's
+9-point isolated win to a 16-point *loss* under shared load (frozen 59% vs unfrozen 43%). Letting the
identity drift to specialise for one operation corrupts the residual every other task depends on; in
isolation there is nothing to corrupt, on a mixed corpus there is. So freezing is not a per-task
generalisation trick (the codec-seeded *initialisation* is what generalises), but under the realistic
shared-corpus regime it is also **not** a cost: it wins. What it unambiguously buys on top of that is
legibility: because the identity stays on the codec geometry, the answer decodes from the frozen bands
with no probe (face-decode column above, and §4.4). I report the frozen configuration as the headline
on both grounds. Reproducible via `dotnet run --project bench -c Release -- --columnar` (isolated) and
`dotnet run --project bench -c Release` (shared-corpus ablation, printed under the main table).

### 4.4 Reading the computation from the inside

Because every activation is a phasor face, PrismFormer's internal state is directly *decodable*; a dense
transformer's is not. I test this on the two operations with a codec homomorphism, single-digit
addition (linear band) and multiplication (log band), training both models to convergence and then
inspecting. On raw single-digit accuracy the two models are comparable (single-digit tables are small
and memorisable; the transformer is competitive or better), so the point here is not accuracy but
*legibility*. Decoding the final-position face against the codebook, with **no trained probe**,
recovers the answer from the frozen identity bands as well as the model's own readout does:

| decode of the final face (no probe) | add | mul |
|---|---|---|
| homomorphic band only | 35.2% | 34.4% |
| full frozen identity (both bands) | 45.5% | 75.0% |
| model's own readout | 45.5% | 75.0% |

Identity == readout: the answer literally lives in the codec geometry of the activation. The only way to
read the dense transformer is to *fit* a linear probe to its hidden state (75.0% ± 22.7% add, 62.5% ±
13.4% mul), and even that is a trained decoder, not a property of the representation. That contrast,
free and probe-free legibility versus a fitted probe, is the concrete pay-off of phasor faces. Run
`dotnet run --project bench -c Release -- --inspect`.

### 4.5 Language modelling

Character-level next-token modelling on a small embedded public-domain text, at matched size, both
temperature-calibrated on a validation slice:

| model | next-char accuracy | bits/char (lower better) |
|-------|--------------------|--------------------------|
| **PrismFormer** | **41.0% ± 1.9%** | **3.399 ± 0.081** |
| transformer     | 21.7% ± 3.6%     | 4.652 ± 0.273 |
| unigram baseline | 19.1%           | 5.322 |

Both models beat the unigram baseline (the transformer on bits/char, so it is learning, not broken),
and PrismFormer's char-face embeddings win decisively at this size: the algebraic architecture is
competitive at language, not only arithmetic. Both are weak in absolute terms at ≈10⁵ parameters on
1,670 characters; the claim is the matched-size comparison, not an absolute language result. Run
`dotnet run --project bench -c Release -- --lm`.

### 4.6 Exact-merge verification

Splitting a batch into `K` shards, computing them sequentially and in parallel, and reducing in a fixed
index order yields gradient tensors with an identical bitwise (FNV) checksum
(`0x945A7A4E8E0B9F3B`), and a re-run reproduces it exactly. The `verify/` kit reproduces this in
seconds; the benchmark's `--gradcheck` extends it to the full production spec. This is the property that
lets a mismatched-CPU colony sum into one coherent model.

### 4.7 Atomic memory capacity

Every experiment so far tests generalisation, where the codec is meant to help. The opposite question is
raw storage: with nothing to generalise from, how many arbitrary facts can each model simply hold? I build
a key→value store with no structure at all. Each key is two symbols over a fixed 128-symbol alphabet and
its value is one of 256 tokens drawn uniformly at random, so nothing can be inferred and every fact must be
memorised. The vocabulary is fixed, so neither model's parameter budget grows with the number of facts N,
and both are pound-for-pound at about 67k parameters (the transformer auto-matched as everywhere else).
Both train with early stopping: each runs until it converges to full recall (it fits N) or plateaus (past
its capacity), which is a truer capacity probe than a fixed budget because it does not conflate how fast a
model learns with how much it can hold. This is a single exploratory sweep at one seed, so I read the shape
of the curve, not a calibrated margin.

| N facts | PrismFormer | transformer |
|--------:|-------------|-------------|
| 512 | 99.0% | 100.0% |
| 1,024 | 98.4% | 98.8% |
| 2,048 | 98.8% | 96.1% |
| 3,584 | 99.3% | 97.8% |
| 4,096 | **98.0%** | 33.0% |
| 5,120 | **98.5%** | 71.3% |
| 6,656 | **98.4%** | 51.7% |
| 8,192 | **98.9%** | 7.2% |

PrismFormer converges to full recall (98–99%) at every N across the whole sweep, out to 8,192 facts; its
ceiling is above the top of the sweep and I did not find it. The matched transformer holds to about
N=3,584, then falls off a cliff at 4,096 (33%) and thrashes beyond it, individual points bouncing between
roughly 5% and 71% as N grows, which is training collapse rather than graceful decay. Taking the largest N
still at ≥95% recall as the atomic capacity, PrismFormer reaches at least 8,192 against the transformer's
3,584: a factor of at least 2.3, and a lower bound, because the phasor side never broke in range. This is
the memorisation counterpart to the generalisation results above, not a substitute for them; a store of
random pairs is the easiest thing there is to overfit, and the only claim is the relative capacity and
stability at matched size. The transformer's exact break point and its post-break values are seed-sensitive,
and that instability is itself part of the finding: at matched size the phasor substrate stores markedly
more and degrades gracefully where the dense baseline destabilises. Run `dotnet run --project bench -c
Release -- --capacity`.

---

## 5. Limitations

I state these plainly. (i) **Arithmetic scope.** I claim generalisation to unseen operand
*combinations within* the trained range (§4.3, worked multi-digit add/sub and multiply/divide by a
digit). I do **not** claim: *out-of-range magnitude* extrapolation (operands larger than any seen is
≈1% for both models, because the end-to-end model learns the mapping statistically, not via the codec's
exact homomorphism); *length* extrapolation (more digits than trained, a hard positional-generalisation
problem I do not target); or full multi-digit × multi-digit multiplication, which needs a
partial-product scratchpad (§4.3). (ii) **Reverse-inference** (inferring a held-out symmetric/reverse
relation from the trained direction) is **0% for both models** at this scale (§4.2). (iii) **Freezing
is regime-dependent**: on a single operation in isolation an unfrozen (still codec-seeded) identity
generalises better; freezing only wins under shared multi-task load, and its unconditional benefit is
legibility (§4.3). I keep it as the headline for the shared-corpus and legibility reasons and state the
isolated-task caveat. (iv) **Scale.** Results are at ≈10⁵ parameters; large-scale
convergence of the full colony is demonstrated to *run* but not yet to *converge* at scale. (v) **The
token economy** described in the repository is a design, not code. (vi) **Novelty.** The phasor codec
and RFF encoding are prior art (§2); my contribution is their use as a trainable, exactly-mergeable
transformer substrate and the swarm it enables.

---

## 6. Future work and open questions

I would rather name the sharpest objections to this work than let a reader find them unaided. Each is a
concrete next step, not a rhetorical hedge.

**A codec-seeded baseline.** PrismFormer's numeric embeddings are seeded from the phasor codec, while
the transformer baseline starts from random initialisation. That is a confound: some of the measured
gap could be initialisation rather than architecture. The control I have not yet run is to seed the
baseline's embeddings from the same codec and re-measure. Until that is done, the head-to-heads should
be read as "codec-seeded algebraic model vs standard transformer," not "architecture alone."

**A stronger baseline, and the right prior art.** My transformer baseline is a hand-derived,
gradient-checked encoder without layer normalisation, matched by parameter count. It fits its training
data, so it is not a strawman, but it is not a modern reference either (no layer norm, no warmup, no
tuned optimiser). A comparison against a well-tuned standard transformer, and against existing
complex-valued and VSA / neuro-symbolic models, which I do not yet compare to, is needed before any
architectural claim is settled.

**The swarm, actually distributed.** I show that sharded gradients merge bit-for-bit and that the
mechanism runs, but every experiment here is single-process. A genuine multi-node run, with a
convergence curve, a fault-injection test, and a wall-clock scaling measurement, is future work; until
it exists, the swarm should be read as a demonstrated *merge property* plus an *architecture*, not a
demonstrated distributed-training result. The value of bit-exactness over ordinary asynchronous
approximate merging is likewise something I argue for but have not yet measured.

**The homomorphism does not extrapolate, and that narrows the thesis.** Within the trained range the
codec makes arithmetic decodable and the model generalises to unseen operand combinations; out of range
it collapses toward chance, and (§5) it is the network learning a statistical mapping on the codec's
representation, not the codec's exact homomorphism, that does the generalising. So the honest thesis is
narrower than "arithmetic as algebra": the codec supplies a decodable, well-conditioned representation
and initialisation, and the network learns on top of it. Making the exact homomorphism itself carry
out-of-range values and multi-digit-by-multi-digit multiplication (via a partial-product scratchpad,
§4.3) is open.

**Scale and statistics.** Everything here is at roughly 10⁵ parameters with 4 to 8 seeds, and several
per-task standard deviations are large relative to their means; I report means and standard deviations
but not confidence intervals or significance tests across the task suite. Whether any of this survives
one or two orders of magnitude more parameters is unknown, and the absence of layer normalisation, which
currently caps trainable depth, is the first thing that has to change to get there.

None of this is settled, and I would rather ship it saying so.

---

## 7. Conclusion

I set out to build a representation that reasons instead of recalls, and the honest result is narrower
and more concrete than that ambition: numbers written as phasors make arithmetic something you can
decode straight out of the representation, and a model small enough to live on one CPU, with gradients
that merge to the bit, makes a fault-tolerant training swarm possible. Those two things came from the
same idea and I think they stand. I have tried to make every claim here something you can run and check
for yourself, including one of my own that the evidence knocked down, and the verification kit and
per-section commands are there so anyone can reproduce, or refute, each number.

---

## Reproducibility

The exact code that produced every number above is bundled with this archive as a frozen source
snapshot (repository tag `paper-v1`), so it does not depend on the live repository, which continues to
evolve. Build with the .NET 8 SDK; each command below is run from the snapshot root.

- `verify/`: the arithmetic-from-the-codec and exact-merge checks in seconds, no training
  (`dotnet run --project verify/Verify -c Release`).
- `bench/`: the parameter-matched head-to-heads, the default run (§4.2), plus `--columnar` (§4.3, with
  the frozen-identity ablation), `--inspect` (§4.4, mechanistic), `--lm` (§4.5), `--gradcheck` (§4.6).
- `PRIOR_ART.md`: a timestamped defensive-publication disclosure of the methods.

## References

- Borzunov, A., Baranchuk, D., Dettmers, T., et al. (2023). *Petals: Collaborative Inference and Fine-tuning of Large Models.* ACL 2023 (System Demonstrations), pp. 558–568.
- Cheng, Y., Yu, F. X., Feris, R. S., Kumar, S., Choudhary, A., & Chang, S.-F. (2015). *An Exploration of Parameter Redundancy in Deep Networks with Circulant Projections.* ICCV, pp. 2857–2865.
- Gayler, R. W. (2003). *Vector Symbolic Architectures answer Jackendoff's challenges for cognitive neuroscience.* Proc. Joint Int. Conf. on Cognitive Science (ICCS/ASCS), Sydney, pp. 133–138.
- Kanerva, P. (2009). *Hyperdimensional Computing: An Introduction to Computing in Distributed Representation with High-Dimensional Random Vectors.* Cognitive Computation, 1(2), 139–159.
- Kingma, D. P., & Ba, J. (2015). *Adam: A Method for Stochastic Optimization.* ICLR.
- Kleyko, D., Rachkovskij, D. A., Osipov, E., & Rahimi, A. (2022). *A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part I: Models and Data Transformations.* ACM Computing Surveys, 55(6), Art. 130. (arXiv:2111.06077)
- Kleyko, D., Rachkovskij, D. A., Osipov, E., & Rahimi, A. (2023). *A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part II: Applications, Cognitive Models, and Challenges.* ACM Computing Surveys, 55(9), Art. 175. (arXiv:2112.15424)
- McMahan, B., Moore, E., Ramage, D., Hampson, S., & Arcas, B. A. (2017). *Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg).* AISTATS, PMLR 54:1273–1282.
- Plate, T. A. (1994). *Distributed Representations and Nested Compositional Structure.* PhD thesis, University of Toronto. (Origin of the unitary / frequency-domain HRR construction.)
- Plate, T. A. (1995). *Holographic Reduced Representations.* IEEE Transactions on Neural Networks, 6(3), 623–641.
- Plate, T. A. (2003). *Holographic Reduced Representation: Distributed Representation for Cognitive Structures.* CSLI Publications, CSLI Lecture Notes 150. (Frequency-domain / "unitary vector" phasor HRR; the name "FHRR" is a later coinage in the VSA literature.)
- Pribram, K. H. (1971). *Languages of the Brain: Experimental Paradoxes and Principles in Neuropsychology.* Prentice-Hall. (Holographic-memory hypothesis; cited here as metaphor only.)
- Pribram, K. H. (1991). *Brain and Perception: Holonomy and Structure in Figural Processing.* Lawrence Erlbaum Associates. (The developed holonomic-brain theory; a contested hypothesis, not relied upon.)
- Rahimi, A., & Recht, B. (2007). *Random Features for Large-Scale Kernel Machines.* NeurIPS (NIPS), pp. 1177–1184.
- Sindhwani, V., Sainath, T. N., & Kumar, S. (2015). *Structured Transforms for Small-Footprint Deep Learning.* NeurIPS (NIPS).
- Smolensky, P. (1990). *Tensor Product Variable Binding and the Representation of Symbolic Structures in Connectionist Systems.* Artificial Intelligence, 46(1–2), 159–216.
- Thawani, A., Pujara, J., Ilievski, F., & Szekely, P. (2021). *Representing Numbers in NLP: a Survey and a Vision.* NAACL-HLT, pp. 644–656.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). *Attention Is All You Need.* NeurIPS (NIPS), pp. 5998–6008.
- Wallace, E., Wang, Y., Li, S., Singh, S., & Gardner, M. (2019). *Do NLP Models Know Numbers? Probing Numeracy in Embeddings.* EMNLP-IJCNLP, pp. 5307–5315.

*Please open an issue if any attribution is incomplete or mis-stated; correct credit to prior work matters to me.*
