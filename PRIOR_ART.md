# PrismFormer — Defensive Publication and Prior‑Art Disclosure

**Inventor / author:** Dongyang Stephen Chen, trading as Evaluated Applications
**Date of public disclosure:** 9 July 2026 (established by the timestamped commit history of this repository)
**Status:** Source‑available (see `LICENSE`). This document is a *defensive publication* intended to place the
methods described below into the public record as prior art from the date above.

Copyright © 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.

---

## Purpose

This document publicly and enablingly discloses the inventions, methods, and designs embodied in this repository so
that they constitute **prior art** as of the date above. The intent is defensive: to establish authorship and priority
and to prevent others from obtaining exclusive rights over the methods disclosed here. The source code in this
repository forms part of this disclosure and is the enabling reference implementation.

## Field

Machine learning; neural sequence models and language models; representation of numeric and symbolic quantities in
neural networks; parameter‑efficient transformer architectures; and their training.

## Summary of the invention

A neural sequence model ("PrismFormer") in which tokens — including both words and numbers — are represented as
**phasor faces** (bundles of unit complex numbers / phase patterns), composed by a single algebraic operation
(per‑component complex multiplication), such that arithmetic is performed by the *same* operation that performs
symbolic association, and computed results are read out by correlation. The model is a transformer‑shaped architecture
in which every dense linear map is replaced by a lean, decodable **relation‑bank**, trained by an exactly‑mergeable,
shardable gradient scheme. The following sections enumerate the disclosed methods and their contemplated variations.

## Disclosed methods (and contemplated variations)

1. **Phasor encoding of a quantity's value.** A numeric value `v` is encoded as a set of phases `[cos(v·θ_k),
   sin(v·θ_k)]` over a set of frequencies `θ_k`. Variations disclosed: (a) a **linear‑phase band** (`θ` applied to `v`)
   that is *additive‑homomorphic* — per‑frequency complex multiplication of the faces of `a` and `b` yields the face of
   `a+b`; (b) a **log‑phase band** (`θ` applied to `ln|v|`) that is *multiplicative‑homomorphic* — the same operation
   yields the face of `a·b`; (c) any number and choice of frequencies, random or structured, including reserving
   specific frequencies (e.g. a period‑2 frequency `θ=π`) to expose specific residues such as parity.

2. **Uniform phasor codec for all tokens.** Every token is a bundle of phasors: a number is encoded by (1); a
   non‑numeric symbol is encoded as a deterministic phasor signature derived from its identity (e.g. a hash of its
   spelling), optionally with a learned "orbital" region. Numbers are therefore not privileged in representation —
   they and symbols occupy one uniform phasor space and are composed by one rule.

3. **One composition operation.** Composition is per‑component **complex multiplication** ("bind"); its conjugate is
   "unbind" (subtraction / division on number bands, and inversion of an association for symbols); component‑wise sum is
   "bundle" (superposition / set formation). Disclosed use: the *same* bind performs symbolic role–filler binding *and*
   arithmetic, with addition on the linear band and multiplication on the log band occurring simultaneously in one
   operation, the appropriate projection being selected at readout.

4. **Arithmetic as algebra over faces, without a calculator.** No lookup table, arithmetic‑logic unit, or dedicated
   arithmetic routing is used. A computed value is produced by composing operand faces with bind/unbind and is named by
   **correlation** (inner product) against the codebook of value faces; addition, subtraction, multiplication, and
   division correspond to bind/unbind on the linear/log bands respectively.

5. **Correlation (retrieval) readout on discriminable faces.** Because distinct values are distinct phase patterns
   (rather than collinear magnitudes), a tied correlation readout can *name* a computed result, resolving the failure
   mode of magnitude‑based numeric embeddings that can only select extreme values. Contemplated: temperature scaling of
   the readout logits (a fixed or learned scalar) to calibrate probabilities for language modelling.

6. **Frozen identity, learned meaning.** The identity portion of each face (the value/signature bands) is held fixed
   ("frozen") during training so numeric ground truth is preserved and generalises to unseen operands, while a separate
   learned region carries adjustable meaning; gradients to the frozen region are suppressed.

7. **Algebraic relation‑bank transformer ("AlgFormer").** A transformer‑shaped model (attention → per‑token
   feed‑forward → residuals, stacked) in which every dense linear map `W·x` is replaced by a **relation‑bank**
   `y[i] = Σ_{k<S} bank[k][i] · x[(i+k) mod d]` — a bundle (`Σ`) of bound (`·`) and cyclically permuted (`(i+k) mod d`)
   terms. `S` is a lean↔full‑rank control: at `S=d` it reparametrises a full matrix; at `S<d` it is a parameter‑efficient
   approximation costing `S·d` parameters per map instead of `d²`. Every parameter is itself a decodable face. The
   readout is tied to the face/embedding table. Disclosed variations: applying this bank form to the query, key, value,
   output, and feed‑forward projections; stacking to depth `L`; and a feed‑forward "combine" that is either a gated
   unit or a per‑complex‑pair multiplication ("bind FFN").

8. **Exactly‑mergeable, shardable training.** Because the substrate is exact and stateless during backpropagation
   (read‑only model), gradients accumulate into a **detached, mergeable buffer**; a minibatch is split into independent
   shards whose gradient buffers are summed and applied with one optimiser step, producing a result identical
   bit‑for‑bit to serial training (the gradient of a sum being the sum of gradients). Disclosed: performing this
   fan‑out across CPU cores and/or across machines through a resource‑gated, adaptively‑tuned parallel scheduler, with
   fallback to in‑process parallelism.

9. **Training recipe for depth.** A warmup followed by a learning rate scaled down as depth grows is disclosed as
   enabling stable training of deep stacks of the above multiplicative/algebraic blocks.

10. **Applicability.** The above methods are disclosed as applicable to character‑, word‑, subword‑, or token‑level
    sequence modelling; to arithmetic, comparison, ordering, and relational tasks; and at any embedding dimension,
    depth, or vocabulary size.

11. **Identity-preserving capacity growth ("upgrade in place").** Increasing a trained model's capacity — more
    relation-bank shift terms `S`, or a longer context — by zero-initialising the new components so the output
    is byte-identical the instant the model is extended (the added terms contribute zero to the algebraic sum),
    after which training fills them in. A saved checkpoint upgrades to the larger shape without retraining; a
    spec signature advertised by each node gates incompatible peers.

12. **Full-replication training and serving colony.** A network in which every node holds the *whole* model
    (not a shard), trains and serves it, and shares with neighbours by (a) exchanging raw or distilled training
    examples ("bleed") — a weight-free transfer of skill — and (b) small elastic/federated weight-slice
    averages; combined with the exactly-mergeable gradients of (8), a spec-signature compatibility gate, and a
    KV-cache causal serve off an immutable snapshot. Fault-tolerant: losing any node leaves every other a
    complete, working model, so no peer is load-bearing — contemplated across heterogeneous CPUs.

13. **Mechanistic decodability of internal state.** Because activations are themselves decodable faces, a value
    computed inside the network can be read directly from a hidden activation by correlation against the value
    codebook, with no trained probe — enabling inspection and verification of the model's internal computation.

14. **Codec-only model with a regenerable representational substrate.** A model in which the *entire* token
    representation (the full embedding of every token, at the full model dimension) is a deterministic function of
    the token's identity under the codec of (1)–(3), held fixed for the whole of training, so that (a) no
    representational parameters are learned at all and every learned parameter resides in the compute (the
    relation-banks of (7)) and the positional and output-bias terms; and (b) the representation need not be stored in
    a checkpoint, being regenerable on demand from the token inventory and the codec, so the deployed artefact reduces
    to the learned compute parameters plus a token list (for a large-vocabulary model this is a small fraction of the
    nominal parameter count, and can be quantised further). Disclosed by setting the frozen-identity region of (6) to
    the whole model dimension, and by deriving the shipped artefact by regeneration rather than storage.

15. **Deterministic superposition crosstalk as a learned computational resource.** A method in which the bounded
    superposition capacity of a fixed codec (the "crosstalk" by which an overloaded bundle correlates most strongly
    with a codebook face other than the ones bound into it) is used not as an error to be avoided but as a fixed,
    deterministic, reproducible transform from an input superposition to an output face, over which a deep learned
    network (the stack of (7)) learns to select and compose. The determinism (the transform is identical on every
    instance because the codec is fixed) is what makes the structure learnable, distinguishing it from stochastic
    noise; the depth is what supplies the cleanup and the composition that make individual free transforms usable.
    Disclosed: a deep network that routes over the deterministic collision structure of a fixed vector-symbolic codec
    as a compute primitive; the observation (measured, see reduction to practice) that decode failure under overload
    is graceful and lands on never-bound faces rather than confusing bound items, and that exact many-to-one collapse
    in the continuous representation is a measure-zero event, so the operative limit is a soft discriminability
    (signal-to-noise) floor rather than a hard capacity wall. Contemplated for any fixed structured or random codec
    exhibiting deterministic superposition collisions.

16. **Externalisation with re-quantisation ("scratchpad cleanup").** A method of unbounded-depth computation on a
    bounded-width substrate in which intermediate results are emitted as tokens into the model's own context and are
    thereby re-quantised onto clean codec faces, each emission acting at once as (a) offloaded working memory and (b)
    error correction that discards accumulated superposition crosstalk before it exceeds the discriminable margin, so
    that an arbitrarily long computation proceeds as a sequence of shallow, clean bind/bundle steps interleaved with
    clean-up writes. Disclosed for arithmetic (a written column/carry scratchpad) and contemplated for general
    step-wise reasoning; the emitted trace is itself decodable per (13), giving a legible record of the computation.

17. **Identity-initialised depth growth.** Extending (11) from width/context to *depth*: adding one or more
    transformer blocks to a trained model by initialising each added block's output projection to zero, so the block
    is an exact identity at the instant of addition (it contributes zero to the residual stream and the model's output
    is byte-identical) yet carries live gradient and begins to specialise immediately. Disclosed uses: (a) building a
    deep stack from a shallow one for trainability (construct one block, then grow to depth L); and (b) *progressive*
    growth, adding a block when a given depth saturates, in place and without resetting the optimiser state of the
    existing blocks, up to the point where the residual-stream width rather than depth becomes limiting.

18. **Trainable-only synchronisation exploiting a regenerable substrate.** In the distributed setting of (12) and in
    accelerator offload, transmitting and merging only the *learned* parameters and never the frozen regenerable
    representation of (14), on the ground that the latter is identical on every node and every instance by
    construction; this reduces both per-exchange network transfer and per-step host-to-accelerator transfer to the
    learned fraction of the model. Disclosed: a weight-slice bleed and an accelerator weight-sync that skip the frozen
    embedding block, the correctness following directly from the substrate's determinism.

19. **Proximity-ranked gossip mesh for skill and weight diffusion.** A peer-to-peer mesh in which each node maintains
    a rolling round-trip-time proximity map of its peers and directs its example-bleeds (12a) and weight-slice
    averages (12b) to its K nearest peers, spreads membership by the same nearest-neighbour gossip, and gates peers by
    the spec signature of (11)/(14); so that skill and weight consensus diffuse across the mesh along low-latency paths
    with no central coordinator, and (as observed in practice) a node that performs no training of its own can acquire
    fragments of a skill purely by absorbing averaged weight slices. Disclosed: RTT-probed proximity, K-nearest
    directed bleed and gossip, roster diffusion, elastic weight-slice averaging at a small merge rate, and
    signature-gated compatibility.

20. **Number-clean subword tokenisation for a phasor codec.** A subword tokenisation constrained so that pure-digit
    tokens are limited to a small fixed width (for example one or two digits) and mixed digit/letter subwords are
    excluded, so that a number of any magnitude tokenises into uniform small-value chunks, each carrying a clean number
    face under (1), keeping arithmetic aligned with the additive and multiplicative homomorphisms of (1)/(3); combined
    with a content hash of the token inventory folded into the spec signature of (11)/(12), so that any change to the
    inventory is a detectable hard fork for the compatibility gate rather than a silent, misaligned "upgrade".

## Relationship to prior art and points of distinction

The methods above draw on several established lines of work. The invention lies in their specific combination and in the
distinguishing features noted for each; this section states the nearest art and those distinctions so that the scope of
what is disclosed here is clear.

- **Holographic Reduced Representations and Vector Symbolic Architectures** (Plate; Gayler; Kanerva; Kleyko et al.).
  PrismFormer takes bind (here per-component complex multiplication), unbind (conjugate), bundle (sum) and correlation
  cleanup from this line. Points of distinction: (i) a *phasor* codec in which a token's numeric *value* is written as
  phase, so one bind performs exact arithmetic on numbers and role–filler association on symbols, selected at readout
  (1)–(4); (ii) the vector-symbolic substrate is held *fixed* and a *deep learned* transformer is stacked on it, so
  cleanup and composition are learned and deep rather than hand-designed and shallow (7); (iii) the deterministic
  collision structure is used as a learned compute resource (15), not treated solely as a capacity limit.

- **Reservoir computing, echo-state and liquid-state machines, random-feature methods, extreme learning machines**
  (Jaeger; Maass; Rahimi & Recht; Huang et al.). PrismFormer shares the pattern of a fixed nonlinear substrate with a
  learned readout. Distinctions: (i) the substrate is not random but a structured, deterministic, decodable algebraic
  codec (arithmetic-capable, quasi-orthogonal by construction); (ii) the readout is not linear but a deep attention
  stack that learns to route; (iii) the substrate's deterministic *collisions* are themselves used as compute (15), a
  use absent from classical reservoirs.

- **Fixed, tied and pretrained embeddings; weight tying** (Press & Wolf; Inan et al.). PrismFormer freezes the *whole*
  embedding at a deterministic *regenerable* codec (14), so no representational parameters are learned or stored,
  distinct from freezing learned or random embeddings, and enabling the trainable-only synchronisation of (18).

- **Identity/residual initialisation for deep networks** (ReZero, Bachlechner et al.; Fixup, Zhang et al.; SkipInit,
  De & Smith). PrismFormer uses zero-output-projection identity init and extends it to in-place *progressive depth
  growth* (17) and to capacity growth on a frozen algebraic substrate (11).

- **Network growing and progressive stacking** (Net2Net, Chen et al.; progressive BERT stacking, Gong et al.).
  Distinction: growth is by exact identity init on a codec-only model, byte-identical at the instant of growth and
  preserving optimiser state (17).

- **Chain-of-thought and scratchpad reasoning** (Nye et al.; Wei et al.). Distinction: a scratchpad write is
  simultaneously an error-correcting re-quantisation onto a clean codec face (16), tying externalisation to VSA
  cleanup memory, and the trace is decodable by construction (13).

- **Elastic-averaging, federated and gossip SGD** (EASGD, Zhang et al.; FedAvg, McMahan et al.; gossip/decentralised
  SGD). Distinctions: (i) *exact*, index-aligned merge is possible because the deterministic codec places every token
  at the same parameter row on every node (8, 12); (ii) *full replication*, every node holds the whole model rather
  than a shard, with fault tolerance and no load-bearing peer; (iii) proximity-ranked mesh routing (19); (iv)
  example-level "bleed" as a weight-free skill transfer alongside weight averaging.

- **Numeric representation in neural networks** (positional/binary numeric embeddings; xVal; abacus embeddings).
  Distinction: arithmetic is the bind operation itself on a phase-encoded value (1)–(4), exact and extrapolating,
  rather than a learned numeric embedding read by a magnitude-based head.

- **Complex- and Fourier-domain neural networks** (deep complex networks; Fourier neural operators). Distinction:
  phase encodes a token's *value* for algebraic binding and arithmetic, not merely complex-valued weights or spectral
  mixing.

Methods (14)–(20) and this section are disclosed as of the commit that introduces this revision; the repository's
version-control history provides the authoritative timestamp for each.

## Reduction to practice

The accompanying source code implements the above and is gradient‑checked. Measured results (see `README.md`) include:
generalisation to held‑out operand combinations on arithmetic and comparison tasks substantially exceeding a
parameter‑matched dense transformer; a temperature‑calibrated character language model that exceeds a
parameter‑matched dense transformer on both next‑character accuracy and bits‑per‑character; and marked parameter
efficiency across model sizes. The `README.md` also documents the methods' current limitations honestly (e.g. failure
to extrapolate beyond the trained numeric range in the end‑to‑end trained model).

Methods (14)–(20) are implemented in this repository and are disclosed as reduced to practice to the following extent.
The codec-only configuration (14) and identity-initialised depth growth (17) are implemented and used to construct and
train a deep (tens of layers) subword model whose learned parameters are a small fraction of its nominal size; a
whole-system language-modelling result for this configuration is under evaluation and is not asserted here. The
crosstalk-as-resource claim (15) is supported by a direct measurement over the real frozen codebook (`bench --bind`):
retrieval from a superposition of bound pairs degrades *gracefully* with the number of pairs rather than at a cliff,
its failures land on faces that were never bound (rather than confusing bound items), and a pure bind chain unbinds
losslessly, locating the fidelity limit in bundling and confirming it as a soft signal-to-noise floor rather than a
hard wall. The proximity-ranked mesh (19) and full-replication colony (12) are implemented, and skill acquisition by a
non-training node via weight averaging alone has been observed. These disclosures are enabling regardless of the
eventual whole-system results; the methods and their contemplated variations are placed in the public record as of the
timestamp of this revision.

## Reference implementation

The complete enabling implementation is contained in this repository (`src/` and `bench/`) as of the commit that
introduces this document. The repository's version‑control history provides the authoritative timestamp of disclosure.
