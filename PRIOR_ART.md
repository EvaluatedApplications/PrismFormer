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

## Reduction to practice

The accompanying source code implements the above and is gradient‑checked. Measured results (see `README.md`) include:
generalisation to held‑out operand combinations on arithmetic and comparison tasks substantially exceeding a
parameter‑matched dense transformer; a temperature‑calibrated character language model that exceeds a
parameter‑matched dense transformer on both next‑character accuracy and bits‑per‑character; and marked parameter
efficiency across model sizes. The `README.md` also documents the methods' current limitations honestly (e.g. failure
to extrapolate beyond the trained numeric range in the end‑to‑end trained model).

## Reference implementation

The complete enabling implementation is contained in this repository (`src/` and `bench/`) as of the commit that
introduces this document. The repository's version‑control history provides the authoritative timestamp of disclosure.
