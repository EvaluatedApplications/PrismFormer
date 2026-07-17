# Paper 2 — scratchpad / running findings

**Status:** raw working notes, NOT prose. To be rewritten in my own voice later (first person, plain,
networking-as-spine — same as paper 1). Everything here is a number from a test I actually ran; where a
hypothesis flipped, I've kept the flip because the honest version is the stronger claim.

**Builds on / cites paper 1:** *PrismFormer* — DOI 10.5281/zenodo.21384525. Paper 1 = one algebraic
transformer (phasor faces, frozen number codec, relation-banks). Paper 2 = **a swarm of them**: what
happens to competency, robustness, and skill when many PrismFormers are coupled into a decentralized mesh.

**Working thesis (candidate):** In a swarm of algebraic transformers, competency is a property of the
*collective, not the individual*, and the frozen algebraic codec acts as a **shared representational
coordinate system** that lets genuinely-separate, independently-initialised nodes share skill through
continuous elastic coupling — near-losslessly, where a plain net only gets partway.

---

## 1. The single network is brittle, distributed-but-not-redundant (Levin-inspired)
Bench: `--lesion`, `--plasticity`.
- **Lesion tolerance:** kill 5% of weights at inference → a mastered task drops 100% → 31%; 30% → 3%.
  Brittle. Exact arithmetic is a precise, low-redundancy code — no spare redundancy to lose.
- **Regeneration:** damage 40%, retrain → heals *slower* than a fresh model from scratch (81% vs 99% at
  +3000 steps). Damaged structure is miscalibrated, not a helpful partial.
- **Plasticity / degeneracy:** amputate the most-critical weight-block and hold it dead → no relocation,
  recovery stalls at chance (~7%). Importance map: **every block 70–98% critical** → the code is
  **distributed (all regions participate) yet non-redundant (none dispensable)**. Holographic-spread ≠
  damage-robust. (Caveat: byte-blocks dominated by the I/O codec, not cleanly the relational banks.)
- **Takeaway:** basal competency does NOT live in one network's weights.

## 2. Competency is collective (the Levin "broken hardware" analog)
Bench: `--colony`, `--emergence`.
- **Broken-hardware colony (exact gradient-merge, = the ABANDONED master/slave path):** holds 100% at 40%
  node failure per round, 90% at 60%, breaks only at 80% (no data left). No failover logic is coded — the
  robustness is emergent.
- **Byzantine limit:** one garbage node averaged into the mean → 31% → 8% at 10% garbage. A trust-free
  average has no defence; real swarms need median/trust.
- **Emergence:** 8 nodes each train only their disjoint slice; some pairs hidden from everyone. Best solo
  node: 20% seen / 7% hidden. Collective (exact-merge): **100% seen, 19% hidden** — the addition *rule*
  appears in the group and in no member. NOTE: this used gradient-sum; see §4 for the faithful mechanism.

## 3. Diversity is a red herring; the win is decorrelated errors
Bench: `--diverse`.
- Homogeneous seed-diversity ensemble: 60% (+12% over best member). Architecturally-diverse (matched):
  53% (+5%). Diverse-with-weak-members: 44%. **Architectural diversity does not beat homogeneous
  seed-diversity;** ensembling helps only when competent members fail on *different* items. Mixing in weak
  configs hurts.

## 4. The REAL mesh mechanism — no gradient summing (faithful replica)
Bench: `--mesh`. Source of truth: `SwarmChatter` / `StudioModel.MergeWeightSlice` / `HeadlessNode`.
- The always-on colony is **NOT** master/slave gradient-sum (that path exists but was abandoned). It is
  autonomous separate models that "chatter": **weight-slice elastic-averaging** (`mine = (1-α)·mine +
  α·theirs`, α=0.05, slice 1024, fanout 3) **+ pair-gossip** (share training examples). No gradients.
- Long horizon (200 ticks): workers cohere in FUNCTION (seen 0→93%, smooth, **no collapse**), the rule
  emerges (hidden 0→28%), but nodes stay **structurally diverged** (~0.34 plateau = "separate but
  bleeding"), and a **passive passenger is NOT carried** (0% for 200 ticks).
- **Correction to §2:** the strong short-horizon emergence + carried-passenger were artifacts of the
  gradient-sum. On the real mechanism, coherence is driven by **pair-gossip (shared data)**, not the
  weight-bleed. The weight-bleed is a gentle tether that transfers skill (§6) but can't build from cold.

## 5. Averaging does NOT damage the holographic algebra
Bench: `--collapse`.
- If averaging corrupted the precise code, algebra accuracy would periodically collapse. It doesn't —
  frozen AND unfrozen codec both stay smooth (max single-tick drop 4–5%, zero collapses).
- Reason (not the codec-pin, which I predicted wrong): gentle per-slice blend (5% of 1024/40k params) +
  shared basin + continuous training repair → perturbations are tiny and self-healing. Contrast `--lesion`
  (5% ZEROING, permanent, unrepaired → collapse) vs bleed (5% blend toward a valid peer, repaired → fine).

## 6. Skill-sharing by weight-averaging — the core result
Bench: `prismnet swarmbleed`, `--average`.
- **Weight-only transfer (no gossip), shared init, continuous coupling:** node A's accuracy on node B's
  skills lifts 16% → 51% at α=0.15 while staying separate (divergence 0.03); skills diffuse through a ring
  2 hops to nodes that never trained them. Pure averaging transfers skill.
- **One-shot averaging of independently-trained models FAILS:** train two to competence, average once →
  frozen same-init 72%, frozen **different-init 9%**, none same-init 8%, none different-init 6%. Different
  basins collapse *even with the codec* (it pins the embedding, not the banks). → You genuinely cannot just
  average two separate models.
- **Continuous coupling BRIDGES different inits** where one-shot fails: pull-to-mean during training →
  frozen different-init **AVERAGE 99%** (divergence 0.022), none different-init **75%** (divergence 0.074).
- **Unified rule:** weight coupling shares skill between models that are (a) both actively training and
  (b) continuously tethered. Continuous coupling — not shared init — is what makes separate models
  averageable; the **frozen codec makes it near-lossless (99% vs 75%)** by fixing a shared coordinate
  system; a passive non-training absorber gets nothing (§4 passenger, 0%).
- **DEFENSIBLE CLAIM (survives a hostile reviewer):** "continuous elastic coupling shares skill across
  independently-initialised nodes — partway on a plain net (75%), near-losslessly with the algebraic codec
  (99%)." NOT "only our codec can average separate models" (false: 75% without) and NOT "impossible in a
  NN" (EASGD is standard). The codec is a powerful, precisely-bounded *enhancer* of cross-init coupling.

## 7. The average-only anchor is a low-resolution echo
Tool: `prismgym probe` on the live anchor model (pulled 2026-07-17).
- Average-only anchor (1.2M params): on the canonical `X + Y =` format it produces the FULL carry-column
  scratchpad skeleton with high confidence (`134 + 267 =` → `4+7+0=0c1 3+6+1=8c0 1+2+0=2c0 = 280`, conf
  5.32; digits wrong). On natural-language prompts it garbles into word-salad (`what is 47 + 38` → "calen
  the he out t").
- Interpretation: averaging keeps what the whole fleet AGREES on (heavily-reinforced skeleton survives)
  and washes out the IDIOSYNCRATIC (variable NL-framing blurs). The anchor is the fleet's **lowest common
  denominator** — a faithful but low-fidelity echo. It never "becomes a model" on its own; it preserves
  co-grown structure, can't create new competence.

## 8. Architecture argument (for the discussion section)
- Weight-averaging + gossip vs master/slave gradient-sum: the classic sync-allreduce vs async/EASGD split.
  Gradient-sum is lossless but needs a live master + lockstep loop (stragglers gate every round, SPOF, big
  bandwidth) — fragile over a public broker on heterogeneous WAN boxes. Elastic-averaging + gossip is
  lossy/slower but decentralized, async, fault-tolerant, O(1) bandwidth, per-node autonomy — and the
  exactness lost doesn't bite (no collapse, still reaches the goal, skill still shares). The deployment
  picks async.

---

## 9. Free head-start representations & self-optimising tokenisation (DESIGN thread — mostly UNBUILT, flagged)

Second arc of the session: can we get useful representations WITHOUT gradient descent, by superposition — and let
the swarm self-optimise them? Build status flagged per item; most is design, not measured.

**9.1 Free prototype learning (bundling) — vision side already validated.** Bundle (superpose) K holographic
encodings of the same class → one prototype vector; classify new images by correlation. No gradient descent: the
fixed codec extracts features, bundling averages examples into a concept. VisionCodecBench already does this
(per-class bundle of ~100 → 100% held-out). NEW `bench/PrototypeBench.cs` (`--prototype`) sweeps K = the few-shot
curve — WRITTEN, not yet wired/run. Claim to show: K=1..few already beats chance (one/few-shot free classifier);
superposition saturates at high K (capacity, cf `--assoc`). Prior art: prototype/mean-embedding nets, VSA bundling.

**9.2 Free token embeddings for text — "meta meaning" by corpus-averaging.** Build a token's vector by (a)
COMPOSITIONAL bind of its char faces (= fastText) AND (b) DISTRIBUTIONAL average of its corpus contexts (= Random
Indexing / BEAGLE). Both in the same phasor space; sum → a token grounded in spelling AND usage, no backprop. Use to
SEED the embedding table = free warm-start. Rare word still placed by its chars; common word arrives meaningful.
UNBUILT. Experiment: small corpus → build token vectors → (a) nearest-neighbours sensible, (b) seed AlgFormer,
measure training head-start vs random init.

**9.3 LOAD-BEARING CAVEAT (measured, `--codec-baseline`):** seeding only helps an architecture that can EXPLOIT the
structure. Ran the paper-1 §6-A control (seed a MiniTransformer's embeddings from the codec vs random): the
transformer got ZERO benefit (both ~chance held-out). BUT it only fit 16–24% of TRAIN (AlgFormer 100%) → undertrained
→ §6-A still NOT conclusively settled (needs a tuned/longer transformer that actually fits). Suggestive: attention
can't use phasor geometry, AlgFormer's bind/correlate can — so the whole head-start thread is conditional on the
exploiting architecture. NOTE: not param-matched here (transformer 805k vs AlgFormer 242k), so no clean pound-for-pound
claim from this bench; the paper's matched head-to-head stands separately.

**9.4 Hybrid char + mixed-length vocab.** Single chars = ATOMIC BASE = complete fallback → no OOV ever → makes it SAFE
to promote the most-frequent sequences of ANY length (BPE / WordPiece / Unigram-LM) to their own tokens, each seeded
(9.2). Framing: bind = entanglement-like (recoverable pair, unbind to read); bundle = superposition (mixture). One new
moving part vs pure char: a segmentation step (greedy longest-match, or Unigram Viterbi). UNBUILT.

**9.5 In-place vocab upgrade = append-and-seed morphogenesis.** New token ids at the END → existing ids unchanged, no
reindex; append SEEDED rows to Emb (+U/C only if the tokens are output targets). Same grow-in-place as Context/Shifts
(LoadUpgrade) but SEED instead of zero-pad. Identity-preserving on old behaviour (softmax dilution minor if new output
bias seeded low). CATCH: Vocab is a NON-growable dim + exact-Signature mesh gate → a bump ORPHANS the old-vocab fleet
(cf PRISM-1→2). So in-place per MODEL, coordinated version-bump per SWARM. UNBUILT (needs a LoadUpgrade append+seed
branch) — but see 9.7, which dissolves the orphaning.

**9.6 Fidelity / capacity — D=256 has huge headroom.** Random face crosstalk std = 1/√256 = 0.0625; the worst-case
(closest pair) grows only ~logarithmically: ≈0.26 @96 tokens (today) → ≈0.37 @9k (all 2-char) → ≈0.42 @857k (all
3-char); random faces don't genuinely collide until ~10¹³. Separability is NOT the bottleneck for a big text vocab.
The TIGHT subspace is the 64-dim FROZEN number band (arithmetic) — but text tokens live in the full-256 learnable
signature space, so arithmetic fidelity is untouched. Analytic only. Experiment (proposed): capacity bench — pack N
real `Encode` faces, measure max-crosstalk + cleanup-decode acc vs N, find the real saturation (Encode is FNV-hash,
not perfectly Gaussian — measure, don't trust the estimate).

**9.7 Self-optimising decentralised vocabulary (the frontier).** Each node DISCOVERS its own frequent tokens locally
(frequency heuristic). The shared char base = a shared coordinate system → per-node vocabs are vector-compatible even
when discovered independently (same codec-as-coordinate-system principle as §6). Swarm WASH-OUT (the anchor dynamic:
common survives, idiosyncratic fades) selects the globally-useful vocabulary — **masked averaging IS the consensus**,
no separate protocol.
- CATCH: naive weight-averaging of MISALIGNED per-node id-tables collapses (cf `--average` different-init = 9%). Needs
  SHARED ADDRESSING.
- SOLUTION (elegant): FIXED LATENT address space + sparse ACTIVE mask. Addresses are CONTENT-DERIVED (composed vector,
  or hash) → same token → same slot on every node → alignment automatic. Fixed table size → signature never changes →
  NO version bump, NO orphaning (dissolves 9.5's catch). MASKED averaging = merge a slot ONLY over nodes where it's
  ACTIVE (else the inactive majority dilutes a real token to zero); the mask applies at inference AND at merge.
  Frequent → activated by many → reinforced; idiosyncratic → few → washes out = self-optimisation.
- ADDRESS EXPLOSION: do NOT pre-book physical slots (95ⁿ: 2-char 9,025 / 3-char 857,375 / 4-char 81M / exponential).
  DERIVE addresses — composition (free, infinite latent space, computed on demand) or hash to a FIXED M. Physical cost
  = the ACTIVE set only, LENGTH-INDEPENDENT. Keep M ≫ active-set to avoid active-token collisions. Prior art: hashing
  trick (Weinberger 2009), Bloom / hash embeddings, content-addressable memory.
- Experiments (proposed): (1) masked-merge dynamic — do frequent tokens reinforce across nodes while idiosyncratic
  ones fade, with NO version bump? (2) collision / M sweep vs active-set size.

---

## Prior art to cite (honest lineage — don't skip these)
- **Model soups** — Wortsman et al. 2022 (averaging fine-tuned weights; works because of shared init).
- **Federated Averaging (FedAvg)** — McMahan et al. 2017 (average weights of models on different data).
- **Elastic Averaging SGD (EASGD)** — Zhang, Choromanska, LeCun 2015 (continuous pull to a center — the
  mesh's actual mechanism).
- **Stochastic Weight Averaging (SWA)** — Izmailov et al. 2018.
- **Git Re-Basin / linear mode connectivity** — the permutation-symmetry barrier that makes independent-net
  averaging fail (motivates why the codec's fixed coordinate system helps).
- **Vector-symbolic architectures / HRR / FHRR / RFF** — carry over from paper 1.
- **Tokenisation / free representations (for §9):** fastText (Bojanowski et al. 2017 — word vec = sum of char-n-gram
  vecs, the compositional seeding); BPE (Sennrich et al. 2016), WordPiece, Unigram-LM (Kudo 2018) — variable-length
  subword vocab over a char base; Random Indexing (Sahlgren 2005) + BEAGLE (Jones & Mewhort 2007) — distributional
  meaning by superposition, no backprop; the hashing trick (Weinberger et al. 2009), Bloom / hash embeddings,
  content-addressable memory — deterministic fixed-footprint addressing for an unbounded token space; Smolensky
  tensor-product representations + quantum-cognition (Aerts) — the "bind = entanglement-like" framing (structural
  analogy, not literal QM).
- **Michael Levin — basal cognition / sorting algorithms as morphogenesis** (Zhang, Goldstein, Levin) — the
  frame for §1–2 (competency despite broken elements; goal-directedness). Cite as *inspiration*, honestly.
- Position vs the above: our contribution is not the averaging (known) but the **frozen algebraic codec as a
  shared coordinate system that makes cross-init continuous coupling near-lossless**, demonstrated in a live
  decentralized mesh.

## Reproducibility (bench modes, all in this repo)
`--lesion --colony --diverse --plasticity --emergence --mesh --collapse --average` (project `bench`),
`prismnet swarmbleed`, and `prismgym probe <bin>`.

## Open threads / TODO before writing
- [ ] Adversarial collapse test: seed each node's codec DIFFERENTLY, then average (isolate "codec protects"
      from "shared-basin regime protects").
- [ ] Continuous coupling on DISJOINT skills across different inits (not just same task) — does §6's 99%
      hold when the two nodes own different skills? (the real skill-transfer claim, cross-init)
- [ ] Scale beyond toy add(0..12) — a real multi-skill / language task to show the mesh result isn't
      arithmetic-specific.
- [ ] Anchor drift over days: seed a fresh anchor from today's prism.bin, re-pull later, measure drift
      toward the fleet mean.
- [ ] Decide the single headline figure. Candidate: the one row — one-shot 9% → continuous 75% →
      continuous+codec 99%.

### §9 head-start / tokenisation experiments to run
- [ ] Wire + run `--prototype` (PrototypeBench, written): few-shot curve, find where superposition saturates.
- [ ] Text token-seeding: small corpus → build token vectors (bind chars + RI contexts) → (a) nearest-neighbours
      sensible, (b) seed AlgFormer, measure training head-start vs random init. (The core "free pretraining" claim.)
- [ ] Finish §6-A properly: a TUNED transformer that actually FITS train, then re-measure the codec-seeding effect
      (current run was undertrained → inconclusive).
- [ ] Capacity bench: pack N real `Encode` faces, plot max-crosstalk + cleanup-decode acc vs N to 9k+ — verify the
      analytic "huge headroom" against the real FNV faces.
- [ ] Masked-merge vocab consensus: fixed latent address space + active mask + content-derived addresses; does
      frequent-token reinforcement / idiosyncratic wash-out happen across nodes with NO version bump? + M/collision sweep.
- [ ] Decide whether self-optimising tokenisation (§9.7) is a paper-2 section or a paper-3 line — it's the most
      novel and the least built.
