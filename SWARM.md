# PrismFormer Swarm — a self-replicating training/inference colony

> Status: **design / hypothetical.** Nothing here is proven at swarm scale yet. Two pieces of the core
> (bit-exact gradient merge, the manifest) exist and are verified locally; everything else is architecture.
> This doc exists to check that the design *closes in code, not just in concept*.

## The one idea

Frontier distributed systems (Petals, Hivemind, `torch.distributed`) are shaped by a constraint PrismFormer
doesn't have: **the model doesn't fit on one machine.** That forces them to *shard* an organism across nodes —
a fragile pipeline where losing a peer severs the model.

PrismFormer is CPU-small. So we invert the whole design:

> **Every node holds the *whole* model.** The network is not a sharded organism — it is a colony of identical
> **cells**, each a complete copy that can train *and* serve the entire model alone.

Smallness stops being a limitation and becomes the enabling property. A colony of full copies is:

- **Homogeneous** — a phone-class ARM chip and a desktop run the *same cell*, no matched hardware.
- **Unkillable** — kill any cell; every survivor still trains and serves the full model.
- **Mathematically coherent across mismatched speeds** — because gradients are *bit-exact mergeable*, a slow
  node's contribution and a fast node's contribution combine into identical math. A ragtag swarm stays exact.

`torch.distributed` cannot say that. It needs matched GPUs. This is the gold mine underneath the gold mine.

---

## The cell

One machine. Identical software on every machine. A cell carries the complete works:

| Organ | What it is | Status in repo |
|---|---|---|
| **Genome** | full model weights (`AlgFormer`) | ✅ `AlgFormer.Serialize/Deserialize` |
| **Program** | the manifest: deterministic data recipe + task + merge cadence | ✅ `IJobSource.Manifest()` / `FromManifest()` |
| **Metabolism** | train loop: pull positions → gradient → gossip delta → merge inbound | ✅ `LocalJobTrainer`, `SerializeGradient`, exact merge |
| **Voice** | HTTP endpoint serving inference from the local copy | ⬜ not built |
| **Reproduction** | on invite, hand {join-code + binary + weights} to a new machine | ⬜ partial (`MqttJob.MakeCode`) |
| **Senses** | gossip membership + discovery; NAT hole-punch + relay fallback | ⬜ have relay star, not mesh |

### The cell state machine

The claim to test is that **train / merge / serve / reproduce interleave on one node without stalling each
other.** They do, if serving reads a snapshot and training owns the writable weights:

```
                    ┌─────────────────────────────────────────────┐
                    │                  CELL                         │
                    │                                               │
   join-code  ──▶   │  BOOT: parse code → seed peers                │
                    │        pull manifest → build IJobSource       │
                    │        pull weights  → AlgFormer.Deserialize  │
                    │                                               │
                    │        ┌────────────── LIVE ──────────────┐   │
                    │        │                                   │   │
   inbound deltas ─▶│   [MERGE]  apply mergeable deltas          │   │
                    │        │      (bit-exact accumulate)        │   │
                    │        │            │                       │   │
                    │        │            ▼                       │   │
                    │   [TRAIN]  K local steps over own shard     │   │
                    │        │      (positions by index)          │   │
                    │        │            │                       │   │
                    │        │            ▼                       │   │
   outbound delta ◀─│   [GOSSIP] emit merged delta every K steps  │   │
                    │        │            │                       │   │
                    │        │            ▼ (async, snapshot)      │   │
   HTTP query   ───▶│   [SERVE]  forward pass on weight snapshot ─┼──▶ tokens
                    │        │            │                       │   │
                    │        │            ▼ (on invite)           │   │
   new machine  ◀───│   [SPAWN]  ship {code + binary + weights}   │   │
                    │        └───────────────────────────────────┘   │
                    └─────────────────────────────────────────────┘
```

Key invariant that makes it close: **`SERVE` reads an immutable snapshot; `TRAIN`/`MERGE` own the mutable
weights.** Double-buffer — publish a fresh read-only snapshot after each merge. Serving never blocks training,
training never corrupts a mid-flight forward pass. `SPAWN` ships the same snapshot `SERVE` reads.

---

## The colony

N identical cells. The only thing that must stay coherent is the weights, and they're kept coherent by
**periodic mergeable-delta gossip** — not per-step sync.

### Convergence: real, not hoped

- **Synchronous limit:** if every cell syncs each step, the colony is **bit-identical** to one giant batch on a
  single machine. Provable — that's what the exact merge buys.
- **Practical regime:** nobody runs synchronous (bandwidth). Relax to **local-SGD / DiLoCo / FedAvg**: each cell
  trains K steps locally, then gossips a merged delta. Known to converge under bounded staleness. Our manifest +
  exact-merge drop straight into this pattern.

So "will a swarm of laptops learn what one server would?" has an answer — *yes, up to the sync cadence you
choose* — not a wish.

### Verifiable work (the anti-poison lever)

Because the manifest makes work **deterministic by index**, any cell can **recompute and check** another cell's
claimed gradient. Most open swarms can't do this. It gives real Byzantine defense options:

- challenge-and-recompute (spot-check a random claimed shard),
- reputation weighted by verification pass-rate,
- permissioned/friends-only colony for the trusted case.

> **Honest tension:** Byzantine-robust aggregation (median / Krum / trimmed-mean) *breaks* bit-exactness. Each
> round you pick **provably identical** *or* **provably robust** — not both. That's a genuine fork, not a free
> lunch. Trusted colony → exact. Open colony → robust.

---

## The honest hard nucleus

Three load-bearing problems. Each has a real handle; none is magic.

1. **Bandwidth.** Naive all-to-all gradient gossip is O(N²) — the classic swarm killer. Handle: local-SGD
   cadence + delta compression + tree/hierarchical gossip. Send merged deltas periodically, not every gradient
   to everyone every step.
2. **Poisoning.** Open membership lets anyone gossip junk. Handle: manifest-verifiable work (above).
3. **Discovery + NAT.** Handle: gossip membership (SWIM) or a DHT for discovery; hole-punch with **relay
   fallback** for the ~20–30% of NAT pairs that can't punch. The join-code bootstraps it — it embeds seed peers,
   and every cell can seed the next. *That is the self-replication mechanism.* Most of this substrate already
   exists in **libp2p**; wrapping it beats rebuilding it in C#.

---

## The one line that keeps it a swarm and not a worm

**Self-replicating must mean *trivially cloneable + viral by invitation*, never *self-propagating without
consent*.** You paste a code, you opt in, you can invite others. A cell must **never auto-install** onto a
machine. Consensual replication → a legitimate colony that can grow explosively. Autonomous propagation →
malware, felony, full stop, regardless of intent. The line is easy to stay on the right side of: no cell ever
installs itself anywhere; a human always pastes the code.

---

## What's real vs. what's new

Already built and verified (locally / toy scale):

- **Bit-exact mergeable gradients** — `AlgFormer.SerializeGradient` / `DeserializeGradient`, exact accumulate.
- **The manifest** — `IJobSource.Manifest()` / `FromManifest()`; workers reconstruct data by index, data never
  crosses the wire (`relaytest2`: 2 workers built a 100k-example source from a manifest, trained by positions).
- **Zero-friction join** — paste a base64 code (`MqttJob.MakeCode`), relay through a broker, works through NAT.
- **Elastic pool + straggler weighting** — the slowest node doesn't stall the round.

What the swarm adds on top (mostly *substrate*, not new science):

- HTTP inference organ (the **Voice**).
- Snapshot double-buffering so SERVE never blocks TRAIN.
- Gossip membership + discovery (SWIM / DHT — candidate: libp2p) to replace the relay *star* with a *mesh*.
- Local-SGD delta cadence + compression (bandwidth).
- Challenge-and-recompute verification (trust).
- Consensual spawn: ship {code + binary + weights} on invite.

**The novel, true claim:** a CPU-small model with exactly-mergeable gradients is the *one* case where a
full-replication swarm — every node a whole cell that both trains and serves — actually works. Big models can't
(they must shard → fragile). PrismFormer can.

---

## De-risking sequence (build order)

Don't build the mesh before proving anyone joins. Cheapest proof first:

1. **Two-network training proof** — a second machine on a *different* network pastes the code and its gradients
   land in your model over the real internet. Until this is real, everything above is architecture on an
   unproven base.
2. **The Voice** — add the HTTP inference endpoint reading a weight snapshot. Now a cell both trains and serves;
   the swarm becomes tangible ("paste a code, and you can query the model over HTTP").
3. **Snapshot double-buffer** — make SERVE/TRAIN coexist on one node without stalls.
4. **Substrate fork** — *only now*, forced by scale or censorship-resistance: build P2P routing vs. stand on
   libp2p. Speculative mesh-building before this step is the trap.

Each step is independently useful and independently demoable. The colony emerges from stacking them — it does
not need to be built all at once.
