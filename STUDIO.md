# Prism Studio — the networked, self-teaching trainer

> Status: **design + foundation.** The unifying data-plane primitive (`PairSource`) is being built first; the rest
> (P2P topology, gossip protocol, UI wiring) sequences off it. See [SWARM.md](SWARM.md) for the swarm internals.

## The one insight: everything is a training pair

Default trainers, user data, and neighbour gossip are all the same thing — a stream of `(context → target)` pairs.
So Gym and BabyLM are not two systems; they are two **pair sources** feeding **one model**. This collapses the whole
vision to: *one model, many pair sources, some of which arrive over the network.*

```
                         ┌──────────── one model (Gym XOR BabyLM, one run at a time) ───────────┐
   default generator ───▶│                                                                       │
   data/  (user pairs) ─▶│   train on the union of all pair sources over a shared char vocab     │──▶ saved model
   gossip/ (neighbours)─▶│                                                                       │
                         └───────────────────────────────────────────────────────────────────────┘
```

## Decisions this commits to

- **One shared vocabulary.** "One model for Gym or BabyLM" only works if both tokenize the same way. Commit to
  **char-level over printable ASCII (32..126, 95 symbols)** — BabyLM is already there, Gym reduces to it, and a
  user's raw text file is trainable as-is. One vocab ⇒ one model ⇒ one checkpoint.
- **One run at a time.** A single active model + a single run lock. Gym and BabyLM share the model; starting one
  stops the other. The saved file is the model, not "the gym model" or "the babylm model."
- **Folders are the interface.** Under the model's directory: `data/` (user drops raw pairs — plain text or
  `prompt<TAB>target`), `gossip/` (what neighbours sent). Both are just more pair sources. Human-inspectable,
  human-editable, trivially shareable.

## Two data planes (they are different on purpose)

1. **Deterministic default** — the built-in curriculum + the shared corpus. Index-addressable, so it rides the
   manifest/position relay: nodes reconstruct it locally, only positions cross the wire (already built, see SWARM.md).
2. **Local + gossip pairs** — `data/` and `gossip/`. NOT deterministic across nodes (everyone's folder differs), so
   these are shared as **actual pairs** — this is the bleed medium. A training pair *is* a fragment of skill:
   human-readable, order-independent, and the receiver trains on it in its own way, so it stays itself.

## Neighbours by ping

- Peers connect P2P; the bootstrap **server we host** just helps them find each other (discovery + relay fallback).
- Each peer pings candidates and keeps the **lowest-latency** ones as neighbours — the ring/mesh from the bleed
  experiment, where local diffusion is strongest. Ping-rating is the topology.

## Bleed = sharing pairs / distillation

When a node bleeds with a neighbour, it does **not** average weights (we measured that's a blunt convergence dial).
It sends a small sample of pairs, which the neighbour writes into its `gossip/` folder and trains on:

- **Raw pairs** — examples from my `data/` or default stream that I found high-value.
- **Distilled pairs** — `(input, MY prediction)`: I teach my behaviour, the neighbour absorbs a fragment of my skill
  without ever copying my weights. This is the mechanism SwarmBleed pointed to as the way to open the "separate but
  bleeding" middle regime.

Selection policy (a real knob, start simple): share pairs I'm **confident** on (good teaching signal); skip ones I'm
unsure of. Later: prefer novel/high-loss-for-the-neighbour pairs (active teaching).

## Trust & limits (honest, needed before opening it up)

- `gossip/` is **capacity-capped** (ring-trim oldest) so it can't explode.
- **Dedup** on arrival; **provenance** tag per pair (which neighbour) so a bad source can be down-weighted/blocked.
- A poisoned pair is lower-blast-radius than a poisoned gradient (it's one example among many, and it's readable) —
  but it's still real. Confident-only sharing + caps + provenance are the first line.

## Build sequence

1. **`PairSource` data-plane** ← *starting here.* Shared char vocab; read `(ctx→target)` pairs from the default
   generator + `data/` + `gossip/`; index-addressable; a `GossipInbox` that appends/caps/dedups received pairs.
2. **One-model refactor** in the studio: single model + run lock; Gym and BabyLM become pair sources over it.
3. **Network chatter** in Gym + BabyLM: during a run, periodically send confident pairs to neighbours and drain the
   inbox into `gossip/` (which is already a training source from step 1).
4. **Ping-rated P2P + bootstrap server**: neighbour selection by latency; the hosted server for discovery/relay.

Each step is independently testable. Step 1 needs no network — it's the substrate everything else feeds.
