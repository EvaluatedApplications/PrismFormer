# Prism Economy — a market of personal currencies over the swarm

> Status: **vision / captured for later.** No code yet — this is phase-2, a separate discipline from the trainer
> (see SWARM.md / STUDIO.md). Written down so the design doesn't evaporate. Prior art noted so future-us doesn't
> reinvent it.

## The core moves

1. **Everyone mints their own currency.** No mining. You self-issue with your own key; issuance is free. Value is
   *not* set by proof-of-work — it's set by a **market** that prices your coin by how good your model's answers are.
   You manage your own inflation (mint more = cheaper/accessible, stay scarce = valuable). Precedent: Hayek's
   *denationalization of money*, mutual-credit / Ripple trust-lines, creator/social tokens, bonding curves. Best read
   as **transferable reputation**.

2. **The peg.** A freely-minted coin is worth something because **people need your coin to query your model.** Demand
   for your coin = demand for your answers = your model's quality. That's the anchor; without it, price is vibes.

3. **Minting = proof-of-useful-interaction.** Coin is minted at a **countersigned query/training event** and held by
   the participants — so the money supply grows in proportion to real useful work, minted where the value was created.
   The counterparty must co-sign that the interaction happened, so you can't mint from a query you faked alone. That
   reduces the whole attack surface to **collusion** (two parties co-signing hollow interactions), which the market
   then punishes (colluders pump coins nobody else values).

4. **Acceptance-policy tiering.** Each node chooses which coins it accepts:
   - *Open* (accepts anyone's coin) → max accessibility, max volume, max passive learning; accumulates a diversified
     basket of everyone's coins → becomes a **market-maker / liquidity provider**, carrying everyone's credit risk.
   - *Premium* (accepts only its own coin) → to query you, a peer must first acquire your coin on the exchange →
     manufactured demand straight into your price; zero counterparty risk; fully sovereign. Only viable if you're
     good enough that people bother — so the policy itself is a priced confidence signal.
   The two archetypes *need each other*: premium nodes are only reachable because open nodes provide the liquidity.
   And the premium tier lands on the same nodes as the competence/routing tier — money flows where the queries do.

5. **Gaming-as-feature (incentive compatibility).** Every exploit is designed to *become* the useful behavior, à la
   Bitcoin (the 51% "attack" is just mining, which secures the chain):
   - wash-trade / self-pump → **self-training** (pumping needs your model to actually improve; self-querying is backprop).
   - hammer the network → **sponsor the swarm** (queries cost tokens; labelled ones teach whoever answers).
   - sybil 1000 identities → **1000 real nodes** (a fake coin is worthless unless its model is good).
   - answer everything to farm → **be right or get better** (wrong answers earn nothing but hand you a backprop pass).
   - print infinite coin → **dilute only yourself**.
   The rule that makes them all flip: *peg value to realized, verifiable learning/quality, so every shortcut routes
   through doing the work.* Residual attacks that DON'T flip: **collusion rings** and **deep-capital manipulation** —
   those still need real defenses (stake, quadratic weighting, order-book surveillance).

## The ledger

- **Agent-centric, not one global chain.** Each agent keeps their own hash-linked chain (their issuance + receipts),
  holds their own key. There is **no global blockchain bottleneck** — most transactions are 2-party (a query), so they
  are **co-signed by both parties and gossiped to a shared DHT** for tamper-evidence and double-spend checks. Global
  consensus is needed only to stop double-spends, which countersigning + the DHT handle. This is *why* "no mining".
- **Closest existing system: Holochain** (agent-centric source chains + DHT validation + mutual-credit currencies as
  its canonical example — almost exactly this). Interop-of-sovereign-chains precedent: **Cosmos** (zones + hub/IBC),
  **Polkadot** (parachains + relay chain). Multi-currency exchange precedent: **Stellar/Ripple** path-payments, **Uniswap**-style AMMs.

## Where Prism has a real edge

The one genuinely hard, unsolved-in-general problem in crypto-ML is **verifiable proof of useful work** (proving a node
did honest training/inference cheaply, no trusted referee). Our architecture already carries the primitive: the
**manifest** makes work deterministic + index-addressable, and gradients are **bit-exact mergeable** — so any claimed
work is **challenge-and-recompute** verifiable to the bit. Most crypto-ML projects (Bittensor/TAO, Gensyn, Ritual) have
to bolt this on; we get it from the way we already ship work.

## Honest scope

This is a **distributed-ledger platform** — consensus, tokenomics, crypto-econ, a liquid multi-currency exchange —
a multi-year build in its own right, orthogonal to the trainer. Holochain is ~a decade of work and still niche. The
residuals (cross-chain double-spend edge cases, collusion, liquidity/cold-start) are where such projects live or die.
Capture now; build only after the swarm itself is real and used.
