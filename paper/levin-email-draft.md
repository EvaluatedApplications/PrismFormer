# Draft email to Michael Levin — DO NOT SEND until numbers confirmed; Stephen sends, not Claude

Subject: Your basal-competency framing, tested in a phasor/VSA substrate — including where our intuition broke

Hi Professor Levin,

I'm a full-stack developer (networking background, not an academic), and your work on
sorting algorithms as morphogenesis — competency that shows up despite broken elements —
sent me down a concrete experimental path I wanted to share, because the result surprised me.

I've been building PrismFormer, a transformer variant whose representations are phasors
(vector-symbolic / holographic algebra) rather than learned dense matrices. I ran two
experiments in the spirit of your "sorting on broken hardware":

1. **Damage one trained network.** I expected the holographic representation to degrade
   gracefully. It did the opposite — killing just 5% of the weights collapses it (100% → ~31%
   accuracy on a task it had mastered). It's a *precise, low-redundancy* code: exact arithmetic
   needs exact phase alignment, so there's no spare redundancy to lose. My hypothesis was wrong.

2. **Damage the collective instead.** A colony of replicas, each on its own data shard, merging
   every round, with nodes randomly failing. Here the competency you describe *does* appear —
   with a lossless gradient-merge the colony holds 100% accuracy even when 40% of its nodes fail
   every round, and 90% at 60% failure. It only breaks when almost no hardware is left alive.

The punchline lands squarely on your thesis: in this substrate, basal competency is a property
of the **collective, not the individual cell**. One network is brittle; the group routes around
dead members and still gets there. I don't think I'd have looked for that without your framing.

It's all synthetic/toy so far, and I'm sharing it as an honest curiosity, not a claim. The
write-up (with a DOI) is here if it's of any interest: [Zenodo 10.5281/zenodo.21384525]

Thank you for making your work — and the way you think about goals and competency — so open.

Best,
Stephen Chen
```
Notes for Stephen before sending:
- Numbers confirmed from --colony (exact-merge: 100% @ 40% node failure, 90% @ 60%).
- Keep it short; he gets a lot of mail. Trim a paragraph if you like.
- Don't overclaim — "toy/synthetic, honest curiosity" is the right register.
- Your call whether to send at all — I think the collective-not-cell result earns it, but no pressure.
```
