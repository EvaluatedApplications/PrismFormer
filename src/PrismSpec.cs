// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

namespace PrismFormer;

/// <summary>
/// THE FROZEN SPEC. Every node in the swarm must run byte-identical architecture or it cannot merge gradients or
/// bleed pairs — so these constants are NOT configurable and are meant to outlive the decade. There is exactly one
/// production Prism model shape; the only knob is how much you train it. (Diagnostics/experiments may still build
/// off-spec tiny models directly via the AlgFormer ctor — those never join the swarm.)
///
/// Interop rule: a model is swarm-compatible iff its full Signature (Version, Vocab, vocab-content Hash, Dim, FrozenPrefix,
/// Context, Layers, Shifts) matches (the mesh gate is exact-Signature). Changing Version, Dim, FrozenPrefix, Layers, or the
/// VOCAB (any content change — the Hash gate makes even a same/larger-count regeneration a hard fork, so a mis-indexed
/// upgrade can't happen) → fresh weights. Only GROWING Context or Shifts is an UPGRADE-IN-PLACE within the same Version +
/// same vocab: old checkpoints carry over (zero-pad up, identity-preserving). See CanUpgradeFrom / AlgFormer.LoadUpgrade.
/// Any change still moves the Signature, so the whole swarm must move to the new build together (older-spec peers gated out).
/// </summary>
public static class PrismSpec
{
    // PRISM-3: the DEEP CODEC-ONLY reset. Hard fork from PRISM-2 (was L8 / S256 / c1024 / f64 learned-tail). Now a 32-layer
    // stack with a FULLY FROZEN codec embedding (FrozenPrefix = Dim → zero learned tail; every token IS its spelling's face),
    // a large 30k subword vocab (near-free because frozen), a shorter 512 window to fund depth, and S64 to keep deep training
    // tractable. The deep stack is IDENTITY-INIT (ReZero/Fixup: layers 1..L start with zeroed residual output projections) so
    // it is trainable from scratch instead of a cold all-random init. Version bump changes Signature → PRISM-2 checkpoints are
    // rejected on load (every node discards + rebuilds fresh on boot); the Studio "Reset model" button wipes them explicitly.
    public const string Version = "PRISM-3";

    public static int Vocab => CharVocab.Total;          // 96 chars + ~30.6k subword n-grams. Char ids 0..95 fixed forever; codec-only means EVERY row is frozen at its codec face → a big vocab costs only storage + softmax, no training.
    public const int Context = 512;                      // 512 characters of context (~85 words) — halved from 1024 to fund depth (context and depth both consume activation memory on a 6GB card)
    public const int Layers = 32;                        // DEPTH — the lever we max. Identity-init (see NewModel) makes a 32-deep stack trainable. ~L16-48 fit the 6GB card (activation-bound); L32 trains ~7.6s/batch on an RTX A3000.
    public const int Shifts = 64;                        // relation-bank rank (S) — kept modest so deep training stays tractable (banks = 7·S·Dim·Layers); shifts don't affect the depth ceiling (activation mem is S-independent)
    public const int InitSeed = 1;                       // canonical init seed (all nodes must init byte-identically for index-based merge)

    public static int Dim => PhasorLayout.Dim;           // 256 reals (= 128 complex components x 2) — frozen by the phasor codec (PhasorLayout)
    public static int FrozenPrefix => Dim;               // CODEC-ONLY: freeze the WHOLE embedding (= Dim) → zero learned tail; every token is exactly its codec face, so vocab is free to grow

    /// <summary>Build THE production model — a DEEP identity-init stack. <paramref name="embedSeed"/> supplies the phasor
    /// codec's per-symbol seed (pass PhasorCodec.Encode over the vocab); pass null for a bare-init model. The stack is built
    /// 1-layer-then-<see cref="AlgFormer.GrowLayers"/>-to-Layers so layers 1..Layers-1 start as the IDENTITY (zeroed residual
    /// output projections, ReZero/Fixup) — this is what makes a 32-deep model trainable from scratch. Deterministic, so every
    /// node produces a byte-identical fresh model (required for index-based gradient/weight merge).</summary>
    public static AlgFormer NewModel(Func<int, double[]>? embedSeed = null)
    {
        var m = new AlgFormer(Vocab, shifts: Shifts, layers: 1, maxContext: Context, dModel: Dim, frozenPrefix: FrozenPrefix, embedSeed: embedSeed, seed: InitSeed);
        return Layers > 1 ? m.GrowLayers(Layers - 1, zeroOutputOnly: true, seed: InitSeed) : m;
    }

    /// <summary>8-hex content hash of the subword table — folds the actual vocab CONTENT into the Signature so a REGENERATED
    /// vocab (different tokens at the same indices, even at the same or a larger count) is a hard fork, NOT a spurious
    /// append-only upgrade. Char rows 0..95 are fixed, so hashing the subwords captures the entire variable part.</summary>
    public static string VocabHash
    {
        get
        {
            ulong h = 1469598103934665603UL;
            foreach (var s in SubwordTable.List) { foreach (var ch in s) { h ^= ch; h *= 1099511628211UL; } h ^= 10; h *= 1099511628211UL; }
            return ((uint)(h ^ (h >> 32))).ToString("x8");
        }
    }

    /// <summary>Compact tag written into checkpoints so an incompatible model is rejected rather than silently mis-merged.</summary>
    public static string Signature => $"{Version}/v{Vocab}/h{VocabHash}/d{Dim}/f{FrozenPrefix}/c{Context}/L{Layers}/S{Shifts}";

    /// <summary>Parsed spec fields from a <see cref="Signature"/> string (null if unparseable). Hash is "" for pre-hash sigs.</summary>
    public sealed record Fields(string Version, int Vocab, string Hash, int Dim, int Frozen, int Context, int Layers, int Shifts);

    public static Fields? Parse(string sig)
    {
        try
        {
            var p = (sig ?? "").Split('/');
            int G(string k) => int.Parse(p.First(x => x.StartsWith(k))[k.Length..]);
            string GS(string k) { var x = p.FirstOrDefault(y => y.StartsWith(k)); return x is null ? "" : x[k.Length..]; }
            return new Fields(p[0], G("v"), GS("h"), G("d"), G("f"), G("c"), G("L"), G("S"));
        }
        catch { return null; }
    }

    /// <summary>Can a checkpoint saved under <paramref name="old"/> be UPGRADED-in-place into the current spec? True iff the
    /// Version, the exact VOCAB CONTENT (Hash), Dim, Frozen and Layers all match, and only Shifts/Context are the-same-or-
    /// smaller (we zero-pad those up, see <see cref="AlgFormer.LoadUpgrade"/>). The Hash gate means ANY vocab regeneration —
    /// even to the same or a larger count — is a HARD FORK (discard + rebuild), never a silent mis-indexed upgrade. A pre-hash
    /// checkpoint has Hash="" ≠ VocabHash, so it too is correctly rejected. Anything else → start fresh.</summary>
    public static bool CanUpgradeFrom(Fields old)
        => old.Version == Version && old.Hash == VocabHash && old.Vocab == Vocab && old.Dim == Dim && old.Frozen == FrozenPrefix
           && old.Layers == Layers && old.Shifts <= Shifts && old.Context <= Context;

    /// <summary>Mesh gate: may we exchange weights with a peer advertising <paramref name="theirSig"/>? Only EXACT-spec peers
    /// merge safely (weight-slices are shape-specific). A peer on an older/newer/incompatible spec is blocked — they must
    /// update; their checkpoint upgrades in place once they run the current build.</summary>
    public static bool MeshCompatible(string theirSig) => theirSig == Signature;
}
