// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

namespace PrismFormer;

/// <summary>
/// THE FROZEN SPEC. Every node in the swarm must run byte-identical architecture or it cannot merge gradients or
/// bleed pairs — so these constants are NOT configurable and are meant to outlive the decade. There is exactly one
/// production Prism model shape; the only knob is how much you train it. (Diagnostics/experiments may still build
/// off-spec tiny models directly via the AlgFormer ctor — those never join the swarm.)
///
/// Interop rule: a model is swarm-compatible iff its (Version, Vocab, Dim, FrozenPrefix, Context, Layers, Shifts) all
/// match (the mesh gate is exact-Signature). Changing a NON-GROWABLE dim (Version, Dim, FrozenPrefix, Layers) is a
/// hard fork → new Version + fresh weights. GROWING Context, Shifts, or Vocab is an UPGRADE-IN-PLACE within the same
/// Version: old checkpoints carry over (Context/Shifts zero-pad up, identity-preserving; Vocab is APPEND-ONLY — the
/// char rows 0..95 carry over and appended subword rows seed from their codec faces). See CanUpgradeFrom /
/// AlgFormer.LoadUpgrade. It still changes the Signature, so the whole swarm must move to the new build together
/// (older-spec peers are gated out).
/// </summary>
public static class PrismSpec
{
    // PRISM-2: CAUSAL attention (was PRISM-1 bidirectional). Same dims — a mask + KV-cache change, not a reshape — but the
    // weights mean something different, so PRISM-2 must not merge with / load PRISM-1 checkpoints. Bumping Version changes
    // Signature, so old checkpoints are rejected on load (start fresh); the Studio "Reset model" button wipes them explicitly.
    public const string Version = "PRISM-2";

    public static int Vocab => CharVocab.Total;          // 96 chars + committed subword n-grams (append-only). Char ids 0..95 are fixed forever; subwords append after, so vocab GROWS in place.
    public const int Context = 1024;                     // 1024 characters of context (~170 words / a paragraph of chat) — grown in place 256→512→1024 (older checkpoints zero-pad up via LoadUpgrade); cheap on params + everyday serve, but O(T²) attention when the window is actually FILLED
    public const int Layers = 8;                         // depth
    public const int Shifts = 256;                       // relation-bank rank (S) — grown 64→256 (max, = Dim) in place via LoadUpgrade (new shift rows zero-pad, identity-preserving). ~4x the S64 bank compute.
    public const int InitSeed = 1;                       // canonical init seed

    public static int Dim => PhasorLayout.Dim;           // 256 reals (= 128 complex components x 2) — frozen by the phasor codec (PhasorLayout)
    public static int FrozenPrefix => PhasorLayout.FrozenReals;  // 64 — frozen identity prefix

    /// <summary>Build THE production model. <paramref name="embedSeed"/> supplies the phasor codec's per-symbol seed
    /// (pass PhasorCodec.Encode over the char vocab); pass null for a bare-init model.</summary>
    public static AlgFormer NewModel(Func<int, double[]>? embedSeed = null)
        => new(Vocab, shifts: Shifts, layers: Layers, maxContext: Context, dModel: Dim, frozenPrefix: FrozenPrefix, embedSeed: embedSeed, seed: InitSeed);

    /// <summary>Compact tag written into checkpoints so an incompatible model is rejected rather than silently mis-merged.</summary>
    public static string Signature => $"{Version}/v{Vocab}/d{Dim}/f{FrozenPrefix}/c{Context}/L{Layers}/S{Shifts}";

    /// <summary>Parsed spec fields from a <see cref="Signature"/> string (null if unparseable).</summary>
    public sealed record Fields(string Version, int Vocab, int Dim, int Frozen, int Context, int Layers, int Shifts);

    public static Fields? Parse(string sig)
    {
        try
        {
            var p = (sig ?? "").Split('/');
            int G(string k) => int.Parse(p.First(x => x.StartsWith(k))[k.Length..]);
            return new Fields(p[0], G("v"), G("d"), G("f"), G("c"), G("L"), G("S"));
        }
        catch { return null; }
    }

    /// <summary>Can a checkpoint saved under <paramref name="old"/> be UPGRADED-in-place into the current spec? True iff the
    /// non-growable dims (Version, Vocab, Dim, Frozen, Layers) match and only Shifts/Context are the same-or-smaller — i.e.
    /// we can zero-pad it up (see <see cref="AlgFormer.LoadUpgrade"/>). Anything else is a hard fork → start fresh.</summary>
    public static bool CanUpgradeFrom(Fields old)
        => old.Version == Version && old.Vocab <= Vocab && old.Dim == Dim && old.Frozen == FrozenPrefix
           && old.Layers == Layers && old.Shifts <= Shifts && old.Context <= Context;   // Vocab may GROW (append-only): old char rows carry over, new subword rows seed from their codec faces

    /// <summary>Mesh gate: may we exchange weights with a peer advertising <paramref name="theirSig"/>? Only EXACT-spec peers
    /// merge safely (weight-slices are shape-specific). A peer on an older/newer/incompatible spec is blocked — they must
    /// update; their checkpoint upgrades in place once they run the current build.</summary>
    public static bool MeshCompatible(string theirSig) => theirSig == Signature;
}
