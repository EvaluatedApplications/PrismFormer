// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

namespace PrismFormer;

/// <summary>
/// The PHASOR face layout — the uniform-rule successor to the poly/log/spell/orbital bands (see PrismLayout). A face
/// is <see cref="Comps"/> complex numbers stored as interleaved (cos, sin) reals. Every face is phasors and the
/// SINGLE operation is per-component complex multiply (bind); numbers are not special. A number fills the
/// linear-phase comps (bind = value ADD) and the log-phase comps (bind = value MULTIPLY); a symbol fills the
/// identity comps with a deterministic random phasor signature. The identity comps [0, <see cref="IdComps"/>) are
/// FROZEN (a concept's ground truth); the orbital tail is the learned meaning.
///
/// Dim/frozen match the legacy layout (64 reals, 32 frozen) so the AlgFormer contract is unchanged. A frequency
/// sweep showed 8 comps per band decode +/× cleanly over the gym's ranges, so the identity fits in 16 comps.
/// </summary>
internal static class PhasorLayout
{
    public const int LinComps = 16;                   // linear phase: bind = value add / unbind = subtract
    public const int LogComps = 16;                   // log phase:    bind = value multiply / unbind = divide
    public const int IdComps = LinComps + LogComps;   // 32 — frozen identity (number value / symbol signature)
    public const int OrbitComps = 96;                 // learned orbital tail. Dim HALVED 512→256: 512 was too heavy for CPU (even an i9 struggled) — pure double[] scalar math, no SIMD/GPU. PRISM-1 spec width = 256.
    public const int Comps = IdComps + OrbitComps;    // 128 complex components
    public const int Dim = 2 * Comps;                 // 256 reals (interleaved cos, sin) — PRISM-1 spec
    public const int FrozenReals = 2 * IdComps;       // 64 — the frozen identity prefix (value + signature survive bit-for-bit; everything past it is learned tail)

    public const int LinStart = 0;                    // comp index of the linear-phase band
    public const int LogStart = LinComps;             // comp index of the log-phase band
    public const int OrbitStart = IdComps;            // comp index of the orbital band
}
