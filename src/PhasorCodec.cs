// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using System.Globalization;

namespace PrismFormer;

/// <summary>
/// The PHASOR codec — the uniform-rule identity codec. One rule for every face: a face is a bundle of phasors and
/// the single operation is per-component complex multiply (<see cref="Bind"/>), its conjugate (<see cref="Unbind"/>)
/// and sum (<see cref="Bundle"/>). Numbers are NOT special: a number's linear-phase comps make <c>bind = value add</c>
/// and its log-phase comps make <c>bind = value multiply</c>; a symbol is a random phasor signature. Values are read
/// back by CORRELATION (every value is its own phase pattern), which is what fixes the colinear-ray problem the
/// poly/log bands had — a computed result is nameable, not just landable.
///
/// It carries NO calculator — arithmetic is <see cref="Bind"/> read through a projection, one operation, no second
/// route. The only place a number is treated differently is ENCODING: a token that parses as a finite number has its
/// VALUE written as phase (so bind = arithmetic); anything else is hashed to a signature. That is representation, not
/// computation — the compose (<see cref="Bind"/>) and readout (correlation) are identical for numbers and symbols.
/// </summary>
public static class PhasorCodec
{
    // Deterministic, fixed random frequencies per band (higher spread → sharper decode peak).
    private static readonly double[] LinTheta = Frequencies(0xC0DEC0DEu, PhasorLayout.LinComps);
    private static readonly double[] LogTheta = Frequencies(0xF00DBABEu, PhasorLayout.LogComps);

    public static int Dim => PhasorLayout.Dim;

    /// <summary>Length of the frozen identity prefix (reals) — pass as AlgFormer's frozenPrefix so number value stays exact.</summary>
    public static int FrozenReals => PhasorLayout.FrozenReals;

    /// <summary>Encode a symbol into a full phasor face — number value (linear+log phase) or symbol signature+orbital.</summary>
    public static double[] Encode(string symbol)
    {
        var f = new double[PhasorLayout.Dim];
        if (IsNumber(symbol, out var v)) { WriteNumber(f, v); return f; }  // orbital stays 0 for numbers
        WriteSignature(f, symbol);
        WriteOrbitalSeed(f, symbol);
        return f;
    }

    /// <summary>Does the token represent a FINITE number literal? If so its value is encoded as phase (bind = arithmetic);
    /// otherwise it is a symbol. Not a calculator — just the on-ramp for a number's value into the phasor face. Non-finite
    /// (NaN/Infinity) is rejected so it can never poison a face with NaN phases.</summary>
    public static bool IsNumber(string? symbol, out double value)
        => double.TryParse(symbol?.Trim(), NumberStyles.Float | NumberStyles.AllowLeadingSign, CultureInfo.InvariantCulture, out value) && double.IsFinite(value);

    /// <summary>The pure value face for a number (identity comps filled, orbital 0).</summary>
    public static double[] NumberFace(double v) { var f = new double[PhasorLayout.Dim]; WriteNumber(f, v); return f; }

    // ---- the ONE rule: per-component complex multiply / conjugate / sum ----

    /// <summary>Bind — per-component complex multiply. exp(iaθ)·exp(ibθ)=exp(i(a+b)θ): linear comps add values, log comps add logs (= multiply values).</summary>
    public static double[] Bind(double[] a, double[] b)
    {
        var h = new double[a.Length];
        for (var i = 0; i < a.Length; i += 2) { double ca = a[i], sa = a[i + 1], cb = b[i], sb = b[i + 1]; h[i] = ca * cb - sa * sb; h[i + 1] = ca * sb + sa * cb; }
        return h;
    }

    /// <summary>Unbind — multiply by the conjugate. exp(iaθ)·exp(-ibθ)=exp(i(a-b)θ): linear comps subtract, log comps divide.</summary>
    public static double[] Unbind(double[] a, double[] b)
    {
        var h = new double[a.Length];
        for (var i = 0; i < a.Length; i += 2) { double ca = a[i], sa = a[i + 1], cb = b[i], sb = b[i + 1]; h[i] = ca * cb + sa * sb; h[i + 1] = sa * cb - ca * sb; }
        return h;
    }

    /// <summary>Bundle — superpose (component-wise sum). The set-forming operation (records, superpositions).</summary>
    public static double[] Bundle(params double[][] xs)
    {
        var h = new double[PhasorLayout.Dim];
        foreach (var x in xs) for (var i = 0; i < h.Length; i++) h[i] += x[i];
        return h;
    }

    // ---- readout by correlation ----

    /// <summary>Full-face correlation (real dot product) — the symbol/associative readout.</summary>
    public static double Correlate(double[] a, double[] b) => Correlate(a, b, 0, PhasorLayout.Comps);

    internal static double Correlate(double[] a, double[] b, int compLo, int compHi)
    {
        var s = 0.0;
        for (var i = 2 * compLo; i < 2 * compHi; i++) s += a[i] * b[i];
        return s;
    }

    /// <summary>Decode the value carried by <paramref name="h"/>'s linear-phase comps (sum band): argmax over 0..max.</summary>
    public static int DecodeSum(double[] h, int max) => DecodeBand(h, PhasorLayout.LinStart, PhasorLayout.LinStart + PhasorLayout.LinComps, 0, max);

    /// <summary>Decode the value carried by <paramref name="h"/>'s log-phase comps (product band): argmax over 1..max.</summary>
    public static int DecodeProduct(double[] h, int max) => DecodeBand(h, PhasorLayout.LogStart, PhasorLayout.LogStart + PhasorLayout.LogComps, 1, max);

    private static int DecodeBand(double[] h, int compLo, int compHi, int lo, int max)
    {
        int best = lo; var bz = double.NegativeInfinity;
        for (var c = lo; c <= max; c++)
        {
            var s = Correlate(NumberFace(c), h, compLo, compHi);
            if (s > bz) { bz = s; best = c; }
        }
        return best;
    }

    // ---- band writers ----

    private static void WriteNumber(double[] f, double v)
    {
        for (var k = 0; k < PhasorLayout.LinComps; k++) { var ph = v * LinTheta[k]; var c = PhasorLayout.LinStart + k; f[2 * c] = Math.Cos(ph); f[2 * c + 1] = Math.Sin(ph); }
        if (v > 0)
        {
            var ln = Math.Log(v);
            for (var k = 0; k < PhasorLayout.LogComps; k++) { var ph = ln * LogTheta[k]; var c = PhasorLayout.LogStart + k; f[2 * c] = Math.Cos(ph); f[2 * c + 1] = Math.Sin(ph); }
        }
    }

    private static void WriteSignature(double[] f, string symbol)
    {
        var phase = Phases(symbol, 0x9E3779B9u, PhasorLayout.IdComps);
        for (var c = 0; c < PhasorLayout.IdComps; c++) { f[2 * c] = Math.Cos(phase[c]); f[2 * c + 1] = Math.Sin(phase[c]); }
    }

    private static void WriteOrbitalSeed(double[] f, string symbol)
    {
        var phase = Phases(symbol, 0x85EBCA77u, PhasorLayout.OrbitComps);
        for (var k = 0; k < PhasorLayout.OrbitComps; k++) { var c = PhasorLayout.OrbitStart + k; f[2 * c] = Math.Cos(phase[k]); f[2 * c + 1] = Math.Sin(phase[k]); }
    }

    // ---- deterministic hashing (FNV-1a → phases in (-π, π]) ----

    private static double[] Frequencies(uint salt, int n) => Phases($"freq:{salt:X8}", salt, n);

    private static double[] Phases(string s, uint salt, int n)
    {
        var v = new double[n];
        var h = 1469598103934665603UL ^ salt;
        foreach (var c in s) { h ^= c; h *= 1099511628211UL; }
        for (var i = 0; i < n; i++)
        {
            h ^= (ulong)(i + 1); h *= 1099511628211UL;
            var u = ((h >> 11) & 0xFFFFF) / (double)0xFFFFF;   // [0,1)
            v[i] = (u * 2.0 - 1.0) * Math.PI;
            h ^= h >> 7;
        }
        return v;
    }
}
