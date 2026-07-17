// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;
using PrismFormer;

namespace PrismFormer.Gpu;

/// <summary>
/// Batched GPU forward pass for AlgFormer (fp32), built from a CPU model's Serialize() bytes (no core-lib changes).
/// Mirrors AlgFormer.ForwardAll exactly: per layer q/k/v = relbank(R*, h); causal attention (scores→softmax→Σa·v);
/// o = relbank(Ro, ctx); m = h+o; GLU z = aval·σ(gt) with aval=relbank(A1,m), gt=relbank(Ag,m); y = m+relbank(Ao,z).
/// Logits at the last position: C[w] + Σ emb[w]·h_last. Gradchecked against AlgFormer.LogitsFor — the CPU stays the
/// oracle; GPU is fp32-close, not bit-identical. (Non-bind GLU path only, the default.)
/// </summary>
public sealed class GpuModel : IDisposable
{
    readonly Accelerator _acc;
    readonly int _v, _s, _layers, _maxT, _d;
    readonly MemoryBuffer1D<float, Stride1D.Dense> _emb, _pos, _cbias;
    readonly MemoryBuffer1D<float, Stride1D.Dense>[] _banks;   // per layer: 7*s*d in order Rq,Rk,Rv,Ro,A1,Ag,Ao

    readonly Action<Index1D, ArrayView<int>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> _embed;
    readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> _relbank;
    readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, float> _scores;
    readonly Action<Index1D, ArrayView<float>, ArrayView<float>, int> _softmax;
    readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> _context;
    readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> _add;
    readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>> _glu;
    readonly Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int> _logits;

    public GpuModel(byte[] serialized)
    {
        _acc = GpuDevice.Accelerator ?? throw new InvalidOperationException("no accelerator");
        using var r = new BinaryReader(new MemoryStream(serialized));
        _v = r.ReadInt32(); _s = r.ReadInt32(); _layers = r.ReadInt32(); _maxT = r.ReadInt32(); _d = r.ReadInt32(); r.ReadInt32();   // frozen (unused in forward)
        float[] Read(int n) { var a = new float[n]; for (var i = 0; i < n; i++) a[i] = (float)r.ReadDouble(); return a; }
        _emb = _acc.Allocate1D(Read(_v * _d));
        _pos = _acc.Allocate1D(Read(_maxT * _d));
        _cbias = _acc.Allocate1D(Read(_v));
        _banks = new MemoryBuffer1D<float, Stride1D.Dense>[_layers];
        for (var l = 0; l < _layers; l++) _banks[l] = _acc.Allocate1D(Read(7 * _s * _d));

        _embed = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<int>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(
            (idx, toks, emb, pos, h, T, d) => { int i = idx % d, t = (idx / d) % T, b = idx / (T * d); var tok = toks[b * T + t]; h[idx] = emb[tok * d + i] + pos[t * d + i]; });
        _relbank = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(
            (idx, x, bank, y, d, s) => { int row = idx / d, i = idx % d; float a = 0f; for (var k = 0; k < s; k++) { var xi = i + k; if (xi >= d) xi -= d; a += bank[k * d + i] * x[row * d + xi]; } y[idx] = a; });
        _scores = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, float>(
            (idx, q, k, sc, T, d, inv) => { int j = idx % T, tt = (idx / T) % T, b = idx / (T * T); if (j > tt) { sc[idx] = -1e30f; return; } float a = 0f; for (var c = 0; c < d; c++) a += q[(b * T + tt) * d + c] * k[(b * T + j) * d + c]; sc[idx] = a * inv; });
        _softmax = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, int>(
            (idx, sc, a, T) => { int t = idx % T, b = idx / T; var baseI = (b * T + t) * T; float mx = -1e30f; for (var j = 0; j <= t; j++) if (sc[baseI + j] > mx) mx = sc[baseI + j]; float sum = 0f; for (var j = 0; j <= t; j++) { var e = XMath.Exp(sc[baseI + j] - mx); a[baseI + j] = e; sum += e; } for (var j = 0; j <= t; j++) a[baseI + j] /= sum; for (var j = t + 1; j < T; j++) a[baseI + j] = 0f; });
        _context = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(
            (idx, a, v, ctx, T, d) => { int i = idx % d, t = (idx / d) % T, b = idx / (T * d); float acc = 0f; for (var j = 0; j <= t; j++) acc += a[(b * T + t) * T + j] * v[(b * T + j) * d + i]; ctx[idx] = acc; });
        _add = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>((idx, x, y, o) => o[idx] = x[idx] + y[idx]);
        _glu = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>>((idx, aval, gt, z) => z[idx] = aval[idx] * (1f / (1f + XMath.Exp(-gt[idx]))));
        _logits = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
            (idx, emb, cbias, h, outp, d, V, T) => { int w = idx % V, b = idx / V; var row = b * T + (T - 1); float a = cbias[w]; for (var i = 0; i < d; i++) a += emb[w * d + i] * h[row * d + i]; outp[idx] = a; });
    }

    /// <summary>Forward a batch of B sequences, each exactly T tokens (T ≤ maxContext). Returns logits[B][V] at the last position.</summary>
    public float[][] Forward(int[][] tokens)
    {
        int B = tokens.Length, T = tokens[0].Length, d = _d, N = B * T * d;
        var flat = new int[B * T];
        for (var b = 0; b < B; b++) for (var t = 0; t < T; t++) flat[b * T + t] = tokens[b][t];

        using var dtok = _acc.Allocate1D(flat);
        MemoryBuffer1D<float, Stride1D.Dense> Buf() => _acc.Allocate1D<float>(N);
        var h = Buf(); var q = Buf(); var k = Buf(); var v = Buf(); var ctx = Buf(); var o = Buf(); var m = Buf(); var aval = Buf(); var gt = Buf(); var z = Buf(); var y = Buf();
        using var sc = _acc.Allocate1D<float>(B * T * T);
        using var att = _acc.Allocate1D<float>(B * T * T);
        var inv = (float)(1.0 / Math.Sqrt(d));

        _embed(N, dtok.View, _emb.View, _pos.View, h.View, T, d); _acc.Synchronize();
        ArrayView<float> Bank(int l, int b) => _banks[l].View.SubView(b * _s * d, _s * d);
        for (var l = 0; l < _layers; l++)
        {
            _relbank(N, h.View, Bank(l, 0), q.View, d, _s);
            _relbank(N, h.View, Bank(l, 1), k.View, d, _s);
            _relbank(N, h.View, Bank(l, 2), v.View, d, _s);
            _scores(B * T * T, q.View, k.View, sc.View, T, d, inv);
            _softmax(B * T, sc.View, att.View, T);
            _context(N, att.View, v.View, ctx.View, T, d);
            _relbank(N, ctx.View, Bank(l, 3), o.View, d, _s);
            _add(N, h.View, o.View, m.View);
            _relbank(N, m.View, Bank(l, 4), aval.View, d, _s);
            _relbank(N, m.View, Bank(l, 5), gt.View, d, _s);
            _glu(N, aval.View, gt.View, z.View);
            _relbank(N, z.View, Bank(l, 6), o.View, d, _s);   // reuse o for fz
            _add(N, m.View, o.View, y.View);
            (h, y) = (y, h);   // y becomes next h; old h reused as scratch next round
        }
        _acc.Synchronize();

        using var dlog = _acc.Allocate1D<float>(B * _v);
        _logits(B * _v, _emb.View, _cbias.View, h.View, dlog.View, d, _v, T);   // reads the last position of each row-group: row = b*T + (T-1)
        _acc.Synchronize();
        var lg = dlog.GetAsArray1D();

        foreach (var buf in new[] { h, q, k, v, ctx, o, m, aval, gt, z, y }) buf.Dispose();
        var outp = new float[B][];
        for (var b = 0; b < B; b++) { outp[b] = new float[_v]; Array.Copy(lg, b * _v, outp[b], 0, _v); }
        return outp;
    }

    public void Dispose()
    {
        _emb.Dispose(); _pos.Dispose(); _cbias.Dispose();
        foreach (var b in _banks) b.Dispose();
    }
}
