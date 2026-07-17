// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using ILGPU;
using ILGPU.Runtime;
using ILGPU.Algorithms;

namespace PrismFormer.Gpu;

/// <summary>
/// Batched GPU BACKWARD for AlgFormer (fp32), mirroring the reverse of ForwardAll. Produces gradients for Emb, Pos, C
/// and every relation bank, in Serialize/Pairs order, so it can be gradchecked against AlgFormer.SerializeGradient.
/// The embedding is tied (used in both the input and the readout), so Emb grad accumulates from both. (Non-bind GLU.)
/// </summary>
public sealed partial class GpuModel
{
    Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int> _gbank = null!;
    Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> _dx = null!;
    Action<Index1D, ArrayView<float>, ArrayView<float>> _addInto = null!, _copy = null!;
    Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>> _gluBack = null!;
    Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int> _daAtt = null!, _dvAtt = null!;
    Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int> _softmaxBack = null!;
    Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, float> _dqAtt = null!, _dkAtt = null!;
    Action<Index1D, ArrayView<float>, ArrayView<float>, int, int> _logitsDC = null!;
    Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int> _logitsDEmb = null!;
    Action<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int> _logitsDh = null!;
    Action<Index1D, ArrayView<float>, ArrayView<float>, int, int, int> _embDPos = null!;
    Action<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, int, int> _embScatter = null!;

    void InitBackward()
    {
        _gbank = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int>(
            (idx, dOut, x, g, BT, d, s) => { int i = idx % d, k = idx / d; float a = 0f; for (var row = 0; row < BT; row++) { var xi = i + k; if (xi >= d) xi -= d; a += dOut[row * d + i] * x[row * d + xi]; } g[idx] = a; });
        _dx = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(
            (idx, dOut, bank, dXo, d, s) => { int p = idx % d, row = idx / d; float a = 0f; for (var k = 0; k < s; k++) { var q = p - k; if (q < 0) q += d; a += dOut[row * d + q] * bank[k * d + q]; } dXo[idx] = a; });
        _addInto = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>((idx, src, acc) => acc[idx] += src[idx]);
        _copy = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>>((idx, src, dst) => dst[idx] = src[idx]);
        _gluBack = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>, ArrayView<float>>(
            (idx, dz, aval, gt, daval, dgt) => { var sig = 1f / (1f + XMath.Exp(-gt[idx])); daval[idx] = dz[idx] * sig; dgt[idx] = dz[idx] * aval[idx] * sig * (1f - sig); });
        _daAtt = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(
            (idx, dctx, v, da, T, d) => { int j = idx % T, t = (idx / T) % T, b = idx / (T * T); if (j > t) { da[idx] = 0f; return; } float a = 0f; for (var i = 0; i < d; i++) a += dctx[(b * T + t) * d + i] * v[(b * T + j) * d + i]; da[idx] = a; });
        _dvAtt = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int>(
            (idx, a, dctx, dv, T, d) => { int i = idx % d, j = (idx / d) % T, b = idx / (T * d); float acc = 0f; for (var t = j; t < T; t++) acc += a[(b * T + t) * T + j] * dctx[(b * T + t) * d + i]; dv[idx] = acc; });
        _softmaxBack = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int>(
            (idx, a, da, ds, T) => { int t = idx % T, b = idx / T; var bs = (b * T + t) * T; float sum = 0f; for (var j = 0; j <= t; j++) sum += a[bs + j] * da[bs + j]; for (var j = 0; j <= t; j++) ds[bs + j] = a[bs + j] * (da[bs + j] - sum); for (var j = t + 1; j < T; j++) ds[bs + j] = 0f; });
        _dqAtt = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, float>(
            (idx, ds, k, dq, T, d, inv) => { int i = idx % d, t = (idx / d) % T, b = idx / (T * d); float acc = 0f; for (var j = 0; j <= t; j++) acc += ds[(b * T + t) * T + j] * inv * k[(b * T + j) * d + i]; dq[idx] = acc; });
        _dkAtt = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, float>(
            (idx, ds, q, dk, T, d, inv) => { int i = idx % d, j = (idx / d) % T, b = idx / (T * d); float acc = 0f; for (var t = j; t < T; t++) acc += ds[(b * T + t) * T + j] * inv * q[(b * T + t) * d + i]; dk[idx] = acc; });
        _logitsDC = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, int, int>(
            (idx, dlog, gC, B, V) => { float a = 0f; for (var b = 0; b < B; b++) a += dlog[b * V + idx]; gC[idx] = a; });
        _logitsDEmb = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int>(
            (idx, dlog, h, gEmb, B, V, d, T) => { int i = idx % d, w = idx / d; float a = 0f; for (var b = 0; b < B; b++) a += dlog[b * V + w] * h[(b * T + (T - 1)) * d + i]; gEmb[idx] = a; });
        _logitsDh = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, ArrayView<float>, int, int, int, int>(
            (idx, dlog, emb, dh, B, V, d, T) => { int i = idx % d, b = idx / d; float a = 0f; for (var w = 0; w < V; w++) a += dlog[b * V + w] * emb[w * d + i]; dh[(b * T + (T - 1)) * d + i] = a; });
        _embDPos = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<float>, int, int, int>(
            (idx, dh, gPos, B, T, d) => { int i = idx % d, t = idx / d; float a = 0f; for (var b = 0; b < B; b++) a += dh[(b * T + t) * d + i]; gPos[idx] = a; });
        _embScatter = _acc.LoadAutoGroupedStreamKernel<Index1D, ArrayView<float>, ArrayView<int>, ArrayView<float>, int, int>(
            (idx, dh, toks, gEmb, T, d) => { int i = idx % d, bt = idx / d; Atomic.Add(ref gEmb[toks[bt] * d + i], dh[bt * d + i]); });
    }

    /// <summary>Forward+backward a batch (B seqs × T tokens, targets[B]). Returns gradients flat in Pairs order
    /// (Emb, Pos, C, then per layer Rq,Rk,Rv,Ro,A1,Ag,Ao) — same layout as AlgFormer.SerializeGradient.</summary>
    public float[] Backward(int[][] tokens, int[] targets)
    {
        int B = tokens.Length, T = tokens[0].Length, d = _d, N = B * T * d, A = B * T * T;
        var flat = new int[B * T];
        for (var b = 0; b < B; b++) for (var t = 0; t < T; t++) flat[b * T + t] = tokens[b][t];
        using var dtok = _acc.Allocate1D(flat);
        var bufs = new List<MemoryBuffer1D<float, Stride1D.Dense>>();
        MemoryBuffer1D<float, Stride1D.Dense> Buf(int n) { var b = _acc.Allocate1D<float>(n); bufs.Add(b); return b; }
        ArrayView<float> Bank(int l, int b) => _banks[l].View.SubView(b * _s * d, _s * d);
        var inv = (float)(1.0 / Math.Sqrt(d));

        // ── cached forward ──
        var Xs = new MemoryBuffer1D<float, Stride1D.Dense>[_layers + 1];
        var qs = new MemoryBuffer1D<float, Stride1D.Dense>[_layers]; var ks = new MemoryBuffer1D<float, Stride1D.Dense>[_layers]; var vs = new MemoryBuffer1D<float, Stride1D.Dense>[_layers];
        var as_ = new MemoryBuffer1D<float, Stride1D.Dense>[_layers]; var ctxs = new MemoryBuffer1D<float, Stride1D.Dense>[_layers]; var ms = new MemoryBuffer1D<float, Stride1D.Dense>[_layers];
        var avals = new MemoryBuffer1D<float, Stride1D.Dense>[_layers]; var gts = new MemoryBuffer1D<float, Stride1D.Dense>[_layers]; var zs = new MemoryBuffer1D<float, Stride1D.Dense>[_layers];
        Xs[0] = Buf(N); _embed(N, dtok.View, _emb.View, _pos.View, Xs[0].View, T, d);
        for (var l = 0; l < _layers; l++)
        {
            var X = Xs[l];
            qs[l] = Buf(N); ks[l] = Buf(N); vs[l] = Buf(N);
            _relbank(N, X.View, Bank(l, 0), qs[l].View, d, _s); _relbank(N, X.View, Bank(l, 1), ks[l].View, d, _s); _relbank(N, X.View, Bank(l, 2), vs[l].View, d, _s);
            var sc = Buf(A); as_[l] = Buf(A);
            _scores(A, qs[l].View, ks[l].View, sc.View, T, d, inv); _softmax(B * T, sc.View, as_[l].View, T);
            ctxs[l] = Buf(N); _context(N, as_[l].View, vs[l].View, ctxs[l].View, T, d);
            var o = Buf(N); _relbank(N, ctxs[l].View, Bank(l, 3), o.View, d, _s);
            ms[l] = Buf(N); _add(N, X.View, o.View, ms[l].View);
            avals[l] = Buf(N); gts[l] = Buf(N); _relbank(N, ms[l].View, Bank(l, 4), avals[l].View, d, _s); _relbank(N, ms[l].View, Bank(l, 5), gts[l].View, d, _s);
            zs[l] = Buf(N); _glu(N, avals[l].View, gts[l].View, zs[l].View);
            var fz = Buf(N); _relbank(N, zs[l].View, Bank(l, 6), fz.View, d, _s);
            Xs[l + 1] = Buf(N); _add(N, ms[l].View, fz.View, Xs[l + 1].View);
        }
        _acc.Synchronize();

        // ── grad buffers (Pairs order) ──
        using var gEmb = _acc.Allocate1D<float>(_v * d); using var gPos = _acc.Allocate1D<float>(_maxT * d); using var gC = _acc.Allocate1D<float>(_v);
        var gBanks = new MemoryBuffer1D<float, Stride1D.Dense>[_layers];
        for (var l = 0; l < _layers; l++) gBanks[l] = _acc.Allocate1D<float>(7 * _s * d);
        gPos.MemSetToZero();

        // ── logits backward ──
        using var dlogB = _acc.Allocate1D<float>(B * _v);
        _logits(B * _v, _emb.View, _cbias.View, Xs[_layers].View, dlogB.View, d, _v, T);
        _acc.Synchronize();
        var lg = dlogB.GetAsArray1D();
        var dlog = new float[B * _v];
        for (var b = 0; b < B; b++)
        {
            var mx = float.NegativeInfinity; for (var w = 0; w < _v; w++) if (lg[b * _v + w] > mx) mx = lg[b * _v + w];
            double sum = 0; for (var w = 0; w < _v; w++) sum += Math.Exp(lg[b * _v + w] - mx);
            for (var w = 0; w < _v; w++) { var p = (float)(Math.Exp(lg[b * _v + w] - mx) / sum); dlog[b * _v + w] = p - (w == targets[b] ? 1f : 0f); }
        }
        using var ddlog = _acc.Allocate1D(dlog);
        _logitsDC(_v, ddlog.View, gC.View, B, _v);
        _logitsDEmb(_v * d, ddlog.View, Xs[_layers].View, gEmb.View, B, _v, d, T);
        var dh = Buf(N); dh.MemSetToZero();
        _logitsDh(B * d, ddlog.View, _emb.View, dh.View, B, _v, d, T);   // writes dh at the last position of each row-group; others stay 0
        _acc.Synchronize();

        // ── per-layer backward ──
        var dm = Buf(N); var dX = Buf(N); var dz = Buf(N); var daval = Buf(N); var dgt = Buf(N); var dctx = Buf(N);
        var dq = Buf(N); var dk = Buf(N); var dv = Buf(N); var da = Buf(A); var ds = Buf(A); var tmp = Buf(N);
        for (var l = _layers - 1; l >= 0; l--)
        {
            var gb = gBanks[l];
            void GBank(int bank, ArrayView<float> dOut, ArrayView<float> x) => _gbank(_s * d, dOut, x, gb.View.SubView(bank * _s * d, _s * d), B * T, d, _s);
            // y = m + fz : dm = dh, dfz = dh
            _copy(N, dh.View, dm.View);
            // fz = relbank(Ao, z)
            GBank(6, dh.View, zs[l].View); _dx(N, dh.View, Bank(l, 6), dz.View, d, _s);   // dz = grad into z
            // z = aval*sig(gt)
            _gluBack(N, dz.View, avals[l].View, gts[l].View, daval.View, dgt.View);
            // aval=relbank(A1,m); gt=relbank(Ag,m) : dm += dx(daval,A1)+dx(dgt,Ag)
            GBank(4, daval.View, ms[l].View); _dx(N, daval.View, Bank(l, 4), tmp.View, d, _s); _addInto(N, tmp.View, dm.View);
            GBank(5, dgt.View, ms[l].View); _dx(N, dgt.View, Bank(l, 5), tmp.View, d, _s); _addInto(N, tmp.View, dm.View);
            // m = X + o : dX = dm (start), do = dm
            _copy(N, dm.View, dX.View);
            // o = relbank(Ro, ctx)
            GBank(3, dm.View, ctxs[l].View); _dx(N, dm.View, Bank(l, 3), dctx.View, d, _s);
            // attention
            _dvAtt(N, as_[l].View, dctx.View, dv.View, T, d);
            _daAtt(A, dctx.View, vs[l].View, da.View, T, d);
            _softmaxBack(B * T, as_[l].View, da.View, ds.View, T);
            _dqAtt(N, ds.View, ks[l].View, dq.View, T, d, inv);
            _dkAtt(N, ds.View, qs[l].View, dk.View, T, d, inv);
            // q/k/v = relbank(R*, X) : dX += dx(dq,Rq)+dx(dk,Rk)+dx(dv,Rv)
            GBank(0, dq.View, Xs[l].View); _dx(N, dq.View, Bank(l, 0), tmp.View, d, _s); _addInto(N, tmp.View, dX.View);
            GBank(1, dk.View, Xs[l].View); _dx(N, dk.View, Bank(l, 1), tmp.View, d, _s); _addInto(N, tmp.View, dX.View);
            GBank(2, dv.View, Xs[l].View); _dx(N, dv.View, Bank(l, 2), tmp.View, d, _s); _addInto(N, tmp.View, dX.View);
            _copy(N, dX.View, dh.View);   // dh for the next-lower layer (grad of this layer's input)
        }
        _acc.Synchronize();

        // ── embedding backward (dh = grad of embedding output) ──
        _embDPos(T * d, dh.View, gPos.View, B, T, d);
        _embScatter(N, dh.View, dtok.View, gEmb.View, T, d);
        _acc.Synchronize();

        // ── assemble flat grads in Pairs order ──
        var outp = new List<float>(_v * d + _maxT * d + _v + _layers * 7 * _s * d);
        outp.AddRange(gEmb.GetAsArray1D()); outp.AddRange(gPos.GetAsArray1D()); outp.AddRange(gC.GetAsArray1D());
        for (var l = 0; l < _layers; l++) { outp.AddRange(gBanks[l].GetAsArray1D()); gBanks[l].Dispose(); }
        foreach (var b in bufs) b.Dispose();
        return outp.ToArray();
    }
}
