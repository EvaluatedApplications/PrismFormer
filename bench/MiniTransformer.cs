// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

namespace PrismFormer.Bench;

/// <summary>
/// A REAL miniature transformer with fully manual gradients — the honest Phase-2 baseline. Encoder-style (full
/// attention), L layers, single head, residuals, tanh FFN, learned token+position embeddings, softmax-CE readout
/// at the last position, Adam. <see cref="GradCheck"/> verifies the backward pass against central finite
/// differences at startup, so a training failure is a capability result, never a gradient bug.
/// </summary>
public sealed class MiniTransformer
{
    private readonly int _v, _d, _f, _layers, _maxT;
    private const double Scale = 0.08;

    internal readonly double[][] Emb, Pos, U;
    internal readonly double[] C;
    internal readonly LayerParams[] Ls;

    private readonly double[][] _gEmb, _gPos, _gU;
    private readonly double[] _gC;
    private readonly LayerParams[] _gLs;
    private readonly List<(double[] p, double[] g, double[] m, double[] v)> _adam = new();
    private long _t;

    internal sealed class LayerParams
    {
        public required double[][] Wq, Wk, Wv, Wo, W1, W2;
        public required double[] B1, B2;
    }

    public MiniTransformer(int vocab, int dModel, int dff, int layers, int maxT, int seed = 42)
    {
        _v = vocab; _d = dModel; _f = dff; _layers = layers; _maxT = maxT;
        var rng = new Random(seed);
        double[] Row(int n, bool zero = false)
        {
            var r = new double[n];
            if (!zero) for (var i = 0; i < n; i++) r[i] = (rng.NextDouble() - 0.5) * 2 * Scale;
            return r;
        }
        double[][] Mat(int rows, int cols, bool zero = false) => Enumerable.Range(0, rows).Select(_ => Row(cols, zero)).ToArray();

        Emb = Mat(_v, _d); Pos = Mat(_maxT, _d); U = Mat(_v, _d); C = Row(_v, true);
        _gEmb = Mat(_v, _d, true); _gPos = Mat(_maxT, _d, true); _gU = Mat(_v, _d, true); _gC = Row(_v, true);
        Ls = new LayerParams[layers]; _gLs = new LayerParams[layers];
        for (var l = 0; l < layers; l++)
        {
            Ls[l] = new LayerParams { Wq = Mat(_d, _d), Wk = Mat(_d, _d), Wv = Mat(_d, _d), Wo = Mat(_d, _d), W1 = Mat(_f, _d), B1 = Row(_f, true), W2 = Mat(_d, _f), B2 = Row(_d, true) };
            _gLs[l] = new LayerParams { Wq = Mat(_d, _d, true), Wk = Mat(_d, _d, true), Wv = Mat(_d, _d, true), Wo = Mat(_d, _d, true), W1 = Mat(_f, _d, true), B1 = Row(_f, true), W2 = Mat(_d, _f, true), B2 = Row(_d, true) };
        }
        void Reg(double[][] p, double[][] g) { for (var i = 0; i < p.Length; i++) _adam.Add((p[i], g[i], new double[p[i].Length], new double[p[i].Length])); }
        Reg(Emb, _gEmb); Reg(Pos, _gPos); Reg(U, _gU);
        _adam.Add((C, _gC, new double[_v], new double[_v]));
        for (var l = 0; l < layers; l++)
        {
            Reg(Ls[l].Wq, _gLs[l].Wq); Reg(Ls[l].Wk, _gLs[l].Wk); Reg(Ls[l].Wv, _gLs[l].Wv); Reg(Ls[l].Wo, _gLs[l].Wo);
            Reg(Ls[l].W1, _gLs[l].W1); Reg(Ls[l].W2, _gLs[l].W2);
            _adam.Add((Ls[l].B1, _gLs[l].B1, new double[_f], new double[_f]));
            _adam.Add((Ls[l].B2, _gLs[l].B2, new double[_d], new double[_d]));
        }
    }

    public long ParamCount => _adam.Sum(x => (long)x.p.Length);

    private sealed class Cache
    {
        public required double[][] X, Q, K, V, Ctx, M, A, Z;
    }

    private (double[][] h, Cache[] caches) ForwardAll(int[] toks)
    {
        var T = toks.Length;
        var h = new double[T][];
        for (var t = 0; t < T; t++)
        {
            h[t] = new double[_d];
            for (var d = 0; d < _d; d++) h[t][d] = Emb[toks[t]][d] + Pos[t][d];
        }
        var caches = new Cache[_layers];
        for (var l = 0; l < _layers; l++)
        {
            var L = Ls[l];
            var X = h;
            var q = Apply(L.Wq, X); var k = Apply(L.Wk, X); var v = Apply(L.Wv, X);
            var a = new double[T][];
            var ctx = new double[T][];
            var inv = 1.0 / Math.Sqrt(_d);
            for (var t = 0; t < T; t++)
            {
                var s = new double[T];
                for (var j = 0; j < T; j++) { var acc = 0.0; for (var d = 0; d < _d; d++) acc += q[t][d] * k[j][d]; s[j] = acc * inv; }
                var max = s.Max(); var sum = 0.0;
                a[t] = new double[T];
                for (var j = 0; j < T; j++) { a[t][j] = Math.Exp(s[j] - max); sum += a[t][j]; }
                for (var j = 0; j < T; j++) a[t][j] /= sum;
                ctx[t] = new double[_d];
                for (var j = 0; j < T; j++) for (var d = 0; d < _d; d++) ctx[t][d] += a[t][j] * v[j][d];
            }
            var o = Apply(L.Wo, ctx);
            var m = new double[T][];
            for (var t = 0; t < T; t++) { m[t] = new double[_d]; for (var d = 0; d < _d; d++) m[t][d] = X[t][d] + o[t][d]; }
            var z = new double[T][];
            var y = new double[T][];
            for (var t = 0; t < T; t++)
            {
                z[t] = new double[_f];
                for (var f = 0; f < _f; f++) { var acc = L.B1[f]; for (var d = 0; d < _d; d++) acc += L.W1[f][d] * m[t][d]; z[t][f] = Math.Tanh(acc); }
                y[t] = new double[_d];
                for (var d = 0; d < _d; d++) { var acc = L.B2[d]; for (var f = 0; f < _f; f++) acc += L.W2[d][f] * z[t][f]; y[t][d] = m[t][d] + acc; }
            }
            caches[l] = new Cache { X = X, Q = q, K = k, V = v, A = a, Ctx = ctx, M = m, Z = z };
            h = y;
        }
        return (h, caches);
    }

    private double[] Logits(double[] hLast)
    {
        var logit = new double[_v];
        for (var v = 0; v < _v; v++) { var acc = C[v]; for (var d = 0; d < _d; d++) acc += U[v][d] * hLast[d]; logit[v] = acc; }
        return logit;
    }

    public int Predict(int[] toks)
    {
        var (h, _) = ForwardAll(toks);
        var logit = Logits(h[^1]);
        var best = 0;
        for (var v = 1; v < _v; v++) if (logit[v] > logit[best]) best = v;
        return best;
    }

    /// <summary>Next-token logits at the final position (for probability/perplexity evaluation).</summary>
    public double[] LogitsFor(int[] toks) { var (h, _) = ForwardAll(toks); return Logits(h[^1]); }

    /// <summary>Interpretability contrast: the opaque hidden vector at the last position — what a linear probe
    /// must be TRAINED to read. Unlike PrismFormer's phasor faces, nothing here is directly decodable.</summary>
    public double[] LastHidden(int[] toks) { var (h, _) = ForwardAll(toks); return h[^1]; }

    public double LossOnly(int[] toks, int answer)
    {
        var (h, _) = ForwardAll(toks);
        var logit = Logits(h[^1]);
        var max = logit.Max(); var sum = 0.0;
        for (var v = 0; v < _v; v++) sum += Math.Exp(logit[v] - max);
        return -(logit[answer] - max - Math.Log(sum));
    }

    internal double AccumulateGrads(int[] toks, int answer)
    {
        var T = toks.Length;
        var (h, caches) = ForwardAll(toks);

        var logit = Logits(h[^1]);
        var max = logit.Max(); var sum = 0.0;
        var p = new double[_v];
        for (var v = 0; v < _v; v++) { p[v] = Math.Exp(logit[v] - max); sum += p[v]; }
        for (var v = 0; v < _v; v++) p[v] /= sum;
        var loss = -Math.Log(Math.Max(p[answer], 1e-300));

        var dh = new double[T][];
        for (var t = 0; t < T; t++) dh[t] = new double[_d];
        for (var v = 0; v < _v; v++)
        {
            var dz = p[v] - (v == answer ? 1.0 : 0.0);
            _gC[v] += dz;
            for (var d = 0; d < _d; d++) { _gU[v][d] += dz * h[^1][d]; dh[^1][d] += dz * U[v][d]; }
        }

        for (var l = _layers - 1; l >= 0; l--)
        {
            var L = Ls[l]; var G = _gLs[l]; var cc = caches[l];
            var dY = dh;

            var dM = new double[T][];
            for (var t = 0; t < T; t++) dM[t] = (double[])dY[t].Clone();
            for (var t = 0; t < T; t++)
            {
                var dZ = new double[_f];
                for (var d = 0; d < _d; d++)
                {
                    var dyd = dY[t][d];
                    if (dyd == 0) continue;
                    G.B2[d] += dyd;
                    for (var f = 0; f < _f; f++) { G.W2[d][f] += dyd * cc.Z[t][f]; dZ[f] += L.W2[d][f] * dyd; }
                }
                for (var f = 0; f < _f; f++)
                {
                    var dP = dZ[f] * (1 - cc.Z[t][f] * cc.Z[t][f]);
                    if (dP == 0) continue;
                    G.B1[f] += dP;
                    for (var d = 0; d < _d; d++) { G.W1[f][d] += dP * cc.M[t][d]; dM[t][d] += L.W1[f][d] * dP; }
                }
            }

            var dX = new double[T][];
            for (var t = 0; t < T; t++) dX[t] = (double[])dM[t].Clone();
            var dCtx = new double[T][];
            for (var t = 0; t < T; t++)
            {
                dCtx[t] = new double[_d];
                for (var r = 0; r < _d; r++)
                {
                    var dor = dM[t][r];
                    if (dor == 0) continue;
                    for (var c = 0; c < _d; c++) { G.Wo[r][c] += dor * cc.Ctx[t][c]; dCtx[t][c] += L.Wo[r][c] * dor; }
                }
            }

            var dV = new double[T][]; var dQ = new double[T][]; var dK = new double[T][];
            for (var t = 0; t < T; t++) { dV[t] = new double[_d]; dQ[t] = new double[_d]; dK[t] = new double[_d]; }
            var inv = 1.0 / Math.Sqrt(_d);
            for (var t = 0; t < T; t++)
            {
                var dA = new double[T];
                for (var j = 0; j < T; j++)
                {
                    var acc = 0.0;
                    for (var d = 0; d < _d; d++) { dV[j][d] += cc.A[t][j] * dCtx[t][d]; acc += dCtx[t][d] * cc.V[j][d]; }
                    dA[j] = acc;
                }
                var dot = 0.0;
                for (var j = 0; j < T; j++) dot += cc.A[t][j] * dA[j];
                for (var j = 0; j < T; j++)
                {
                    var dS = cc.A[t][j] * (dA[j] - dot) * inv;
                    if (dS == 0) continue;
                    for (var d = 0; d < _d; d++) { dQ[t][d] += dS * cc.K[j][d]; dK[j][d] += dS * cc.Q[t][d]; }
                }
            }

            BackProject(Ls[l].Wq, G.Wq, cc.X, dQ, dX);
            BackProject(Ls[l].Wk, G.Wk, cc.X, dK, dX);
            BackProject(Ls[l].Wv, G.Wv, cc.X, dV, dX);
            dh = dX;
        }

        for (var t = 0; t < T; t++)
            for (var d = 0; d < _d; d++)
            {
                _gEmb[toks[t]][d] += dh[t][d];
                _gPos[t][d] += dh[t][d];
            }
        return loss;
    }

    public void Step(double lr = 1e-3, double b1 = 0.9, double b2 = 0.999, double eps = 1e-8)
    {
        _t++;
        var c1 = 1 - Math.Pow(b1, _t);
        var c2 = 1 - Math.Pow(b2, _t);
        foreach (var (p, g, m, v) in _adam)
            for (var i = 0; i < p.Length; i++)
            {
                var gi = g[i];
                if (gi == 0) continue;
                m[i] = b1 * m[i] + (1 - b1) * gi;
                v[i] = b2 * v[i] + (1 - b2) * gi * gi;
                p[i] -= lr * (m[i] / c1) / (Math.Sqrt(v[i] / c2) + eps);
                g[i] = 0;
            }
    }

    public double TrainStep(int[] toks, int answer, double lr = 1e-3)
    {
        var loss = AccumulateGrads(toks, answer);
        Step(lr);
        return loss;
    }

    private double[][] Apply(double[][] w, double[][] x)
    {
        var T = x.Length;
        var y = new double[T][];
        for (var t = 0; t < T; t++)
        {
            y[t] = new double[_d];
            for (var r = 0; r < _d; r++)
            {
                var acc = 0.0; var row = w[r];
                for (var c = 0; c < _d; c++) acc += row[c] * x[t][c];
                y[t][r] = acc;
            }
        }
        return y;
    }

    private void BackProject(double[][] w, double[][] gw, double[][] x, double[][] dOut, double[][] dX)
    {
        var T = x.Length;
        for (var t = 0; t < T; t++)
            for (var r = 0; r < _d; r++)
            {
                var d = dOut[t][r];
                if (d == 0) continue;
                var row = w[r]; var grow = gw[r];
                for (var c = 0; c < _d; c++) { grow[c] += d * x[t][c]; dX[t][c] += row[c] * d; }
            }
    }

    /// <summary>Startup self-verification: analytic vs central-finite-difference gradients on a tiny config.</summary>
    public static bool GradCheck(out double worstRel)
    {
        var xf = new MiniTransformer(vocab: 9, dModel: 4, dff: 5, layers: 2, maxT: 4, seed: 6);
        int[] toks = { 3, 1, 7, 2 };
        const int ans = 5; const double eps = 1e-5;
        xf.AccumulateGrads(toks, ans);

        worstRel = 0;
        var probes = new (double[] p, double[] g, string name)[]
        {
            (xf.Emb[3], xf._gEmb[3], "Emb"), (xf.Pos[0], xf._gPos[0], "Pos"), (xf.U[5], xf._gU[5], "U"), (xf.C, xf._gC, "C"),
            (xf.Ls[0].Wq[1], xf._gLs[0].Wq[1], "Wq0"), (xf.Ls[1].Wk[0], xf._gLs[1].Wk[0], "Wk1"),
            (xf.Ls[0].Wv[2], xf._gLs[0].Wv[2], "Wv0"), (xf.Ls[1].Wo[3], xf._gLs[1].Wo[3], "Wo1"),
            (xf.Ls[0].W1[2], xf._gLs[0].W1[2], "W1"), (xf.Ls[1].W2[1], xf._gLs[1].W2[1], "W2"),
            (xf.Ls[0].B1, xf._gLs[0].B1, "B1"), (xf.Ls[1].B2, xf._gLs[1].B2, "B2"),
        };
        foreach (var (p, g, _) in probes)
            for (var i = 0; i < Math.Min(2, p.Length); i++)
            {
                var keep = p[i];
                p[i] = keep + eps; var up = xf.LossOnly(toks, ans);
                p[i] = keep - eps; var dn = xf.LossOnly(toks, ans);
                p[i] = keep;
                var numeric = (up - dn) / (2 * eps);
                var rel = Math.Abs(numeric - g[i]) / Math.Max(1e-6, Math.Abs(numeric) + Math.Abs(g[i]));
                if (Math.Abs(numeric - g[i]) > 1e-7) worstRel = Math.Max(worstRel, rel);
            }
        return worstRel < 1e-4;
    }
}
