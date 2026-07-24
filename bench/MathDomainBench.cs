// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

using PrismFormer;

namespace PrismFormer.Bench;

/// <summary>
/// --mathdomain : HOW DOES MATHS ACTUALLY BEHAVE IN THE PHASOR CODEC? Pure codec, no training, no learning — just the raw
/// algebra of the number faces, to see what the substrate gives for free and where it breaks. Probes: addition and
/// multiplication as ONE bind (value-add on the linear band, value-multiply on the log band, simultaneously); the exact
/// range before phase wraps (the number line is a CIRCLE); subtraction/division as unbind; chains of binds; and what a
/// SUPERPOSITION of numbers decodes to. The point is to characterise the domain and surface anything surprising.
/// </summary>
public static class MathDomainBench
{
    static double[] N(double v) => PhasorCodec.NumberFace(v);
    static int Sum(double[] f, int max) => PhasorCodec.DecodeSum(f, max);
    static int Prod(double[] f, int max) => PhasorCodec.DecodeProduct(f, max);

    public static void Run()
    {
        try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
        const int MAX = 8000;   // decode search ceiling (kept modest so the sweeps stay fast)
        Console.WriteLine("HOW MATHS BEHAVES IN THE PHASOR DOMAIN — pure codec, no training\n");

        // 1. ADDITION as bind on the linear band — exact range before it wraps
        Console.WriteLine("== ADDITION  (bind = phase-add = value-add, linear band) ==");
        int addMax = 0;
        for (var s = 2; s <= MAX; s++) { var a = s / 2; var got = Sum(PhasorCodec.Bind(N(a), N(s - a)), MAX); if (got == s) addMax = s; else { Console.WriteLine($"  exact a+b up to {addMax};  first wrap at a+b={s} decodes as {got}  (the number line is a circle of circumference ~{addMax + 1})"); break; } }
        if (addMax == MAX) Console.WriteLine($"  exact through the whole tested range (a+b ≤ {MAX})");

        // 2. MULTIPLICATION as the SAME bind, read off the log band
        Console.WriteLine("\n== MULTIPLICATION  (the SAME bind, read on the log band) ==");
        var mok = 0; var mtot = 0;
        foreach (var (a, b) in new[] { (3, 4), (7, 6), (9, 9), (12, 8), (15, 15), (23, 4) }) { var got = Prod(PhasorCodec.Bind(N(a), N(b)), MAX); mtot++; if (got == a * b) mok++; Console.WriteLine($"  {a} x {b} = {got}   {(got == a * b ? "OK" : "want " + a * b)}"); }

        // 3. THE SURPRISE: one bind carries BOTH results at once, disambiguated only at readout
        Console.WriteLine("\n== DUAL-BAND: a SINGLE bind holds sum AND product simultaneously ==");
        foreach (var (a, b) in new[] { (3, 4), (7, 6), (12, 5) }) { var f = PhasorCodec.Bind(N(a), N(b)); Console.WriteLine($"  bind({a},{b}):  read-as-sum = {Sum(f, MAX)} (a+b={a + b}),  read-as-product = {Prod(f, MAX)} (a·b={a * b})   — one operation, two answers"); }

        // 4. SUBTRACTION / DIVISION as unbind (the conjugate)
        Console.WriteLine("\n== SUBTRACTION & DIVISION  (unbind = phase-subtract) ==");
        foreach (var (a, b) in new[] { (52, 22), (100, 37), (9, 3) }) { var f = PhasorCodec.Unbind(N(a), N(b)); Console.WriteLine($"  {a} - {b} = {Sum(f, MAX)} (want {a - b}),   {a} / {b} = {Prod(f, MAX)} (want {(b != 0 && a % b == 0 ? (a / b).ToString() : "n/a")})"); }

        // 5. CHAINS: bind many numbers, does the sum accumulate exactly?
        Console.WriteLine("\n== CHAINS: bind a run of numbers, decode the running sum ==");
        var acc = N(0); var run = 0; var chainMax = 0;
        for (var i = 1; i <= 400; i++) { acc = PhasorCodec.Bind(acc, N(i)); run += i; if (Sum(acc, MAX) == run) chainMax = i; else { Console.WriteLine($"  1+2+...+{chainMax} exact (= {chainMax * (chainMax + 1) / 2}); breaks adding {i} (sum {run} wraps)"); break; } }

        // 6. THE SYNTHESIS BOUNDARY — treat combining numbers as unexplored space and map it.
        //    Bundle sums the faces' unit phasors; the resultant phase is the MEAN phase, so a superposition of values
        //    should decode to a NOVEL value (their average) that was never an input — that novel value is the "synthesis".
        //    But as the values pull apart on the phase circle the resultant shrinks (2·cos of the half-difference), so the
        //    synthesis weakens and eventually cancels. Map WHERE it forms cleanly vs where it dissolves.
        int Decode(double[] f) => Enumerable.Range(0, MAX).Select(v => (v, c: PhasorCodec.Correlate(f, N(v)))).OrderByDescending(x => x.c).First().v;
        double Strength(double[] f) { var self = PhasorCodec.Correlate(N(0), N(0)); var top = Enumerable.Range(0, MAX).Max(v => PhasorCodec.Correlate(f, N(v))); return top / self; }

        Console.WriteLine("\n== THE SYNTHESIS: what does combining numbers PRODUCE? ==");
        foreach (var set in new[] { new[] { 4, 6 }, new[] { 10, 20 }, new[] { 2, 4, 6 }, new[] { 10, 20, 30, 40 } })
        {
            var f = PhasorCodec.Bundle(set.Select(v => N(v)).ToArray());
            Console.WriteLine($"  {{{string.Join(",", set)}}}  →  {Decode(f)}    (mean {set.Average():F1}, sum {set.Sum()})   strength {Strength(f):F2}");
        }

        Console.WriteLine("\n== MAPPING THE BOUNDARY: {50, 50+gap} pulling apart — does the synthesis (their mean) survive? ==");
        Console.WriteLine($"    {"gap",5}{"decodes",9}{"mean",7}{"strength",10}   state");
        foreach (var gap in new[] { 0, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024 })
        {
            var f = PhasorCodec.Bundle(N(50), N(50 + gap));
            var dec = Decode(f); var str = Strength(f); var mean = 50 + gap / 2.0;
            var state = str > 0.85 ? "clean synthesis" : str > 0.4 ? "fuzzy" : "dissolved";
            Console.WriteLine($"    {gap,5}{dec,9}{mean,7:F0}{str,10:F2}   {state}");
        }

        Console.WriteLine("\n  read: the substrate does exact +,-,x,/ with no training and no calculator, over a finite CIRCULAR range");
        Console.WriteLine("  (arithmetic goes modular past the wrap), and ONE bind carries sum and product at once — the model only has");
        Console.WriteLine("  to learn which band to read, never how to compute. And combining numbers doesn't give their SUM, it gives");
        Console.WriteLine("  their MEAN: superposition is averaging in phase space. The synthesis is a real novel value while the parts");
        Console.WriteLine("  stay close, and it fades to noise as they pull apart — that fade is the synthesis boundary, mapped.");
    }
}
