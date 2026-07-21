// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

namespace PrismFormer;

/// <summary>
/// Builds the deterministic subword list for <see cref="SubwordVocab"/> from a corpus. Counts n-grams of length 2..4
/// (folded chars), keeps the top-K PER LENGTH by frequency, and returns them in a CANONICAL order (by length, then
/// ordinal) so the list is reproducible and identical on every node regardless of counting noise or file order. The
/// full permutation of 2-digit pairs (00..99) is always included; longer number tokens (e.g. "470", "2024") come in
/// via frequency. Numeric tokens get number faces (value-as-phase) from the codec, so 2..4-digit numbers become
/// atomic, arithmetic-capable tokens.
///
/// <para>Determinism is the load-bearing property: every node must derive the same list so the vocab layout (hence the
/// index-based exact-merge) matches. Generate once, commit the list, ship it; do not recompute per node.</para>
/// </summary>
public static class SubwordBuilder
{
    static char Fold(char c) => c >= CharVocab.Lo && c <= CharVocab.Hi ? c : ' ';

    /// <summary>Count n-gram frequencies (len 2..4) across <paramref name="segments"/> and return the top-K per length
    /// (plus all 2-digit pairs), canonically ordered. N-grams never span a segment boundary.</summary>
    public static List<string> FromSegments(IEnumerable<string> segments, int topBi, int topTri, int topQuad, bool includeAllDigit2 = true)
    {
        var budget = new Dictionary<int, int> { [2] = topBi, [3] = topTri, [4] = topQuad };
        var counts = new Dictionary<int, Dictionary<string, long>> { [2] = new(StringComparer.Ordinal), [3] = new(StringComparer.Ordinal), [4] = new(StringComparer.Ordinal) };

        foreach (var seg in segments)
        {
            var f = new char[seg.Length];
            for (var i = 0; i < seg.Length; i++) f[i] = Fold(seg[i]);
            for (var len = 2; len <= SubwordVocab.MaxLen; len++)
            {
                if (budget[len] <= 0) continue;
                var c = counts[len];
                for (var pos = 0; pos + len <= f.Length; pos++)
                {
                    var g = new string(f, pos, len);
                    c[g] = c.GetValueOrDefault(g) + 1;
                }
            }
        }

        var chosen = new HashSet<string>(StringComparer.Ordinal);
        for (var len = 2; len <= SubwordVocab.MaxLen; len++)
            foreach (var g in counts[len].OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key, StringComparer.Ordinal).Take(Math.Max(0, budget[len])).Select(kv => kv.Key))
                chosen.Add(g);

        if (includeAllDigit2)
            for (var a = '0'; a <= '9'; a++)
                for (var b = '0'; b <= '9'; b++)
                    chosen.Add($"{a}{b}");

        // canonical order → reproducible, and stable as counts shift between runs
        return chosen.OrderBy(w => w.Length).ThenBy(w => w, StringComparer.Ordinal).ToList();
    }
}
