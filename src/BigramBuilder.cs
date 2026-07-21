// Copyright (c) 2026 Dongyang Stephen Chen, trading as Evaluated Applications. All rights reserved.
// Source-available for study and verification; commercial use, redistribution, or derivatives require permission. See LICENSE.

namespace PrismFormer;

/// <summary>
/// Builds the deterministic bigram list for <see cref="BigramVocab"/> from a corpus. Counts adjacent (folded) char
/// pairs, keeps the top-K by frequency, and returns them in a CANONICAL order (sorted by the packed pair, NOT by
/// count) so the list is reproducible and identical on every node regardless of counting noise or file order. The
/// full permutation of DIGIT pairs (00..99) is always included: they are cheap (100 slots) and each gets a number
/// face, so two-digit numbers become atomic, arithmetic-capable tokens.
///
/// <para>Determinism is the load-bearing property: every node must derive the same list so the vocab layout (hence
/// the index-based exact-merge) matches. Generate once, commit the list, ship it; do not recompute per node from a
/// possibly-different corpus.</para>
/// </summary>
public static class BigramBuilder
{
    static char Fold(char c) => c >= CharVocab.Lo && c <= CharVocab.Hi ? c : ' ';

    /// <summary>Count char-bigram frequencies across <paramref name="segments"/> and return the top-<paramref name="topK"/>
    /// (plus all digit pairs), canonically ordered. Pairs never span a segment boundary.</summary>
    public static List<string> FromSegments(IEnumerable<string> segments, int topK, bool includeAllDigitPairs = true)
    {
        var count = new Dictionary<int, long>();
        foreach (var seg in segments)
        {
            var have = false; char prev = ' ';
            foreach (var raw in seg)
            {
                var c = Fold(raw);
                if (have) { var k = (prev << 8) | c; count[k] = count.GetValueOrDefault(k) + 1; }
                prev = c; have = true;
            }
        }

        // top-K by count, tie-broken by pair for stability
        var chosen = count.OrderByDescending(kv => kv.Value).ThenBy(kv => kv.Key)
                          .Take(Math.Max(0, topK)).Select(kv => kv.Key).ToHashSet();
        if (includeAllDigitPairs)
            for (var a = '0'; a <= '9'; a++)
                for (var b = '0'; b <= '9'; b++)
                    chosen.Add((a << 8) | b);

        // canonical order → reproducible, and stable as counts shift between runs
        return chosen.OrderBy(k => k)
                     .Select(k => new string(new[] { (char)(k >> 8), (char)(k & 0xFF) }))
                     .ToList();
    }
}
