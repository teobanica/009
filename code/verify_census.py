#!/usr/bin/env python3
"""Independently check the return-count fingerprints of the 353 bundled tables.

Usage: python3 verify_census.py /path/to/census-tables.json.gz

For a normal basis element of a ring of rank r, the scalar law has at most r
atoms. Two such measures with matching holomorphic moments in degrees 0..2r-1
are equal: on their combined support of at most 2r points, use Vandermonde
interpolation. No eigenvalue computation or polynomial factorization is needed.
"""
import argparse
import gzip
import json
from collections import Counter, defaultdict


def return_counts(table, i, max_degree):
    """Return the unit coefficients of b_i^n in degrees 0..max_degree."""
    rank = len(table)
    vector = [1] + [0] * (rank - 1)
    result = [1]
    for _ in range(max_degree):
        vector = [sum(a * row[k] for a, row in zip(vector, table[i]))
                  for k in range(rank)]
        result.append(vector[0])
    return tuple(result)


def fingerprints(record):
    table = record["table"]
    rank = record["rank"]
    returns = []
    basic_laws = []
    for i in range(rank):
        dual, = (j for j in range(rank) if table[i][j][0])
        counts = return_counts(table, i, max(5, 2 * rank - 1))
        returns.append(counts)
        # The empty tuple is the manuscript's marker for a nonnormal element.
        normal = table[i][dual] == table[dual][i]
        basic_laws.append(counts[:2 * rank] if normal else ())
    raw4 = tuple(sorted(v[2:5] for v in returns))
    raw5 = tuple(sorted(v[2:6] for v in returns))
    laws = tuple(sorted(basic_laws))
    cubic = sum(value for plane in table for row in plane for value in row)
    return raw4, raw5, laws, cubic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tables", help="bundled census-tables.json.gz")
    args = parser.parse_args()
    with gzip.open(args.tables, "rt") as stream:
        records = json.load(stream)

    print("rank  rings  raw<=4  raw<=5  laws  laws+cubic")
    rows = []
    by_rank9_law = defaultdict(list)
    by_rank9_raw4 = defaultdict(list)
    for rank in range(1, 10):
        layer = [rec for rec in records if rec["rank"] == rank]
        data = [fingerprints(rec) for rec in layer]
        row = (len(layer), len({a for a, b, c, d in data}),
               len({b for a, b, c, d in data}), len({c for a, b, c, d in data}),
               len({(c, d) for a, b, c, d in data}))
        rows.append(row)
        print(f"{rank:4d}" + "".join(f"{x:7d}" for x in row))
        if rank == 9:
            # Keep the same nonnormal marker while truncating the normal moments.
            for cutoff, expected_counts in [(4, (135, 138)), (5, (139, 142))]:
                truncated = []
                for _, _, laws, cubic in data:
                    signature = tuple(sorted(v[:cutoff + 1] for v in laws))
                    truncated.append((signature, cubic))
                totals = len({key for key, _ in truncated}), len(set(truncated))
                assert totals == expected_counts
                print(f"      Marked through degree {cutoff}: {totals[0]}, with cubic: {totals[1]}")
            for rec, (raw4, raw5, laws, cubic) in zip(layer, data):
                line = rec["source_line"]
                by_rank9_law[laws].append((line, cubic))
                by_rank9_raw4[raw4].append(line)

    counts = [1, 2, 4, 10, 16, 39, 43, 96, 142]
    expected = [(n, n, n, n, n) for n in counts[:-1]]
    expected.append((142, 137, 142, 139, 142))
    assert rows == expected and len(records) == sum(counts)

    collisions = sorted(sorted(group) for group in by_rank9_law.values() if len(group) > 1)
    assert collisions == [[(1923, 100), (2051, 116), (2075, 164)], [(1936, 163), (2080, 227)]]
    print("Rank-9 basic-law collisions (source line, cubic):", collisions)
    pairs = sorted(sorted(group) for group in by_rank9_raw4.values() if len(group) > 1)
    assert pairs == [[30, 114], [31, 33], [236, 280], [1019, 1021], [1301, 1465]]
    print("Rank-9 raw degree-4 collisions:", pairs)

    example = [r for r in records if r["rank"] == 9 and r["source_line"] in (30, 114)]
    example.sort(key=lambda r: r["source_line"])
    (a4, a5, _, ca), (b4, b5, _, cb) = map(fingerprints, example)
    assert a4 == b4 and ca == cb == 132 and a5 != b5
    for entry, signature in zip(example, (a5, b5)):
        print(f"L{entry['source_line']} (cubic 132), degree-2..5 tuples with multiplicities:",
              sorted(Counter(signature).items()))
    print("PASS: all 353 records separated by the multiset of raw degree-2..5 tuples.")


if __name__ == "__main__":
    main()
