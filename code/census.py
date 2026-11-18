#!/usr/bin/env python3
"""Reproduce the multiplicity-free fusion-ring census calculations offline.

The bundled 350 rank-3--9 tables are the 0/1 tables selected from the pinned
Fusion Ring Census files identified in data/census-manifest.json. Three
explicit rank-1--2 tables complete the list of 353 rings. Source line numbers
are one-based. Each table has entries N[i][j][k] = [b_i b_j : b_k], with b_0=1.

For each basis element we record its reduced scalar resolvent at the unit,
provided its left multiplication matrix is normal, and otherwise record a
non-normal marker. A commutative ring is also tested for a proper nontrivial
based subring and for ANY based subring admitting a nontrivial Z/2 grading.
The grading test solves exact homogeneous linear equations over F_2.
For commutative tables, we also enumerate two-colorings of each integral
nonunit fusion graph. Each surviving graph has a recorded basis index,
Perron root and coloring. The standalone BFS verifier checks the same count.

Run python3 census.py. Outputs are census-output.txt and census-results.json.
The optional command python3 census.py --extract /path/to/source-files first
rebuilds data/census-tables.json.gz from rank3.txt.gz,...,rank9.txt.gz after
checking each source SHA-256. Normal operation requires no network access.
"""

import argparse
from collections import defaultdict
import gzip
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
import sympy as sp


DATA = Path(__file__).parent / "data"
Z = sp.Symbol("z")
MULTIPLE = re.compile(r"[2-9]|[0-9]{2}")


def small_tables():
    return [
        {"id": "r1:trivial", "rank": 1, "source_line": None, "table": [[[1]]]},
        {"id": "r2:C2", "rank": 2, "source_line": None,
         "table": [[[1, 0], [0, 1]], [[0, 1], [1, 0]]]},
        {"id": "r2:Fibonacci", "rank": 2, "source_line": None,
         "table": [[[1, 0], [0, 1]], [[0, 1], [1, 1]]]},
    ]


def encode_bundle(tables):
    """Use deterministic JSON and a gzip header independent of the platform."""
    encoded = json.dumps(tables, separators=(",", ":"), ensure_ascii=True).encode()
    stream = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=stream, mtime=0) as compressed:
        compressed.write(encoded)
    return stream.getvalue()


def extract_tables(source_directory, manifest):
    """Rebuild the small bundle from the authenticated original gzip files."""
    tables = small_tables()
    for source in manifest["sources"]:
        rank = source["rank"]
        path = source_directory / f"rank{rank}.txt.gz"
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if checksum != source["sha256"]:
            raise ValueError(f"Source checksum mismatch for {path}")
        with gzip.open(path, "rt") as stream:
            for line_number, line in enumerate(stream, 1):
                if MULTIPLE.search(line):
                    continue
                table = json.loads(line.replace("{", "[").replace("}", "]"))
                tables.append({"id": f"r{rank}:L{line_number}", "rank": rank,
                               "source_line": line_number, "table": table})
    compressed = encode_bundle(tables)
    if hashlib.sha256(compressed).hexdigest() != manifest["bundle_sha256"]:
        raise ValueError("Rebuilt data differ from the authenticated bundle")
    (DATA / "census-tables.json.gz").write_bytes(compressed)


def load_tables():
    manifest = json.loads((DATA / "census-manifest.json").read_text())
    compressed = (DATA / "census-tables.json.gz").read_bytes()
    if hashlib.sha256(compressed).hexdigest() != manifest["bundle_sha256"]:
        raise ValueError("Bundled table checksum mismatch")
    return json.loads(gzip.decompress(compressed))


def validate_table(table):
    """Verify the unit, duality, reciprocity, and associativity exactly."""
    rank = len(table)
    assert table.shape == (rank, rank, rank)
    assert np.all((table == 0) | (table == 1))
    assert np.array_equal(table[0], np.eye(rank, dtype=np.int64))
    assert np.array_equal(table[:, 0, :], np.eye(rank, dtype=np.int64))
    assert np.all(table[:, :, 0].sum(axis=1) == 1)
    dual = np.argmax(table[:, :, 0], axis=1)
    assert np.array_equal(dual[dual], np.arange(rank))
    matrices = [table[i].T for i in range(rank)]
    assert all(np.array_equal(matrices[i].T, matrices[dual[i]]) for i in range(rank))
    for i in range(rank):
        for j in range(rank):
            product = sum((table[i, j, k] * matrices[k] for k in range(rank)),
                          np.zeros((rank, rank), dtype=np.int64))
            assert np.array_equal(matrices[i] @ matrices[j], product)
    return dual, matrices


def matrix_key(matrix):
    """The exact reduced numerator and denominator of the scalar resolvent."""
    denominator = sp.Poly(sp.Matrix(matrix.tolist()).charpoly(Z).as_expr(), Z)
    numerator = sp.Poly(sp.Matrix(matrix[1:, 1:].tolist()).charpoly(Z).as_expr(), Z)
    divisor = sp.gcd(denominator, numerator)
    return (tuple(int(value) for value in numerator.exquo(divisor).all_coeffs()),
            tuple(int(value) for value in denominator.exquo(divisor).all_coeffs()))


def based_subrings(table, dual):
    """List all unit-containing, dual-closed, product-closed basis subsets."""
    rank = len(table)
    subrings = []
    for mask in range(1, 1 << rank, 2):
        basis = [i for i in range(rank) if mask & (1 << i)]
        if any(not mask & (1 << int(dual[i])) for i in basis):
            continue
        if any(table[i, j, k] and not mask & (1 << k)
               for i in basis for j in basis for k in range(rank)):
            continue
        subrings.append(basis)
    return subrings


def parity_grading(table, basis):
    """Return the odd basis indices in a nontrivial grading, or None.

    For every N[i][j][k]>0 impose x_i+x_j+x_k=0 in F_2 and set x_0=0.
    Row reduction with the highest nonzero bit as pivot gives a null vector
    whenever there is a free variable. The returned vector is rechecked.
    """
    variable_mask = sum(1 << i for i in basis if i)
    rows = set()
    for i in basis:
        for j in basis:
            for k in basis:
                if table[i, j, k]:
                    rows.add(((1 << i) ^ (1 << j) ^ (1 << k)) & variable_mask)
    pivots = {}
    for original in sorted(rows):
        row = original
        while row:
            pivot = row.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = row
                break
            row ^= pivots[pivot]
    free = next((i for i in basis if i and i not in pivots), None)
    if free is None:
        return None
    assignment = 1 << free
    for pivot in sorted(pivots):
        if (pivots[pivot] & assignment).bit_count() % 2:
            assignment ^= 1 << pivot
    assert all((row & assignment).bit_count() % 2 == 0 for row in rows)
    return [i for i in basis if assignment & (1 << i)]


def grading_exists_by_enumeration(table, basis):
    """Independent check by enumerating every nonzero parity assignment."""
    variables = basis[1:]
    for assignment in range(1, 1 << len(variables)):
        odd = {index for position, index in enumerate(variables)
               if assignment & (1 << position)}
        if all(((i in odd) + (j in odd) + (k in odd)) % 2 == 0
               for i in basis for j in basis for k in basis if table[i, j, k]):
            return True
    return False


def integral_bipartite_witness(table):
    """Find a nonunit basic element of integral dimension with bipartite graph.

    Fixing the unit color zero loses no solutions. Enumerating all colorings
    checks disconnected components as well as loops. The spectral radius is
    the largest real root by Perron--Frobenius. Sturm counts test this exactly.
    """
    rank = len(table)
    for index in range(1, rank):
        matrix = table[index]
        rows, columns = np.nonzero(matrix)
        if np.any(rows == columns):
            continue
        edges = list(zip(rows.tolist(), columns.tolist()))
        for colors in range(0, 1 << rank, 2):
            if all(((colors >> i) ^ (colors >> j)) & 1 for i, j in edges):
                break
        else:
            continue
        polynomial = sp.Poly(sp.Matrix(matrix.tolist()).charpoly(Z).as_expr(), Z)
        for d in range(1, int(matrix.sum(axis=1).max()) + 1):
            if polynomial.eval(d):
                continue
            remaining = polynomial
            linear = sp.Poly(Z - d, Z)
            while remaining.eval(d) == 0:
                remaining = remaining.exquo(linear)
            if remaining.degree() == 0 or remaining.count_roots(d, sp.oo) == 0:
                return {"basis_index": index, "FP_dimension": d,
                        "color_one": [i for i in range(rank) if (colors >> i) & 1]}
    return None


def analyze(tables, log, cross_check_gradings=False):
    summaries = []
    details = []
    all_collisions = []
    spectral_cache = {}
    grading_checks = 0
    for rank in range(1, 10):
        records = [record for record in tables if record["rank"] == rank]
        signatures = defaultdict(list)
        refined_signatures = set()
        commutative_count = all_normal_positive = self_dual_count = nonnormal_count = 0
        simple_ids = []
        excluded_ids = []
        graph_excluded_ids = []
        for record in records:
            table = np.array(record["table"], dtype=np.int64)
            dual, matrices = validate_table(table)
            commutative = np.array_equal(table, table.transpose(1, 0, 2))
            commutative_count += int(commutative)
            self_dual_count += int(np.count_nonzero(dual[1:] == np.arange(1, rank)))
            central_symmetric_parts = all(
                np.array_equal((matrices[i] + matrices[dual[i]]) @ matrices[j],
                               matrices[j] @ (matrices[i] + matrices[dual[i]]))
                for i in range(rank) for j in range(rank))
            all_normal_positive += int(central_symmetric_parts)
            signature = []
            nonnormal_indices = []
            for i, matrix in enumerate(matrices):
                if np.array_equal(matrix @ matrix.T, matrix.T @ matrix):
                    cache_key = (rank, matrix.tobytes())
                    if cache_key not in spectral_cache:
                        spectral_cache[cache_key] = matrix_key(matrix)
                    signature.append((1, spectral_cache[cache_key]))
                else:
                    nonnormal_count += 1
                    nonnormal_indices.append(i)
                    signature.append((0,))
            signature = tuple(sorted(signature))
            total_matrix = sum(matrices, np.zeros((rank, rank), dtype=np.int64))
            cubic = int((total_matrix @ total_matrix @ total_matrix)[0, 0])
            signatures[signature].append({"id": record["id"], "cubic": cubic})
            refined_signatures.add((signature, cubic))
            subrings = based_subrings(table, dual)
            if cross_check_gradings:
                for basis in subrings:
                    assert bool(parity_grading(table, basis)) == grading_exists_by_enumeration(table, basis)
                    grading_checks += 1
            fusion_simple = not any(1 < len(basis) < rank for basis in subrings)
            if commutative and fusion_simple:
                simple_ids.append(record["id"])
            parity_witness = None
            for basis in subrings:
                odd = parity_grading(table, basis)
                if odd:
                    parity_witness = {"basis": basis, "odd": odd}
                    break
            if commutative and parity_witness is None:
                excluded_ids.append(record["id"])
            graph_witness = integral_bipartite_witness(table) if commutative else None
            if commutative and graph_witness is None:
                graph_excluded_ids.append(record["id"])
            details.append({"id": record["id"], "rank": rank,
                            "source_line": record["source_line"],
                            "commutative": bool(commutative),
                            "fusion_simple": fusion_simple,
                            "non_normal_basis_indices": nonnormal_indices,
                            "total_basic_sum_cubic": cubic,
                            "parity_subring_witness": parity_witness,
                            "braided_nonsemisimple_excluded_by_parity":
                                bool(commutative and parity_witness is None),
                            "integral_bipartite_witness": graph_witness,
                            "braided_nonsemisimple_excluded_by_graph":
                                bool(commutative and graph_witness is None)})
        collisions = [group for group in signatures.values() if len(group) > 1]
        all_collisions.extend(collisions)
        summary = {"rank": rank, "rings": len(records),
                   "commutative": commutative_count,
                   "all_normal_positive": all_normal_positive,
                   "self_dual_nonunit_basis": self_dual_count,
                   "nonnormal_basis": nonnormal_count,
                   "distinct_basic_signatures": len(signatures),
                   "distinct_cubic_refinements": len(refined_signatures),
                   "commutative_fusion_simple": len(simple_ids),
                   "parity_filter_exclusions": len(excluded_ids),
                   "graph_filter_exclusions": len(graph_excluded_ids),
                   "fusion_simple_ids": simple_ids,
                   "parity_exclusion_ids": excluded_ids,
                   "graph_exclusion_ids": graph_excluded_ids,
                   "collisions": collisions}
        summaries.append(summary)
        log(json.dumps({key: value for key, value in summary.items()
                        if not key.endswith("_ids") and key != "collisions"}))
    expected = [(1, 1, 1, 0, 0, 1), (2, 2, 2, 2, 0, 2),
                (4, 4, 4, 6, 0, 4), (10, 10, 10, 22, 0, 10),
                (16, 16, 16, 50, 0, 16), (39, 37, 37, 141, 0, 39),
                (43, 40, 40, 190, 0, 43), (96, 77, 80, 480, 10, 96),
                (142, 116, 119, 802, 40, 139)]
    fields = ("rings", "commutative", "all_normal_positive", "self_dual_nonunit_basis",
              "nonnormal_basis", "distinct_basic_signatures")
    assert [tuple(row[key] for key in fields) for row in summaries] == expected
    assert sum(row["rings"] for row in summaries) == 353
    assert all(row["distinct_cubic_refinements"] == row["rings"] for row in summaries)
    expected_cubics = {"r9:L1923": 100, "r9:L2051": 116, "r9:L2075": 164,
                       "r9:L1936": 163, "r9:L2080": 227}
    assert {entry["id"]: entry["cubic"] for group in all_collisions for entry in group} == expected_cubics
    log("PASS all 353 tables satisfy the checked fusion-ring axioms")
    log("PASS all spectral counts agree with the stated census")
    log("PASS the total-basic-sum cubic separates every colliding signature")
    if cross_check_gradings:
        log(f"PASS independent exhaustive parity checks for all {grading_checks} based subrings")
    for group in all_collisions:
        log("Collision resolved by cubic: " + json.dumps(group))
    combined_exclusions = [entry["id"] for entry in details if entry["commutative"]
                           and (entry["braided_nonsemisimple_excluded_by_parity"] or entry["rank"] == 3)]
    assert len(combined_exclusions) == 41
    log("PASS 41 commutative rings satisfy the parity or rank-three exclusion")
    graph_exclusions = [entry["id"] for entry in details if entry["braided_nonsemisimple_excluded_by_graph"]]
    assert [row["graph_filter_exclusions"] for row in summaries] == [1, 1, 4, 5, 14, 22, 36, 32, 96]
    assert len(graph_exclusions) == 211 and set(combined_exclusions) <= set(graph_exclusions)
    log("PASS 211 of 303 commutative rings satisfy the integral bipartite graph exclusion")
    log("Ring IDs, grading witnesses, graph witnesses and exclusions are in census-results.json")
    return {"summaries": summaries, "rings": details,
            "combined_parity_or_rank3_exclusion_ids": combined_exclusions,
            "integral_bipartite_exclusion_ids": graph_exclusions}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--extract", type=Path, help="rebuild the selected data from original source files")
    parser.add_argument("--cross-check-gradings", action="store_true",
                        help="independently enumerate parity assignments of all based subrings")
    args = parser.parse_args()
    manifest = json.loads((DATA / "census-manifest.json").read_text())
    if args.extract:
        extract_tables(args.extract, manifest)
    with open("census-output.txt", "w") as stream:
        def log(message):
            print(message, flush=True)
            stream.write(message + "\n")
            stream.flush()
        log("Fusion Ring Census commit " + manifest["commit"])
        log("Structural exclusions concern braided finite complex categorifications of fusion type.")
        results = analyze(load_tables(), log, args.cross_check_gradings)
    Path("census-results.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
