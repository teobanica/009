#!/usr/bin/env python3
"""Summarize the archived searches of Section 7.5 without rerunning them.

Run from code/: python3 summarize_searches.py [--check]
The default writes data/search-coverage.json.gz and search-coverage.tsv.
--check regenerates the summary in memory and compares it with those files.
Only the Python standard library is needed.

Each candidate is identified by its transcript, group index and first occurrence.
Repeated pass-two records update the attained depth; their source lines remain
listed. Exact negative-square polynomials are parsed and evaluated against the
archived integer moments. No missing stop reason or unperformed test is inferred.
"""
import argparse
import ast
from collections import Counter
from fractions import Fraction
import gzip
import hashlib
import json
from math import gcd, lcm
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
SOURCES = ("families-output.txt", "euclidean-output.txt",
           "euclidean-search-output.txt")
JSON_FILE = HERE / "data/search-coverage.json.gz"
TSV_FILE = HERE / "search-coverage.tsv"
MODULE = re.compile(r"^(.*?)   \[dimension (\d+), ([\d.]+) s\]$")
GROUP = re.compile(r"^(.*?): H = ")
CANDIDATES = re.compile(r"^(\d+) candidate modules$")
PASS = re.compile(r"^Pass (\d+), degrees up to (\d+)$")
POSITIVE = re.compile(r"^    positive semidefinite through size (\d+), definite through size (\d+)")
NEGATIVE = re.compile(r"^    NO LAW: the Hankel matrix of size (\d+) has inertia (\([^)]*\))")
STOP = re.compile(r"^    attained degree (\d+) of target (\d+); stop: (\S+)$")


def polynomial(text):
    """Read the rational polynomial format emitted by weights.poly_string."""
    coefficients = {}
    for term in text.replace(" - ", " + -").split(" + "):
        term = term.strip()
        if "t" in term:
            coefficient, power = term.split("t", 1)
            coefficient = coefficient.strip()
            coefficient = Fraction(-1 if coefficient == "-" else
                                   1 if not coefficient else coefficient)
            degree = int(power[1:]) if power else 1
        else:
            coefficient, degree = Fraction(term), 0
        coefficients[degree] = coefficients.get(degree, Fraction(0)) + coefficient
    return [coefficients.get(i, Fraction(0)) for i in range(max(coefficients) + 1)]


def canonical_group(name):
    if name in ("C^1 x| Spin_1", "G_a (identity component)"):
        return "G_a (identity component in Spin_3)"
    return name


def recover_negative_square(moments, size):
    """Find an integer negative direction by exact congruence, including zero diagonals."""
    original = [[Fraction(moments[i + j]) for j in range(size)] for i in range(size)]
    matrix = [row[:] for row in original]
    basis = [[Fraction(i == j) for i in range(size)] for j in range(size)]
    direction = None
    while matrix:
        n = len(matrix)
        pivot = next((i for i in range(n) if matrix[i][i]), None)
        if pivot is not None:
            d = matrix[pivot][pivot]
            if d < 0:
                direction = basis[pivot]
                break
            rest = [i for i in range(n) if i != pivot]
            basis = [[basis[j][k] - matrix[pivot][j] / d * basis[pivot][k]
                      for k in range(size)] for j in rest]
            matrix = [[matrix[i][j] - matrix[i][pivot] * matrix[pivot][j] / d
                       for j in rest] for i in rest]
        else:
            pair = next(((i, j) for i in range(n) for j in range(i + 1, n)
                         if matrix[i][j]), None)
            if pair is None:
                raise ValueError("Reported negative inertia has no negative direction")
            i, j = pair
            sign = 1 if matrix[i][j] > 0 else -1
            direction = [basis[i][k] - sign * basis[j][k] for k in range(size)]
            break
    if direction is None:
        raise ValueError("Reported negative inertia has no negative direction")
    denominator = lcm(*(a.denominator for a in direction))
    coefficients = [int(a * denominator) for a in direction]
    divisor = gcd(*coefficients)
    coefficients = [a // divisor for a in coefficients]
    value = sum(a * b * moments[i + j] for i, a in enumerate(coefficients)
                for j, b in enumerate(coefficients))
    if value >= 0:
        raise ValueError("Recovered square is not negative")
    return coefficients, value


def parse_source(filename):
    path = HERE / filename
    raw = path.read_bytes()
    lines = raw.decode("utf-8").splitlines()
    groups = []
    current = None
    candidate_records = {}
    pass_number = target = None
    record = None

    for index, line in enumerate(lines):
        line_number = index + 1
        match = GROUP.match(line)
        if match:
            current = {
                "group_id": f"{Path(filename).stem}:g{len(groups) + 1:02d}",
                "group": canonical_group(match.group(1)),
                "archived_group_name": match.group(1),
                "description_source_line": line_number,
                "announced_candidates": None,
                "group_summary_recorded": False,
                "passes_recorded": [],
                "candidates": [],
            }
            groups.append(current)
            candidate_records = {}
            pass_number = target = None
            record = None
            continue
        match = CANDIDATES.match(line)
        if match and current is not None:
            current["announced_candidates"] = int(match.group(1))
            continue
        match = PASS.match(line)
        if match and current is not None:
            pass_number, target = map(int, match.groups())
            current["passes_recorded"].append({"pass": pass_number, "target_degree": target})
            continue
        if line.startswith("SUMMARY for ") and current is not None:
            current["group_summary_recorded"] = True
            record = None
            continue
        match = MODULE.match(line)
        if match and current is not None:
            name, dim, seconds = match.groups()
            if name not in candidate_records:
                candidate = {
                    "candidate_id": f'{current["group_id"]}:c{len(candidate_records) + 1:04d}',
                    "module": name,
                    "dimension": int(dim),
                    "source_records": [],
                }
                candidate_records[name] = candidate
                current["candidates"].append(candidate)
            candidate = candidate_records[name]
            if candidate["dimension"] != int(dim):
                raise ValueError(f"{filename}:{line_number}: ambiguous candidate name")
            if any(r["pass"] == pass_number for r in candidate["source_records"]):
                raise ValueError(f"{filename}:{line_number}: repeated candidate in one pass")
            record = {"source_line": line_number, "pass": pass_number,
                      "target_degree": target, "elapsed_seconds": float(seconds)}
            candidate["source_records"].append(record)
            continue
        if record is None:
            continue
        if line.startswith("    c = "):
            moments = ast.literal_eval(line.split(" = ", 1)[1])
            if not isinstance(moments, list) or not moments or moments[0] != 1:
                raise ValueError(f"{filename}:{line_number}: invalid moments")
            if not all(isinstance(x, int) and x >= 0 for x in moments):
                raise ValueError(f"{filename}:{line_number}: nonintegral moments")
            record["_moments"] = moments
            record["moment_source_line"] = line_number
            record["terminal_degree"] = len(moments) - 1
            record["hankel_size_available"] = (len(moments) + 1) // 2
            record["stop_reason"] = ("target_reached" if len(moments) - 1 == target
                                     else "early_stop_reason_not_recorded")
            continue
        match = POSITIVE.match(line)
        if match:
            size, definite = map(int, match.groups())
            if size != record["hankel_size_available"]:
                raise ValueError(f"{filename}:{line_number}: positive depth mismatch")
            record.update(hankel_size_tested=size, definite_through_size=definite,
                          first_negative_size=None, negative_inertia=None,
                          negative_square=None, test_source_line=line_number)
            continue
        match = NEGATIVE.match(line)
        if match:
            size = int(match.group(1))
            inertia = ast.literal_eval(match.group(2))
            if sum(inertia) != size or inertia[1] < 1 or size > record["hankel_size_available"]:
                raise ValueError(f"{filename}:{line_number}: invalid negative inertia")
            record.update(hankel_size_tested=size, definite_through_size=None,
                          first_negative_size=size, negative_inertia=list(inertia),
                          negative_square=None, test_source_line=line_number)
            if "; p = " in line:
                poly, value = line.split("; p = ", 1)[1].split(" gives L(p^2) = ")
                coefficients, expected = polynomial(poly), Fraction(value)
                if len(coefficients) > size or expected >= 0:
                    raise ValueError(f"{filename}:{line_number}: invalid negative square")
                moments = record["_moments"]
                actual = sum(a * b * moments[i + j]
                             for i, a in enumerate(coefficients)
                             for j, b in enumerate(coefficients))
                if actual != expected:
                    raise ValueError(f"{filename}:{line_number}: negative square mismatch")
                record["negative_square"] = {
                    "coefficients_low_to_high": list(map(str, coefficients)),
                    "value": str(actual),
                    "source_line": line_number,
                    "origin": "archived_polynomial",
                }
            else:
                coefficients, value = recover_negative_square(record["_moments"], size)
                record["negative_square"] = {
                    "coefficients_low_to_high": list(map(str, coefficients)),
                    "value": str(value),
                    "source_line": record["moment_source_line"],
                    "origin": "recovered_from_archived_moments",
                }
            continue
        match = STOP.match(line)
        if match:
            attained, stated_target = map(int, match.groups()[:2])
            if (attained, stated_target) != (record["terminal_degree"], record["target_degree"]):
                raise ValueError(f"{filename}:{line_number}: stopping metadata mismatch")
            record["stop_reason"] = match.group(3)

    for group in groups:
        candidates = group["candidates"]
        if group["announced_candidates"] is None:
            raise ValueError(f"{filename}: group lacks announced candidate count")
        for candidate in candidates:
            for record in candidate["source_records"]:
                if "hankel_size_tested" not in record:
                    raise ValueError(f"{filename}:{record['source_line']}: incomplete record")
            best = max(candidate["source_records"], key=lambda r: r["terminal_degree"])
            negatives = [r for r in candidate["source_records"] if r["first_negative_size"]]
            candidate.update(terminal_degree=best["terminal_degree"],
                             hankel_size_available=best["hankel_size_available"],
                             hankel_size_tested=best["hankel_size_tested"],
                             first_negative_size=min(
                                 (r["first_negative_size"] for r in negatives), default=None),
                             obstruction_found=bool(negatives),
                             negative_square=next(
                                 (r["negative_square"] for r in negatives if r["negative_square"]), None))
            for record in candidate["source_records"]:
                record.pop("_moments")
                record.pop("negative_square")
        depths = Counter(c["terminal_degree"] for c in candidates)
        group.update(recorded_candidates=len(candidates),
                     unrecorded_candidates=group["announced_candidates"] - len(candidates),
                     terminal_degree_histogram={str(d): depths[d] for d in sorted(depths)},
                     reached_degree_12=sum(d >= 12 for c in candidates for d in [c["terminal_degree"]]),
                     reached_degree_40=sum(d >= 40 for c in candidates for d in [c["terminal_degree"]]),
                     obstructions_found=sum(c["obstruction_found"] for c in candidates),
                     negative_squares_checked=sum(c["negative_square"] is not None for c in candidates),
                     negative_squares_recovered=sum(c["negative_square"] is not None and
                         c["negative_square"]["origin"] == "recovered_from_archived_moments"
                         for c in candidates))
        if group["unrecorded_candidates"] < 0:
            raise ValueError(f"{filename}: more records than announced candidates")
    return {"source": filename, "sha256": hashlib.sha256(raw).hexdigest(),
            "final_summary_recorded": "FINAL SUMMARY" in lines,
            "groups": groups}


def build():
    runs = [parse_source(filename) for filename in SOURCES]
    return {
        "format_version": 1,
        "scope": "The three archived exploratory search transcripts used in Section 7.5.",
        "notes": [
            "Historical transcripts are unchanged; the m=1 spin label is corrected in the group field.",
            "Terminal degrees are attained degrees, not search targets.",
            "A negative inertia is an exact obstruction; archived polynomial witnesses are checked with Fraction arithmetic.",
            "Three records give negative inertia without a polynomial; integer negative directions are recovered by exact congruence and checked against their archived moments.",
            "No inference of a full law is made from a passing finite test.",
            "Old early stopping reasons were not logged; time and size limits cannot be distinguished retrospectively.",
            "No final summary means that completion of the announced run is not recorded, without assigning a cause.",
        ],
        "runs": runs,
    }


def make_tsv(summary):
    fields = ("source", "group", "announced", "recorded", "minimum_degree",
              "maximum_degree", "reached_12", "reached_40", "obstructions",
              "negative_squares_checked", "group_summary_recorded", "degree_histogram")
    rows = ["\t".join(fields)]
    for run in summary["runs"]:
        for group in run["groups"]:
            histogram = group["terminal_degree_histogram"]
            degrees = list(map(int, histogram))
            row = (run["source"], group["group"], group["announced_candidates"],
                   group["recorded_candidates"], min(degrees), max(degrees),
                   group["reached_degree_12"], group["reached_degree_40"],
                   group["obstructions_found"], group["negative_squares_checked"],
                   group["group_summary_recorded"],
                   ",".join(f"{d}:{histogram[str(d)]}" for d in sorted(degrees)))
            rows.append("\t".join(map(str, row)))
    return "\n".join(rows) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="compare with the distributed summary")
    args = parser.parse_args()
    summary = build()
    tsv = make_tsv(summary)
    if args.check:
        with gzip.open(JSON_FILE, "rt", encoding="utf-8") as handle:
            archived = json.load(handle)
        if archived != summary or TSV_FILE.read_text(encoding="utf-8") != tsv:
            raise SystemExit("The distributed search summary does not match its transcripts.")
    else:
        JSON_FILE.parent.mkdir(exist_ok=True)
        data = (json.dumps(summary, ensure_ascii=True, separators=(",", ":")) + "\n").encode()
        # Deterministic gzip header: no source filename or modification timestamp.
        with JSON_FILE.open("wb") as output:
            with gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as handle:
                handle.write(data)
        TSV_FILE.write_text(tsv, encoding="utf-8")
    groups = [g for run in summary["runs"] for g in run["groups"]]
    print("Search coverage: %d group-runs, %d recorded candidates, %d exact obstructions; "
          "%d negative squares checked (%d recovered)." %
          (len(groups), sum(g["recorded_candidates"] for g in groups),
           sum(g["obstructions_found"] for g in groups),
           sum(g["negative_squares_checked"] for g in groups),
           sum(g["negative_squares_recovered"] for g in groups)))
    for run in summary["runs"]:
        for group in run["groups"]:
            if group["unrecorded_candidates"]:
                print("%s: %d of %d announced candidates recorded; %d unrecorded." %
                      (run["source"], group["recorded_candidates"],
                       group["announced_candidates"], group["unrecorded_candidates"]))
    print("Distributed summary matches." if args.check else
          "Wrote data/search-coverage.json.gz and search-coverage.tsv.")


if __name__ == "__main__":
    main()

