# Computations for *Invariant tensor dimensions as Hamburger moments*

Run the commands from this directory with Python 3.10 or later. The full suite uses NumPy and SymPy. The bundled input tables and sequences suffice for all verification commands below.

## Exact verification commands

| Command | Result checked | Dependencies |
|---|---|---|
| `python3 verify_exterior.py --compare data/lambda2-172.json` | All 173 invariant dimensions of the restricted exterior square, the Hankel and shifted Hankel signs through size 43, and the upper-endpoint localizing matrices for the support bound 36 | SymPy |
| `python3 verify_exterior_pieri.py data/lambda2-172.json` | Independently recomputes all 173 dimensions by the GL(4) vertical Pieri rule using positive integer multiplicities | Standard library |
| `python3 verify_census.py data/census-tables.json.gz` | Spectral fingerprints, both collision groups and their cubics, the degree-four and degree-five return-count fingerprints for all 353 rings, and sharpness at rank nine | Standard library |
| `python3 check_bipartite_census.py data/census-tables.json.gz` | Counts the 211 of 303 commutative rings with no integral nonunit basic element whose entire undirected fusion graph is bipartite | SymPy |
| `python3 verify_nichols.py` | Nonvanishing of the first four powers of the degree-two invariant for the spin-five generator in the even part of SU(2) at level 13 | Standard library |

The transcripts are `verify-exterior-output.txt`, `verify-pieri-output.txt`, `verify-census-output.txt`, `check-bipartite-output.txt` and `verify-nichols-output.txt`.

### The doubled exterior square

`verify_exterior.py` uses Weyl alternation for type D3 and unrestricted-walk counts given by binomial sums. It recomputes every dimension from scratch in exact integer arithmetic. Let d[k] = c[2k]. The Hankel, shifted Hankel, and upper-endpoint forms on indices 0 through 41 are positive definite. The size-43 unshifted determinant has 1,603 digits and is negative. The manuscript proves that a probability measure on (0,36) realizes d[0] through d[83], and that adjoining d[84] precludes a Hamburger measure.

`verify_exterior_pieri.py` iterates the vertical Pieri rule on the three consecutive row differences of a four-row partition. At each degree it sums the multiplicities at labels (a,0,0). It reads the archived sequence only for the final comparisons. The independent `exterior_square_walks.py` counts chamber walks modulo nine primes and reconstructs the counts by CRT. The product of the moduli exceeds the a priori bound 6^172. The sixth-order recurrence in that program is a proposed identity tested for starting indices 0 through 166. All three computations obtain the dimensions independently of that recurrence. `doubling.py --extended` checks the degree-80 prefix.

### The census and the braided categorification obstructions

`verify_census.py` iterates multiplication by each basis element from the unit and compares the sorted tuples of unit coefficients. The tuples in degrees two through five distinguish the 353 bundled rings. In rank nine the degree-four cutoff gives 137 signatures for 142 rings, including two having identical total cubic 132.

`census.py` checks the unit, duality, reciprocity and associativity of the tables and computes the reduced unit-vector resolvent for each normal basis element by exact characteristic polynomials. Its multiset uses a marker for each nonnormal basis element. There are 139 such multisets at rank nine. The cubic of the sum of all basis elements separates the two collision groups. Each algorithm tests recognition on this specified census.

The graph obstruction rules out nonsemisimple braided finite categorifications for 211 of the 303 commutative rings. The exclusions in ranks one through nine are 1, 1, 4, 5, 14, 22, 36, 32 and 96. A passing ring has at least one integral nonunit basic element with a bipartite fusion graph. Existence of a categorification is a separate question.

`check_bipartite_census.py` reads the compressed tables directly, checks bipartiteness by breadth-first search on every connected component, and recognizes an integral Perron root by shifting the exact characteristic polynomial. Independently, `census.py` enumerates two-colorings and uses Sturm root counts. Its `census-results.json` includes the 211 excluded table identifiers and a basis index, integer Frobenius–Perron dimension and coloring for each of the 92 remaining commutative tables. The unit has color zero.

`python3 census.py --cross-check-gradings` also enumerates parity assignments on all 1,850 based subrings and checks them against elimination over the field of two elements. The graded-subring obstruction excludes 39 commutative tables, rising to 41 with the rank-three criterion. All 41 belong to the 211 graph exclusions. The counts and the two rank-nine spectral collisions are recorded in `census-output.txt`.

The data under `data/` include the selected 0/1 tables and a manifest of the selection, commit, source paths and hashes. The rank-one and rank-two tables are explicit. To regenerate the remaining 350 selected records from their seven original inputs, obtain the manifest-identified files under the names `rank3.txt.gz` through `rank9.txt.gz` and run `python3 census.py --extract /path/to/source-files`. Source revision [1b8494093de41c3ca23bb8e4b836fe5be649944e](https://github.com/sebastienpalcoux/fusion-ring-census/tree/1b8494093de41c3ca23bb8e4b836fe5be649944e).

### The rank-seven even Verlinde ring

`verify_nichols.py` works in a rational fusion-path gauge and specializes q to 3 in the field with 31 elements. It verifies denominators and braid relations and obtains nonzero diagonal entries 2, 30, 23 and 26 in degrees 2, 4, 6 and 8 on the paths of adjacent invariant cups. These residues imply nonvanishing in characteristic zero.

In the even part of SU(2) at level 13, the twist residues a(a+1) modulo 15 are 0, 2, 6, 12, 5, 0, 12, for spins a from zero through six. The four ordered pairs of distinct spins with equal residues are (0,5), (5,0), (3,6) and (6,3). Reversing braidings exchanges the factors. The manuscript proves infinite-dimensionality of the Nichols algebras of the two representative generators by the central Hopf-module argument and the all-level semisimplicity theorem. The residue calculation above establishes separately that the particular fourth power of the invariant cup survives. Higher powers of the cup require their own argument.

`nichols_verlinde.py` estimates graded multiplicities via singular values of quantum symmetrizers in the unitary realization. It omits braid-matrix entries smaller than 10^-14 and counts singular values larger than 10^-8 times the maximum of one and the largest singular value. The following are the resulting Frobenius–Perron dimension estimates. The full component multiplicities are in `nichols-verlinde-output.txt`.

| Generator | Degree 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---:|---:|---:|---:|---:|---:|---:|
| X5 | 3.57 | 12.78 | 42.09 | 138.67 | 449.99 | 1,451.81 | 4,661.89 |
| X3 paired with X6 in the reverse factor | 9.36 | 83.99 | 730.24 | 6,323.01 | 54,272.79 | | |

## Archived exploratory searches: actual coverage

The three transcripts used in Section 7.5 are `families-output.txt`,
`euclidean-output.txt` and `euclidean-search-output.txt`. Their search targets were
degree 12 in the first pass and degree 40 in the optional second pass. Time and
size limits often stopped individual computations earlier. A candidate with no
detected obstruction has passed only the finite test actually recorded.

Run the following short, standard-library check without rerunning any search:

```sh
python3 summarize_searches.py --check
```

This regenerates the summary from the transcripts, checks its agreement with
`data/search-coverage.json.gz` and `search-coverage.tsv`, and evaluates all 1,150
negative-square certificates exactly. Of these, 1,147 polynomials were printed
in the original transcripts. For the remaining three, which reported negative
inertia after a singular smaller form, the summarizer recovers integer negative
directions by rational congruence. Each recovered certificate is identified as
such and checked against the archived moments. This check validates the recorded
obstructions from their input moments; it does not recompute those invariants.
To regenerate the two summary files, omit `--check`.

The compressed JSON contains 6,611 candidate records across 29 group-runs, with
stable identifiers, dimensions, transcript line numbers, every recorded pass,
actual terminal degrees, Hankel sizes available and tested, obstruction status,
and negative polynomials in ascending coefficient order. The TSV gives the full
group-by-group depth histograms. Repeated second-pass records are combined with
the first pass, retaining their source lines. These are counts of archived
candidate records, not counts of distinct representations across all runs.

| Run/family | Recorded candidates | Terminal degrees | Reached 12 | Reached 40 |
|---|---:|---:|---:|---:|
| `families.py`: C^3 x\| Spin_3 | 63 | 23--40 | 63 | 29 |
| `families.py`: C^4 x\| SL_4 | 151 | 5--12 | 11 | 0 |
| `families.py`: C^4 x\| GL_4 | 610 | 3--12 | 61 | 0 |
| `euclidean.py`: double cover of E_3 | 1491 | 12 | 1491 | 0 |
| `euclidean_search.py`: E_3 | 465 | 10--12 | 454 | 0 |

The `euclidean.py` transcript announces 1,719 candidates but contains 1,491
records and no group or final summary: 228 announced candidates have no recorded
result, and no second pass is recorded. The archive does not establish why the
run ended. Likewise, old records below their pass target do not distinguish the
time and size stopping conditions. Their reason is explicitly marked as not
recorded. Future `weights.search` output includes the attained degree, target and
actual stopping condition.

The historical label `C^1 x| Spin_1` in `families-output.txt` denotes **the identity
component G_a** of the isotropic-vector stabilizer in Spin_3. The full inverse
image is G_a x mu_2 and is not what was computed: the spinor has one invariant for
G_a and none for the full inverse image. The corrected program and summary use
the connected-component label. The original transcripts are preserved unchanged
and their hashes are recorded in the JSON. For m >= 2 the C^m x| Spin_m labels
retain their usual meaning.

The search's `NO LAW` messages come from exact rational inertia, not from
floating-point eigenvalue thresholds. Uniform existence results for the additive
group and the specified transported double-cover family are proved in the
manuscript independently of finite search coverage.

## Other computations

| Program | Computation |
|---|---|
| `verify.py` | 93 checks on Hopf modules, algebraic groups, projective covers, Jacobi parameters, recognition and mixed moments. Some tensor-matrix kernels use floating-point ranks. The character recurrences and Hankel checks use exact arithmetic. |
| `hankel.py` | Full inertia of rational symmetric matrices by congruence elimination, including two-by-two pivots when the residual diagonal is zero. |
| `levi.py`, `doubling.py` | Stabilizer examples and finite Hankel, even-Hankel and shifted even-Hankel diagnostics. |
| `independent_levi_check.py` | Laurent-character and Weyl-denominator computations for the two Levi examples, through degrees four and twelve. |
| `weights.py` | Common character engine for highest-weight stabilizers, tensor products, exact Hankel inertia, and searches. |
| `vector_stabilizers.py` | Uniform vector-stabilizer witnesses through degree ten for the finite ranges of ranks used in the proof. |
| `euclidean_groups.py` | Character calculations for the complex Euclidean groups, including the 14-dimensional witness at m = 3. |
| `contraction.py` | Character transport from the spin cover of the Euclidean three-space group to SU(2) × SU(2). |
| `independent_euclidean_check.py`, `independent_euclidean_groups_check.py` | Sparse Lie-algebra kernel calculations over the field of order 2^31 − 1. The modular nullities bound the characteristic-zero nullities from above. The character calculations supply the characteristic-zero dimensions. |
| `euclidean_search.py`, `euclidean.py`, `families.py`, `linear.py`, `orthogonal.py` | Exploratory Hankel searches for self-dual modules. A negative direction excludes a law. A passing finite test reports the reached size. |

Run a named program with `python3`, for example `python3 euclidean_groups.py`. For the optional time-budgeted scans, see each script's usage string. The programs' recorded text outputs are included. Some optional Lie-algebra and exploratory runs take hours. Running an original program writes or replaces its own transcript. The five stand-alone verification commands at the top print to standard output.
