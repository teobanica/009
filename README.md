# Invariant tensor dimensions as Hamburger moments

**T. Banica and S. Palcoux**, November 2026

For a self-dual object $X$ of a complex tensor category, the invariant tensor dimensions $c_n(X)=\dim Hom(1,X^{\otimes n})$ generalize the moments $\int_G\chi_V^n$ of the character of a representation of a compact group. This paper studies when the $c_n(X)$ are the moments of a positive measure on the real line, and what this implies for the category. Among the results:

- the $c_n(X)$ are Hamburger moments if and only if $h(A,B)^2\leq h(A,A)h(B,B)$ holds for all finite sums $A,B$ of tensor powers of $X$, where $h(A,B)=\dim Hom(A,B)$; in a nonsemisimple category, the global version of this inequality gives Jordan blocks $J_r$ of every length, with the tensor rules of $Rep(G_a)$;
- for connected complex affine algebraic groups: the even invariant dimensions determine the Jordan type for $G_a$; in the solvable case every self-dual module has a law exactly for the tori and their products with $G_a$; the groups $\mathbb C^N\rtimes SL_N$, $\mathbb C^N\rtimes GL_N$ and the complex Euclidean groups $\mathbb C^m\rtimes SO_m$ have self-dual modules without a law, while for the double cover $\mathbb C^3\rtimes SL_2\subset Sp_4$ the modules built from $Sp_4$ and $SL_2$ have laws, carried by $SU(2)\times SU(2)$;
- in a finite tensor category of fusion type, Hamburger positivity for the projective cover of the unit forces unimodularity, and is equivalent to semisimplicity when the identity and double dual are isomorphic as additive functors;
- every finite categorification of the even Verlinde ring of $\mathfrak{sl}_2$ at odd level is semisimple, the corresponding Nichols algebras are infinite-dimensional, and a bipartite graph obstruction excludes nonsemisimple braided realizations for 211 of the 303 commutative multiplicity-free fusion rings through rank 9;
- passing from $X$ to $X\otimes X^*$ gives a Stieltjes criterion, and for $X=\Lambda^2(\mathbb C^4)$ restricted to $\mathbb C^3\rtimes SL_3$ the doubled sequence has a 42-atom representing measure through degree 83, and fails at degree 84.

## Contents

- `009.pdf` — the paper (50 pages)
- `009.tex` — its LaTeX source (self-contained, bibliography included; compile with `pdflatex`, 3 times)
- `code/` — the Python programs used in the paper, with their output files next to them, and the input data in `code/data/`; `code/README.md` gives the exact verification commands and the coverage of the searches

| file | content | paper |
|---|---|---|
| `weights.py` | character engine for stabilizers of highest weight vectors: tensor products, restriction, exact Hankel inertia, searches | Section 7 |
| `hankel.py` | inertia of rational symmetric matrices by congruence elimination | all |
| `verify.py` | 93 checks on Hopf modules, algebraic groups, projective covers, Jacobi parameters, recognition and mixed moments | Sections 4 to 12 |
| `levi.py`, `independent_levi_check.py` | stabilizer examples, and the 2 Levi examples independently, by Laurent characters | Sections 7.2, 7.4 |
| `vector_stabilizers.py` | the uniform witness for $\mathbb C^N\rtimes SL_N$ and $\mathbb C^N\rtimes GL_N$, Theorem 7.4 | Section 7.2 |
| `euclidean_groups.py` | the complex Euclidean groups, Theorem 7.6 | Section 7.3 |
| `contraction.py` | the double cover $\widetilde E_3$ and $SU(2)\times SU(2)$, Theorem 7.7, for 51 modules | Sections 7.4, 7.5 |
| `linear.py`, `orthogonal.py`, `families.py`, `euclidean.py`, `euclidean_search.py` | the searches for modules without a law, table of Section 7.5 | Sections 7.4, 7.5 |
| `summarize_searches.py` | coverage audit of the 3 search transcripts, checking all 1150 negative squares, output in `search-coverage.tsv` and `data/search-coverage.json.gz` | Section 7.5 |
| `census.py`, `check_bipartite_census.py` | the census of multiplicity-free fusion rings: gradings, the bipartite obstruction of Theorem 9.8, spectral multisets | Sections 9.3, 12.3 |
| `verify_nichols.py` | the 4th power of the invariant cup for the rank 7 even Verlinde ring, exact modulo 31, Proposition 9.17 | Section 9.4 |
| `nichols_verlinde.py` | growth of the Nichols algebras in degrees 1 to 7, numerical | Section 9.4 |
| `doubling.py` | doubled, even and shifted Hankel diagnostics | Section 11 |
| `verify_exterior.py`, `verify_exterior_pieri.py`, `exterior_square_walks.py` | the doubled exterior square through degree 172, by 3 independent methods, Theorem 11.5 | Section 11.2 |
| `verify_census.py` | spectral fingerprints and return counts for the 353 rings of the census | Section 12.3 |

The long computations ran on 2 laptops, with the following programs.

| laptop program | computation | paper |
|---|---|---|
| `families.py` | the 27 groups behind the table of Section 7.5, modules up to dimension 400, 3.99 hours | Section 7.5 |
| `euclidean.py` | 1491 modules of $\widetilde E_3\subset Sp_4$, stopped by hand after 1491 of its 1719 candidates, once replaced by `euclidean_search.py` | Section 7.5 |
| `euclidean_search.py` | 465 modules of $E_3=\mathbb C^3\rtimes SO_3\subset SL_4$, 1.54 hours | Sections 7.4, 7.5 |
| `independent_euclidean_check.py` | Lie algebra kernels for $E_3$, through degree 8, modulo $2^{31}-1$ | Section 7.3 |
| `independent_euclidean_groups_check.py` | Lie algebra kernels for $E\oplus W$ on $\mathbb C^m\rtimes SO_m$, $m=2,\ldots,7$, modulo $2^{31}-1$ | Section 7.3 |

## Running the code

Python 3.10 or later, with `numpy` and `sympy`. Run the programs from the `code/` folder, for instance `python3 verify_exterior.py --compare data/lambda2-172.json`, `python3 check_bipartite_census.py data/census-tables.json.gz` or `python3 verify_nichols.py`; the compressed data files are read as they are. The audit `python3 summarize_searches.py --check` takes seconds, and checks the search transcripts without rerunning the searches. Most programs run in seconds or minutes; the laptop programs take hours, and rerunning one replaces its output file.
