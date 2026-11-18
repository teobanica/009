#!/usr/bin/env python3
"""
Doubling tests (the paper's doubling section). For a self-dual object X with invariant
dimensions c_n = c_n(X), the doubled sequence is c'_k = c_k(X (x) X) = c_{2k} = dim End(X^{(x)k}).
We compute exact inertia for available finite truncations of
   H : Hankel matrix (c_{i+j})
   E : Hankel matrix (c_{2i+2j})
   O : Hankel matrix (c_{2i+2j+2}).
Positivity of every truncation of H characterizes a law for X on the line.
Positivity of every E and O characterizes a law for X (x) X on [0, infinity).
Each output reports the finite truncation tested for that module.
Run:  python3 doubling.py     Results are printed and written to doubling-output.txt.
Optional expensive audit: python3 doubling.py --extended
This recomputes Lambda^2(C^4) through degree 80, checks its previously recorded
exact sequence, and tests E of size 21 and O of size 20.
"""
import argparse
import itertools, math
import json
from fractions import Fraction as Q
from collections import Counter
from pathlib import Path
import numpy as np
from hankel import hankel_inertia, hankel_summary

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--extended", action="store_true", help="recompute the degree-80 Levi audit")
args = parser.parse_args()

# ---- Taft algebra modules (matrices of g and x)
def taft_module(m, l, r):
    q = np.exp(2j * np.pi / m)
    g = np.diag([q ** (r - k) for k in range(l)]); x = np.zeros((l, l), complex)
    for k in range(l - 1): x[k + 1, k] = 1
    return (g, x)
def taft_tensor(A, B):
    (gA, xA), (gB, xB) = A, B
    return (np.kron(gA, gB), np.kron(np.eye(len(gA)), xB) + np.kron(xA, gB))
def taft_sum(A, B):
    def bd(a, b):
        o = np.zeros((len(a) + len(b),) * 2, complex); o[:len(a), :len(a)] = a; o[len(a):, len(a):] = b; return o
    return (bd(A[0], B[0]), bd(A[1], B[1]))
def taft_moments(X, N):
    c, P = [1], X
    for n in range(1, N + 1):
        g, x = P; d = len(g)
        idx = np.where(np.abs(np.diag(g) - 1) < 1e-9)[0]          # g is diagonal
        c.append(int(len(idx) - np.linalg.matrix_rank(x[:, idx], tol=1e-8)) if len(idx) else 0)
        if n < N: P = taft_tensor(P, X)
    return c

# ---- G_3 = C^3 x| SL_3, X = Lambda^2(C^4)
def lambda2_moments(N):
    dec = Counter({(0, 0, 0, 0): 1}); c = [1]
    for _ in range(N):
        new = Counter()
        for lam, m in dec.items():
            for i, j in itertools.combinations(range(4), 2):
                mu = list(lam); mu[i] += 1; mu[j] += 1
                if all(mu[t] >= mu[t + 1] for t in range(3)): new[tuple(x - mu[3] for x in mu)] += m
        dec = new; c.append(sum(m for lam, m in dec.items() if lam[1] == 0))
    return c

# ---- G_2 = C^2 x| SL_2, words in M, D = M^*, and 1,2,3 = V_1, V_2, V_3
def g2_moments(words, N):
    dec = Counter({(0, 0, 0): 1}); c = [1]
    for _ in range(N):
        new = Counter()
        for w in words:
            cur = dec
            for ch in w:
                nxt = Counter()
                for (a, b, s), m in cur.items():
                    if ch in 'MD':
                        for (x, y) in ([(a + 1, b), (a - 1, b + 1), (a, b - 1)] if ch == 'M' else [(a, b + 1), (a + 1, b - 1), (a - 1, b)]):
                            if x >= 0 and y >= 0: nxt[(x, y, s)] += m
                    else:
                        d = int(ch)
                        for e in range(abs(s - d), s + d + 1, 2): nxt[(a, b, e)] += m
                cur = nxt
            new.update(cur)
        dec = new; c.append(sum(m for (a, b, s), m in dec.items() if b == s))
    return c

# ---- affine group: Y = V_3|_B (x) (D^5 + D^-5),  c_n = sum_i C(n,i) a_{n,5(n-2i)}
def affine_moments(N):
    dec = Counter({0: 1}); c = [1]
    for n in range(1, N + 1):
        new = Counter()
        for m, mu in dec.items():
            for e in range(abs(m - 3), m + 4, 2): new[e] += mu
        dec = new; c.append(sum(math.comb(n, i) * dec.get(5 * (n - 2 * i), 0) for i in range(n + 1)))
    return c

# ---- G_a: torus formula, weights of the lift
def ga_moments(blocks, N):
    w = Counter()
    for r in blocks:
        for k in range(r): w[r - 1 - 2 * k] += 1
    P, c = Counter({0: 1}), []
    for _ in range(N + 1):
        c.append(P.get(0, 0) + Q(1, 2) * (P.get(1, 0) + P.get(-1, 0)))
        R = Counter()
        for a, ca in P.items():
            for b, cb in w.items(): R[a + b] += ca * cb
        P = R
    return [int(x) for x in c]

out = open("doubling-extended-output.txt" if args.extended else "doubling-output.txt", "w")
def log(s=""):
    print(s); out.write(s + "\n"); out.flush()
def report(name, c):
    log(f"{name}\n   c = {c[:11]}")
    for label, sequence in (("H", c), ("E", c[0::2]), ("O", c[2::2])):
        log(f"   {label}: {hankel_summary(sequence)}")

log("modules without a law"); log("-" * 70)
report("Taft T_3, M(2,2)", taft_moments(taft_module(3, 2, 2), 12))
report("affine group, V_3 (x) (D^5 + D^-5)", affine_moments(24))
report("C^2 x| SL_2, M (x) M^* (x) V", g2_moments(['MD1'], 10))
report("C^2 x| SL_2, V + M + M^*", g2_moments(['1', 'M', 'D'], 20))
report("C^2 x| SL_2, (M + M^*) (x) V_2", g2_moments(['M2', 'D2'], 14))
report("C^3 x| SL_3, Lambda^2(C^4)", lambda2_moments(28))
log(); log("modules with a law"); log("-" * 70)
report("G_a, J_2", ga_moments([2], 20))
report("G_a, J_2 (x) J_2 = J_1 + J_3", ga_moments([1, 3], 20))
report("G_a, J_4", ga_moments([4], 20))
report("C^2 x| SL_2, M + M^*", g2_moments(['M', 'D'], 20))
if args.extended:
    log(); log("Optional exact degree-80 Levi audit")
    extended = lambda2_moments(80)
    certificate = json.loads((Path(__file__).parent / "data" / "lambda2-80.json").read_text())
    assert extended == certificate, "The recomputed sequence differs from the stored certificate"
    assert hankel_inertia(extended[0::2]) == (21, 0, 0)
    assert hankel_inertia(extended[2::2]) == (20, 0, 0)
    log("PASS all 81 exact moments agree with data/lambda2-80.json")
    log("E: " + hankel_summary(extended[0::2]))
    log("O: " + hankel_summary(extended[2::2]))
    log("The degree-80 prefix passes these tests. verify_exterior.py finds the first impossible doubled moment at degree 84.")
out.close()
