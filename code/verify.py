#!/usr/bin/env python3
"""
Computer checks for the paper "Invariant tensor dimensions as Hamburger moments"
(T. Banica and S. Palcoux).

Checks include tensor-matrix calculations, exact Hankel inertia, the quaternion
example, even moments over G_a, and census collision invariants. The total
number of executed checks is printed at the end.
Requirements: Python 3 and numpy. Run:   python3 verify.py
Results are printed and also written to the file verify-output.txt.
"""
import sys, math
import gzip
import json
from fractions import Fraction as Q
from collections import Counter
from pathlib import Path
import numpy as np
from hankel import first_non_psd_size, hankel_inertia, inertia

OUT = open("verify-output.txt", "w")
FAILS = []
CHECKS = 0
def log(*a):
    s = " ".join(str(x) for x in a)
    print(s); OUT.write(s + "\n"); OUT.flush()
def check(name, got, expected):
    global CHECKS
    CHECKS += 1
    ok = (got == expected)
    if not ok: FAILS.append(name)
    log(("PASS " if ok else "FAIL ") + name + ": " + str(got) + ("" if ok else "   expected " + str(expected)))

# ---------------------------------------------------------------- exact linear algebra
def det(M):
    M = [[Q(x) for x in r] for r in M]; n = len(M); d = Q(1)
    for i in range(n):
        p = next((r for r in range(i, n) if M[r][i] != 0), None)
        if p is None: return Q(0)
        if p != i: M[i], M[p] = M[p], M[i]; d = -d
        d *= M[i][i]
        for r in range(i + 1, n):
            f = M[r][i] / M[i][i]
            for c in range(i, n): M[r][c] -= f * M[i][c]
    return d
def hankel_dets(c, K):
    out = []
    for k in range(1, K + 1):
        if 2 * k - 2 >= len(c): break
        out.append(det([[c[i + j] for j in range(k)] for i in range(k)]))
    return [int(x) if x.denominator == 1 else x for x in out]
def jacobi(c, N):
    """Jacobi parameters (beta_n, lambda_n) of the functional L(t^n)=c[n], Stieltjes procedure."""
    L = lambda p: sum(Q(a) * c[i] for i, a in enumerate(p))
    mul = lambda p, q: [sum(p[i] * q[k - i] for i in range(len(p)) if 0 <= k - i < len(q)) for k in range(len(p) + len(q) - 1)]
    sub = lambda p, q: [(p[i] if i < len(p) else 0) - (q[i] if i < len(q) else 0) for i in range(max(len(p), len(q)))]
    prev, p, betas, lams, nprev = [Q(0)], [Q(1)], [], [], None
    for n in range(N):
        nn = L(mul(p, p))
        if nn == 0: break
        b = L(mul([0] + p, p)) / nn; betas.append(b)
        lam = nn / nprev if n > 0 else Q(0)
        if n > 0: lams.append(lam)
        new = sub(sub([0] + p, [b * x for x in p]), [lam * x for x in prev])
        prev, p, nprev = p, new, nn
    f = lambda x: int(x) if x.denominator == 1 else x
    return [f(x) for x in betas], [f(x) for x in lams]

# ---------------------------------------------------------------- Hopf algebra modules (numerical)
def rank(M): return np.linalg.matrix_rank(M, tol=1e-8)
def blockdiag(*Ms):
    n = sum(len(M) for M in Ms); out = np.zeros((n, n), complex); k = 0
    for M in Ms: out[k:k+len(M), k:k+len(M)] = M; k += len(M)
    return out
def kron(*Ms):
    out = Ms[0]
    for M in Ms[1:]: out = np.kron(out, M)
    return out

# Taft algebra T_m: g^m=1, x^m=0, xg=qgx, Delta(g)=g(x)g, Delta(x)=1(x)x+x(x)g
def taft_module(m, l, r):
    """M(l,r): basis v_0..v_{l-1}, g v_k = q^(r-k) v_k, x v_k = v_{k+1}."""
    q = np.exp(2j * np.pi / m)
    g = np.diag([q ** (r - k) for k in range(l)]); x = np.zeros((l, l), complex)
    for k in range(l - 1): x[k + 1, k] = 1
    return (g, x)
def taft_tensor(A, B):
    (gA, xA), (gB, xB) = A, B
    return (np.kron(gA, gB), np.kron(np.eye(len(gA)), xB) + np.kron(xA, gB))
def taft_sum(*Ms): return (blockdiag(*[M[0] for M in Ms]), blockdiag(*[M[1] for M in Ms]))
def taft_invariants(M):
    g, x = M
    diagonal = np.diag(g)
    assert np.array_equal(g, np.diag(diagonal))  # all displayed module bases diagonalize g
    invariant_weights = np.flatnonzero(np.abs(diagonal - 1) < 1e-9)
    return len(invariant_weights) - rank(x[:, invariant_weights]) if len(invariant_weights) else 0
def taft_moments(X, N):
    c, P = [1], X
    for n in range(1, N + 1):
        c.append(int(taft_invariants(P)))
        if n < N: P = taft_tensor(P, X)
    return c

# H_r: g^(2r)=1, gx=zeta xg, x^2=0, Delta(x)=x(x)1+g^r(x)x
def Hr_simple(r, j): return (np.array([[np.exp(2j*np.pi*j/(2*r))]]), np.zeros((1, 1), complex))
def Hr_proj(r, j):
    z = np.exp(2j * np.pi / (2 * r)); g = np.diag([z ** j, z ** (j + 1)]); x = np.zeros((2, 2), complex); x[1, 0] = 1
    return (g, x)
def Hr_tensor(r, A, B):
    (gA, xA), (gB, xB) = A, B
    return (np.kron(gA, gB), np.kron(xA, np.eye(len(gB))) + np.kron(np.linalg.matrix_power(gA, r), xB))
def Hr_dual(r, M):
    g, x = M; gi = np.linalg.inv(g)
    return (gi.T, (-np.linalg.matrix_power(gi, r) @ x).T)   # antipode S(g)=g^-1, S(x)=-g^-r x

# Lie algebra actions on tensor powers
def lie_power(N, n):
    d = len(N); I = np.eye(d); tot = np.zeros((d ** n, d ** n))
    for k in range(n): tot = tot + kron(*([I] * k + [N] + [I] * (n - k - 1))) if n > 1 else N
    return tot

# torus characters for G_a and SU(2)
def V(d): return Counter({d - 2 * j: 1 for j in range(d + 1)})
def add(*ws):
    c = Counter()
    for w in ws:
        for k, v in w.items(): c[k] += v
    return c
def power_moments(weights, n, functional):
    P, out = Counter({0: 1}), []
    for k in range(n):
        out.append(functional(P))
        R = Counter()
        for a, ca in P.items():
            for b, cb in weights.items(): R[a + b] += ca * cb
        P = R
    return [int(x) if Q(x).denominator == 1 else x for x in out]
ga_functional = lambda P: P.get(0, 0) + Q(1, 2) * (P.get(1, 0) + P.get(-1, 0))      # (1+cos) dtheta/2pi
su2_functional = lambda P: P.get(0, 0) - Q(1, 2) * (P.get(2, 0) + P.get(-2, 0))     # Weyl density 1-cos(2theta)

# ================================================================ checks
log("=" * 70); log("Radical square zero and Taft algebras"); log("=" * 70)
c = taft_moments(taft_module(3, 2, 2), 6)
check("T_3, X=M(2,2), c_0..c_6", c, [1, 0, 1, 1, 2, 4, 7])
check("Hankel det (c_{i+j})_{0<=i,j<=3}", hankel_dets(c, 4)[-1], -1)

for (a, b, cc) in [(1, 0, 1), (0, 1, 1), (2, 1, 1), (1, 2, 0), (0, 0, 1)]:
    parts = [taft_module(2, 1, 0)] * a + [taft_module(2, 1, 1)] * b + [taft_module(2, 2, 0), taft_module(2, 2, 1)] * cc
    X = taft_sum(*parts); N = 5 if len(X[0]) <= 5 else 4
    got = taft_moments(X, N)
    pred = [Q(1, 2) * (a - b) ** n + Q(1, 4) * (a + b) ** n + Q(1, 4) * (a + b + 4 * cc) ** n for n in range(N + 1)]
    check(f"Sweedler, (a,b,c)={(a,b,cc)}, moments vs measure", got, [int(x) for x in pred])

m = 3; Y = taft_sum(taft_module(m, m, 0), taft_module(m, m, m - 1)); c = taft_moments(Y, 4)
check("T_3, Y=P_0+P_2, c_0..c_4", c, [1, 1, 4, 8 * m, 16 * m * m])
hAB, hAA, hBB = c[1] + c[3], c[0] + 2 * c[2] + c[4], c[2]
check("Taft violation h(A,B)^2-h(A,A)h(B,B) = 16m-35 at m=3", hAB ** 2 - hAA * hBB, 16 * m - 35)

r = 2; ok = True
for X0 in [taft_sum(Hr_proj(r, 0), Hr_simple(r, 1)), Hr_proj(r, 1)]:
    X = taft_sum(X0, Hr_dual(r, X0)); c, P = [1], X
    for n in range(1, 5):
        c.append(int(taft_invariants(P)))
        if n < 4: P = Hr_tensor(r, P, X)
    ev = min(np.linalg.eigvalsh(np.array([[c[i + j] for j in range(3)] for i in range(3)], float)))
    log("     H_2, X+X^*: moments", c, " min eigenvalue of Hankel H_2:", round(ev, 4)); ok = ok and ev > 0
check("spot check in Rep(H_2): Hankel matrices positive", ok, True)

log("=" * 70); log("Algebraic groups"); log("=" * 70)
J = np.array([[0, 1], [0, 0]]); Z = np.zeros((2, 2))
N1 = np.block([[J, Z, Z], [Z, Z, Z], [Z, Z, J]]); N2 = np.block([[Z, Z, Z], [Z, J, Z], [Z, Z, J]])
c = [1] + [6 ** n - rank(np.vstack([lie_power(N1, n), lie_power(N2, n)])) for n in range(1, 5)]
check("G_a^2 module, c_0..c_4", [int(x) for x in c], [1, 3, 12, 57, 270])
b = lambda n: math.comb(n, n // 2)
ga3 = [sum(math.factorial(n) // (math.factorial(i) * math.factorial(j) * math.factorial(n - i - j)) * b(i) * b(j) * b(n - i - j)
           for i in range(n + 1) for j in range(n - i + 1)) for n in range(5)]
check("G_a^3 module, c_0..c_4", ga3, [1, 3, 12, 51, 234])

def sl2(d):
    H = np.diag([d - 2 * k for k in range(d + 1)]).astype(float); E = np.zeros((d + 1, d + 1))
    for k in range(1, d + 1): E[k - 1, k] = k * (d - k + 1)
    return H, E
H3, E3 = sl2(3); HY = np.block([[H3 + 5 * np.eye(4), np.zeros((4, 4))], [np.zeros((4, 4)), H3 - 5 * np.eye(4)]])
EY = np.block([[E3, np.zeros((4, 4))], [np.zeros((4, 4)), E3]]); c = [1]
for n in range(1, 5):
    Hn, En = lie_power(HY, n), lie_power(EY, n); idx = np.where(np.abs(np.diag(Hn)) < 1e-9)[0]
    c.append(int(len(idx) - rank(En[:, idx])) if len(idx) else 0)
check("affine group, c_0..c_4", c, [1, 0, 2, 9, 36])

J2 = [1] + [2 ** n - rank(lie_power(J, n)) for n in range(1, 9)]
check("J_2 invariants = binomial(n, floor(n/2))", [int(x) for x in J2], [b(n) for n in range(9)])
check("Tilted torus formula for J_2 (exact)", power_moments(V(1), 9, ga_functional), [b(n) for n in range(9)])
th = np.linspace(0, 2 * np.pi, 400001)[:-1]
dens = [round(float(np.mean((1 + np.cos(th)) * (2 * np.cos(th)) ** n)), 6) for n in range(9)]
check("density (1+x/2)/(pi sqrt(4-x^2)) moments (numerical)", dens, [float(b(n)) for n in range(9)])

log("=" * 70); log("Hankel determinants and Jacobi parameters"); log("=" * 70)
cat = lambda n: math.comb(2 * n, n) // (n + 1)
fib = [1, 0]
for n in range(2, 16): fib.append(fib[-1] + fib[-2])
s3 = [Q(1, 6) * 2 ** n + Q(1, 2) * (0 ** n) + Q(1, 3) * (-1) ** n for n in range(14)]
seqs = {
    "semicircle":      ([cat(n // 2) if n % 2 == 0 else 0 for n in range(16)], [1] * 7),
    "free Poisson":    ([cat(n) for n in range(16)], [1] * 7),
    "SU(2), V_2":      (power_moments(V(2), 16, su2_functional), [1] * 7),
    "Rep(S_3), rho":   ([int(x) for x in s3], [1, 1, 1, 0, 0, 0, 0]),
    "Fibonacci":       (fib, [1, 1, 0, 0, 0, 0, 0]),
    "G_a, J_2":        (power_moments(V(1), 16, ga_functional), [1] * 7),
    "G_a, J_3":        (power_moments(V(2), 16, ga_functional), [2 ** k for k in range(7)]),
    "G_a, J_4":        (power_moments(V(3), 16, ga_functional), [1, 3, 20, 485, 60709]),
    "G_a, J_2+J_3":    (power_moments(add(V(1), V(2)), 16, ga_functional), [1, 5, 59, 1717, 121737]),
    "T_3, M(2,2)":     ([1, 0, 1, 1, 2, 4, 7], [1, 1, 0, -1]),
    "G_a^2 module":    ([1, 3, 12, 57, 270], [1, 3, -63]),
    "affine group":    ([1, 0, 2, 9, 36], [1, 2, -17]),
}
for name, (c, expected) in seqs.items():
    check("Delta_k, " + name, hankel_dets(c, len(expected)), expected)
check("SU(2) V_2 moments are Riordan numbers", seqs["SU(2), V_2"][0][:10], [1, 0, 1, 1, 3, 6, 15, 36, 91, 232])
check("J_3 moments are central trinomial coefficients", seqs["G_a, J_3"][0][:8], [1, 1, 3, 7, 19, 51, 141, 393])
for name, beta, lam in [("G_a, J_2", [1, 0, 0, 0, 0], [1, 1, 1, 1]), ("G_a, J_3", [1, 1, 1, 1, 1], [2, 1, 1, 1]),
                        ("free Poisson", [1, 2, 2, 2, 2], [1, 1, 1, 1]), ("SU(2), V_2", [0, 1, 1, 1, 1], [1, 1, 1, 1]),
                        ("Rep(S_3), rho", [0, 1, 0], [1, 1])]:
    got = jacobi(seqs[name][0], 5)
    check("Jacobi parameters (beta, lambda), " + name, (got[0][:len(beta)], got[1][:len(lam)]), (beta, lam))
check("J_4 lambda_1..lambda_3", jacobi(seqs["G_a, J_4"][0], 4)[1][:3], [3, Q(20, 9), Q(291, 80)])

log("=" * 70); log("Fusion rings and complex measures"); log("=" * 70)
phi = (1 + 5 ** 0.5) / 2
mom = [round(1 / (phi + 2) * phi ** n + (phi + 1) / (phi + 2) * (-1 / phi) ** n, 9) for n in range(10)]
check("Fibonacci measure reproduces tau(x^n)", mom, [float(x) for x in fib[:10]])
# Z/3: tau(g^a g*^b) = [a=b mod 3] versus Plancherel formula, characters w^k, f=3
w = np.exp(2j * np.pi / 3)
pl = [[float(round(abs(sum((w ** k) ** a * np.conj(w ** k) ** b / 3 for k in range(3))), 9)) for b in range(4)] for a in range(4)]
check("Uniform circle subgroup law, Z/3 generator", pl, [[1.0 if (a - b) % 3 == 0 else 0.0 for b in range(4)] for a in range(4)])
P0, P1 = taft_module(2, 2, 0), taft_module(2, 2, 1)
check("Sweedler: (c_{1,0}, c_{0,1}) for X=P_0", (int(taft_invariants(P0)), int(taft_invariants(P1))), (0, 1))

# ---------------------------------------------------- additional character calculations
import itertools
def matpow_entry(J, n):
    v = [1] + [0] * (len(J) - 1)
    for _ in range(n): v = [sum(J[i][j] * v[j] for j in range(len(J))) for i in range(len(J))]
    return v[0]
def Jl(l):
    J = [[0] * (l + 1) for _ in range(l + 1)]
    for i in range(l + 1):
        if i: J[i][i] = 1; J[i][i - 1] = J[i - 1][i] = 1
    return J
def pivots(c):
    """Unpivoted pivots for the displayed strict/rank-terminating examples.

    General semidefiniteness is tested with hankel_inertia, including zero pivots.
    """
    n = (len(c) + 1) // 2
    M = [[Q(c[i + j]) for j in range(n)] for i in range(n)]; out = []
    for i in range(n):
        out.append(M[i][i])
        if M[i][i] == 0: break
        for r in range(i + 1, n):
            f = M[r][i] / M[i][i]
            for cc in range(i, n): M[r][cc] -= f * M[i][cc]
    return out

log("-" * 70); log("Semisimple Levi factors")
# Lemma [levi]: G_N = Stab_{SL_{N+1}}(e_1);  dim (Y|_G (x) W)^G = sum [Y:V_lambda][W^*:V_{lambda bar}]
def sl3_times(dec, kind):                      # tensor with M = C^3 ('M') or its dual ('D'); weights (a,b,c), c for SL_2
    out = Counter()
    for (a, b, c), m in dec.items():
        for (x, y) in ([(a + 1, b), (a - 1, b + 1), (a, b - 1)] if kind == 'M' else [(a, b + 1), (a + 1, b - 1), (a - 1, b)]):
            if x >= 0 and y >= 0: out[(x, y, c)] += m
    return out
def sl2_times(dec, d):
    out = Counter()
    for (a, b, c), m in dec.items():
        for e in range(abs(c - d), c + d + 1, 2): out[(a, b, e)] += m
    return out
def g2_moments(words, N):                      # X = sum of words in M, D (=M^*), 1,2,3 (= V_1,V_2,V_3 of the Levi SL_2)
    dec = Counter({(0, 0, 0): 1}); c = [1]
    for _ in range(N):
        new = Counter()
        for w in words:
            cur = dec
            for ch in w: cur = sl3_times(cur, ch) if ch in 'MD' else sl2_times(cur, int(ch))
            new.update(cur)
        dec = new; c.append(sum(m for (a, b, cc), m in dec.items() if b == cc))
    return c
c = g2_moments(['MD1'], 4)
check("G = C^2 x| SL_2, X = M (x) M^* (x) V: c_0..c_4", c, [1, 1, 4, 46, 356])
check("violation c_3^2 - c_2 c_4", c[3] ** 2 - c[2] * c[4], 692)
# independent check by Lie algebra kernels (no highest weight theory)
def Eij(i, j):
    A = np.zeros((3, 3)); A[i - 1, j - 1] = 1; return A
gM = {'E': Eij(2, 3), 'P1': Eij(1, 2), 'P2': Eij(1, 3)}; gD = {k: -v.T for k, v in gM.items()}
def hw_vectors(n):                             # highest weight vectors of ((M (x) M^*)^{(x) n})^U, by SL_2-weight
    reps = [gM, gD] * n; ws = [[0, 1, -1], [0, -1, 1]] * n; k = 2 * n
    W = np.zeros(1, dtype=int)
    for w in ws: W = (W[:, None] + np.array(w)[None, :]).reshape(-1)
    ops = []
    for x in gM:
        tot = np.zeros((3 ** k, 3 ** k))
        for j in range(k):
            t = np.eye(1)
            for i in range(k): t = np.kron(t, reps[i][x] if i == j else np.eye(3))
            tot += t
        ops.append(tot)
    m = {}
    for w in range(0, 2 * n + 1):
        idx = np.where(W == w)[0]
        if len(idx):
            A = np.vstack([o[:, idx] for o in ops]); d = len(idx) - rank(A)
            if d: m[w] = d
    return m
def v1_power(n):
    d = {0: 1}
    for _ in range(n):
        e = {}
        for cc, mu in d.items():
            for f in (cc - 1, cc + 1):
                if f >= 0: e[f] = e.get(f, 0) + mu
        d = e
    return d
brute = [1] + [sum(hw_vectors(n).get(w, 0) * mu for w, mu in v1_power(n).items()) for n in (1, 2, 3)]
check("c_0..c_3 again, from Lie algebra kernels", brute, [1, 1, 4, 46])
# tilted Haar law for restricted self-dual SL_3-modules, Y = M + M^*
def E_cos(m): return Q(math.comb(m, m // 2), 2 ** m) if m % 2 == 0 else Q(0)
def tilted(n): return 2 * sum(math.comb(n, j) * 2 ** (n - j) * 4 ** j * (E_cos(j) - E_cos(j + 1) - E_cos(j + 2) + E_cos(j + 3)) for j in range(n + 1))
c = g2_moments(['M', 'D'], 12)
check("Y = M + M^* restricted to C^2 x| SL_2, c_0..c_12", c, [1, 1, 4, 12, 48, 192, 832, 3712, 17152, 81152, 392192, 1928192, 9621504])
check("same numbers from the density (1/pi) sin^2 t (1 - cos t)", [int(tilted(n)) for n in range(13)], c)
check("Hankel pivots all positive (sizes 1..7)", all(p > 0 for p in pivots(c)), True)
# G = C^3 x| SL_3 in SL_4, X = Lambda^2(C^4): Pieri rule, vertical 2-strips, count shapes (k+a,a,a,a)
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
c = lambda2_moments(6)
check("G = C^3 x| SL_3, X = Lambda^2(C^4): c_0..c_6", c, [1, 0, 1, 1, 3, 10, 21])
p = [5, 3, -6, 1]
check("L(p^2) for p = t^3-6t^2+3t+5", sum(p[i] * p[j] * c[i + j] for i in range(4) for j in range(4)), -25)

log("-" * 70); log("Projective covers and Cartan matrices")
for m_ in (2, 4):
    d = 2 ** m_
    c = [1, 1] + [Q(d ** (n - 1), 2) for n in range(2, 5)]       # c_n = tau(a^{n-1}), a = (d/2)(1+s)
    check(f"exterior algebra, dim V = {m_}: (c_0..c_4)", c, [1, 1, Q(d, 2), Q(d * d, 2), Q(d ** 3, 2)])
    check(f"dim V = {m_}: det of 3x3 Hankel matrix", det([[c[i + j] for j in range(3)] for i in range(3)]), Q(-d ** 3, 8))
    hAA, hAB, hBB = c[2], c[1] + c[3], c[0] + 2 * c[2] + c[4]
    check(f"dim V = {m_}: h(A,B)^2 - h(A,A)h(B,B)", hAB ** 2 - hAA * hBB, 1 + Q(d * d - d, 2))

log("-" * 70); log("Twists of the even Verlinde categories")
good = [l for l in range(1, 31) if len({a * (a + 1) % (2 * l + 3) for a in range(l + 1)}) == l + 1]
def isprime(n): return n > 1 and all(n % p for p in range(2, int(n ** .5) + 1))
check("Distinct-twist residue test for l <= 30", good, [l for l in range(1, 31) if isprime(2 * l + 3) or 2 * l + 3 == 9])
log("     admissible l:", good)

log("-" * 70); log("Recognition of fusion rules")
rows = {l: [matpow_entry(Jl(l), n) for n in range(2 * l + 3)] for l in (1, 2, 3, 4)}
check("Even Verlinde moments, row l=1", rows[1], [1, 0, 1, 1, 2])
check("Even Verlinde moments, row l=2", rows[2], [1, 0, 1, 1, 3, 6, 14])
for l in rows: log("     l =", l, " moments", rows[l], " Hankel pivots", [str(x) for x in pivots(rows[l])])
check("Even Verlinde unit Hankel determinants then zero, l=1..4", all(pivots(rows[l]) == [1] * (l + 1) + [0] for l in rows), True)
# rank 3 tables (m,k,l,n): x^2=1+mx+ky, xy=kx+ly, y^2=1+lx+ny
def rank3_moments(m, k, l, n, N):
    Nx = [[0, 1, 0], [1, m, k], [0, k, l]]
    return [matpow_entry(Nx, j) for j in range(N + 1)]
def recover(s):
    m = s[3]; k = math.isqrt(s[4] - 1 - m * m); l = Q(s[5] - 2 * m - m ** 3 - 2 * m * k * k, k * k); n = (k * k + l * l - l * m - 1) / k
    return (m, k, l, n)
for tab in [(0, 1, 1, 0), (0, 1, 0, 0), (1, 1, 1, 0), (2, 2, 3, 3), (1, 1, 0, 0)]:
    if tab[1] ** 2 + tab[2] ** 2 == tab[1] * tab[3] + tab[2] * tab[0] + 1:
        check(f"recovery of the table {tab} from s_3, s_4, s_5", recover(rank3_moments(*tab, 5)), tab)
check("Rep(S_3) vs even level 5: s_0..s_5", (rank3_moments(1, 1, 0, 0, 5), rank3_moments(1, 1, 1, 0, 5)), ([1, 0, 1, 1, 3, 5], [1, 0, 1, 1, 3, 6]))
# third moments of J_1 + J_5 and J_2 + J_2 over G_a
check("third moments of J_1+J_5 and of 2 J_2", (power_moments(add(V(0), V(4)), 4, ga_functional)[3], power_moments(add(V(1), V(1)), 4, ga_functional)[3]), (38, 24))

log("-" * 70); log("Doubling")
cT = taft_moments(taft_module(3, 2, 2), 8)
check("Taft M(2,2): doubled sequence d_0..d_4", cT[0::2], [1, 1, 2, 7, 28])
check("Taft M(2,2): Hankel determinant of size 3 of the doubled sequence", det([[cT[2 * (i + j)] for j in range(3)] for i in range(3)]), -1)
J2 = power_moments(V(1), 17, ga_functional)
check("J_2: doubled sequence = binomial(2k,k) = that of the torus module", J2[0::2], [math.comb(2 * k, k) for k in range(9)])
check("J_2: odd moments = binomial(2k+1,k)", J2[1::2], [math.comb(2 * k + 1, k) for k in range(8)])
check("law of J_1+J_3 = doubled law of J_2", power_moments(add(V(0), V(2)), 9, ga_functional), J2[0::2])
for m_ in (2, 4):
    d = 2 ** m_
    dk = [1] + [Q(d ** (2 * k - 1), 2) for k in range(1, 6)]
    law = [(1 - Q(1, 2 * d)) * (1 if k == 0 else 0) + Q(1, 2 * d) * d ** (2 * k) for k in range(6)]
    check(f"exterior algebra, dim V = {m_}: d_k(P) = moments of (1-1/2d) delta_0 + (1/2d) delta_(d^2)", dk, law)
sc = [cat(n // 2) if n % 2 == 0 else 0 for n in range(26)]
bj, lj = jacobi(sc[0::2], 6)
check("semicircle -> free Poisson: (beta', lambda') of the doubled sequence", (bj, lj), ([1, 2, 2, 2, 2, 2], [1, 1, 1, 1, 1]))
g4 = power_moments(V(3), 25, ga_functional)                      # a symmetric test law: symmetrize J_4
sym = [g4[n] if n % 2 == 0 else 0 for n in range(25)]
b1, l1 = jacobi(sym, 9); b2, l2 = jacobi(sym[0::2], 4)
check("beta'_n = lambda_2n + lambda_2n+1 and lambda'_n = lambda_2n-1 lambda_2n (test sequence)",
      (b2, l2), ([l1[0]] + [l1[2 * n - 1] + l1[2 * n] for n in (1, 2, 3)], [l1[2 * n - 2] * l1[2 * n - 1] for n in (1, 2, 3)]))
for l in (1, 2, 3):
    A = [[1 if abs(i - j) == 1 else 0 for j in range(2 * l + 2)] for i in range(2 * l + 2)]
    dx = [matpow_entry(A, 2 * k) for k in range(2 * l + 3)]
    check(f"Verlinde level {2 * l + 1}: tau(X_1^n) = sum (-1)^(n-k) C(n,k) d_k(x)",
          [sum((-1) ** (n - k) * math.comb(n, k) * dx[k] for k in range(n + 1)) for n in range(2 * l + 3)], rows[l])

log("-" * 70); log("Exact Hankel diagnostics, including singular pivots")
check("zero diagonal with a nonzero off-diagonal entry", inertia([[0, 1], [1, 0]]), (1, 1, 0))
check("zero residual row followed by a negative direction", inertia([[1, 0, 0], [0, 0, 0], [0, 0, -1]]), (1, 1, 1))
check("singular positive-semidefinite matrix", inertia([[1, 1], [1, 1]]), (1, 0, 1))
check("Taft zero-pivot regression, inertia of H_4", hankel_inertia([1, 0, 1, 1, 2, 4, 7]), (3, 1, 0))
check("Taft zero-pivot regression, first non-PSD Hankel size", first_non_psd_size([1, 0, 1, 1, 2, 4, 7]), 4)

log("-" * 70); log("Quaternion exterior bosonization, dimension 32")
# Rep(Q_8) has four invertible basis elements forming C_2 x C_2 and a
# two-dimensional V satisfying V^2=1+a+b+c. The unit cover has class 2+V.
N_V = [[0] * 5 for _ in range(5)]
for i in range(4):
    N_V[i][4] = N_V[4][i] = 1
C_Q8 = [[N_V[i][j] + 2 * (i == j) for j in range(5)] for i in range(5)]
q8_moments = [1] + [matpow_entry(C_Q8, n) for n in range(8)]
check("Q_8 unit cover, c_0..c_8", q8_moments, [1, 1, 2, 5, 14, 44, 152, 560, 2144])
check("Q_8 Cartan spectral law (delta_4+delta_0+6 delta_2)/8",
      [matpow_entry(C_Q8, n) for n in range(9)],
      [Q(4 ** n + (n == 0) + 6 * 2 ** n, 8) for n in range(9)])
check("Q_8 unit cover, leading Hankel determinants sizes 1..4", hankel_dets(q8_moments, 4), [1, 1, 1, -3])
check("Q_8 unit cover, first non-PSD Hankel size", first_non_psd_size(q8_moments), 4)
check("Q_8 doubled law 19/32 delta_0+3/8 delta_4+1/32 delta_16",
      q8_moments[::2],
      [Q(19, 32) * (n == 0) + Q(3, 8) * 4 ** n + Q(1, 32) * 16 ** n for n in range(5)])

log("-" * 70); log("G_a reconstruction from even tensor degrees")
torus_functional = lambda polynomial: polynomial.get(0, 0)
check("J_3 full moments equal those of the torus weights -2,0,2",
      power_moments(V(2), 13, ga_functional), power_moments(V(2), 13, torus_functional))
check("even moments distinguish J_1+J_5 and 2 J_2 beyond their common c_2",
      (power_moments(add(V(0), V(4)), 5, ga_functional)[2::2],
       power_moments(add(V(1), V(1)), 5, ga_functional)[2::2]),
      ([8, 196], [8, 96]))

def partitions(total, minimum=1):
    if total == 0:
        yield ()
    for first in range(minimum, total + 1):
        for rest in partitions(total - first, first):
            yield (first,) + rest

jordan_types = [blocks for dimension in range(11) for blocks in partitions(dimension)]
even_signatures = {
    tuple(power_moments(add(*(V(size - 1) for size in blocks)), 13, ga_functional)[::2])
    for blocks in jordan_types
}
check("all Jordan types of dimension at most 10 distinguished by c_0,c_2,...,c_12",
      (len(jordan_types), len(even_signatures)), (139, 139))

log("-" * 70); log("Census collision invariant")
data_path = Path(__file__).parent / "data" / "census-tables.json.gz"
tables = {record["id"]: record["table"] for record in json.loads(gzip.decompress(data_path.read_bytes()))}
cubic_expected = {"r9:L1923": 100, "r9:L2051": 116, "r9:L2075": 164,
                  "r9:L1936": 163, "r9:L2080": 227}
for ring_id, expected in cubic_expected.items():
    table = tables[ring_id]
    rank_ = len(table)
    total_matrix = [[sum(table[i][j][k] for i in range(rank_))
                     for j in range(rank_)] for k in range(rank_)]
    check(f"{ring_id}, tau((sum of all basis elements)^3)", matpow_entry(total_matrix, 3), expected)

log("=" * 70)
log(f"Executed {CHECKS} checks; {CHECKS - len(FAILS)} passed; {len(FAILS)} failed")
log("ALL CHECKS PASSED" if not FAILS else "FAILED: " + ", ".join(FAILS))
OUT.close()
sys.exit(1 if FAILS else 0)
