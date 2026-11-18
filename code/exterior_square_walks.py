#!/usr/bin/env python3
"""
The doubled exterior square.  Let X = Lambda^2(C^4), restricted to the
stabilizer C^3 x| SL_3 of a vector in SL_4.  By the highest-weight vector stabilizer formula, c_n(X) is the sum of
the multiplicities of the modules Sym^k(C^4) in X^{(x)n}.  Through SL_4 -> SO_6, X is the
vector module of SO_6, with weights +-e_1, +-e_2, +-e_3, and Sym^k(C^4) has highest weight
(k/2)(1,1,1).  Only k = 2j occurs, and the Brauer-Klimyk rule gives

    c_n(X) = number of walks of length n in Z^3, with steps +-e_i, staying in the chamber
             x_1 >= x_2 >= |x_3|, from 0 to a point (j, j, j).

The program
  1. counts these walks exactly through n = 172, modulo 9 primes below 2^59 with NumPy,
     and recovers the integers by the Chinese remainder theorem;
  2. compares them with the 81 values of data/lambda2-80.json (Pieri rule, doubling.py);
  3. tests a proposed order-six recurrence, with degree-five polynomial coefficients,
     for the finite range of starting indices 0 through 166;
  4. computes exactly, by fraction-free elimination, the leading Hankel determinants of
     the doubled sequence d_k = c_2k(X) and of the shifted sequence d_(k+1), through
     size 43: the first 42 are positive, and the one of size 43 is negative.
So X (x) X has neither a Hamburger nor a Stieltjes law.

Run:   python3 exterior_square_walks.py        (about 10 minutes, 300 MB of memory)
Output: exterior-square-walks-output.txt, and the values in data/lambda2-172.json.
Needs Python 3.8 or later and NumPy.
"""
import os, json, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = open(os.path.join(HERE, "exterior-square-walks-output.txt"), "w")
NMAX = 172
SIZE = 43
RECURRENCE = [   # p_i(n) = sum_d RECURRENCE[i][d] n^d, and sum_i p_i(n) c_(n+i) = 0
    [11546924544, 25977884544, 20842997760, 7613831520, 1281868416, 80074656],
    [21162788352, 30215327616, 16482476160, 4319360640, 547473168, 27085104],
    [-33024257280, -40120265088, -19062748368, -4421069136, -500170752, -22111776],
    [-37216976832, -35066930544, -12974270976, -2360117736, -211774032, -7523640],
    [66138156, 1193656690, 705289872, 154154716, 14718372, 519634],
    [4883582160, 3352330114, 893936405, 116535875, 7459195, 188091],
    [460003500, 193949600, 31458315, 2404289, 82905, 911],
]


def say(s=""):
    print(s, flush=True)
    OUT.write(s + "\n")
    OUT.flush()


def is_prime(n):
    if n < 2 or n % 2 == 0:
        return n == 2
    i = 3
    while i * i <= n:
        if n % i == 0:
            return False
        i += 2
    return True


def primes_below(b, k):
    out, n = [], b - 1
    while len(out) < k:
        if is_prime(n):
            out.append(n)
        n -= 1
    return out


def walk_counts_mod(p, nmax):
    S = nmax + 2
    off = S
    i1, i2, i3 = np.meshgrid(np.arange(S), np.arange(S), np.arange(2 * S + 1) - off, indexing="ij")
    outside = ~((i1 >= i2) & (i2 >= np.abs(i3)))
    A = np.zeros((S, S, 2 * S + 1), dtype=np.int64)
    A[0, 0, off] = 1
    vals = [1]
    for n in range(1, nmax + 1):
        B = np.zeros_like(A)
        B[1:] += A[:-1]
        B[:-1] += A[1:]
        B[:, 1:] += A[:, :-1]
        B[:, :-1] += A[:, 1:]
        B[:, :, 1:] += A[:, :, :-1]
        B[:, :, :-1] += A[:, :, 1:]
        B %= p
        B[outside] = 0
        A = B
        vals.append(sum(int(A[j, j, off + j]) for j in range(n // 3 + 1)) % p)
    return vals


def leading_minors(mu, size):
    """Leading principal minors of the Hankel matrix (mu_(i+j)), by Bareiss elimination."""
    A = [[mu[i + j] for j in range(size)] for i in range(size)]
    dets, prev = [], 1
    for k in range(size):
        piv = A[k][k]
        dets.append(piv)
        if piv == 0:
            break
        for i in range(k + 1, size):
            for j in range(k + 1, size):
                A[i][j] = (A[i][j] * piv - A[i][k] * A[k][j]) // prev
        prev = piv
    return dets


if __name__ == "__main__":
    t0 = time.time()
    say("Lambda^2(C^4) on C^3 x| SL_3: walks in the Weyl chamber of SO_6")
    ps = primes_below(2 ** 59, 9)
    residues = []
    for p in ps:
        residues.append(walk_counts_mod(p, NMAX))
        say("  counted modulo %d   (%.0f s)" % (p, time.time() - t0))
    M = 1
    for p in ps:
        M *= p
    # There are at most 6**n unrestricted walks of length n. The residue
    # modulo M therefore determines the actual count at every tested length.
    assert 6**NMAX < M
    c = []
    for n in range(NMAX + 1):
        x = 0
        for p, r in zip(ps, residues):
            Mi = M // p
            x += r[n] * Mi * pow(Mi, -1, p)
        c.append(x % M)
    assert all(value <= 6**n for n, value in enumerate(c))
    say("  c_0, ..., c_12 = %s" % c[:13])
    seb = [int(x) for x in json.load(open(os.path.join(HERE, "data", "lambda2-80.json")))]
    assert seb == c[:81]
    say("  agrees with the 81 values of data/lambda2-80.json")
    R = len(RECURRENCE) - 1
    for n in range(NMAX + 1 - R):
        assert sum(sum(a * n ** d for d, a in enumerate(RECURRENCE[i])) * c[n + i]
                   for i in range(R + 1)) == 0
    say("  the proposed order-six recurrence matches the data for 0 <= n <= %d" % (NMAX - R))
    json.dump([str(x) for x in c], open(os.path.join(HERE, "data", "lambda2-172.json"), "w"))
    say("  values written to data/lambda2-172.json")
    d = [c[2 * k] for k in range(NMAX // 2 + 1)]
    for name, mu in [("doubled sequence d_k = c_2k", d), ("shifted sequence d_(k+1)", d[1:])]:
        D = leading_minors(mu, SIZE)
        signs = "".join("+" if x > 0 else "-" if x < 0 else "0" for x in D)
        say("  %s: signs of the Hankel determinants of sizes 1..%d" % (name, SIZE))
        say("      %s" % signs)
        say("      the determinant of size %d is negative, with %d digits" % (SIZE, len(str(-D[-1]))))
        assert signs == "+" * (SIZE - 1) + "-"
    say("so X (x) X has neither a Hamburger law nor a Stieltjes law   (%.0f s)" % (time.time() - t0))
    say("all assertions passed")
    OUT.close()
