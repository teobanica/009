#!/usr/bin/env python3
"""
Numerical Nichols component dimensions for the even Verlinde ring of rank 7. The fusion category
D categorifying the ring K_6 is the even part of SU(2)_13, with simples X_a of spin a,
0 <= a <= 6, and its Drinfeld center is D [x] D^rev. We study X_5 in D and
X_3 [x] X_6 in the center. Both Nichols algebras are infinite-dimensional by
the central Hopf-module argument in the manuscript.

The program builds the 6j symbols and the R-matrices of SU(2)_13 (q = exp(i pi/15)),
checks the orthogonality of the F-matrices and the braid relations numerically, and
estimates the ranks in each graded piece B^n(V) using the quantum symmetrizers
S_n = (S_(n-1) (x) 1)(1 + c_(n-1) + c_(n-1)c_(n-2) + ... + c_(n-1)...c_1) on the
multiplicity spaces Hom(Z, V^(x)n), in the fusion tree basis.  It reports numerical multiplicity estimates
and their corresponding Frobenius-Perron dimension estimates.  A test object,
the invertible simple of spin 13/2 with braiding i, gives the 4-dimensional C[x]/(x^4).
The computed norms of the quadratic invariant powers through degree 8 are compared
with a numerical tolerance. The separate verify_nichols.py certifies nonvanishing.

Run:   python3 nichols_verlinde.py        (under a minute)
Output: nichols-verlinde-output.txt.  Needs NumPy.
"""
import os, time
import numpy as np
from fractions import Fraction as Fr
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = open(os.path.join(HERE, "nichols-verlinde-output.txt"), "w")
def say(s=""):
    print(s, flush=True)
    OUT.write(s + "\n")
    OUT.flush()
from functools import lru_cache
K=13
q=np.exp(1j*np.pi/(K+2))
def qn(n): return np.sin(n*np.pi/(K+2))/np.sin(np.pi/(K+2))
@lru_cache(None)
def qf(n):
    r=1.0
    for i in range(1,int(n)+1): r*=qn(i)
    return r
def adm(a,b,c):
    # spins (Fractions), admissible triple at level K
    if (a+b+c)%1!=0: return False
    if c<abs(a-b) or c>a+b: return False
    if a+b+c>K: return False
    return True
def Delta(a,b,c):
    return np.sqrt(qf(a+b-c)*qf(a-b+c)*qf(-a+b+c)/qf(a+b+c+1))
def sixj(j1,j2,j12,j3,j,j23):
    if not (adm(j1,j2,j12) and adm(j12,j3,j) and adm(j2,j3,j23) and adm(j1,j23,j)): return 0.0
    lo=max(j1+j2+j12, j12+j3+j, j2+j3+j23, j1+j23+j)
    hi=min(j1+j2+j3+j, j1+j3+j12+j23, j2+j+j12+j23)
    s=0.0; z=lo
    while z<=hi:
        s+=(-1)**int(z)*qf(z+1)/(qf(z-j1-j2-j12)*qf(z-j12-j3-j)*qf(z-j2-j3-j23)*qf(z-j1-j23-j)
                                  *qf(j1+j2+j3+j-z)*qf(j1+j3+j12+j23-z)*qf(j2+j+j12+j23-z))
        z+=1
    return Delta(j1,j2,j12)*Delta(j12,j3,j)*Delta(j2,j3,j23)*Delta(j1,j23,j)*s
def spins(): return [Fr(n,2) for n in range(K+1)]
def fuse(a,b): return [c for c in spins() if adm(a,b,c)]
def Fmat(j1,j2,j3,j):
    rows=[x for x in fuse(j1,j2) if adm(x,j3,j)]
    cols=[y for y in fuse(j2,j3) if adm(j1,y,j)]
    M=np.array([[(-1)**int(j1+j2+j3+j)*np.sqrt(qn(2*x+1)*qn(2*y+1))*sixj(j1,j2,x,j3,j,y) for y in cols] for x in rows])
    return rows,cols,M
def R(a,b,c,inv=False):
    C=lambda j: j*(j+1)
    v=(-1)**int(a+b-c)*q**(C(c)-C(a)-C(b))
    return np.conj(v) if inv else v

def paths(v,n):
    P=[(v,)]
    for _ in range(n-1):
        P=[p+(c,) for p in P for c in fuse(p[-1],v)]
    return P
def generators(v,n,inv=False):
    P=paths(v,n); idx={p:i for i,p in enumerate(P)}; N=len(P)
    gens=[]
    cache={}
    for i in range(1,n):   # sigma_i acts on strands i, i+1 (1-based); changes a_{i+1}? see below
        S=np.zeros((N,N),complex)
        for p in P:
            col=idx[p]
            if i==1:
                S[col,col]=R(v,v,p[1],inv)
            else:
                a=p[i-2]; x=p[i-1]; b=p[i]       # a_{i-1}, a_i, a_{i+1} (0-based shift)
                key=(a,b)
                if key not in cache:
                    rows,cols,F=Fmat(a,v,v,b)
                    M=F@np.diag([R(v,v,y,inv) for y in cols])@F.T
                    cache[key]=(rows,M)
                rows,M=cache[key]
                xi=rows.index(x)
                for k,x2 in enumerate(rows):
                    if abs(M[k,xi])>1e-14:
                        S[idx[p[:i-1]+(x2,)+p[i:]],col]+=M[k,xi]
        gens.append(S)
    return P,gens

def dims(): return {s: qn(int(2*s+1)) for s in spins()}
def symmetrizer_rank(gens, n, block_idx):
    """rank of quantum symmetrizer S_n restricted to the given index block"""
    G=[g[np.ix_(block_idx,block_idx)] for g in gens]
    m=len(block_idx); I=np.eye(m,dtype=complex)
    S=I.copy()
    for k in range(2,n+1):
        T=I.copy(); P=I.copy()
        for j in range(k-1,0,-1):
            P=P@G[j-1]; T=T+P
        S=S@T
    if m==0: return 0
    sv=np.linalg.svd(S,compute_uv=False)
    return int(np.sum(sv>1e-8*max(1,sv.max())))
def nichols_single(v, nmax, inv=False):
    d=dims(); out=[]
    for n in range(1,nmax+1):
        P,g=generators(v,n,inv)
        mult={}
        for Z in spins():
            blk=[i for i,p in enumerate(P) if p[-1]==Z]
            if blk:
                r=symmetrizer_rank(g,n,blk) if n>1 else len(blk)
                if r: mult[Z]=r
        fp=sum(m*d[Z] for Z,m in mult.items())
        out.append((n,mult,fp))
        say("  degree %d   FPdim %.3f   %s"%(n,fp," ".join("%s:%d"%(Z,m) for Z,m in sorted(mult.items()))))
    return out
def nichols_pair(v1,v2,nmax):
    """V = v1 [x] v2 in D [x] D^rev"""
    d=dims()
    for n in range(1,nmax+1):
        P1,g1=generators(v1,n,False); P2,g2=generators(v2,n,True)
        mult={}
        for Z1 in spins():
            b1=[i for i,p in enumerate(P1) if p[-1]==Z1]
            if not b1: continue
            for Z2 in spins():
                b2=[i for i,p in enumerate(P2) if p[-1]==Z2]
                if not b2: continue
                G=[np.kron(x[np.ix_(b1,b1)],y[np.ix_(b2,b2)]) for x,y in zip(g1,g2)]
                m=len(b1)*len(b2)
                if n==1: r=m
                else:
                    I=np.eye(m,dtype=complex); S=I.copy()
                    for k in range(2,n+1):
                        T=I.copy(); Pm=I.copy()
                        for j in range(k-1,0,-1):
                            Pm=Pm@G[j-1]; T=T+Pm
                        S=S@T
                    sv=np.linalg.svd(S,compute_uv=False); r=int(np.sum(sv>1e-8*max(1,sv.max())))
                if r: mult[(Z1,Z2)]=r
        fp=sum(m*d[a]*d[b] for (a,b),m in mult.items())
        say("  degree %d   FPdim %.3f   %s"%(n,fp," ".join("%s|%s:%d"%(a,b,m) for (a,b),m in sorted(mult.items()))))

def qpowers(v, nmax):
    for n2 in range(2, nmax + 1, 2):
        P, g = generators(v, n2)
        blk = [i for i, p in enumerate(P) if p[-1] == 0]
        pos = {P[i]: k for k, i in enumerate(blk)}
        G = [x[np.ix_(blk, blk)] for x in g]
        m = len(blk)
        I = np.eye(m, dtype=complex)
        S = I.copy()
        for k in range(2, n2 + 1):
            T = I.copy()
            Pm = I.copy()
            for j in range(k - 1, 0, -1):
                Pm = Pm @ G[j - 1]
                T = T + Pm
            S = S @ T
        e = np.zeros(m, complex)
        e[pos[tuple([v, Fr(0)] * (n2 // 2))]] = 1
        say("  degree %d: |S(Q^%d)| = %.4g (computed norm)" % (n2, n2 // 2, np.linalg.norm(S @ e)))
        assert np.linalg.norm(S @ e) > 1e-6


if __name__ == "__main__":
    t0 = time.time()
    say("SU(2)_13, even part: numerical estimates for Nichols symmetrizer ranks and graded FP dimensions")
    bad = 0
    for j1 in spins()[:8]:
        for j2 in spins()[:8]:
            for j3 in spins()[:8]:
                for j in spins():
                    r, c, M = Fmat(j1, j2, j3, j)
                    if len(r) != len(c) or (len(r) and np.abs(M @ M.T - np.eye(len(r))).max() > 1e-9):
                        bad += 1
    assert bad == 0
    for v in [Fr(5), Fr(3), Fr(6)]:
        P, g = generators(v, 4)
        assert np.abs(g[0] @ g[1] @ g[0] - g[1] @ g[0] @ g[1]).max() < 1e-10
        assert np.abs(g[1] @ g[2] @ g[1] - g[2] @ g[1] @ g[2]).max() < 1e-10
    say("numerical checks: orthogonality and four-strand braid residuals within tolerance")
    say("test: the invertible object of spin 13/2")
    nichols_single(Fr(13, 2), 5)
    say("V = X_5, the simple of spin 5 (twist 1, FPdim %.3f)" % qn(11))
    nichols_single(Fr(5), 7)
    say("powers of the quadratic invariant Q of B(X_5)")
    qpowers(Fr(5), 8)
    say("V = X_3 [x] X_6 in D [x] D^rev (twist 1, FPdim %.3f)" % (qn(7) * qn(13)))
    nichols_pair(Fr(3), Fr(6), 5)
    say("(%.0f s)" % (time.time() - t0))
    OUT.close()
