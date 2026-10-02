"""Volume III, Paper 5, open problem (d)/(e) of Remark 8.1: star cages of nonlinear chamber games
for every thickness q (chapters/V3P5_add_generalq.tex).

  (B1) balls of the Bruhat-Tits building of PGL_3(F_q((s))) for q = 2, 3, 4, 5, built from lattice
       classes s^m L_0 < L < L_0 (submodules of (F_q[s]/s^m)^3, normalized by L not< s L_0);
       sphere sizes 1, 2k, k q (3q+1) (and 560 for q = 2, radius 3); links = PG(2,q) incidence
       graphs at the centre and at every radius-1 vertex; buildings are flag complexes there;
  (B2) common neighbours of two vertices at distance 2: exactly 2 (bent, same type) or 1 (straight);
  (S1) chamber counts around a star: boundary vertex (q+1, q(q+1), q^2(q+1)); most dangerous
       defector (1, 2q); star threshold by brute force = closed form r_star(q) for q = 2, 3, 4, 5,
       and q = 1 on the thin torus T_12;
  (S2) closed forms 3k/(q+1)^2, 3k(q+1)/(2q^2+5q+1), 3k(q+1)/(q^3+5q^2+11q+5); values at q = 1, 2;
       monotonicity for q >= 1 (sympy certificate); limits 3, infinity, 3;
  (R2) radius-2 ball for q = 2 and 3: spreading thresholds from the anchored ball and stability
       of the unanchored ball (q = 2: 21/10, 3, 21/20 and 7/3, 63/19, 63/55 as in Prop. 7.4);
  (W)  half-apartment (Weyl wall) in the thick building: counts 3, 4q-2, q^3+2q^2-2q at a wall
       vertex, worst defector (1, 2q), threshold r_wall(q) = 3k(q+1)/(3phi3+(4q-3)phi2+(q^3+2q^2-4q)phi1),
       equal to the star value at q = 2 and above it for q >= 3; far-point lonely counts;
  (MC) Fermi dynamics, K = 0.1, threshold game: transition window on random D_2 = 52 hypergraphs
       (q = 3 proxy) with n = 2000 and 8000; coexistence (interfaces) on Y_768 (level 4 of the
       canonical 2-tower), on the random hypergraph and on the thin torus T_24; survival of
       cooperator balls / half-tori after 200 sweeps.

Run from code/:  python v3p5_nonlinear_general_q.py      (numpy, sympy; a few minutes)
"""
import itertools
import math
import sys
import time
from fractions import Fraction

import numpy as np
import sympy as sp

from a2_common import GIDX, T, thin_torus

t0 = time.time()
PHI = {"linear": [0, 1, 2, 3], "threshold": [0, 0, 2, 3], "quadratic": [0, 1, 4, 9]}
KAP = lambda q: q * q + q + 1


def banner(tag, msg):
    print(f"({tag}) {msg}", flush=True)


# ================================================================ finite fields F_q, q = 2,3,4,5
class GF:
    """Tables for F_q (prime q, or q = 4 with x^2 + x + 1)."""

    def __init__(self, q):
        self.q = q
        if q == 4:
            def mul(a, b):
                r = 0
                for i in range(2):
                    if (b >> i) & 1:
                        r ^= a << i
                if r & 4:
                    r ^= 0b111
                return r
            self.ADD = np.array([[a ^ b for b in range(4)] for a in range(4)], np.int64)
            self.MUL = np.array([[mul(a, b) for b in range(4)] for a in range(4)], np.int64)
            self.NEG = np.arange(4)
        else:
            self.ADD = np.array([[(a + b) % q for b in range(q)] for a in range(q)], np.int64)
            self.MUL = np.array([[(a * b) % q for b in range(q)] for a in range(q)], np.int64)
            self.NEG = np.array([(-a) % q for a in range(q)], np.int64)
        self.INV = np.array([0] + [next(b for b in range(1, q) if self.MUL[a, b] == 1)
                                   for a in range(1, q)], np.int64)

    def rref(self, M):
        """Reduced row echelon form (nonzero rows only) of an integer array over F_q."""
        M = np.array(M, np.int64).reshape(-1, M.shape[-1]) if len(M) else np.zeros((0, 0), np.int64)
        rows, cols = M.shape
        r = 0
        for c in range(cols):
            if r == rows:
                break
            piv = np.nonzero(M[r:, c])[0]
            if not len(piv):
                continue
            p = r + piv[0]
            if p != r:
                M[[r, p]] = M[[p, r]]
            M[r] = self.MUL[self.INV[M[r, c]], M[r]]
            for i in range(rows):
                if i != r and M[i, c]:
                    M[i] = self.ADD[M[i], self.MUL[self.NEG[M[i, c]], M[r]]]
            r += 1
        return M[:r]

    def reduce(self, v, E):
        """Residual of v modulo the row space of the RREF E."""
        v = np.array(v, np.int64)
        for row in E:
            c = np.nonzero(row)[0][0]
            if v[c]:
                v = self.ADD[v, self.MUL[self.NEG[v[c]], row]]
        return v

    def nullspace(self, M):
        """Basis of {x : M x = 0} (M with columns = unknowns)."""
        M = np.array(M, np.int64)
        n = M.shape[1]
        E = self.rref(M)
        pivs = [np.nonzero(row)[0][0] for row in E]
        free = [c for c in range(n) if c not in pivs]
        basis = []
        for f in free:
            x = np.zeros(n, np.int64)
            x[f] = 1
            for row, pc in zip(E, pivs):
                x[pc] = self.NEG[row[f]]
            basis.append(x)
        return np.array(basis, np.int64).reshape(-1, n)

    def proj_points(self, k):
        """Canonical representatives of the points of P^{k-1}(F_q) (first nonzero entry 1)."""
        pts = []
        for i in range(k):
            for tail in itertools.product(range(self.q), repeat=k - i - 1):
                pts.append(np.array([0] * i + [1] + list(tail), np.int64))
        return pts

    def subspaces(self, k, dims=(1, 2)):
        """Bases (coefficient rows) of the subspaces of F_q^k of the given dimensions, k <= 3."""
        out = []
        if 1 in dims:
            out += [p.reshape(1, k) for p in self.proj_points(k)]
        if 2 in dims and k >= 2:
            if k == 2:
                out.append(np.eye(2, dtype=np.int64))
            else:
                for nrm in self.proj_points(k):              # hyperplane normal vectors
                    i = np.nonzero(nrm)[0][0]
                    rows = []
                    for j in range(k):
                        if j != i:
                            e = np.zeros(k, np.int64)
                            e[j] = 1
                            e[i] = self.NEG[nrm[j]]
                            rows.append(e)
                    out.append(np.array(rows, np.int64))
        return out


# ================================================================ lattice-class balls of the building
class BuildingBall:
    """Vertices at distance <= R from L_0 in the A~_2 building of order q, as lattice classes.

    Coordinates: F_q^{3m}, m = R + 1, index 3*k + i <-> coefficient of s^k e_i. A lattice
    s^m L_0 < L < L_0 is an F_q[s]/s^m-submodule of F_q^{3m}; the class representative is
    normalized by L not< s L_0 (minimal elementary divisor 0); then graph distance = maximal
    elementary divisor.  Adjacency: s L < L' < L or s L' < L < L'.  Chambers at u (u at
    distance <= R - 1) are the flags line < plane of L_u / s L_u.
    """

    def __init__(self, q, R):
        self.q, self.R, self.m = q, R, R + 1
        self.F = GF(q)
        self.N = 3 * self.m
        self.kap = KAP(q)
        m = self.m
        self.shift = np.zeros((self.N, self.N), np.int64)           # x -> s x
        for k in range(m - 1):
            for i in range(3):
                self.shift[3 * k + i, 3 * (k + 1) + i] = 1
        self.keys, self.ids, self.basis, self.dist = {}, [], [], []
        self.nbrs, self.chambers = {}, {}
        self.build()

    # ---- lattice primitives
    def key(self, E):
        return E.shape[0], E.tobytes()

    def apply_s(self, E):
        if not len(E):
            return E
        return self.F.rref((E @ self.shift) % self.q) if self.q != 4 else self.F.rref(self.shift_q4(E))

    def shift_q4(self, E):
        return E @ self.shift                                        # 0/1 matrix: no field arithmetic needed

    def preimage_s(self, E):
        """{x : s x in L}."""
        F = self.F
        res = []
        for j in range(self.N):
            e = np.zeros(self.N, np.int64)
            e[j] = 1
            res.append(F.reduce(e @ self.shift, E))
        return F.rref(F.nullspace(np.array(res, np.int64).T))

    def complement(self, Esmall, Ebig):
        """Vectors Q with L_big = L_small + span(Q) (RREF of the residuals)."""
        res = np.array([self.F.reduce(v, Esmall) for v in Ebig], np.int64).reshape(-1, self.N)
        return self.F.rref(res)

    def span_with(self, E, Q, coeffs):
        """RREF of L + span(coeffs @ Q)."""
        add = (coeffs @ Q)
        if self.q != 4:
            add = add % self.q
        else:
            add = self.combine_q4(coeffs, Q)
        return self.F.rref(np.vstack([E, add]) if len(E) else add)

    def combine_q4(self, coeffs, Q):
        F = self.F
        out = np.zeros((coeffs.shape[0], Q.shape[1]), np.int64)
        for a in range(coeffs.shape[0]):
            v = np.zeros(Q.shape[1], np.int64)
            for b in range(Q.shape[0]):
                if coeffs[a, b]:
                    v = F.ADD[v, F.MUL[coeffs[a, b], Q[b]]]
            out[a] = v
        return out

    def in_sL0(self, E):
        return not E[:, :3].any()

    def normalize(self, E):
        while self.in_sL0(E):
            E = self.preimage_s(E)
        return E

    def contains(self, Ebig, Esmall):
        return all(not self.F.reduce(v, Ebig).any() for v in Esmall)

    def adjacent(self, a, b):
        Ea, Eb = self.basis[a], self.basis[b]
        if a == b:
            return False
        if Ea.shape[0] > Eb.shape[0]:
            return self.contains(Ea, Eb) and self.contains(Eb, self.apply_s(Ea))
        return self.contains(Eb, Ea) and self.contains(Ea, self.apply_s(Eb))

    def vid(self, E):
        k = self.key(E)
        if k not in self.keys:
            self.keys[k] = len(self.ids)
            self.ids.append(k)
            self.basis.append(E)
            self.dist.append(None)
        return self.keys[k]

    def vtype(self, v):
        return (-self.basis[v].shape[0]) % 3

    # ---- neighbours and chambers of a normalized lattice at distance <= R - 1
    def neighbours_of(self, v):
        E = self.basis[v]
        F = self.F
        sE = self.apply_s(E)
        Q = self.complement(sE, E)
        assert Q.shape[0] == 3
        out = set()
        for S in F.subspaces(3):
            out.add(self.vid(self.normalize(self.span_with(sE, Q, S))))
        P = self.preimage_s(E)
        Q2 = self.complement(E, P)
        for S in F.subspaces(Q2.shape[0]):
            out.add(self.vid(self.span_with(E, Q2, S)))
        out.discard(v)
        return out

    def chambers_of(self, v):
        E = self.basis[v]
        sE = self.apply_s(E)
        Q = self.complement(sE, E)
        planes = self.F.subspaces(3, dims=(2,))
        lines = self.F.subspaces(3, dims=(1,))
        pid = {}
        for H in planes:
            pid[H.tobytes()] = self.vid(self.normalize(self.span_with(sE, Q, H)))
        lid = {l.tobytes(): self.vid(self.normalize(self.span_with(sE, Q, l))) for l in lines}
        ch = set()
        for H in planes:
            EH = self.F.rref(H)
            for l in lines:
                if not self.F.reduce(l[0], EH).any():
                    ch.add(frozenset((v, pid[H.tobytes()], lid[l.tobytes()])))
        return ch

    def build(self):
        E0 = np.eye(self.N, dtype=np.int64)
        c = self.vid(E0)
        self.dist[c] = 0
        frontier = [c]
        for d in range(self.R):
            new = []
            for v in frontier:
                nb = self.neighbours_of(v)
                assert len(nb) == 2 * self.kap, (self.q, v, len(nb))
                self.nbrs[v] = nb
                for w in nb:
                    if self.dist[w] is None:
                        self.dist[w] = d + 1
                        new.append(w)
            frontier = new
        self.sphere = [sum(1 for d in self.dist if d == r) for r in range(self.R + 1)]
        self.n = len(self.ids)
        self.frontier = frontier
        for v in range(self.n):
            if self.dist[v] <= self.R - 1:
                ch = self.chambers_of(v)
                assert len(ch) == self.kap * (self.q + 1)
                self.chambers[v] = ch

    def chambers_meeting(self, v):
        """All chambers containing v, for v at distance R as well (through its inner neighbours)."""
        if v in self.chambers:
            return self.chambers[v]
        ch = set()
        for u in self.chambers:
            for t in self.chambers[u]:
                if v in t:
                    ch.add(t)
        return ch


def check_pg_link(ball, v):
    """The link of v (its 2 kappa neighbours with chamber adjacency) is the PG(2,q) incidence graph."""
    q, kap = ball.q, ball.kap
    nb = sorted(ball.nbrs[v])
    link = {u: set() for u in nb}
    for t in ball.chambers[v]:
        a, b = [x for x in t if x != v]
        link[a].add(b)
        link[b].add(a)
    types = {ball.vtype(u) for u in nb}
    assert len(types) == 2 and all(ball.vtype(u) != ball.vtype(v) for u in nb)
    side = {t: [u for u in nb if ball.vtype(u) == t] for t in types}
    assert all(len(side[t]) == kap for t in types)
    assert all(len(link[u]) == q + 1 for u in nb)
    assert all(ball.vtype(w) != ball.vtype(u) for u in nb for w in link[u])        # bipartite
    for t in types:                                                               # projective plane axioms
        for a, b in itertools.combinations(side[t], 2):
            assert len(link[a] & link[b]) == 1
    for u in nb:                                                                  # flag complex: link edges are edges
        for w in link[u]:
            assert ball.adjacent(u, w)
    return True


# ---------------------------------------------------------------- (B1) build the balls
BALLS = {}
for q, R in ((2, 3), (3, 3), (4, 2), (5, 2)):
    tb = time.time()
    B = BuildingBall(q, R)
    BALLS[q] = B
    kap = KAP(q)
    assert B.sphere[:3] == [1, 2 * kap, kap * q * (3 * q + 1)], (q, B.sphere)
    if q == 2:
        assert B.sphere[3] == 560
    check_pg_link(B, 0)
    for v in range(B.n):
        if B.dist[v] == 1:
            check_pg_link(B, v)
    # neighbour sets are symmetric where both ends are expanded
    for v in B.nbrs:
        for w in B.nbrs[v]:
            if w in B.nbrs:
                assert v in B.nbrs[w]
    banner("B1", f"q={q}: spheres {B.sphere} (1, 2k, kq(3q+1)); link = PG(2,{q}) incidence graph "
                 f"at the centre and at the {2 * kap} radius-1 vertices; flag complex   [{time.time() - tb:.1f} s]")
