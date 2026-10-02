"""Shared constructions for the Volume III verification scripts (v3p2_verify.py onward).

Builds the abelian voltage covers of the three-vertex quotient X_3 of the CMSZ group of the
triangle presentation of Vol. I-II, Ch. X, Construction 1.4, and the thin tori of
Construction 1.3, together with exact integer linear algebra (Smith invariants, Bareiss
determinants, saturated integer kernels).  v3p1_verify.py is kept self-contained and
does not import this module.
"""
import math
from fractions import Fraction

import numpy as np

q = 2
KAPPA = q * q + q + 1

ORBITS = [(0, 1, 3), (0, 2, 6), (0, 4, 5), (1, 2, 4), (1, 5, 6), (2, 3, 5), (3, 4, 6)]
T = sorted({t for (x, y, z) in ORBITS for t in [(x, y, z), (y, z, x), (z, x, y)]})
GIDX = {(s, x): 3 * x + s for s in range(3) for x in range(7)}


def relator_matrix():
    rels = []
    for s in range(3):
        for (x, y, z) in T:
            r = [0] * 21
            r[GIDX[(s, x)]] += 1
            r[GIDX[((s + 1) % 3, y)]] += 1
            r[GIDX[((s + 2) % 3, z)]] += 1
            rels.append(r)
    for triv in [(0, 0), (1, 0)]:
        r = [0] * 21
        r[GIDX[triv]] = 1
        rels.append(r)
    return rels


# ------------------------------------------------------------------ exact linear algebra
def invariant_factors(M):
    """Nonzero invariant factors of an integer matrix."""
    A = [list(map(int, row)) for row in M]
    m, n = len(A), len(A[0])
    inv, r = [], 0
    for c in range(n):
        if r >= m:
            break
        while True:
            piv = None
            for i in range(r, m):
                for j in range(c, n):
                    if A[i][j] and (piv is None or abs(A[i][j]) < abs(A[piv[0]][piv[1]])):
                        piv = (i, j)
            if piv is None:
                return inv
            i, j = piv
            A[r], A[i] = A[i], A[r]
            for row in A:
                row[c], row[j] = row[j], row[c]
            p = A[r][c]
            clean = True
            for i in range(r + 1, m):
                f = A[i][c] // p
                if f:
                    A[i] = [a - f * b for a, b in zip(A[i], A[r])]
                clean &= A[i][c] == 0
            for j in range(c + 1, n):
                f = A[r][j] // p
                if f:
                    for row in A:
                        row[j] -= f * row[c]
                clean &= A[r][j] == 0
            if not clean:
                continue
            bad = next((i for i in range(r + 1, m)
                        if any(A[i][j] % p for j in range(c + 1, n))), None)
            if bad is not None:
                A[r] = [a + b for a, b in zip(A[r], A[bad])]
                continue
            inv.append(abs(p))
            r += 1
            break
    return inv


def torsion(M):
    return [d for d in invariant_factors(M) if d > 1]


def bareiss_det(M):
    M = [list(map(int, row)) for row in M]
    m, sign, prev = len(M), 1, 1
    if m == 0:
        return 1
    for i in range(m - 1):
        if M[i][i] == 0:
            sw = next((r for r in range(i + 1, m) if M[r][i]), None)
            if sw is None:
                return 0
            M[i], M[sw] = M[sw], M[i]
            sign = -sign
        for r in range(i + 1, m):
            for c in range(i + 1, m):
                M[r][c] = (M[r][c] * M[i][i] - M[r][i] * M[i][c]) // prev
        prev = M[i][i]
    return sign * M[m - 1][m - 1]


def integer_kernel(M):
    """Basis (list of integer vectors) of the saturated lattice ker_Z M, by unimodular
    column reduction of [M ; I]."""
    rows, cols = len(M), len(M[0])
    C = [[int(M[i][j]) for i in range(rows)] + [int(i == j) for i in range(cols)]
         for j in range(cols)]                     # columns, each = (M col ; identity col)
    r = 0
    for i in range(rows):
        while True:
            nz = [j for j in range(r, cols) if C[j][i]]
            if not nz:
                break
            p = min(nz, key=lambda j: abs(C[j][i]))
            C[r], C[p] = C[p], C[r]
            done = True
            for j in range(r + 1, cols):
                if C[j][i]:
                    f = C[j][i] // C[r][i]
                    C[j] = [a - f * b for a, b in zip(C[j], C[r])]
                    done &= C[j][i] == 0
            if done:
                r += 1
                break
    return [c[rows:] for c in C[r:]]


def rational_solve(M, b):
    """Exact solution of M x = b over Q (M square nonsingular)."""
    n = len(M)
    A = [[Fraction(int(v)) for v in row] + [Fraction(int(bb))] for row, bb in zip(M, b)]
    for c in range(n):
        p = next(r for r in range(c, n) if A[r][c] != 0)
        A[c], A[p] = A[p], A[c]
        pv = A[c][c]
        A[c] = [v / pv for v in A[c]]
        for r in range(n):
            if r != c and A[r][c] != 0:
                f = A[r][c]
                A[r] = [a - f * b for a, b in zip(A[r], A[c])]
    return [A[r][n] for r in range(n)]


def nullspace_mod(M, p):
    A = [[a % p for a in row] for row in M]
    n, piv, r = len(A[0]), [], 0
    for c in range(n):
        pr = next((i for i in range(r, len(A)) if A[i][c]), None)
        if pr is None:
            continue
        A[r], A[pr] = A[pr], A[r]
        iv = pow(A[r][c], -1, p)
        A[r] = [a * iv % p for a in A[r]]
        for i in range(len(A)):
            if i != r and A[i][c]:
                f = A[i][c]
                A[i] = [(a - f * b) % p for a, b in zip(A[i], A[r])]
        piv.append(c)
        r += 1
    basis = []
    for fc in (c for c in range(n) if c not in piv):
        v = [0] * n
        v[fc] = 1
        for i, pc in enumerate(piv):
            v[pc] = (-A[i][fc]) % p
        basis.append(v)
    return basis


# ------------------------------------------------------------------ complexes
class Complex:
    """Typed 2-complex: vertices 0..n-1 with type tau, type-increasing adjacency A1,
    edges oriented type t -> t+1, triangles (a,b,c) with a->b->c->a."""

    def __init__(self, name, n, tau, A1, tris, q_order):
        self.name, self.n, self.tau, self.A1, self.q = name, n, tau, A1, q_order
        self.edges = [(i, j) for i in range(n) for j in range(n) if A1[i, j]]
        self.eidx = {e: k for k, e in enumerate(self.edges)}
        self.tris = []
        for t in tris:
            a = next(w for w in t if sum(A1[w, u] for u in t) == 1 and
                     sum(A1[u, w] for u in t) == 1)
            b = next(u for u in t if A1[a, u])
            c = next(u for u in t if u not in (a, b))
            self.tris.append((a, b, c))
        E, F = len(self.edges), len(self.tris)
        self.d1 = np.zeros((n, E), dtype=np.int64)
        for (a, b), k in self.eidx.items():
            self.d1[a, k] -= 1
            self.d1[b, k] += 1
        self.d2 = np.zeros((E, F), dtype=np.int64)
        for fi, (a, b, c) in enumerate(self.tris):
            for (x, y) in [(a, b), (b, c), (c, a)]:
                if (x, y) in self.eidx:
                    self.d2[self.eidx[(x, y)], fi] += 1
                else:
                    self.d2[self.eidx[(y, x)], fi] -= 1
        assert not (self.d1 @ self.d2).any()
        self.L0 = self.d1 @ self.d1.T
        self.L1 = self.d1.T @ self.d1 + self.d2 @ self.d2.T
        self.U1 = self.d2 @ self.d2.T
        self.L2 = self.d2.T @ self.d2
        A2 = A1.T
        self.DH = (q_order ** 3 - 1) * np.eye(n, dtype=np.int64) + A1 - q_order * A2


def voltage_cover(name, theta, mods):
    add = lambda a, b: tuple((u + v) % m for u, v, m in zip(a, b, mods))
    zero = tuple([0] * len(mods))
    elems, frontier = {zero}, [zero]
    while frontier:
        new = []
        for a in frontier:
            for g in theta:
                b = add(a, g)
                if b not in elems:
                    elems.add(b)
                    new.append(b)
        frontier = new
    verts = [(a, s) for a in sorted(elems) for s in range(3)]
    vid = {v: i for i, v in enumerate(verts)}
    step = lambda v, x: (add(v[0], theta[GIDX[(v[1], x)]]), (v[1] + 1) % 3)
    n = len(verts)
    A1 = np.zeros((n, n), dtype=np.int64)
    tris = set()
    for v in verts:
        for x in range(7):
            A1[vid[v], vid[step(v, x)]] += 1
        for (x, y, z) in T:
            v1 = step(v, x)
            v2 = step(v1, y)
            assert step(v2, z) == v
            tris.add(frozenset((vid[v], vid[v1], vid[v2])))
    assert A1.max() == 1 and not np.diag(A1).any() and not (A1 * A1.T).any()
    return Complex(name, n, [v[1] for v in verts], A1, tris, q)


def thin_torus(m):
    """Construction 1.3 of Vol. I-II, Ch. X: the A2 Coxeter complex mod m Z^2 (3 | m)."""
    assert m % 3 == 0 and m >= 6
    idx = lambda a, b: (a % m) * m + (b % m)
    n = m * m
    A1 = np.zeros((n, n), dtype=np.int64)
    ds = [(1, 0), (0, 1), (-1, -1)]
    tris = set()
    for a in range(m):
        for b in range(m):
            for (da, db) in ds:
                A1[idx(a, b), idx(a + da, b + db)] = 1
            for i in range(3):
                for j in range(3):
                    if i != j:
                        (p1, p2), (r1, r2) = ds[i], ds[j]
                        tris.add(frozenset((idx(a, b), idx(a + p1, b + p2),
                                            idx(a + p1 + r1, b + p2 + r2))))
    tau = [(v // m + v % m) % 3 for v in range(n)]
    return Complex(f"T_{m}", n, tau, A1, tris, 1)


def standard_covers():
    """Y_21, Y_24, one Y_42, one Y_84 and Y_168 (the maximal abelian cover)."""
    R = relator_matrix()
    N7, N2 = nullspace_mod(R, 7), nullspace_mod(R, 2)
    th7 = [(v,) for v in N7[0]]
    f2 = lambda k: [tuple(N2[j][i] for j in range(k)) for i in range(21)]
    out = {
        "Y_21": voltage_cover("Y_21", th7, (7,)),
        "Y_24": voltage_cover("Y_24", f2(3), (2, 2, 2)),
        "Y_42": voltage_cover("Y_42", [(a[0], b[0]) for a, b in zip(f2(1), th7)], (2, 7)),
        "Y_84": voltage_cover("Y_84", [a + b for a, b in zip(f2(2), th7)], (2, 2, 7)),
        "Y_168": voltage_cover("Y_168", [a + b for a, b in zip(f2(3), th7)], (2, 2, 2, 7)),
    }
    return out


# ------------------------------------------------------------------ canonical 2-tower (Paper 6)
def _f8mul(x, y):
    r = 0
    for i in range(3):
        if (y >> i) & 1:
            r ^= x << i
    for d in (4, 3):
        if (r >> d) & 1:
            r ^= 0b1011 << (d - 3)
    return r


_F8 = [[_f8mul(x, y) for y in range(8)] for x in range(8)]
_FROB = [_F8[x][x] for x in range(8)]
_ALPHA = [1]
for _ in range(6):
    _ALPHA.append(_F8[_ALPHA[-1]][2])


class TowerRing:
    """F_8[[phi; sigma]] / (phi^m), phi a = a^2 phi, phi^3 = t; units modulo 1 + t F_2[[t]]."""

    def __init__(self, m):
        self.m = m

    def mul(self, a, b):
        c = [0] * self.m
        for i, ai in enumerate(a):
            if ai:
                for j in range(self.m - i):
                    bj = b[j]
                    for _ in range(i % 3):
                        bj = _FROB[bj]
                    if bj:
                        c[i + j] ^= _F8[ai][bj]
        return tuple(c)

    def one(self):
        return tuple([1] + [0] * (self.m - 1))

    def inv(self, a):
        x = tuple([0] + list(a[1:]))
        res, p = self.one(), self.one()
        for _ in range(self.m):
            p = self.mul(p, x)
            res = tuple(u ^ v for u, v in zip(res, p))
        return res

    def normalize(self, a):
        a = list(a)
        for i in range(1, (self.m - 1) // 3 + 1):
            if a[3 * i] & 1:
                z = [0] * self.m
                z[0] = z[3 * i] = 1
                a = list(self.mul(tuple(a), tuple(z)))
        return tuple(a)

    def voltages(self):
        """Images of the Schreier generators gamma(s,x) = u_0^s u_x u_0^-((s+1) mod 3)."""
        def u(x):
            e = [0] * self.m
            e[0] = 1
            if self.m > 1:
                e[1] = _ALPHA[x]
            return tuple(e)
        pw = {0: self.one(), 1: u(0), 2: self.mul(u(0), u(0))}
        th = [None] * 21
        for s in range(3):
            for x in range(7):
                th[GIDX[(s, x)]] = self.normalize(
                    self.mul(self.mul(pw[s], u(x)), self.inv(pw[(s + 1) % 3])))
        return th


class LocalBuilding:
    """Lazy exploration of level m of the canonical 2-tower; for m >= 12 its balls of radius 4
    coincide with those of the Bruhat-Tits building of PGL_3(F_2((s)))."""

    def __init__(self, m=14):
        self.R = TowerRing(m)
        self.th = self.R.voltages()
        self.thi = [self.R.normalize(self.R.inv(g)) for g in self.th]
        self.v0 = (self.R.normalize(self.R.one()), 0)
        self._out, self._in = {}, {}

    def out(self, v):
        if v not in self._out:
            self._out[v] = [(self.R.normalize(self.R.mul(v[0], self.th[GIDX[(v[1], x)]])), (v[1] + 1) % 3)
                            for x in range(7)]
        return self._out[v]

    def inn(self, v):
        if v not in self._in:
            s = (v[1] - 1) % 3
            self._in[v] = [(self.R.normalize(self.R.mul(v[0], self.thi[GIDX[(s, x)]])), s) for x in range(7)]
        return self._in[v]

    def nbrs(self, v):
        return self.out(v) + self.inn(v)

    def chambers(self, v):
        ch = set()
        for w in [v] + self.inn(v) + [u for x in self.inn(v) for u in self.inn(x)]:
            for (x, y, z) in T:
                a = w
                b = self.out(a)[x]
                c = self.out(b)[y]
                if v in (a, b, c):
                    ch.add(frozenset((a, b, c)))
        return ch

    def ball(self, radius):
        dist, frontier = {self.v0: 0}, [self.v0]
        for r in range(radius):
            new = []
            for v in frontier:
                for w in self.nbrs(v):
                    if w not in dist:
                        dist[w] = r + 1
                        new.append(w)
            frontier = new
        return dist
