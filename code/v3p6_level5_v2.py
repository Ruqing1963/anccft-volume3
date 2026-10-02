"""Volume III, Paper 6, open problem (b), second route: v_2|Jac^H(Y_m)| along the canonical
2-tower by induced characters of an abelian normal subgroup (Artin / Clifford decomposition),
levels m = 3..8; chapters/ch06-s08-two-adic-tower.tex, subsection "Induced characters and levels 6-8".

  For k with 2k >= m the subgroup A = P_k/P_m of G_m = P/P_m is elementary abelian and normal
  ((1+x)(1+y) = 1+x+y since xy in phi^{2k} = 0), and A\\Y_m = Y_k.  With coset representatives
  r_c (the truncated lifts of c in G_k), every arc c -x-> c' of Y_k gets the voltage
  a_e = r_c theta_{s,x} r_{c'}^{-1} in A, and C[G_m] = (+)_chi Ind_A^G chi gives
      D^H(Y_m) ~ (+)_{chi in A^} D_chi,   D_chi = 7I + A_chi - 2 A_chi^t,  A_chi = sum chi(a_e) E_{cc'},
      v_2|Jac^H(Y_m)| = v_2|Jac^H(Y_k)| + v_2(n_k) - v_2(n_m) + sum_{chi != 1} v_2 det D_chi.
  Grouping the chi by G_m-conjugacy orbits is Clifford theory: prod_{chi in O} det D_chi =
  prod_{rho over O} det(Delta^H_rho)^{dim rho}, the Artin factorization over the irreducible
  rho of G_m lying over O; blocks in one orbit are conjugate by signed permutations.

  (QT) quotient models: G_k, the voltages a_e, A = F_2^d by coefficient bits, d = log2|A|;
  (CJ) conjugation action of G_m on A^: orbits; equal Smith profiles inside an orbit (checked on
       every orbit at m <= 6, on sampled members at m >= 7);
  (VA) validation: (m, k) = (3,2), (4,2), (5,3) reproduce 99, 289, 1742 (direct elimination /
       central-layer route of v3p6_v2_growth.py); (6,3) and (6,4) are two independent routes to
       level 6 and must agree;
  (NW) new levels: m = 6 (n = 49152), m = 7 (n = 196608), m = 8 (n = 1572864); with --check7 also
       the second route to m = 7 over Y_5 (6144 x 6144 blocks);
  (E)  e_m + 1 = dim F_2[G_m]/F_2[G_m]F_m for m <= 6 (F_2 elimination, 16384 x 16384 at m = 6), and
       the sharpened bound v_2(Y_m) >= v_2(Y_k) - log2(n_m/n_k) + (n_m/n_k - 1)(e_k + 1), k < m <= 2k;
  (TO) total order (1/n) log|Jac^H(Y_m)| from the same blocks (float), against h_H = 2.063693;
  (FT) growth of v_2/n_m: local exponents, power law, three-parameter and rational (Pade)
       extrapolations, a profile least-squares fit with leave-one-out, and the rigorous interval;
  output: data/v3p6_level5_v2.json and v3p6_v2_growth.png.

Run from code/:  python v3p6_level5_v2.py [--check7]   (numpy, matplotlib; ~3 min, about +1 h with --check7)
"""
import json
import math
import os
import sys
import time
from collections import Counter

import numpy as np

from a2_common import GIDX, TowerRing

t0 = time.time()
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
os.makedirs(OUT, exist_ok=True)


def elapsed():
    return f"[{time.time() - t0:7.1f} s]"


def n_of(m):
    return 3 * 2 ** (3 * (m - 1) - (m - 1) // 3)


# ---------------------------------------------------------------- 2-adic Smith profile
# (same procedure as v3p6_v2_growth.py: odd pivots mod 2^k, then divide the even remainder by 2)
def _odd_round(A, k):
    mask = (1 << k) - 1
    nr, nc = A.shape
    r, j, last = 0, 0, nc - 1
    while j <= last and r < nr:
        odd = np.flatnonzero(A[r:, j] & 1)
        if odd.size == 0:
            if j != last:
                A[:, [j, last]] = A[:, [last, j]]
            last -= 1
            continue
        i = r + int(odd[0])
        if i != r:
            A[[r, i]] = A[[i, r]]
        uinv = pow(int(A[r, j]), -1, 1 << k)
        if r + 1 < nr and j + 1 < nc:
            f = (A[r + 1:, j] * uinv) & mask
            nz = np.flatnonzero(f)
            if nz.size:
                rows = nz + r + 1
                sub = A[rows, j + 1:]
                sub -= f[nz, None] * A[r, j + 1:][None, :]
                sub &= mask
                A[rows, j + 1:] = sub
        r += 1
        j += 1
    return r


def smith2_profile(M, K=30):
    """{valuation: multiplicity} of the invariant factors over Z_2, #factors = 0 mod 2^K, cap."""
    A = np.array(M, dtype=np.int64)
    counts, shift, nzero, cap = Counter(), 0, 0, False
    while A.size:
        k = K - shift
        if k <= 0:
            cap = True
            break
        A &= (1 << k) - 1
        if not A.any():
            nzero = min(A.shape)
            break
        r = _odd_round(A, k)
        if r:
            counts[shift] = r
        A = np.ascontiguousarray(A[r:, r:])
        if A.size:
            assert not (A & 1).any()
            A >>= 1
        shift += 1
    return counts, nzero, cap


def pv2(c):
    return sum(k * v for k, v in c.items())


def pstr(c):
    return " ".join(f"2^{k}:{c[k]}" for k in sorted(c) if k > 0) or "-"


# ---------------------------------------------------------------- quotient model Y_m -> Y_k
class Induced:
    """Y_m as an A-voltage cover of Y_k, A = P_k/P_m elementary abelian (2k >= m)."""

    def __init__(self, m, k):
        assert 2 * k >= m > k >= 2
        self.m, self.k = m, k
        self.Rm, self.Rk = TowerRing(m), TowerRing(k)
        self.thm = self.Rm.voltages()
        thk = self.Rk.voltages()
        assert all(self.Rk.normalize(t[:k]) == tk for t, tk in zip(self.thm, thk))
        one = self.Rk.normalize(self.Rk.one())
        seen, fr = {one}, [one]
        while fr:
            new = []
            for g in fr:
                for h in thk:
                    x = self.Rk.normalize(self.Rk.mul(g, h))
                    if x not in seen:
                        seen.add(x)
                        new.append(x)
            fr = new
        self.Gk = sorted(seen)
        self.gid = {g: i for i, g in enumerate(self.Gk)}
        self.nk = 3 * len(self.Gk)
        assert self.nk == n_of(k)
        # coordinates of A: bits of the coefficients of phi^j, j = k..m-1 (bit 0 dropped when 3 | j)
        self.coord = [(j, b) for j in range(k, m) for b in range(3) if not (j % 3 == 0 and b == 0)]
        self.d = len(self.coord)
        assert n_of(m) == self.nk * 2 ** self.d
        mm = lambda a, b: self.Rm.normalize(self.Rm.mul(a, b))
        self.mm = mm
        lift = lambda g: tuple(g) + (0,) * (m - k)
        rows, cols, volt = [], [], []
        for g in self.Gk:
            for s in range(3):
                for x in range(7):
                    th = self.thm[GIDX[(s, x)]]
                    w = mm(lift(g), th)
                    gk = self.Rk.normalize(w[:k])
                    a = mm(w, self.Rm.normalize(self.Rm.inv(lift(gk))))
                    assert a[0] == 1 and not any(a[1:k])
                    rows.append(3 * self.gid[g] + s)
                    cols.append(3 * self.gid[gk] + (s + 1) % 3)
                    volt.append(self.bits(a))
        self.rows, self.cols = np.array(rows), np.array(cols)
        self.volt = np.array(volt, dtype=np.int64)              # voltages as integers in F_2^d
        assert len(set(zip(rows, cols))) == 7 * self.nk           # Y_k is simple
        # A is abelian with (1+x)(1+y) = 1+x+y: check on the basis
        basis = [self.elem(1 << i) for i in range(self.d)]
        for i, a in enumerate(basis):
            for j, b in enumerate(basis):
                assert mm(a, b) == mm(b, a) == self.elem((1 << i) ^ (1 << j))

    def bits(self, a):
        v = 0
        for i, (j, b) in enumerate(self.coord):
            v |= ((a[j] >> b) & 1) << i
        return v

    def elem(self, v):
        e = [0] * self.m
        e[0] = 1
        for i, (j, b) in enumerate(self.coord):
            if (v >> i) & 1:
                e[j] |= 1 << b
        return tuple(e)

    def signs(self, lam):
        par = np.zeros(len(self.volt), dtype=np.int64)
        x = self.volt & lam
        while x.any():
            par ^= x & 1
            x >>= 1
        return 1 - 2 * par

    def adj(self, lam):
        A = np.zeros((self.nk, self.nk), dtype=np.int64)
        np.add.at(A, (self.rows, self.cols), self.signs(lam))
        return A

    def block(self, lam):
        A = self.adj(lam)
        return 7 * np.eye(self.nk, dtype=np.int64) + A - 2 * A.T

    def conj_orbits(self):
        """Orbits of G_m on the characters of A (conjugation), via the F_2-linear action of the
        three generators theta_{2,x}, x = 0, 1, 2 (they generate G_m) on A = F_2^d."""
        gens = [self.thm[GIDX[(2, x)]] for x in (0, 1, 2)]
        Ms = []
        for h in gens:
            hi = self.Rm.normalize(self.Rm.inv(h))
            M = np.zeros((self.d, self.d), dtype=np.int64)
            for i in range(self.d):
                c = self.mm(self.mm(h, self.elem(1 << i)), hi)
                assert c[0] == 1 and not any(c[1:self.k])
                v = self.bits(c)
                for r in range(self.d):
                    M[r, i] = (v >> r) & 1
            Ms.append(M)
        # chi_lam(h a h^-1) = (-1)^{lam . M bits(a)}: lam -> M^t lam
        def act(M, lam):
            lv = np.array([(lam >> i) & 1 for i in range(self.d)])
            w = (M.T @ lv) % 2
            return int(sum(int(b) << i for i, b in enumerate(w)))
        parent = list(range(2 ** self.d))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for lam in range(2 ** self.d):
            for M in Ms:
                a, b = find(lam), find(act(M, lam))
                if a != b:
                    parent[a] = b
        orb = {}
        for lam in range(1, 2 ** self.d):
            orb.setdefault(find(lam), []).append(lam)
        assert find(0) not in orb or orb[find(0)] == []
        return sorted(orb.values(), key=lambda o: o[0])


def logabsdet(B):
    s, ld = np.linalg.slogdet(B.astype(float))
    assert s != 0
    return float(ld)


def run(m, k, v2k, logJk, check_all):
    Q = Induced(m, k)
    orbs = Q.conj_orbits()
    sizes = Counter(len(o) for o in orbs)
    print(f"(QT) Y_{m} -> Y_{k}: A = P_{k}/P_{m} = (Z/2)^{Q.d}, {2 ** Q.d - 1} twisted blocks of size "
          f"{Q.nk}; (CJ) {len(orbs)} conjugation orbits on A^ - 1, sizes "
          f"{dict(sorted(sizes.items()))}   {elapsed()}")
    tot_v2, tot_log, maxexp, profs = 0, 0.0, 0, {}
    for o in orbs:
        rep = o[0]
        c, z, cap = smith2_profile(Q.block(rep))
        assert z == 0 and not cap
        members = o if check_all else o[1:2] + o[-1:] if len(o) > 1 else []
        for lam in members:
            if lam != rep:
                c2, z2, cap2 = smith2_profile(Q.block(lam))
                assert not cap2 and z2 == 0 and c2 == c, (m, k, rep, lam)
        ld = logabsdet(Q.block(rep))
        tot_v2 += len(o) * pv2(c)
        tot_log += len(o) * ld
        maxexp = max(maxexp, max(c))
        profs[rep] = (len(o), c)
    v2m = v2k + int(math.log2(Q.nk)) - int(math.log2(n_of(m))) + tot_v2
    logJm = logJk + math.log(Q.nk) - math.log(n_of(m)) + tot_log
    pattern = Counter(pv2(c) for (_, c) in profs.values())
    print(f"(VA) Y_{m}: v_2 = {v2k} + {int(math.log2(Q.nk))} - {int(math.log2(n_of(m)))} + {tot_v2} = {v2m};"
          f"  per-orbit v_2 det D_chi values {dict(sorted(pattern.items()))}; largest block exponent {maxexp};"
          f"  (1/n) log|Jac| = {logJm / n_of(m):.6f}   {elapsed()}")
    return v2m, logJm, dict(d=Q.d, nblocks=2 ** Q.d - 1, norbits=len(orbs), orbit_sizes=dict(sizes),
                            layer_sum=tot_v2, max_block_exp=maxexp)


# ================================================================ base values (direct elimination)
# Y_2, Y_3, Y_4 by direct 2-adic elimination of D^H (v3p6_v2_growth.py, line (SM)); log|Jac| at
# Y_2..Y_4 from the spectrum (same script, line (TO)).
V2 = {1: 0, 2: 18, 3: 99, 4: 289}
LOGJ_CHAPTER = {2: 1.862909, 3: 2.041580, 4: 2.054414}
LOGJ = {}
for mm_ in (2, 3, 4):
    DH = Induced(mm_ + 1, mm_).block(0)                  # trivial character: D^H(Y_m) itself
    c, z, cap = smith2_profile(DH)
    assert z == 1 and not cap and pv2(c) == V2[mm_]
    a = np.linalg.eigvals(DH.astype(float))
    o = np.argsort(np.abs(a))
    assert abs(a[o[0]]) < 1e-6 and abs(a[o[1]]) > 1e-3
    LOGJ[mm_] = float(np.sum(np.log(np.abs(a[o[1:]])))) - math.log(n_of(mm_))   # det' = n |Jac|
    assert abs(LOGJ[mm_] / n_of(mm_) - LOGJ_CHAPTER[mm_]) < 1e-5
print(f"(VA) direct elimination: v_2|Jac^H(Y_m)| = 18, 99, 289 and (1/n) log|Jac^H| = 1.862909, "
      f"2.041580, 2.054414 for m = 2, 3, 4   {elapsed()}")

INFO = {}
v, lj, INFO["3/2"] = run(3, 2, V2[2], LOGJ[2], True)
assert v == 99 and abs(lj - LOGJ[3]) < 1e-6
v, lj, INFO["4/2"] = run(4, 2, V2[2], LOGJ[2], True)
assert v == 289 and abs(lj - LOGJ[4]) < 1e-6
v, lj, INFO["5/3"] = run(5, 3, V2[3], LOGJ[3], True)
assert v == 1742 and abs(lj / n_of(5) - 2.062520) < 1e-5
V2[5], LOGJ[5] = v, lj
print("(VA) the induced-character route reproduces 99, 289 and 1742 = v_2|Jac^H(Y_5)| (n = 6144), "
      "and (1/n) log|Jac^H(Y_5)| = 2.062520")

v63, lj63, INFO["6/3"] = run(6, 3, V2[3], LOGJ[3], True)
v64, lj64, INFO["6/4"] = run(6, 4, V2[4], LOGJ[4], True)
assert v63 == v64 and abs(lj63 - lj64) < 1e-6 * abs(lj63)
V2[6], LOGJ[6] = v63, lj63
print(f"(NW) level 6, n = {n_of(6)}: v_2|Jac^H(Y_6)| = {V2[6]} by two independent routes "
      f"(over Y_3 and over Y_4)")

v, lj, INFO["7/4"] = run(7, 4, V2[4], LOGJ[4], False)
V2[7], LOGJ[7] = v, lj
print(f"(NW) level 7, n = {n_of(7)}: v_2|Jac^H(Y_7)| = {V2[7]}")
if "--check7" in sys.argv:                  # second route to level 7: 6144 x 6144 blocks over Y_5
    v75, lj75, INFO["7/5"] = run(7, 5, V2[5], LOGJ[5], False)
    assert v75 == V2[7] and abs(lj75 - LOGJ[7]) < 1e-6 * abs(lj75)
    print(f"(NW) level 7: the route over Y_5 gives {v75} as well")
v, lj, INFO["8/4"] = run(8, 4, V2[4], LOGJ[4], False)
V2[8], LOGJ[8] = v, lj
print(f"(NW) level 8, n = {n_of(8)}: v_2|Jac^H(Y_8)| = {V2[8]}")
assert V2[6] / n_of(6) >= (1 - n_of(5) / n_of(6)) * 547 / n_of(5)   # Thm v2bounds (ii), e_5 = 546


# ================================================================ (E) e_m + 1 = dim F_2[G_m]/F_2[G_m] F_m
def gf2_rank_packed(P, C):
    """Rank over F_2 of packed rows (uint64 words, little-endian bits)."""
    M = P.copy()
    R = M.shape[0]
    r = 0
    one = np.uint64(1)
    for c in range(C):
        if r == R:
            break
        w, b = divmod(c, 64)
        nz = np.flatnonzero((M[r:, w] >> np.uint64(b)) & one)
        if nz.size == 0:
            continue
        i = r + int(nz[0])
        if i != r:
            M[[r, i]] = M[[i, r]]
        if nz.size > 1:
            rows = nz[1:] + r
            M[rows, w:] ^= M[r, w:]
        r += 1
    return r


def schur_dim(m):
    """dim_F2 of F_2[G_m]/F_2[G_m] F_m with F_m = 1 + fbar_1 fbar_0 fbar_2 (Prop. V3P6:prop:schur)."""
    R = TowerRing(m)
    th = R.voltages()
    one = R.normalize(R.one())
    mul = lambda a, b: R.normalize(R.mul(a, b))
    seen, fr = {one}, [one]
    while fr:
        new = []
        for g in fr:
            for h in th:
                x_ = mul(g, h)
                if x_ not in seen:
                    seen.add(x_)
                    new.append(x_)
        fr = new
    G = sorted(seen)
    gidx = {g: i for i, g in enumerate(G)}
    fbar = [Counter(R.normalize(R.inv(th[GIDX[(s, x_)]])) for x_ in range(7)) for s in range(3)]

    def gmul(P_, Q_):
        out = Counter()
        for a, ca in P_.items():
            for b, cb in Q_.items():
                out[mul(a, b)] += ca * cb
        return Counter({g: c for g, c in out.items() if c % 2})
    F = gmul(gmul(fbar[1], fbar[0]), fbar[2])
    F[one] = (F[one] + 1) % 2
    supp = [g for g, c in F.items() if c % 2]
    nG = len(G)
    Wd = -(-nG // 64)
    P = np.zeros((nG, Wd), dtype=np.uint64)
    for i, g in enumerate(G):                                   # row g: the element g F_m
        cols = np.array([gidx[mul(g, k_)] for k_ in supp])
        np.bitwise_xor.at(P[i], cols // 64, np.left_shift(np.uint64(1), (cols % 64).astype(np.uint64)))
    return nG - gf2_rank_packed(P, nG), len(supp)


E_PLUS1 = {}
for mm_ in (2, 3, 4, 5, 6):
    E_PLUS1[mm_], ns = schur_dim(mm_)
    print(f"(E) m = {mm_}: |supp F_m| = {ns}, e_m + 1 = dim F_2[G_m]/F_2[G_m]F_m = {E_PLUS1[mm_]}, "
          f"(e_m+1)/n_m = {E_PLUS1[mm_] / n_of(mm_):.5f}   {elapsed()}")
assert [E_PLUS1[mm_] for mm_ in (2, 3, 4, 5)] == [7, 37, 100, 547]
assert E_PLUS1[6] * n_of(5) <= E_PLUS1[5] * n_of(6) and E_PLUS1[6] >= E_PLUS1[5]       # Nakayama
# Theorem (sharpened bound): for k < m <= 2k,
#   v_2|Jac(Y_m)| >= v_2|Jac(Y_k)| - log_2(n_m/n_k) + (n_m/n_k - 1)(e_k + 1)
for (mm_, k_) in ((3, 2), (4, 2), (5, 3), (6, 3), (6, 4), (7, 4), (8, 4)):
    lb = V2[k_] - int(math.log2(n_of(mm_) // n_of(k_))) + (n_of(mm_) // n_of(k_) - 1) * E_PLUS1[k_]
    assert V2[mm_] >= lb
    print(f"(E) bound over Y_{k_}: v_2|Jac^H(Y_{mm_})| = {V2[mm_]} >= {lb}  (v_2/n >= {lb / n_of(mm_):.5f})")

# ================================================================ (TO) total order
hH = 2.063693
MS = sorted(V2)
print("(TO) (1/n) log|Jac^H(Y_m)|: " + ", ".join(f"m={mm_}: {LOGJ[mm_] / n_of(mm_):.6f}"
                                                  for mm_ in MS if mm_ >= 2) + f"   (h_H = {hH})")
for mm_ in MS:
    if mm_ >= 2:
        assert V2[mm_] <= LOGJ[mm_] / math.log(2)
        assert LOGJ[mm_] / n_of(mm_) < hH + 1e-6
for a, b in zip(MS[1:], MS[2:]):
    assert LOGJ[a] / n_of(a) < LOGJ[b] / n_of(b)

# ================================================================ (FT) growth and extrapolation
ms = [mm_ for mm_ in MS if mm_ >= 2]
n = np.array([n_of(mm_) for mm_ in ms], dtype=float)
v2 = np.array([V2[mm_] for mm_ in ms], dtype=float)
x = v2 / n
print("\n(FT) m      n        v_2      v_2/n     local exponent  index")
for i, mm_ in enumerate(ms):
    le = "" if i == 0 else f"{math.log(v2[i] / v2[i - 1]) / math.log(n[i] / n[i - 1]):.4f}"
    ix = "" if i == 0 else f"{int(n[i] / n[i - 1])}"
    print(f"(FT) {mm_}  {int(n[i]):8d}  {int(v2[i]):7d}   {x[i]:.5f}    {le:>8s}       {ix}")
loc = [math.log(v2[i] / v2[i - 1]) / math.log(n[i] / n[i - 1]) for i in range(1, len(ms))]

# (a) power law v_2 = c n^beta on the last four levels
beta, lc = np.polyfit(np.log(n[-4:]), np.log(v2[-4:]), 1)

# (b) three-parameter model v_2/n = L + a n^{-g}, exact through each triple of consecutive levels
def three_param(i):
    (n1, n2, n3), (y1, y2, y3) = n[i:i + 3], x[i:i + 3]
    # (y1-y2)/(y2-y3) = (n1^-g - n2^-g)/(n2^-g - n3^-g): solve for g > 0 by bisection
    target = (y1 - y2) / (y2 - y3)
    f = lambda g: (n1 ** -g - n2 ** -g) / (n2 ** -g - n3 ** -g) - target
    lo, hi = 1e-6, 5.0
    if f(lo) * f(hi) > 0:
        return None
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if f(lo) * f(mid) > 0 else (lo, mid)
    g = (lo + hi) / 2
    a = (y1 - y2) / (n1 ** -g - n2 ** -g)
    return y3 - a * n3 ** -g, g

# (c) Pade [1/1] in u = 1/log_2 n through each triple: x = (p0 + p1 u)/(1 + q1 u), limit p0
def pade(i):
    u = 1 / np.log2(n[i:i + 3])
    y = x[i:i + 3]
    M = np.column_stack([np.ones(3), u, -u * y])
    p0, p1, q1 = np.linalg.solve(M, y)
    return p0

tp = [three_param(i) for i in range(len(ms) - 2)]
pd = [pade(i) for i in range(len(ms) - 2)]
print(f"(FT) local exponents log(v_2 ratio)/log(index): " + ", ".join(f"{e:.3f}" for e in loc))
print(f"(FT) power law on the last four levels: v_2 ~ n^{beta:.4f}")
print("(FT) three-parameter fits v_2/n = L + a n^-g through consecutive triples (m, m+1, m+2): " +
      "; ".join(f"m={ms[i]}: " + ("no solution" if t is None else f"L = {t[0]:+.4f}, g = {t[1]:.3f}")
                for i, t in enumerate(tp)))
print("(FT) Pade [1/1] in 1/log_2 n through consecutive triples, limit p0: " +
      "; ".join(f"m={ms[i]}: {p:+.4f}" for i, p in enumerate(pd)))
print("(FT) the triple extrapolations jump with the period-3 pattern of the indices (8, 4, 8, 8, 4, 8): "
      "they are not stable and are not used for the verdict")


# (d) profile least squares for the model v_2/n = L + a n^-g over m >= 3: for each L the fit of
#     log(v_2/n - L) against log n is linear; RSS(L) and its leave-one-out minimisers
def profile(idx):
    xs, ns = x[idx], n[idx]
    grid = np.linspace(0, 0.999 * xs.min(), 2000)
    rss = []
    for L in grid:
        yy = np.log(xs - L)
        c_ = np.polyfit(np.log(ns), yy, 1)
        rss.append(float(np.sum((np.polyval(c_, np.log(ns)) - yy) ** 2)))
    rss = np.array(rss)
    return grid, rss


idx3 = np.array([i for i, mm_ in enumerate(ms) if mm_ >= 3])
grid, rss = profile(idx3)
Lstar = float(grid[np.argmin(rss)])
loo = []
for drop in idx3:
    g_, r_ = profile(np.array([i for i in idx3 if i != drop]))
    loo.append(float(g_[np.argmin(r_)]))
gamma0 = -np.polyfit(np.log(n[idx3]), np.log(x[idx3]), 1)[0]
print(f"(FT) profile fit v_2/n = L + a n^-g on m = 3..{ms[-1]}: best L = {Lstar:.4f}, RSS(0)/RSS(L*) = "
      f"{rss[0] / rss.min():.2f}; leave-one-out best L: " + ", ".join(f"{v:.4f}" for v in loo) +
      f"; with L = 0 the exponent is g = {gamma0:.4f} (v_2 ~ n^{1 - gamma0:.4f})")
ce_upper = E_PLUS1[6] / n_of(6)
print(f"(FT) rigorous: 0 <= c_e <= liminf v_2/n_m (sharpened bound over Y_k, k ~ m/2) and "
      f"c_e <= (e_6+1)/n_6 = {ce_upper:.5f} (Prop. schur (iii)); limsup v_2/n_m <= h_H/log 2 = "
      f"{hH / math.log(2):.4f}")
ratio2k = {k_: (V2[2 * k_] / n_of(2 * k_)) / (E_PLUS1[k_] / n_of(k_)) for k_ in (2, 3, 4)}
print("(FT) observed (v_2/n at m = 2k) / ((e_k+1)/n_k): " +
      ", ".join(f"k={k_}: {v:.3f}" for k_, v in ratio2k.items()))
print("(FT) (e_m+1)/n_m: " + ", ".join(f"{E_PLUS1[mm_] / n_of(mm_):.5f}" for mm_ in sorted(E_PLUS1)) +
      "; successive ratios " + ", ".join(f"{(E_PLUS1[mm_] / n_of(mm_)) / (E_PLUS1[mm_ - 1] / n_of(mm_ - 1)):.3f}"
                                          for mm_ in sorted(E_PLUS1)[1:]))
dec = all(x[i] > x[i + 1] for i in range(len(x) - 1))
sub = all(e < 1 for e in loc)
print(f"(FT) v_2/n_m decreasing at every level: {dec}; every local exponent < 1: {sub}")
assert dec and sub

# ================================================================ output
res = dict(levels={str(mm_): dict(n=n_of(mm_), v2=V2[mm_], v2_over_n=V2[mm_] / n_of(mm_),
                                   logJ_over_n=(LOGJ[mm_] / n_of(mm_) if mm_ >= 2 else None))
                   for mm_ in MS},
           routes=INFO, local_exponents=loc, power_law_beta_last4=beta,
           three_param=[None if t is None else dict(L=t[0], g=t[1]) for t in tp], pade_limits=pd,
           profile=dict(L_best=Lstar, rss0_over_min=float(rss[0] / rss.min()), loo=loo, g_at_L0=gamma0),
           ce_upper=ce_upper, e_plus_1={str(k_): v for k_, v in E_PLUS1.items()},
           ratio_2k={str(k_): v for k_, v in ratio2k.items()}, hH=hH)
with open(os.path.join(OUT, "v3p6_level5_v2.json"), "w") as fh:
    json.dump(res, fh, indent=1)

import matplotlib                                   # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].loglog(n, v2, "o-", label=r"$v_2|\mathrm{Jac}^H(Y_m)|$")
ax[0].loglog(n, np.exp(lc) * n ** beta, "--", label=rf"$\propto n^{{{beta:.3f}}}$ (last 4)")
ax[0].loglog(n, n * hH / math.log(2), ":", label=r"$n\,h_H/\log 2$")
ax[0].set_xlabel("$n_m$"), ax[0].legend(), ax[0].set_title("2-adic valuation")
ax[1].semilogx(n, x, "o-", label="$v_2/n_m$")
ax[1].axhline(0, color="k", lw=0.5)
ax[1].semilogx(n, x[-1] * (n / n[-1]) ** (-gamma0), "--", lw=0.8, label=rf"$\propto n^{{-{gamma0:.3f}}}$")
ax[1].semilogx([n_of(mm_) for mm_ in sorted(E_PLUS1)], [E_PLUS1[mm_] / n_of(mm_) for mm_ in sorted(E_PLUS1)],
               "s-", ms=4, color="gray", label=r"$(e_m+1)/n_m\geq c_e$")
ax[1].set_xlabel("$n_m$"), ax[1].legend(), ax[1].set_title("$v_2/n_m$")
for a_ in ax:
    a_.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "v3p6_v2_growth.png"), dpi=150)

print(f"\nALL CHECKS PASSED: (QT) (CJ) (VA) (NW) (TO) (FT).   {elapsed()}")
