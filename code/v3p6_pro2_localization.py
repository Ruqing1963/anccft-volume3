"""Volume III, Paper 6, Section "Localization over the non-analytic group Z_2[[P]]"
(chapters/ch06-s09-pro2-localization.tex): finite-level descent along the canonical 2-tower.

Levels Y_m = U_m \\ B, m = 1..5, with n = 3, 24, 192, 768, 6144 and |P/P_m| = 1, 8, 64, 256, 2048.
The layer Y_{m+1} -> Y_m is a regular cover with elementary abelian group H_m = P_m/P_{m+1}
of order 8, 8, 4, 8 (m = 1, 2, 3, 4).

  (OR) |P/P_m| = 2^(3(m-1) - floor((m-1)/3)) for m <= 5; H_m is central and elementary abelian;
  (PF) pushforward compatibility Pi Delta^H(Y_{m+1}) = Delta^H(Y_m) Pi and
       Delta^H(Y_{m+1}) Pi^t = Pi^t Delta^H(Y_m) exactly (dense, m+1 <= 4; at the level of
       labelled arcs for all four layers, which implies both identities);
  (BL) block decomposition Delta^H(Y_{m+1}) E_chi = E_chi Delta^H_chi exactly for every character
       chi of H_m, with E^t E = |H_m|; the trivial block is Delta^H(Y_m) itself;
  (SE) sector lemma at every level m <= 4 (one eigenvalue 0, all others Re > 0) and for every
       twisted block (Re > 0, exact Bareiss determinant != 0; numeric at the layer Y_5/Y_4);
  (CN) relative class number formula prod_{chi != 1} det Delta^H_chi(Y_{m+1}/Y_m)
       = |H_m| |Jac^H(Y_{m+1})| / |Jac^H(Y_m)| with exact integers for the layers Y_2/Y_1 and
       Y_3/Y_2; for Y_4/Y_3 the value |Jac^H(Y_4)| is reconstructed EXACTLY as the (767 x 767)
       cofactor of Delta^H(Y_4) by Chinese remaindering over 31-bit primes (Hadamard bound), and
       the formula is then an exact identity of integers; v_2|Jac^H(Y_4)| = 289 is recovered;
  (SM) coker Delta^H(Y_m) = Z + Jac^H(Y_m) (exactly n - 1 nonzero invariant factors) for m = 2, 3;
       the finite-level torsion module X_m^nt = coker[Delta^H(Y_m) | 1_{V_1} 1_{V_2} 1_{V_3}] is
       finite of order |P/P_m| |Jac^H(Y_m)| / |Jac^H(X_3)|;
  (KR) Jac^H(Y_{m+1}) -> Jac^H(Y_m) (pushforward) is surjective, and the order of its kernel,
       computed independently as a lattice index, equals prod_{chi != 1} det Delta^H_chi / |H_m|.

Run:  python code/v3p6_pro2_localization.py      (numpy only; about three minutes)
"""
import math
import time
from fractions import Fraction

import numpy as np

from a2_common import GIDX, T, bareiss_det, invariant_factors, torsion

t0 = time.time()


# ---------------------------------------------------------------- F_8 and the tower ring
# (copied from v3p6_deep_verify.py; importing that module would run its checks)
def gmul(x, y):
    r = 0
    for i in range(3):
        if (y >> i) & 1:
            r ^= x << i
    for d in (4, 3):
        if (r >> d) & 1:
            r ^= 0b1011 << (d - 3)
    return r


MUL = [[gmul(x, y) for y in range(8)] for x in range(8)]
FROB = [MUL[x][x] for x in range(8)]
TR = [x ^ FROB[x] ^ FROB[FROB[x]] for x in range(8)]          # absolute trace F_8 -> F_2
assert all(TR[x] in (0, 1) for x in range(8)) and sum(TR) == 4


def sig(a, i):
    for _ in range(i % 3):
        a = FROB[a]
    return a


ALPHA = [1]
for _ in range(6):
    ALPHA.append(MUL[ALPHA[-1]][2])


class Ring:
    """F_8[[phi; sigma]] / (phi^m), phi a = sigma(a) phi, phi^3 = t central."""

    def __init__(self, m):
        self.m = m

    def mul(self, a, b):
        c = [0] * self.m
        for i, ai in enumerate(a):
            if ai:
                for j in range(self.m - i):
                    if b[j]:
                        c[i + j] ^= MUL[ai][sig(b[j], i)]
        return tuple(c)

    def one(self):
        return tuple([1] + [0] * (self.m - 1))

    def inv(self, a):
        x = tuple([0] + list(a[1:]))
        res, p = self.one(), self.one()
        for _ in range(self.m):
            p = self.mul(p, x)
            res = tuple(r ^ s for r, s in zip(res, p))
        return res

    def normalize(self, a):
        """Representative modulo the central units 1 + sum c_i t^i, c_i in F_2."""
        a = list(a)
        for i in range(1, (self.m - 1) // 3 + 1):
            if a[3 * i] & 1:
                z = [0] * self.m
                z[0] = z[3 * i] = 1
                a = list(self.mul(tuple(a), tuple(z)))
        return tuple(a)


def u(R, x):
    e = [0] * R.m
    e[0] = 1
    if R.m > 1:
        e[1] = ALPHA[x]
    return tuple(e)


def schreier(R):
    u0 = u(R, 0)
    pw = {0: R.one(), 1: u0, 2: R.mul(u0, u0)}
    th = [None] * 21
    for s in range(3):
        for x in range(7):
            th[GIDX[(s, x)]] = R.normalize(R.mul(R.mul(pw[s], u(R, x)), R.inv(pw[(s + 1) % 3])))
    return th


def closure(R, gens):
    one = R.normalize(R.one())
    seen, frontier = {one}, [one]
    while frontier:
        new = []
        for g in frontier:
            for h in gens:
                k = R.normalize(R.mul(g, h))
                if k not in seen:
                    seen.add(k)
                    new.append(k)
        frontier = new
    return seen


# ---------------------------------------------------------------- levels of the tower
class Level:
    """Level m: vertices (g, s), g in P/P_m, s a type; arcs (g, s) -> (g th(s, x), s + 1)."""

    def __init__(self, m):
        self.m = m
        self.R = Ring(m)
        self.th = schreier(self.R)
        self.G = sorted(closure(self.R, self.th))
        self.verts = [(g, s) for g in self.G for s in range(3)]
        self.vid = {v: i for i, v in enumerate(self.verts)}
        self.n = len(self.verts)
        self.out = np.zeros((self.n, 7), dtype=np.int64)
        for i, (g, s) in enumerate(self.verts):
            for x in range(7):
                w = (self.R.normalize(self.R.mul(g, self.th[GIDX[(s, x)]])), (s + 1) % 3)
                self.out[i, x] = self.vid[w]
        for i in range(self.n):                      # the triangle relations close up
            for (x, y, z) in T:
                assert self.out[self.out[self.out[i, x], y], z] == i
        self._DH = None

    def DH(self):
        if self._DH is None:
            A1 = np.zeros((self.n, self.n), dtype=np.int64)
            for i in range(self.n):
                for x in range(7):
                    A1[i, self.out[i, x]] += 1
            if self.m >= 2:                          # X_3 = Y_1 has 7 parallel arcs per type pair
                assert A1.max() == 1 and not np.diag(A1).any() and not (A1 * A1.T).any()
            else:
                assert (A1 == 7 * np.roll(np.eye(3, dtype=np.int64), 1, axis=1)).all()
            self._DH = 7 * np.eye(self.n, dtype=np.int64) + A1 - 2 * A1.T
        return self._DH


LEV = {m: Level(m) for m in range(1, 6)}
orders = [len(LEV[m].G) for m in range(1, 6)]
assert orders == [1, 8, 64, 256, 2048]
assert orders == [2 ** (3 * (m - 1) - (m - 1) // 3) for m in range(1, 6)]
assert [LEV[m].n for m in range(1, 6)] == [3, 24, 192, 768, 6144]


def projection(Lc, Lb):
    """pi: V(Y_{m+1}) -> V(Y_m), truncation of the group element."""
    m = Lb.m
    pi = np.zeros(Lc.n, dtype=np.int64)
    for j, (g, s) in enumerate(Lc.verts):
        pi[j] = Lb.vid[(Lb.R.normalize(tuple(g[:m])), s)]
    return pi


def layer_group(Lc, Lb):
    """H_m = P_m / P_{m+1} inside P / P_{m+1}: elements 1 + c phi^m."""
    m = Lb.m
    H = [h for h in Lc.G if all(h[i] == 0 for i in range(1, m))]
    return H


LAYERS = {}
for m in range(1, 5):
    Lb, Lc = LEV[m], LEV[m + 1]
    pi = projection(Lc, Lb)
    H = layer_group(Lc, Lb)
    k = {1: 3, 2: 3, 3: 2, 4: 3}[m]
    assert len(H) == 2 ** k
    R = Lc.R
    for h in H:                                      # central, exponent 2
        assert R.normalize(R.mul(h, h)) == R.normalize(R.one())
        for g in Lc.G:
            assert R.normalize(R.mul(h, g)) == R.normalize(R.mul(g, h))
    counts = np.bincount(pi, minlength=Lb.n)
    assert (counts == len(H)).all()                   # fibres of constant size |H_m|
    # the truncated Schreier generators are those of the lower level
    assert [Lb.R.normalize(tuple(g[:m])) for g in Lc.th] == Lb.th
    LAYERS[m] = (pi, H)
print("(OR) |P/P_m| = 1, 8, 64, 256, 2048 (m = 1..5); H_m = P_m/P_{m+1} central elementary abelian "
      "of order 8, 8, 4, 8; fibres of Y_{m+1} -> Y_m have |H_m| vertices")

# ---------------------------------------------------------------- (PF) pushforward compatibility
for m in range(1, 5):
    Lb, Lc = LEV[m], LEV[m + 1]
    pi, H = LAYERS[m]
    # labelled covering property: pi(step(v, x)) = step(pi v, x) for every arc label x
    assert (pi[Lc.out] == Lb.out[pi]).all()
    if m + 1 <= 4:
        Pi = np.zeros((Lb.n, Lc.n), dtype=np.int64)
        Pi[pi, np.arange(Lc.n)] = 1
        DHc, DHb = Lc.DH(), Lb.DH()
        assert (Pi @ DHc == DHb @ Pi).all()           # pushforward:  Z[P/P_{m+1}] -> Z[P/P_m]
        assert (DHc @ Pi.T == Pi.T @ DHb).all()       # pullback
        assert (Pi @ DHc @ Pi.T == len(H) * DHb).all()
print("(PF) pi(step(v, x)) = step(pi v, x) on all arcs of Y_5 -> Y_4 -> Y_3 -> Y_2 -> Y_1; "
      "Pi DH_{m+1} = DH_m Pi, DH_{m+1} Pi^t = Pi^t DH_m and Pi DH_{m+1} Pi^t = |H_m| DH_m exactly "
      "(dense, m + 1 <= 4)")


# ---------------------------------------------------------------- (BL) twisted blocks of a layer
def twisted_blocks(m, with_E=True):
    """Characters chi of H_m, the voltage of every arc of Y_m, the twisted matrices DH_chi,
    and the isotypic matrices E_chi with DH_{m+1} E_chi = E_chi DH_chi."""
    Lb, Lc = LEV[m], LEV[m + 1]
    pi, H = LAYERS[m]
    R = Lc.R
    pad = lambda g: R.normalize(tuple(g) + (0,))      # the section s(v)
    vol = np.zeros((Lb.n, 7), dtype=np.int64)
    sec = np.zeros(Lb.n, dtype=np.int64)
    for i, (g, s) in enumerate(Lb.verts):
        sec[i] = Lc.vid[(pad(g), s)]
        for x in range(7):
            j = Lc.out[sec[i], x]
            w = Lc.verts[j]
            kk = pi[j]
            assert kk == Lb.out[i, x]
            h = R.normalize(R.mul(w[0], R.inv(pad(Lb.verts[kk][0]))))
            assert h[0] == 1 and all(h[t] == 0 for t in range(1, m)) and h in H
            vol[i, x] = h[m]
    # the voltage of each vertex of the cover relative to the section
    cvol = np.zeros(Lc.n, dtype=np.int64)
    for j, (g, s) in enumerate(Lc.verts):
        h = R.normalize(R.mul(g, R.inv(pad(Lb.verts[pi[j]][0]))))
        assert h in H
        cvol[j] = h[m]
    lambdas = [l for l in range(8) if m % 3 or TR[l] == 0]   # characters chi_l(c) = (-1)^Tr(l c)
    assert len(lambdas) == len(H)
    out = {}
    for lam in lambdas:
        chi = np.array([1 - 2 * TR[MUL[lam][c]] for c in range(8)], dtype=np.int64)
        A = np.zeros((Lb.n, Lb.n), dtype=np.int64)
        for i in range(Lb.n):
            for x in range(7):
                A[i, Lb.out[i, x]] += chi[vol[i, x]]
        DHchi = 7 * np.eye(Lb.n, dtype=np.int64) + A - 2 * A.T
        E = None
        if with_E:
            E = np.zeros((Lc.n, Lb.n), dtype=np.int64)
            E[np.arange(Lc.n), pi] = chi[cvol]
        out[lam] = (DHchi, E)
    return out


BLOCKS = {}
for m in range(1, 4):
    Lb, Lc = LEV[m], LEV[m + 1]
    pi, H = LAYERS[m]
    blocks = twisted_blocks(m)
    DHc = Lc.DH()
    Es = []
    for lam, (DHchi, E) in blocks.items():
        assert (DHc @ E == E @ DHchi).all()
        Es.append(E)
    Eall = np.concatenate(Es, axis=1)
    assert (Eall.T @ Eall == len(H) * np.eye(Lc.n, dtype=np.int64)).all()
    assert (blocks[0][0] == Lb.DH()).all()            # trivial character block = DH(Y_m)
    BLOCKS[m] = blocks
# the layer Y_5/Y_4: twisted matrices only (DH(Y_5) is not formed densely)
BLOCKS[4] = twisted_blocks(4, with_E=False)
assert (BLOCKS[4][0][0] == LEV[4].DH()).all()
print("(BL) DH(Y_{m+1}) E_chi = E_chi DH_chi exactly for all characters chi of H_m, E^t E = |H_m| I "
      "(m = 1, 2, 3); the trivial block is DH(Y_m) (m = 1..4)")

# ---------------------------------------------------------------- (SE) sector lemma
for m in range(1, 5):
    mu = np.linalg.eigvals(LEV[m].DH().astype(float))
    order = np.argsort(np.abs(mu))
    assert abs(mu[order[0]]) < 1e-8
    remin = float(np.min(np.real(mu[order[1:]])))
    assert remin > 0.5
    print(f"     level {m}: n = {LEV[m].n}, one eigenvalue 0, min Re of the others = {remin:.4f}")
DETS = {}
for m in range(1, 5):
    DETS[m] = {}
    for lam, (DHchi, E) in BLOCKS[m].items():
        if lam == 0:
            continue
        mu = np.linalg.eigvals(DHchi.astype(float))
        assert float(np.min(np.real(mu))) > 0.5
        if m <= 3:
            d = bareiss_det(DHchi.tolist())
            assert d > 0
            assert abs(math.log(d) - float(np.sum(np.log(np.abs(mu))))) < 1e-8 * math.log(d)
        else:
            sgn, ld = np.linalg.slogdet(DHchi.astype(float))
            assert sgn > 0 and ld > 0
            d = None
        DETS[m][lam] = (d, float(np.sum(np.log(np.abs(mu)))))
print("(SE) every twisted block DH_chi (chi != 1) of the four layers has all eigenvalues with "
      "Re > 0.5; det DH_chi > 0, exact (Bareiss) for the layers Y_2/Y_1, Y_3/Y_2, Y_4/Y_3")


# ---------------------------------------------------------------- Hecke Jacobians, exact
def v_p(x, p):
    k = 0
    while x % p == 0:
        x //= p
        k += 1
    return k


JAC = {}
for m in (1, 2, 3):
    DH = LEV[m].DH()
    inv = invariant_factors(DH.tolist())
    assert len(inv) == LEV[m].n - 1                      # coker = Z + finite
    JAC[m] = math.prod(d for d in inv if d > 1)
assert JAC[1] == 147
assert JAC[2] == 2 ** 18 * 3 * 7 ** 16                 # (Z/7)^2 + Z/49 + (Z/392)^5 + Z/1176
assert v_p(JAC[3], 2) == 99
print(f"(SM) coker DH(Y_m) = Z + Jac^H(Y_m) for m = 1, 2, 3 (Smith form); |Jac^H| = 147, "
      f"2^18 3 7^16, and a {len(str(JAC[3]))}-digit number with v_2 = 99, v_7 = {v_p(JAC[3], 7)}")

# X_m^nt = coker of [DH_m | 1_{V_1} 1_{V_2} 1_{V_3}]: finite of order |P/P_m| |Jac_m| / 147
for m in (2, 3):
    L = LEV[m]
    ind = np.zeros((L.n, 3), dtype=np.int64)
    for i, (g, s) in enumerate(L.verts):
        ind[i, s] = 1
    M = np.concatenate([L.DH(), ind], axis=1)
    inv = invariant_factors(M.tolist())
    assert len(inv) == L.n                              # finite cokernel
    assert math.prod(inv) * 147 == len(L.G) * JAC[m]
print("     X_m^nt = coker[DH_m | type indicators] is finite of order |P/P_m| |Jac^H(Y_m)| / 147 "
      "for m = 2, 3 (Smith form)")

# ---------------------------------------------------------------- (CN) layers 1 and 2, exact
PROD = {}
for m in (1, 2):
    prod = math.prod(d for d, _ in DETS[m].values())
    PROD[m] = prod
    H = LAYERS[m][1]
    assert Fraction(prod) == Fraction(len(H) * JAC[m + 1], JAC[m])
    print(f"(CN) layer Y_{m + 1}/Y_{m} (|H| = {len(H)}): prod_(chi != 1) det DH_chi = "
          f"{len(H)} |Jac^H(Y_{m + 1})| / |Jac^H(Y_{m})| exactly; the product has "
          f"{len(str(prod))} digits, v_2 = {v_p(prod, 2)}, v_3 = {v_p(prod, 3)}, v_7 = {v_p(prod, 7)}")
assert PROD[1] == 2 ** 21 * 7 ** 14


# ---------------------------------------------------------------- (CN) layer 3: |Jac^H(Y_4)| by CRT
def is_prime(n):
    if n < 2:
        return False
    for p in (2, 3, 5, 7, 11, 13):
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for a in (2, 3, 5, 7):                  # deterministic below 3.2e9
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def det_mod(M, p):
    A = np.array(M, dtype=np.int64) % p
    n = A.shape[0]
    det = 1
    for k in range(n):
        piv = np.nonzero(A[k:, k])[0]
        if len(piv) == 0:
            return 0
        i = k + piv[0]
        if i != k:
            A[[k, i]] = A[[i, k]]
            det = -det
        det = det * int(A[k, k]) % p
        inv = pow(int(A[k, k]), -1, p)
        row = (A[k, k + 1:] * inv) % p
        col = A[k + 1:, k].copy()
        A[k + 1:, k + 1:] = (A[k + 1:, k + 1:] - np.outer(col, row)) % p
    return det % p


DH4 = LEV[4].DH()
C4 = DH4[1:, 1:]                                        # cofactor: adj(DH_4) = c J, |Jac^H(Y_4)| = |c|
rows = np.sqrt((C4.astype(float) ** 2).sum(axis=1))
hadamard_bits = float(np.sum(np.log2(rows)))            # log2 of the Hadamard bound
primes, p, modulus = [], 2 ** 31 - 1, 1
while math.log2(modulus) < hadamard_bits + 2:           # product of primes > 2 |c|
    if is_prime(p):
        primes.append(p)
        modulus *= p
    p -= 2
t1 = time.time()
residues = [det_mod(C4, p) for p in primes]
c4 = 0                                                   # Chinese remaindering
for p, r in zip(primes, residues):
    Mp = modulus // p
    c4 = (c4 + r * Mp * pow(Mp, -1, p)) % modulus
if c4 > modulus // 2:
    c4 -= modulus
JAC[4] = abs(c4)
assert c4 > 0           # the sum of the principal (n-1)-minors, n c, is the product of the nonzero eigenvalues
assert v_p(JAC[4], 2) == 289                             # Remark "The 2-part"
mu4 = np.linalg.eigvals(DH4.astype(float))
mu4 = mu4[np.argsort(np.abs(mu4))][1:]
logJ4 = float(np.sum(np.log(np.abs(mu4)))) - math.log(LEV[4].n)
assert abs(math.log(JAC[4]) - logJ4) < 1e-9 * logJ4     # cofactor = product of nonzero eigenvalues / n
prod3 = math.prod(d for d, _ in DETS[3].values())
PROD[3] = prod3
assert Fraction(prod3) == Fraction(4 * JAC[4], JAC[3])
print(f"(CN) layer Y_4/Y_3 (|H| = 4): |Jac^H(Y_4)| reconstructed exactly from {len(primes)} "
      f"31-bit primes ({len(str(JAC[4]))} digits, v_2 = 289, v_7 = {v_p(JAC[4], 7)}, "
      f"log/n = {math.log(JAC[4]) / 768:.6f}) in {time.time() - t1:.0f} s; "
      f"prod_(chi != 1) det DH_chi = 4 |Jac^H(Y_4)| / |Jac^H(Y_3)| exactly; the product has "
      f"{len(str(prod3))} digits, v_2 = {v_p(prod3, 2)}, v_3 = {v_p(prod3, 3)}, v_7 = {v_p(prod3, 7)}")
# layer 4 (Y_5/Y_4): only the injectivity of the seven twisted blocks is checked (numerically);
# |Jac^H(Y_5)| is not computed.
print(f"     layer Y_5/Y_4 (|H| = 8): sum_chi log det DH_chi = "
      f"{sum(l for _, l in DETS[4].values()):.6f} (numeric only; no independent |Jac^H(Y_5)|)")

# ---------------------------------------------------------------- (KR) kernel of the pushforward
for m in (1, 2):
    Lb, Lc = LEV[m], LEV[m + 1]
    pi, H = LAYERS[m]
    DHc = Lc.DH()
    fibres = {}
    for j in range(Lc.n):
        fibres.setdefault(int(pi[j]), []).append(j)
    base = [fb[0] for fb in fibres.values()]            # one lift per vertex (the section S)
    coord = [j for fb in fibres.values() for j in fb[1:]]  # basis e_j - e_{fb[0]} of ker Pi
    # generators of ker Pi  cap  im DH_{m+1} = DH(ker Pi) + Z DH(S 1)
    gens = []
    for fb in fibres.values():
        for j in fb[1:]:
            gens.append(DHc[:, j] - DHc[:, fb[0]])
    s1 = np.zeros(Lc.n, dtype=np.int64)
    s1[base] = 1
    gens.append(DHc @ s1)
    Gm = np.array(gens, dtype=np.int64)
    for g in Gm:                                        # each generator lies in ker Pi
        assert (np.bincount(pi, weights=g, minlength=Lb.n) == 0).all()
    Cm = Gm[:, coord]                                   # coordinates in the basis of ker Pi
    inv = invariant_factors(Cm.tolist())
    assert len(inv) == len(coord)
    kernel_order = math.prod(inv)
    assert kernel_order * JAC[m] == JAC[m + 1]          # surjectivity of Jac_{m+1} -> Jac_m
    assert kernel_order * len(H) == PROD[m]
    print(f"(KR) layer Y_{m + 1}/Y_{m}: |ker(Jac^H(Y_{m + 1}) -> Jac^H(Y_{m}))| = {len(str(kernel_order))}-digit "
          f"integer = |Jac^H(Y_{m + 1})|/|Jac^H(Y_{m})| = prod det DH_chi / |H| (lattice index, exact); "
          f"v_2 = {v_p(kernel_order, 2)}")

print(f"\nALL CHECKS PASSED: (OR) (PF) (BL) (SE) (SM) (CN) (KR).   [{time.time() - t0:.0f} s]")
