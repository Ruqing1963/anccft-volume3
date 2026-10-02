"""Deepening of Volume III, Paper 6: the Cartwright-Steger model of the CMSZ group and the
canonical 2-tower over X_3 (chapters/ch06-hecke-iwasawa.tex, Sections 4-5).

  (CS) u_x = 1 + alpha^x phi in D = (F_8(t)/F_2(t), sigma, t): x + 2y + 4z = 0 mod 7 on every
       triple of T, and u_x u_y u_z = 1 + t exactly;
  (NB) at s = 1 + t, the reductions y -> y + beta y^2 (beta = alpha^x e, any e != 0) have
       7 distinct images (all hyperplanes of F_8) and 7 distinct kernels (all lines), and
       "kernel of y inside image of x" holds iff a_x a_z a_y = 1 for some z: the link map is an
       isomorphism of Heawood graphs;
  (IO) u_0^(2^k) = 1 + phi^(2^k) is never central (k <= 40);
  (FR) Frattini: generators of Gamma^0 span P/P_2 = F_8; squares and commutators fill every level
       k >= 2 (modulo the centre when 3 | k), k = 2..13;
  (IM) the image of Gamma^0 in P/P_m is all of P/P_m, m <= 6; d(P_k) >= dim P_k/P_2k grows;
  (L2) level 2 of the tower has the Hecke Jacobian of Y_24;
  (J)  level 3: Jac^H by Smith form, v_2 = 99; level 4: v_2 = 289 by 2-adic elimination;
       (1/n) log|Jac^H| = 1.862909, 2.041580, 2.054414; all levels Ramanujan, min Re mu > 0;
  (PL) Macdonald's spherical Plancherel measure: mass 1, E a = 0, E|a|^2 = 7, E a^3 = 21, and
       h_H = E log|7 + a - 2 abar| = 2.063693 (the limit of (1/n) log|Jac^H| along any exhausting tower).

Run:  python code/v3p6_deep_verify.py      (numpy only; under a minute)
"""
import math
import time

import numpy as np

from a2_common import GIDX, ORBITS, T, standard_covers, torsion

t0 = time.time()


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
INV = [0] + [next(y for y in range(1, 8) if MUL[x][y] == 1) for x in range(1, 8)]


def sig(a, i):
    for _ in range(i % 3):
        a = FROB[a]
    return a


ALPHA = [1]
for _ in range(6):
    ALPHA.append(MUL[ALPHA[-1]][2])
assert len(set(ALPHA)) == 7


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


# ---------------------------------------------------------------- (CS) relations
for (x, y, z) in T:
    assert (x + 2 * y + 4 * z) % 7 == 0
    b1, b2, b3 = ALPHA[x], ALPHA[y], ALPHA[z]
    assert b1 ^ b2 ^ b3 == 0                                            # phi-coefficient
    assert MUL[b1][FROB[b2]] ^ MUL[b1][FROB[b3]] ^ MUL[b2][FROB[b3]] == 0   # phi^2-coefficient
    assert MUL[MUL[b1][FROB[b2]]][FROB[FROB[b3]]] == 1                  # phi^3 = t coefficient
R7 = Ring(7)
for (x, y, z) in T:
    assert R7.mul(R7.mul(u(R7, x), u(R7, y)), u(R7, z)) == (1, 0, 0, 1, 0, 0, 0)   # 1 + t
print("(CS) x + 2y + 4z = 0 (mod 7) on T; u_x u_y u_z = 1 + t exactly (21 triples)")

# ---------------------------------------------------------------- (NB) neighbours and link
adjB = {(x, y) for (x, z, y) in T}                  # a_x v0 ~ a_y^{-1} v0 in the building of Gamma
for e in range(1, 8):
    images, kernels = [], []
    for x in range(7):
        beta = MUL[ALPHA[x]][e]
        img = frozenset(y ^ MUL[beta][FROB[y]] for y in range(8))
        ker = frozenset(y for y in range(8) if y ^ MUL[beta][FROB[y]] == 0)
        assert len(img) == 4 and len(ker) == 2 and ker == {0, INV[beta]}
        images.append(img)
        kernels.append(ker)
    assert len(set(images)) == 7 and len(set(kernels)) == 7
    adjBp = {(x, y) for x in range(7) for y in range(7) if kernels[y] <= images[x]}
    assert adjBp == adjB and len(adjB) == 21
print("(NB) for every e: 7 distinct hyperplanes, 7 distinct lines, and the incidence equals the "
      "link of the CMSZ building")

# ---------------------------------------------------------------- (IO) infinite order
for k in range(41):
    j = 2 ** k
    assert j % 3 != 0                               # phi^j is central iff 3 | j
print("(IO) u_0^(2^k) = 1 + phi^(2^k), never central: u_0 has infinite order modulo the centre")

# ---------------------------------------------------------------- (FR) Frattini levels
def span_f2(vectors):
    basis = []
    for v in vectors:
        for b in basis:
            v = min(v, v ^ b)
        if v:
            basis.append(v)
    return len(basis)


R2 = Ring(2)
u0 = u(R2, 0)
pw = {0: R2.one(), 1: u0, 2: R2.mul(u0, u0)}
lvl1 = []
for s in range(3):
    for x in range(7):
        g = R2.mul(R2.mul(pw[s], u(R2, x)), R2.inv(pw[(s + 1) % 3]))
        lvl1.append(g[1])
assert span_f2(lvl1) == 3
for k in range(2, 14):
    vecs = []
    for i in range(1, k):
        for a in range(8):
            for b in range(8):
                vecs.append(MUL[a][sig(b, i)] ^ MUL[b][sig(a, k - i)])   # [1+a phi^i, 1+b phi^(k-i)]
    if k % 2 == 0:
        vecs += [MUL[a][sig(a, k // 2)] for a in range(8)]             # (1 + a phi^(k/2))^2
    if k % 3:
        assert span_f2(vecs) == 3
    else:
        assert span_f2(vecs + [1]) == 3             # together with the centre F_2
print("(FR) generators of Gamma^0 span P/P_2 = F_8; squares and commutators fill levels 2..13 "
      "(mod centre): Phi(P) = P_2, d(P) = 3")

# ---------------------------------------------------------------- (IM) images and rank
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


for m in range(1, 7):
    R = Ring(m)
    full = 2 ** (3 * (m - 1) - (m - 1) // 3)
    assert len(closure(R, schreier(R))) == full
ranks = [3 * k - ((2 * k - 1) // 3 - (k - 1) // 3) for k in range(1, 9)]   # dim P_k/P_2k
print(f"(IM) |image of Gamma^0 in P/P_m| = |P/P_m| = 2^(3(m-1) - floor((m-1)/3)) for m <= 6; "
      f"d(P_k) >= dim P_k/P_2k = {ranks}: infinite rank")


# ---------------------------------------------------------------- tower levels
def level(m):
    R = Ring(m)
    th = schreier(R)
    G = sorted(closure(R, th))
    verts = [(g, s) for g in G for s in range(3)]
    vid = {v: i for i, v in enumerate(verts)}
    n = len(verts)
    step = lambda v, x: (R.normalize(R.mul(v[0], th[GIDX[(v[1], x)]])), (v[1] + 1) % 3)
    A1 = np.zeros((n, n), dtype=np.int64)
    for v in verts:
        for x in range(7):
            A1[vid[v], vid[step(v, x)]] += 1
        for (x, y, z) in T:
            assert step(step(step(v, x), y), z) == v
    assert A1.max() == 1 and not np.diag(A1).any() and not (A1 * A1.T).any()
    return n, A1


def v2_smith(M, K=30):
    mod = np.int64(1 << K)
    sub = np.array(M, dtype=np.int64) % mod
    vals = []
    while sub.shape[0] and sub.shape[1]:
        low = sub & (-sub)
        V = np.where(sub == 0, K, np.log2(np.where(low == 0, 1, low)).astype(np.int64))
        k = int(V.argmin())
        i, j = divmod(k, sub.shape[1])
        vmin = int(V[i, j])
        if vmin >= K:
            break
        vals.append(vmin)
        uinv = pow(int(sub[i, j]) >> vmin, -1, 1 << K)
        colfac = ((sub[:, j] >> vmin) * uinv) % mod
        sub = (sub - (colfac[:, None] * sub[i][None, :]) % mod) % mod
        sub = np.delete(np.delete(sub, i, 0), j, 1)
    return vals


Y24 = standard_covers()["Y_24"]
RESULTS = {}
for m in (2, 3, 4):
    n, A1 = level(m)
    DH = 7 * np.eye(n, dtype=np.int64) + A1 - 2 * A1.T
    a = np.linalg.eigvals(A1.astype(float))
    mu = 7 + a - 2 * np.conj(a)
    order = np.argsort(np.abs(mu))
    logJ = float(np.sum(np.log(np.abs(mu[order[1:]])))) - math.log(n)
    ram = all(np.abs(np.abs(np.roots([1, -x / 2, np.conj(x) / 2, -1])) - 1).max() < 1e-6
              for x in a if abs(abs(x) - 7) > 1e-6)
    remin = float(np.min(np.real(mu[order[1:]])))
    v2 = sum(v2_smith(DH))
    RESULTS[m] = (n, logJ / n, v2, ram, remin)
    if m == 2:
        assert sorted(torsion(DH.tolist())) == sorted(torsion(Y24.DH.tolist()))
        assert abs(math.log(math.prod(torsion(DH.tolist()))) - logJ) < 1e-6
    if m == 3:
        J = torsion(DH.tolist())
        assert abs(math.log(math.prod(J)) - logJ) < 1e-6
        assert sum(((d & -d).bit_length() - 1) for d in J) == v2
    print(f"(J) level {m}: n = {n}, (1/n) log|Jac^H| = {logJ / n:.6f}, v_2|Jac^H| = {v2}, "
          f"Ramanujan = {ram}, min Re mu = {remin:.4f}")
assert RESULTS[2][2] == 18 and RESULTS[3][2] == 99 and RESULTS[4][2] == 289
assert abs(RESULTS[2][1] - 1.862909) < 1e-5 and abs(RESULTS[3][1] - 2.041580) < 1e-5
assert abs(RESULTS[4][1] - 2.054414) < 1e-5
assert all(r[3] and r[4] > 0.76 for r in RESULTS.values())
print("(L2) level 2 has the Hecke Jacobian of Y_24: (Z/7)^2 + Z/49 + (Z/392)^5 + Z/1176")

# ---------------------------------------------------------------- (PL) Plancherel limit
N = 1200
th = np.linspace(0, 2 * math.pi, N, endpoint=False) + math.pi / N
T1, T2 = np.meshgrid(th, th, indexing="ij")
Z = [np.exp(1j * T1), np.exp(1j * T2)]
Z.append(1 / (Z[0] * Z[1]))
w = np.ones_like(T1)
for i in range(3):
    for j in range(i + 1, 3):
        r = Z[j] / Z[i]
        w = w * np.abs(1 - r) ** 2 / np.abs(1 - r / 2) ** 2
w /= w.sum()
a = 2 * (Z[0] + Z[1] + Z[2])
assert abs((w * a).sum()) < 1e-9 and abs((w * np.abs(a) ** 2).sum() - 7) < 1e-9
assert abs((w * a ** 3).sum() - 21) < 1e-9
hH = float((w * np.log(np.abs(7 + a - 2 * np.conj(a)))).sum())
assert abs(hH - 2.063693) < 1e-6
assert RESULTS[2][1] < RESULTS[3][1] < RESULTS[4][1] < hH
print(f"(PL) Plancherel: E|a|^2 = 7, E a^3 = 21; h_H = {hH:.6f}; tower values "
      f"{[round(RESULTS[m][1], 6) for m in (2, 3, 4)]} increase towards it")

print(f"\nALL CHECKS PASSED: (CS) (NB) (IO) (FR) (IM) (L2) (J) (PL).   [{time.time() - t0:.0f} s]")


