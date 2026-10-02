"""Verification battery for Volume III, Paper 2 (chapters/ch02-simplicial-sandpiles.tex).

  (S) sign coherence: which of Delta_0, Delta_1, U_1 = d2 d2^T, Delta_2 are switching
      equivalent to Z-matrices (Y_21, Y_24, Y_42, thin torus T_6); Delta^H has
      Delta^H_xy Delta^H_yx = -q < 0 on every edge;
  (N) the edge toppling rule of Delta_1 is not abelian: explicit two-site example, and
      randomized stabilizations on Y_21 (fixed seed);
  (K) the 1-skeleton of Y_21 is K_{7,7,7}; Tr A1^3 = 3 * 7^3;
  (H) Hodge factorization |Jac_1| = n |Jac_0| |Jac_2| disc H_2(Y;Z) on Y_21, Y_24, and its
      b_1 > 0 form |Jac_1| disc(ker Delta_1) = n |Jac_0| |Jac_2| disc H_2 on T_6;
  (D) Dhar's mean avalanche size 1^T G 1 / (n-1), exact rationals on Y_21, Y_24, Y_42,
      and a Monte Carlo check on Y_21;
  (E) entropy per vertex (1/n) log|Jac_0| on the five covers; Jac_0 of Y_84, Y_168 against
      Vol. I-II, Ch. X.

Run:  python code/v3p2_verify.py      (numpy only; a few minutes)
"""
import math
import time
from fractions import Fraction

import numpy as np

from a2_common import (bareiss_det, integer_kernel, rational_solve, standard_covers,
                       thin_torus, torsion)

t0 = time.time()
COV = standard_covers()
Y21, Y24, Y42 = COV["Y_21"], COV["Y_24"], COV["Y_42"]
T6 = thin_torus(6)
q = 2


def switchable(M):
    """True iff some s in {+-1}^N has s_i s_j M_ij <= 0 for all i != j."""
    N = M.shape[0]
    s = [0] * N
    for st in range(N):
        if s[st]:
            continue
        s[st] = 1
        stack = [st]
        while stack:
            i = stack.pop()
            for j in np.nonzero(M[i])[0]:
                if j == i:
                    continue
                want = -s[i] if M[i, j] > 0 else s[i]
                if s[j] == 0:
                    s[j] = want
                    stack.append(j)
                elif s[j] != want:
                    return False
    return True


# ---------------------------------------------------------------- (S) sign coherence
print("(S) switching equivalent to a Z-matrix?")
for X in (Y21, Y24, Y42, T6):
    res = {k: switchable(getattr(X, k)) for k in ("L0", "L1", "U1", "L2")}
    print(f"    {X.name:6s} {res}")
    thick = X.q >= 2
    assert res["L0"] and not res["L1"] and not res["U1"]
    assert res["L2"] == (not thick)          # Delta_2 coherent exactly in the thin case
    if thick:
        DH = X.DH
        prods = {int(DH[x, y] * DH[y, x]) for (x, y) in X.edges}
        assert prods == {-q}
# structure of Delta_1: diagonal q+3, entries 0/+-1, 2q(2q+1) = 20 neighbours
L1 = Y21.L1
E = L1.shape[0]
off = L1[~np.eye(E, dtype=bool)]
assert set(np.diag(L1)) == {q + 3} and set(off.tolist()) == {-1, 0, 1}
assert set(((L1 != 0).sum(1) - 1).tolist()) == {2 * q * (2 * q + 1)}
for (a, b, c) in Y21.tris:                        # no exchange inside a chamber
    e1, e2 = Y21.eidx[(a, b)], Y21.eidx[(b, c)]
    assert L1[e1, e2] == 0

# ---------------------------------------------------------------- (N) non-abelian toppling
THR = q + 3
e, f = map(int, np.argwhere(np.triu(L1, 1) > 0)[0])
z = np.zeros(E, dtype=np.int64)
z[e] = z[f] = THR
z_ef = z - L1[:, e]                               # topple e first: f drops to THR - 1
z_fe = z - L1[:, f]                               # topple f first: e drops to THR - 1
assert z_ef.max() < THR and z_fe.max() < THR and not np.array_equal(z_ef, z_fe)
assert z_ef.min() == -1                           # a neighbour with entry +1 goes negative


# Hecke-Laplacian example of Corollary V3P2:cor:nonabelian on every thick cover
for X in (Y21, Y24, Y42):
    DH, th = X.DH, q ** 3 - 1
    x, y = X.edges[0]
    z = np.zeros(X.n, dtype=np.int64)
    z[x] = z[y] = th
    zA = z - DH[:, y]
    zB = z - DH[:, x]
    assert zB[y] >= th
    zB = zB - DH[:, y]
    assert zA.max() < th and zB.max() < th and not np.array_equal(zA, zB)
    assert zA[x] == th - 1 and zB[x] == -1 and zB[y] == q


def stabilize(z, rng, cap=200000):
    z = z.copy()
    for cnt in range(cap):
        uns = np.nonzero(z >= THR)[0]
        if len(uns) == 0:
            return z, True
        k = uns[0] if rng is None else rng.choice(uns)
        z -= L1[:, k]
    return z, False


rng = np.random.default_rng(1)
st = {"trials": 300, "nonterminating": 0, "order_dependent": 0, "negative_final": 0}
for trial in range(300):
    z = rng.integers(0, THR, E)
    z[rng.integers(E)] += THR
    z1, ok1 = stabilize(z, None)
    z2, ok2 = stabilize(z, np.random.default_rng(trial))
    if not (ok1 and ok2):
        st["nonterminating"] += 1
    elif not np.array_equal(z1, z2):
        st["order_dependent"] += 1
    if ok1 and z1.min() < 0:
        st["negative_final"] += 1
print(f"(N) explicit pair e={e}, f={f}: final states differ; random trials on Y_21: {st}")
assert st["order_dependent"] > 0 and st["negative_final"] > 0

# ---------------------------------------------------------------- (K) skeleton of Y_21
types = np.array(Y21.tau)
sk = (Y21.A1 + Y21.A1.T) > 0
assert np.array_equal(sk, types[:, None] != types[None, :])        # K_{7,7,7}
assert int(np.trace(Y21.A1 @ Y21.A1 @ Y21.A1)) == 3 * 7 ** 3
print("(K) 1-skeleton of Y_21 = K_{7,7,7}; Tr A1^3 = 3*343 = 1029")


# ---------------------------------------------------------------- (H) Hodge factorization
def disc(basis):
    B = np.array(basis, dtype=object)
    return bareiss_det((B @ B.T).tolist()) if len(basis) else 1


o = math.prod
HODGE = {}
for X in (Y21, Y24, T6):
    J0, J1, J2 = torsion(X.L0.tolist()), torsion(X.L1.tolist()), torsion(X.L2.tolist())
    H2 = integer_kernel(X.d2.tolist())
    K1 = integer_kernel(X.L1.tolist())
    b1, b2 = len(K1), len(H2)
    lhs = o(J1) * disc(K1)
    rhs = X.n * o(J0) * o(J2) * disc(H2)
    assert lhs == rhs, X.name
    if b1 == 0:
        assert bareiss_det(X.L1.tolist()) == o(J1)
    HODGE[X.name] = (b1, b2, o(J0), o(J1), o(J2), disc(H2), disc(K1))
    print(f"(H) {X.name}: b1={b1}, b2={b2}, |Jac0|={o(J0)}, |Jac2|={o(J2)}, "
          f"disc H2={disc(H2)}, disc ker D1={disc(K1)}, |Jac1|={o(J1)}  OK")
assert HODGE["Y_21"][5] == 3 * 7 ** 29
assert HODGE["Y_21"][2] == 3 * 2 ** 18 * 7 ** 19 == 21 * 14 ** 18   # tau(K_{7,7,7})
assert torsion(Y21.L0.tolist()) == [2] + [14] * 15 + [98, 294]          # Vol. I-II, Ch. X

# ---------------------------------------------------------------- (D) Dhar mean avalanche
DHAR = {}
for X in (Y21, Y24, Y42):
    Lt = X.L0[1:, 1:]
    sol = rational_solve(Lt.tolist(), [1] * (X.n - 1))
    DHAR[X.name] = sum(sol) / (X.n - 1)
    print(f"(D) {X.name}: <s> = {DHAR[X.name]} = {float(DHAR[X.name]):.6f}")
    # closed form n^2 L+_ss/(n-1) and spectral bounds (Theorem V3P2:thm:mean)
    ev = np.linalg.eigvalsh(X.L0.astype(float))
    Lp = np.linalg.pinv(X.L0.astype(float))
    closed = X.n ** 2 * Lp[0, 0] / (X.n - 1)
    assert abs(closed - float(DHAR[X.name])) < 1e-9
    assert X.n / ev.max() - 1e-9 <= closed <= X.n / sorted(ev)[1] + 1e-9
assert DHAR["Y_21"] == Fraction(29, 20)
assert DHAR["Y_24"] == Fraction(7031, 4186) and DHAR["Y_42"] == Fraction(489107, 157768)

n, d = Y21.n, 2 * (q * q + q + 1)
nb = [np.nonzero(Y21.L0[v] < 0)[0] for v in range(n)]
h = np.zeros(n, dtype=np.int64)
mc = np.random.default_rng(7)


def add(x):
    h[x] += 1
    cnt, stack = 0, [x]
    while stack:
        v = stack.pop()
        if v == 0 or h[v] < d:
            continue
        h[v] -= d
        cnt += 1
        for w in nb[v]:
            h[w] += 1
            if h[w] >= d:
                stack.append(w)
        if h[v] >= d:
            stack.append(v)
    h[0] = 0
    return cnt


for _ in range(20000):
    add(mc.integers(1, n))
N = 200000
mean_mc = sum(add(mc.integers(1, n)) for _ in range(N)) / N
print(f"(D) Monte Carlo on Y_21: <s> = {mean_mc:.5f} (exact 1.45)")
assert abs(mean_mc - 1.45) < 0.02

# ---------------------------------------------------------------- (E) entropy per vertex
X_JAC0 = {
    "Y_84": [(2, 12), (4, 7), (16, 2), (1184, 1), (8288, 5), (16576, 6), (215488, 4),
             (1508416, 1), (4525248, 1)],
    "Y_168": [(2, 31), (16, 6), (592, 6), (1184, 18), (4736, 1), (33152, 3), (430976, 12),
              (3016832, 1), (9050496, 1)],
}
for name, X in COV.items():
    ev = np.sort(np.linalg.eigvalsh(X.L0.astype(float)))
    logk = float(np.sum(np.log(ev[1:])) - math.log(X.n))
    line = f"(E) {name}: gap(Delta_0) = {ev[1]:.6f}, (1/n) log|Jac_0| = {logk / X.n:.6f}"
    if name in X_JAC0:
        logx = sum(m * math.log(a) for a, m in X_JAC0[name])
        line += f"   [Ch. X value {logx / X.n:.6f}]"
        if name == "Y_168":                       # the maximal abelian cover is unique
            assert abs(logx - logk) < 1e-6 * logk
    print(line)
print(f"    log 14 = {math.log(14):.6f}")

print(f"\nALL CHECKS PASSED: (S) (N) (K) (H) (D) (E).   [{time.time() - t0:.0f} s]")
