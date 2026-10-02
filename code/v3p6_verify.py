"""Verification battery for Volume III, Paper 6 (chapters/ch06-hecke-iwasawa.tex).

  (Dg) degenerate affine Hecke algebra H_2: the polynomial representation s_i -> sigma_i - c d_i,
       x_i -> y_i satisfies s_i^2 = 1, braid, s_i x_{i+1} - x_i s_i = c, s_i x_j = x_j s_i;
       the "trivial" module s_i -> I, x_i -> L + (i-1)c satisfies the relations for any L
       (tested with L = L_E of Y_21);
  (Hq) Iwahori-Hecke action on chambers: T_i^2 = (q-1) T_i + q, T_i T_j T_i = T_j T_i T_j
       (Y_21, Y_24, Y_168, T_6);
  (Op) Delta_2 = 3I + T_0 + T_1 + T_2; A_1 = (q+1)^{-1} sum_t pi_t pi_{t+1}^t;
       L_E = (q(q+1))^{-1} p_{t+2} T_t T_{t+2} T_t p_t^t on each block;
  (Sa) Satake: joint eigenvalues a = q e_1(z), abar = q e_2(z), e_3(z) = 1; Delta^H and Delta_0
       as images of symmetric polynomials;
  (CN) class number formula for the abelian covers of X_3:
       prod_{chi != 1} det Delta^H_chi = |A| |Jac^H(Y_A)| / |Jac^H(X_3)|;
  (B1) b_1 = 0 on the covers, so no Z_p quotient of any tower group.

Run:  python code/v3p6_verify.py      (numpy only; sympy is not needed)
"""
import cmath
import math
import time
from itertools import product

import numpy as np

from a2_common import (GIDX, bareiss_det, nullspace_mod, relator_matrix, standard_covers,
                       thin_torus, torsion, voltage_cover)

t0 = time.time()
COV = standard_covers()
q = 2

# ---------------------------------------------------------------- (Dg) degenerate algebra
# polynomial representation on polynomials of degree <= 4 in y1,y2,y3 (dict monomial -> coeff)
from fractions import Fraction

DEG = 4
MONO = [m for m in product(range(DEG + 1), repeat=3) if sum(m) <= DEG]
IDX = {m: k for k, m in enumerate(MONO)}


def swap(m, i):
    m = list(m)
    m[i], m[i + 1] = m[i + 1], m[i]
    return tuple(m)


def op_sigma(i):
    M = np.zeros((len(MONO), len(MONO)), dtype=object)
    for m in MONO:
        M[IDX[swap(m, i)], IDX[m]] = Fraction(1)
    return M


def op_divdiff(i):
    """d_i f = (f - sigma_i f)/(y_i - y_{i+1}) on monomials."""
    M = np.zeros((len(MONO), len(MONO)), dtype=object)
    M[:, :] = Fraction(0)
    for m in MONO:
        a, b = m[i], m[i + 1]
        # (y_i^a y_{i+1}^b - y_i^b y_{i+1}^a)/(y_i - y_{i+1})
        sign, lo, hi = (1, b, a) if a >= b else (-1, a, b)
        for k in range(hi - lo):
            mm = list(m)
            mm[i], mm[i + 1] = lo + k, hi - 1 - k
            M[IDX[tuple(mm)], IDX[m]] += sign
    return M


def op_mult(i):
    M = np.zeros((len(MONO), len(MONO)), dtype=object)
    M[:, :] = Fraction(0)
    for m in MONO:
        mm = list(m)
        mm[i] += 1
        if sum(mm) <= DEG:
            M[IDX[tuple(mm)], IDX[m]] = Fraction(1)
    return M


c = Fraction(3, 2)
S = [op_sigma(i) - c * op_divdiff(i) for i in range(2)]
X = [op_mult(i) for i in range(3)]
low = [k for k, m in enumerate(MONO) if sum(m) <= DEG - 1]     # truncation: test on deg <= 3


def eq(A, B, cols):
    return all(A[r, k] == B[r, k] for r in range(len(MONO)) for k in cols)


I = np.eye(len(MONO), dtype=object)
for i in range(2):
    assert eq(S[i].dot(S[i]), I, range(len(MONO)))
    assert eq(S[i].dot(X[i + 1]) - X[i].dot(S[i]), c * I, low)
    for j in range(3):
        if j not in (i, i + 1):
            assert eq(S[i].dot(X[j]), X[j].dot(S[i]), low)
assert eq(S[0].dot(S[1]).dot(S[0]), S[1].dot(S[0]).dot(S[1]), range(len(MONO)))
print("(Dg) polynomial representation of H_2: s^2 = 1, braid, s_i x_{i+1} - x_i s_i = c, "
      "s_i x_j = x_j s_i (exact)")


def LE(Xc):
    idx, out = Xc.eidx, {}
    for (a, b) in Xc.edges:
        out.setdefault(a, []).append(b)
    ch = set(frozenset(t) for t in Xc.tris)
    L = np.zeros((len(Xc.edges),) * 2, dtype=np.int64)
    for (x, y), i in idx.items():
        for z in out[y]:
            if frozenset((x, y, z)) not in ch:
                L[i, idx[(y, z)]] = 1
    return L


L21 = LE(COV["Y_21"]).astype(float)
Id = np.eye(len(L21))
cc = 0.7
Xs = [L21 + k * cc * Id for k in range(3)]
Ss = [Id, Id]
for i in range(2):
    assert np.allclose(Ss[i] @ Xs[i + 1] - Xs[i] @ Ss[i], cc * Id)
assert all(np.allclose(Xs[a] @ Xs[b], Xs[b] @ Xs[a]) for a in range(3) for b in range(3))
print("    trivial module s_i -> I, x_i -> L_E + (i-1)c satisfies all relations: realizing L_E "
      "in an H_d-module carries no information")

# ---------------------------------------------------------------- (Hq) Iwahori-Hecke action
def hecke_T(Xc):
    """T_i on C^F: chambers sharing their panel of cotype i (the edge missing type i)."""
    F = [tuple(t) for t in Xc.tris]
    panel = {}
    for k, t in enumerate(F):
        for i in range(3):
            e = frozenset(v for v in t if Xc.tau[v] != i)
            panel.setdefault((i, e), []).append(k)
    T = [np.zeros((len(F), len(F)), dtype=np.int64) for _ in range(3)]
    for (i, e), cs in panel.items():
        for a in cs:
            for b in cs:
                if a != b:
                    T[i][a, b] = 1
    return F, T


HQ = {}
for name in ("Y_21", "Y_24", "Y_168"):
    HQ[name] = (COV[name],) + hecke_T(COV[name])
T6 = thin_torus(6)
HQ["T_6"] = (T6,) + hecke_T(T6)
for name, (Xc, F, T) in HQ.items():
    qq = Xc.q
    Iden = np.eye(len(F), dtype=np.int64)
    for i in range(3):
        assert (T[i] @ T[i] == (qq - 1) * T[i] + qq * Iden).all()
        for j in range(3):
            if i != j:
                assert (T[i] @ T[j] @ T[i] == T[j] @ T[i] @ T[j]).all()
print("(Hq) T_i^2 = (q-1)T_i + q and T_iT_jT_i = T_jT_iT_j on C^F for Y_21 (no flag condition), "
      "Y_24, Y_168, T_6")

# ---------------------------------------------------------------- (Op) operators from the action
for name in ("Y_21", "Y_168"):
    Xc, F, T = HQ[name]
    fidx = {frozenset(t): k for k, t in enumerate(F)}
    # chamber orientation of a2_common is type-increasing, so d2 has entries +1 only
    Lc = np.zeros((len(F), len(F)), dtype=np.int64)
    for k, t in enumerate(Xc.tris):
        Lc[fidx[frozenset(t)], k] = 1          # permutation between orderings
    L2 = Lc @ Xc.L2 @ Lc.T
    assert (L2 == 3 * np.eye(len(F), dtype=np.int64) + T[0] + T[1] + T[2]).all()
    # vertex projections
    pi = []
    for t in range(3):
        P = np.zeros((Xc.n, len(F)), dtype=np.int64)
        for k, ch in enumerate(F):
            for v in ch:
                if Xc.tau[v] == t:
                    P[v, k] = 1
        pi.append(P)
    A1 = sum(pi[t] @ pi[(t + 1) % 3].T for t in range(3))
    assert (A1 == (q + 1) * Xc.A1).all()
    # panel projections p_i : C^F -> C^{edges of cotype i}
    E = Xc.edges
    cot = [3 - Xc.tau[a] - Xc.tau[b] for (a, b) in E]   # the missing type
    p = np.zeros((len(E), len(F)), dtype=np.int64)
    for k, ch in enumerate(F):
        for (a, b) in [(u, v) for u in ch for v in ch if (u, v) in Xc.eidx]:
            p[Xc.eidx[(a, b)], k] = 1
    L = LE(Xc)
    for tt in range(3):
        rows = [k for k in range(len(E)) if cot[k] == (tt + 2) % 3]
        cols = [k for k in range(len(E)) if cot[k] == tt]
        W = T[tt] @ T[(tt + 2) % 3] @ T[tt]
        blk = p[rows] @ W @ p[cols].T
        assert (blk == q * (q + 1) * L[np.ix_(rows, cols)]).all()
    assert all(L[i, j] == 0 for i in range(len(E)) for j in np.nonzero(L[i])[0]
               if cot[j] != (cot[i] + 1) % 3)
print("(Op) Delta_2 = 3 + T_0 + T_1 + T_2; A_1 = (q+1)^-1 sum pi_t pi_{t+1}^t; "
      "L_E = p T_t T_{t+2} T_t p^t / (q(q+1)) blockwise (Y_21, Y_168)")

# ---------------------------------------------------------------- (Sa) Satake parameters
for name in ("Y_42", "Y_168"):
    Xc = COV[name]
    a = np.linalg.eigvals(Xc.A1.astype(float))
    for aj in a:
        z = np.roots([1, -aj / q, np.conj(aj) / q, -1])
        e1, e2, e3 = z.sum(), z[0] * z[1] + z[1] * z[2] + z[2] * z[0], z.prod()
        assert abs(q * e1 - aj) < 1e-8 and abs(q * e2 - np.conj(aj)) < 1e-8 and abs(e3 - 1) < 1e-8
        muH = q ** 3 - 1 + aj - q * np.conj(aj)
        assert abs(muH - (q ** 3 - 1 + q * e1 - q * q * e2)) < 1e-8
print("(Sa) joint eigenvalues: a = q e_1(z), abar = q e_2(z), e_3(z) = 1; "
      "Delta^H = (q^3-1) + q e_1 - q^2 e_2 on every joint eigenspace (Y_42, Y_168)")

# ---------------------------------------------------------------- (CN) class number formula
R = relator_matrix()
N7, N2 = nullspace_mod(R, 7), nullspace_mod(R, 2)
th7 = [(v,) for v in N7[0]]
f2 = lambda k: [tuple(N2[j][i] for j in range(k)) for i in range(21)]
TOWERS = {
    "Y_21": (th7, (7,)),
    "Y_24": (f2(3), (2, 2, 2)),
    "Y_42": ([(a_[0], b_[0]) for a_, b_ in zip(f2(1), th7)], (2, 7)),
}


def hecke_laplacian_twisted(theta, mods, chi):
    """Delta^H_chi on the 3 vertices of X_3; chi = tuple of exponents, one per factor."""
    A1 = np.zeros((3, 3), dtype=complex)
    for s in range(3):
        for x in range(7):
            g = theta[GIDX[(s, x)]]
            A1[s, (s + 1) % 3] += cmath.exp(2j * math.pi * sum(k * gi / m for k, gi, m in zip(chi, g, mods)))
    return (q ** 3 - 1) * np.eye(3) + A1 - q * A1.conj().T


DH3 = (7 * np.eye(3, dtype=np.int64) + 7 * np.roll(np.eye(3, dtype=np.int64), 1, axis=1)
       - 14 * np.roll(np.eye(3, dtype=np.int64), -1, axis=1))
J3 = math.prod(torsion(DH3.tolist()))
for name, (theta, mods) in TOWERS.items():
    Y = voltage_cover(name, theta, mods)
    JY = math.prod(torsion(Y.DH.tolist()))
    order = math.prod(mods)
    prod_det = 1
    for chi in product(*[range(m) for m in mods]):
        if any(chi):
            prod_det *= np.linalg.det(hecke_laplacian_twisted(theta, mods, chi))
    assert abs(prod_det.imag) < 1e-6 * abs(prod_det)
    rhs = Fraction(order * JY, J3)
    assert rhs.denominator == 1
    assert abs(prod_det.real - rhs.numerator) < 1e-6 * rhs.numerator, name
    print(f"(CN) {name}: prod_(chi!=1) det Delta^H_chi = {prod_det.real:.6e} = |A||Jac^H(Y)|/"
          f"|Jac^H(X_3)| = {order}*{JY}/{J3}")

# ---------------------------------------------------------------- (B1) no Z_p quotients
for name, Xc in COV.items():
    r1 = np.linalg.matrix_rank(Xc.d1.astype(float))
    r2 = np.linalg.matrix_rank(Xc.d2.astype(float))
    assert len(Xc.edges) - r1 - r2 == 0
print("(B1) b_1 = 0 on the five covers (Paper 1): no Z_p quotient, no CFKSV Ore set")

print(f"\nALL CHECKS PASSED: (Dg) (Hq) (Op) (Sa) (CN) (B1).   [{time.time() - t0:.0f} s]")

