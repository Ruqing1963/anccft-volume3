"""Verification battery for Volume III, Paper 1 (chapters/ch01-higher-rank-buildings.tex).

Rebuilds the abelian voltage covers Y_21, Y_24, Y_42 of the three-vertex quotient X_3 of the
Cartwright--Mantero--Steger--Zappa group attached to the triangle presentation of Vol. I-II,
Ch. X, Construction 1.4, and checks every number used in Vol. III, Paper 1 exactly:

  (P) presentation axioms, H_1(X_3) = Z/2 + Z/2 + Z/14 (Reidemeister--Schreier);
  (G) simplicial, A1 A2 = A2 A1, Heawood links, |E| = |F| = 7n;
  (S) normalized link gap = 1 - sqrt(2)/3 > 1/2 at every vertex;
  (B) b_1 = 0 from ranks of d_1, d_2; H_1(Y;Z) = Tors coker d_2 by Smith form;
  (T) Tr(C^(3))^k = 0 for 3 !| k, and Tr(C^(3))^3 = Tr A1^3 - 3q|E| + 3nq^3;
  (H) Jac^H by Smith form;
  (D) d/du det(I - u C^(3)) at u = 1, computed by Jacobi's formula as n Bareiss
      determinants, equals (-1)^n n (q-1)^2 (q+1) |Jac^H(Y)|.

Run:  python code/v3p1_verify.py     (pure Python + numpy; about a minute)
"""
import itertools
import math

import numpy as np

q = 2
KAPPA = q * q + q + 1  # 7 points / 7 lines of PG(2,2)

# ---------------------------------------------------------------------------------------
# (P) triangle presentation over PG(2,2), compatible with the Singer cycle
# ---------------------------------------------------------------------------------------
ORBITS = [(0, 1, 3), (0, 2, 6), (0, 4, 5), (1, 2, 4), (1, 5, 6), (2, 3, 5), (3, 4, 6)]
T = sorted({t for (x, y, z) in ORBITS for t in [(x, y, z), (y, z, x), (z, x, y)]})
LAM = {x: {(x + 1) % 7, (x + 2) % 7, (x + 4) % 7} for x in range(7)}

assert len(T) == 21
assert all((y, z, x) in T for (x, y, z) in T)                              # (A)
assert all({y for (a, y, _) in T if a == x} == LAM[x] for x in range(7))   # (B)
assert len({(x, y) for (x, y, _) in T}) == 21                             # (C)

# Reidemeister--Schreier for Gamma^0 = ker(Gamma -> Z/3), transversal t_s = a_0^s:
# generators gamma(s,x) = t_s a_x t_{s+1}^{-1}; gamma(0,0) = gamma(1,0) = 1.
GIDX = {(s, x): 3 * x + s for s in range(3) for x in range(7)}
RELS = []
for s in range(3):
    for (x, y, z) in T:
        r = [0] * 21
        r[GIDX[(s, x)]] += 1
        r[GIDX[((s + 1) % 3, y)]] += 1
        r[GIDX[((s + 2) % 3, z)]] += 1
        RELS.append(r)
for triv in [(0, 0), (1, 0)]:
    r = [0] * 21
    r[GIDX[triv]] = 1
    RELS.append(r)


# ---------------------------------------------------------------------------------------
# exact integer linear algebra
# ---------------------------------------------------------------------------------------
def invariant_factors(M):
    """Nonzero invariant factors of an integer matrix (Smith normal form diagonal)."""
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
            if bad is not None:               # enforce divisibility of the next factor
                A[r] = [a + b for a, b in zip(A[r], A[bad])]
                continue
            inv.append(abs(p))
            r += 1
            break
    return inv


def torsion(M):
    return [d for d in invariant_factors(M) if d > 1]


def bareiss_det(M):
    """Exact determinant of an integer matrix by fraction-free Bareiss elimination."""
    M = [list(map(int, row)) for row in M]
    m, sign, prev = len(M), 1, 1
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


def nullspace_mod(M, p):
    """Basis of {v : M v = 0 mod p}, p prime."""
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


H1_X3 = torsion(RELS)
assert H1_X3 == [2, 2, 14] and len(invariant_factors(RELS)) == 21, H1_X3
N7, N2 = nullspace_mod(RELS, 7), nullspace_mod(RELS, 2)
assert len(N7) == 1 and len(N2) == 3
print(f"(P) H_1(X_3;Z) = Z/2 + Z/2 + Z/14;  dim Hom(H_1,F_7) = 1, dim Hom(H_1,F_2) = 3")


# ---------------------------------------------------------------------------------------
# voltage cover: vertices (alpha, s) in A x Z/3, arcs (alpha,s) -> (alpha + theta(gamma(s,x)), s+1)
# ---------------------------------------------------------------------------------------
def build(theta, mods):
    add = lambda a, b: tuple((u + v) % m for u, v, m in zip(a, b, mods))
    elems, frontier = {tuple([0] * len(mods))}, [tuple([0] * len(mods))]
    while frontier:                                   # subgroup generated by the voltages
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
        for (x, y, z) in T:                           # chamber {g, g a_x, g a_x a_y}
            v1 = step(v, x)
            v2 = step(v1, y)
            assert step(v2, z) == v
            tris.add(frozenset((vid[v], vid[v1], vid[v2])))
    return n, A1, sorted(tuple(sorted(t)) for t in tris)


def analyse(name, theta, mods):
    n, A1, tris = build(theta, mods)
    A2 = A1.T
    res = {"n": n}

    # (G) simplicial, commuting Hecke operators, counts
    assert A1.max() == 1 and not np.diag(A1).any() and not (A1 * A1.T).any()
    assert all(len(set(t)) == 3 for t in tris)
    assert np.array_equal(A1 @ A2, A2 @ A1)
    edges = [(i, j) for i in range(n) for j in range(n) if A1[i, j]]   # type-increasing orientation
    E, F = len(edges), len(tris)
    assert E == KAPPA * n and 3 * F == n * KAPPA * (q + 1)
    res.update(E=E, F=F)

    # (G)+(S) vertex links: Heawood graph, normalized gap 1 - sqrt(q)/(q+1)
    lam = 1 - math.sqrt(q) / (q + 1)
    for v in range(n):
        nb = [w for w in range(n) if A1[v, w] or A1[w, v]]
        pos = {w: i for i, w in enumerate(nb)}
        L = np.zeros((len(nb), len(nb)))
        for t in tris:
            if v in t:
                a, b = [w for w in t if w != v]
                L[pos[a], pos[b]] = L[pos[b], pos[a]] = 1
        deg = L.sum(1)
        assert len(nb) == 2 * KAPPA and np.all(deg == q + 1) and L.sum() == 2 * KAPPA * (q + 1)
        # bipartite points/lines by type, girth 6 (no 4-cycles: common neighbours <= 1)
        pts = [pos[w] for w in nb if A1[v, w]]
        assert not L[np.ix_(pts, pts)].any()
        common = L @ L
        assert (common - np.diag(np.diag(common))).max() <= 1
        ev = np.sort(np.linalg.eigvalsh(np.eye(len(nb)) - L / (q + 1)))
        assert abs(ev[0]) < 1e-12 and abs(ev[1] - lam) < 1e-12
    res["gap"] = lam

    # (T) traces of the cubic companion
    I, Z = np.eye(n, dtype=np.int64), np.zeros((n, n), dtype=np.int64)
    C = np.block([[A1, -q * A2, q ** 3 * I], [I, Z, Z], [Z, I, Z]]).astype(object)
    Ck, trs = np.eye(3 * n, dtype=np.int64).astype(object), []
    for _ in range(6):
        Ck = Ck.dot(C)
        trs.append(int(np.trace(Ck)))
    trA13 = int(np.trace(A1 @ A1 @ A1))
    assert int(np.trace(A1 @ A2)) == E
    assert all(trs[k - 1] == 0 for k in (1, 2, 4, 5))
    assert trs[2] == trA13 - 3 * q * E + 3 * n * q ** 3            # general identity
    flag_part = 3 * F - 3 * q * E + 3 * n * q ** 3
    assert flag_part == n * (q - 1) ** 2 * (q + 1)
    assert (trA13 - 3 * F) % 3 == 0
    res.update(trA13=trA13, threeF=3 * F, flag_part=flag_part, excess=trA13 - 3 * F,
               s=(trA13 - 3 * F) // 3, trC3=trs[2], trC6=trs[5])

    # (B) boundary maps, Betti numbers, H_1(Y;Z)
    eidx = {e: i for i, e in enumerate(edges)}
    d1 = [[0] * E for _ in range(n)]
    for (a, b), i in eidx.items():
        d1[a][i] -= 1
        d1[b][i] += 1
    d2 = [[0] * F for _ in range(E)]
    for fi, t in enumerate(tris):
        a = next(w for w in t if sum(A1[w, u] for u in t) == 1 and
                 sum(A1[u, w] for u in t) == 1)
        b = next(u for u in t if A1[a, u])
        c = next(u for u in t if u not in (a, b))
        for (x, y) in [(a, b), (b, c), (c, a)]:                     # boundary a->b->c->a
            if (x, y) in eidx:
                d2[eidx[(x, y)]][fi] += 1
            else:
                d2[eidx[(y, x)]][fi] -= 1
    D1, D2 = np.array(d1), np.array(d2)
    assert not (D1 @ D2).any()
    r1 = len(invariant_factors(d1))
    inv2 = invariant_factors(d2)
    r2 = len(inv2)
    b1, b2 = E - r1 - r2, F - r2
    assert r1 == n - 1 and b1 == 0 and b2 == n - E + F - 1
    res.update(b=(1, b1, b2), chi=n - E + F, H1=[d for d in inv2 if d > 1])

    # (H) Hecke critical group
    DH = (q ** 3 - 1) * I + A1 - q * A2
    JH = torsion(DH.tolist())
    JHord = math.prod(JH)
    assert len(invariant_factors(DH.tolist())) == n - 1

    # (D) derivative identity by Jacobi's formula: sum of n Bareiss determinants
    P1 = (-DH).astype(object)
    Pp = (-A1 + 2 * q * A2 - 3 * q ** 3 * I).astype(object)
    deriv = 0
    for i in range(n):
        M = P1.copy()
        M[:, i] = Pp[:, i]
        deriv += bareiss_det(M.tolist())
    rhs = (-1) ** n * n * (q - 1) ** 2 * (q + 1) * JHord
    assert deriv == rhs
    den = -3 * (1 - q ** 3) * (1 - q ** 6)
    assert deriv % den == 0
    res.update(JH=JH, JHord=JHord, deriv=deriv, Q0_1=deriv // den)
    return res


def report(name, r):
    print(f"--- {name}: n={r['n']}, |E|={r['E']}, |F|={r['F']}, link gap={r['gap']:.6f}")
    print(f"    Tr A1^3={r['trA13']} = 3|F| {r['threeF']} + 3s {r['excess']} (s={r['s']});"
          f"  flag part={r['flag_part']};  Tr C^3={r['trC3']};  Tr C^6={r['trC6']}")
    print(f"    (b0,b1,b2)={r['b']}, chi={r['chi']}, H_1 torsion={r['H1']}")
    print(f"    Jac^H invariants={r['JH']}")
    print(f"    d/du det(I-uC)|_1 = {r['deriv']} = (-1)^n n(q-1)^2(q+1)|Jac^H|;  Q_0(1)={r['Q0_1']}")


# ---------------------------------------------------------------------------------------
# the complexes and the values quoted in Vol. III, Paper 1, Table V3P1:tab:examples and Sec. 6
# ---------------------------------------------------------------------------------------
Y21 = analyse("Y_21", [(v,) for v in N7[0]], (7,))
report("Y_21  (A = Z/7)", Y21)
assert (Y21["n"], Y21["E"], Y21["F"]) == (21, 147, 147)
assert (Y21["trA13"], Y21["flag_part"], Y21["excess"], Y21["trC3"]) == (1029, 63, 588, 651)
assert Y21["s"] == 196 and Y21["trC6"] == 15939
assert Y21["b"] == (1, 0, 20) and Y21["chi"] == 21 and Y21["H1"] == [2] * 6
assert Y21["JH"] == [7] * 17 + [147] and Y21["JHord"] == 3 * 7 ** 19
assert Y21["deriv"] == -2154391190035524027 and Y21["Q0_1"] == 7 ** 18

Y24 = analyse("Y_24", [tuple(N2[j][i] for j in range(3)) for i in range(21)], (2, 2, 2))
report("Y_24  (A = (Z/2)^3)", Y24)
assert (Y24["trA13"], Y24["flag_part"], Y24["trC3"], Y24["trC6"]) == (1008, 72, 576, 12840)
assert Y24["b"] == (1, 0, 23) and Y24["H1"] == [2, 2, 2, 2, 14]
assert Y24["JH"] == [7, 7, 49] + [392] * 5 + [1176]

Y42_JH = [7] * 18 + [299593, 2097151, 176160684]
for coeffs in itertools.product(range(2), repeat=3):            # the seven Z/14 covers
    if not any(coeffs):
        continue
    th2 = [sum(c * N2[j][i] for j, c in enumerate(coeffs)) % 2 for i in range(21)]
    Y42 = analyse("Y_42", [(a, b) for a, b in zip(th2, N7[0])], (2, 7))
    report(f"Y_42  (A = Z/14, F_2-functional {coeffs})", Y42)
    assert (Y42["n"], Y42["E"], Y42["F"]) == (42, 294, 294)
    assert (Y42["trA13"], Y42["flag_part"], Y42["excess"], Y42["trC3"]) == (1386, 126, 504, 630)
    assert Y42["s"] == 168 and Y42["trC6"] == 15414
    assert Y42["b"] == (1, 0, 41) and Y42["H1"] == [2, 2, 2, 4, 4]
    assert Y42["JH"] == Y42_JH
    assert Y42["deriv"] == 22709397752337736246719794532064846488

print("\nALL CHECKS PASSED: (P) (G) (S) (B) (T) (H) (D) on Y_21, Y_24 and the seven Y_42.")
