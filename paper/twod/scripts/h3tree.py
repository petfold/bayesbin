"""Bayesian binning over the H3 cell hierarchy, with Poisson counts and a Gamma prior per bin.

Each node (an H3 cell, or the whole world) is one bin with probability rho, or its children are
binned the same way; a bin's rate is integrated out. A child with no events below it is one empty
bin (an empty region has nothing to split on). Exposure is area × days, a region's area being the
sum of its finest cells' areas, so every partition sees the same exposures.
"""

import heapq

import h3
import numpy as np
from scipy.special import gammaln

from common import RES

ALL = sorted(c for r0 in h3.get_res0_cells() for c in h3.cell_to_children(r0, RES))
AREA = {c: h3.cell_area(c, "km^2") for c in ALL}
EARTH = sum(AREA.values())

# the prior's grid: shape alpha, prior mean = global rate / scale, leaf probability rho
ALPHAS = [0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0]
SCALES = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0]
RHOS = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 0.7]


def region_area(c):
    return AREA[c] if h3.get_resolution(c) == RES else sum(AREA[d] for d in h3.cell_to_children(c, RES))


def logL(Y, E, a, b):
    """log ∫ λ^Y e^{-λE} Gamma(λ; a, b) dλ, without the data-only factor."""
    return a * np.log(b) - gammaln(a) + gammaln(a + Y) - (a + Y) * np.log(b + E)


class Tree:
    """The H3 tree over `cells` (finest cells with events) and their ancestors."""

    def __init__(self, cells):
        levels = [set(cells)]
        for r in range(RES - 1, -1, -1):
            levels.insert(0, {h3.cell_to_parent(c, r) for c in levels[0]})
        self.cells = ["world"] + [c for lv in levels for c in sorted(lv)]
        self.res = np.array([-1] + [h3.get_resolution(c) for c in self.cells[1:]])
        self.index = {c: i for i, c in enumerate(self.cells)}
        self.parent = np.full(len(self.cells), -1)
        ep, ecell = [], []
        for i, c in enumerate(self.cells):
            if c != "world" and self.res[i] == RES:
                continue
            kids = h3.get_res0_cells() if c == "world" else h3.cell_to_children(c, int(self.res[i]) + 1)
            for k in kids:
                if k in self.index:
                    self.parent[self.index[k]] = i
                else:
                    ep.append(i)
                    ecell.append(k)
        self.ep, self.ecell = np.array(ep), ecell
        self.earea = np.array([region_area(k) for k in ecell])
        self.leaf = np.nonzero(self.res == RES)[0]
        self.larea = np.array([AREA[self.cells[i]] for i in self.leaf])

    def totals(self, counts, days):
        """Y, E per node and E per empty child, for counts per finest cell over `days`."""
        Y, E = np.zeros(len(self.cells)), np.zeros(len(self.cells))
        Y[self.leaf] = [counts.get(self.cells[i], 0.0) for i in self.leaf]
        E[self.leaf] = self.larea * days
        np.add.at(E, self.ep, self.earea * days)
        for r in range(RES, -1, -1):
            ii = np.nonzero(self.res == r)[0]
            np.add.at(Y, self.parent[ii], Y[ii])
            np.add.at(E, self.parent[ii], E[ii])
        return Y, E, self.earea * days

    def logZ(self, Y, E, eE, a, b, rho, maximise=False):
        """(log Z of the whole tree, log Z per node, split decisions of the best partition,
        log L per node). maximise=True: the best partition's value instead of the sum."""
        lr, l1r = np.log(rho), np.log1p(-rho)
        L = logL(Y, E, a, b)
        split = np.zeros(len(self.cells))
        np.add.at(split, self.ep, logL(0.0, eE, a, b))
        Z = np.zeros(len(self.cells))
        cut = np.zeros(len(self.cells), bool)
        for r in range(RES, -2, -1):
            ii = np.nonzero(self.res == r)[0]
            if r == RES:
                Z[ii] = L[ii]
            else:
                s, lf = l1r + split[ii], lr + L[ii]
                Z[ii] = np.maximum(lf, s) if maximise else np.logaddexp(lf, s)
                cut[ii] = s > lf
            p = self.parent[ii]
            np.add.at(split, p[p >= 0], Z[ii][p >= 0])
        return Z[0], Z, cut, L

    def posterior(self, Y, E, eE, a, b, rho):
        """P(node is a bin | D) per node and per empty child, from one pass down the tree."""
        _, Z, _, L = self.logZ(Y, E, eE, a, b, rho)
        pleaf = np.where(self.res == RES, 1.0, np.exp(np.log(rho) + L - Z))
        reach = np.zeros(len(self.cells))
        reach[0] = 1.0
        for r in range(0, RES + 1):
            ii = np.nonzero(self.res == r)[0]
            reach[ii] = reach[self.parent[ii]] * (1 - pleaf[self.parent[ii]])
        return reach * pleaf, reach[self.ep] * (1 - pleaf[self.ep])

    def search_prior(self, Y, E, eE, rate0):
        """(alpha, beta, rho, log Z) maximising the evidence on the grid."""
        best = (-np.inf,)
        for a in ALPHAS:
            for s in SCALES:
                b = a / rate0 * s
                for rho in RHOS:
                    z = self.logZ(Y, E, eE, a, b, rho)[0]
                    if z > best[0]:
                        best = (z, a, b, rho)
        return best[1], best[2], best[3], best[0]

    def best_bins(self, cut):
        """The best partition's bins, as H3 cells (mixed resolutions)."""
        out, stack = [], [0]
        while stack:
            i = stack.pop()
            if cut[i]:
                stack.extend(np.nonzero(self.parent == i)[0].tolist())
                out.extend(self.ecell[j] for j in np.nonzero(self.ep == i)[0])
            else:
                out.append(self.cells[i])
        return out


def finest(cells):
    """Each bin (an H3 cell) as the list of its finest cells."""
    return [[c] if h3.get_resolution(c) == RES else list(h3.cell_to_children(c, RES)) for c in cells]


def grid_bins(r):
    groups = {}
    for c in ALL:
        groups.setdefault(h3.cell_to_parent(c, r) if r < RES else c, []).append(c)
    return list(groups.values())


def bin_totals(bins, counts, days):
    Y = np.array([sum(counts.get(c, 0.0) for c in b) for b in bins])
    E = np.array([sum(AREA[c] for c in b) for b in bins]) * days
    return Y, E


_NEIGHBOURS = {}


def neighbours(c):
    if c not in _NEIGHBOURS:
        _NEIGHBOURS[c] = [g for g in h3.grid_disk(c, 1) if g != c]
    return _NEIGHBOURS[c]


def merge(bins, Y, E, a, b, gamma=0.1):
    """Greedy merging of neighbouring bins (lists of finest cells): merge the pair that most
    raises Σ log L(bin) + n_bins·log gamma, while that rises. Lazy: a popped pair is
    re-scored and merged only if it still beats the next stored gain."""
    cell_bin = {c: k for k, bn in enumerate(bins) for c in bn}
    parent = list(range(len(bins)))
    Y, E = list(Y), list(E)
    L = [logL(y, e, a, b) for y, e in zip(Y, E)]
    nbr = [set() for _ in bins]
    for c, k in cell_bin.items():
        for g in neighbours(c):
            if cell_bin[g] != k:
                nbr[k].add(cell_bin[g])

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    def gain(i, j):
        return logL(Y[i] + Y[j], E[i] + E[j], a, b) - L[i] - L[j] - np.log(gamma)

    heap = [(-gain(i, j), i, j) for i in range(len(bins)) for j in nbr[i] if i < j]
    heapq.heapify(heap)
    while heap:
        _, i, j = heapq.heappop(heap)
        i, j = find(i), find(j)
        if i == j:
            continue
        g = gain(i, j)
        if heap and g < -heap[0][0] - 1e-12:
            heapq.heappush(heap, (-g, i, j))
            continue
        if g <= 0:
            if not heap or -heap[0][0] <= 0:
                break
            continue
        if len(nbr[i]) < len(nbr[j]):
            i, j = j, i
        parent[j] = i
        Y[i] += Y[j]
        E[i] += E[j]
        L[i] = logL(Y[i], E[i], a, b)
        for x in nbr[j]:
            x = find(x)
            if x != i:
                nbr[i].add(x)
                heapq.heappush(heap, (-gain(i, x), i, x))
        nbr[j] = set()
    groups = {}
    for k, bn in enumerate(bins):
        groups.setdefault(find(k), []).extend(bn)
    return list(groups.values())
