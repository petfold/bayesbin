/***************************************************************************
 *   Forward-backward computation of the SDF for binsdfc.                  *
 *   Added 2026 to binsdfc 0.1 (Copyright (C) 2007 by Dominik Endres).     *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 ***************************************************************************/
#ifndef FORWARDBACKWARD_H
#define FORWARDBACKWARD_H

#include "spikecounter.h"
#include <vector>
#include <cmath>
#include <cfloat>
#include <limits>
#include <algorithm>
#include <memory>
#if defined(__SSE__) || defined(__x86_64__)
#include <xmmintrin.h>
#endif

void vexpInPlace(double *x, int n);  // vmath.cpp: x[i] = exp(x[i]), vectorised

namespace fb {

const double NEG = -std::numeric_limits<double>::infinity();
/** vexpInPlace/vexpMul return exactly 0 below exp(-700): a term dropped that way is < LOST */
const double LOST = 1e-304;

/** Flush-to-zero and denormals-are-zero for this thread while in scope. Products of the scaled
    factors can underflow into subnormals, and subnormal arithmetic takes a slow microcode path on
    x86. Anything flushed is < DBL_MIN, within the LOST bound of the callers. */
struct FlushDenormals {
#if defined(__SSE__) || defined(__x86_64__)
	unsigned csr;
	FlushDenormals() : csr(_mm_getcsr()) { _mm_setcsr(csr | 0x8040); }
	~FlushDenormals() { _mm_setcsr(csr); }
#endif
};

/** false: every computation as in the original binsdfc 0.1 (set by --virtual-spike) */
extern bool enabled;

/** log(sum_i exp(x_i)), max-shifted; -inf if all are -inf */
inline double logSumExp(const double *x, int n)
{
	double mx = NEG;
	for (int i = 0; i < n; i++) mx = fmax(mx, x[i]);
	if (mx == NEG) return NEG;
	double s = 0.0;
	for (int i = 0; i < n; i++) s += exp(x[i] - mx);
	return mx + log(s);
}

/**
\brief The central iteration (the paper's subE[]) for all m <= mmax, as matrix-vector products.

fwd[m][k] = log evidence of time indexes 0..k as m+1 bins, the last one ending at k,
for bin evidence contributions iec(a,b) (bin a..b inclusive).

Relative to the one-bin evidence b[k] = iec(0,k), one step is
    phi_m[k] = log sum_{r<k} exp(phi_{m-1}[r] + G[r][k]),   G[r][k] = iec(r+1,k) - b[k] + b[r],
and exp(G[r][k] - max_r G[r][k]) is computed once. Each step scales exp(phi) by its maximum
and multiplies: T^2/2 multiply-adds, no exp or log in the inner loop. A term lost to
underflow or to the exp cut-off (exp(-700)) is below LOST, so a column whose sum is not far above
2*T*LOST could be off by more than tol (relative); such columns are recomputed exactly with
log-sum-exp.
*/
template<class IEC>
void fastForward(int K, int mmax, const IEC &iec, std::vector<std::vector<double> > &fwd,
		 double tol = 1e-13)
{
	using std::vector;
	fwd.assign(mmax + 1, vector<double>(K, NEG));
	vector<double> b(K);
	for (int k = 0; k < K; k++) b[k] = iec(0, k);
	fwd[0] = b;
	if (mmax == 0 || K == 1) return;

	// E(r,k) = exp(G[r][k] - c[k]) for r < k, stored block-major: column block j (columns
	// k0 = 64j .. k1-1) holds its rows r = 0..k1-2 contiguously, 64 entries each, zero where k <= r.
	// A step on one block is then a sequential vector-matrix product with a fixed inner length.
	const int CB = 64, nb = (K + CB - 1) / CB;
	vector<size_t> boff(nb + 1, 0);  // block j starts at boff[j]
	for (int j = 0; j < nb; j++) {
		const int k1 = std::min(K, (j + 1) * CB);
		boff[j + 1] = boff[j] + (size_t)std::max(0, k1 - 1) * CB;
	}
	std::unique_ptr<double[]> E(new double[boff[nb] ? boff[nb] : 1]);
	auto rowOf = [&](int j, int r) -> double * { return E.get() + boff[j] + (size_t)r * CB; };

	// pass 1, tile by tile (64 rows of one block: contiguous; iec may be read along rows or, reversed,
	// along columns, both local within a tile): G and the column maxima
	vector<std::pair<int, int> > tiles;
	for (int j = 0; j < nb; j++)
		for (int i = 0; i * CB < std::min(K, (j + 1) * CB) - 1; i++) tiles.push_back(std::make_pair(i, j));
	vector<double> c(K, NEG);
	#pragma omp parallel
	{
		vector<double> cl(K, NEG);  // this thread's column maxima
		#pragma omp for schedule(dynamic)
		for (int t = 0; t < (int)tiles.size(); t++) {
			const int j = tiles[t].second, k0 = j * CB, k1 = std::min(K, k0 + CB);
			const int r0 = tiles[t].first * CB, r1 = std::min(k1 - 1, r0 + CB);
			for (int r = r0; r < r1; r++) {
				double *row = rowOf(j, r) - k0;  // row[k], k in [k0, k1)
				for (int k = k0; k < k1; k++) {
					if (k <= r) { row[k] = NEG; continue; }
					const double g = iec(r + 1, k) - b[k] + b[r];
					row[k] = g;
					cl[k] = fmax(cl[k], g);
				}
				for (int k = k1; k < k0 + CB; k++) row[k] = NEG;  // padding of the last block
			}
		}
		#pragma omp critical
		for (int k = 0; k < K; k++) c[k] = fmax(c[k], cl[k]);
	}
	for (int k = 0; k < K; k++) if (c[k] == NEG) c[k] = 0.0;
	// pass 2: E = exp(G - c), vectorised (vmath.cpp); the k <= r and padding entries become 0
	#pragma omp parallel for schedule(dynamic)
	for (int t = 0; t < (int)tiles.size(); t++) {
		const int j = tiles[t].second, k0 = j * CB, k1 = std::min(K, k0 + CB);
		const int r0 = tiles[t].first * CB, r1 = std::min(k1 - 1, r0 + CB);
		for (int r = r0; r < r1; r++) {
			double *row = rowOf(j, r);
			for (int u = 0; u < CB; u++) {
				const int k = k0 + u;
				row[u] = (k < k1 && k > r) ? fmax(row[u] - c[k], -1000.0) : -1000.0;  // -> 0
			}
			vexpInPlace(row, CB);
		}
	}

	// the steps, blocked over columns: block [k0,k1) at step m needs phi_{m-1}[r] for r < k1 only,
	// which is final (earlier blocks: all m; this block: m-1), so a block of E stays in cache for all
	// m instead of E streaming from memory once per m. The scale p is the maximum of phi_{m-1} over
	// r < k1 (per block and m: consistent within each column's sum).
	// a term lost (to underflow or the exp cut-off) is < LOST in the scaled sum
	const double floor = 2.0 * K * LOST / tol;
	vector<vector<double> > phi(mmax + 1, vector<double>(K, NEG));
	std::fill(phi[0].begin(), phi[0].end(), 0.0);
	vector<double> v(K);
	double sums[CB];
	for (int j = 0; j < nb; j++) {
		const int k0 = j * CB, k1 = std::min(K, k0 + CB), nr = k1 - 1;  // rows r < k1 - 1
		const double *Eb = E.get() + boff[j];
		for (int m = 1; m <= mmax; m++) {
			const double *prev = phi[m - 1].data();
			double p = NEG;
			for (int r = 0; r < nr; r++) p = fmax(p, prev[r]);
			if (p == NEG) { for (int k = k0; k < k1; k++) phi[m][k] = NEG; continue; }
			for (int r = 0; r < nr; r++) v[r] = prev[r] == NEG ? -1000.0 : fmax(prev[r] - p, -1000.0);
			vexpInPlace(v.data(), nr);
			for (int u = 0; u < CB; u++) sums[u] = 0.0;
			#pragma omp parallel
			{
				FlushDenormals ftz;
				double loc[CB];
				for (int u = 0; u < CB; u++) loc[u] = 0.0;
				#pragma omp for schedule(static) nowait
				for (int r = 0; r < nr; r++) {
					const double vr = v[r];
					const double *row = Eb + (size_t)r * CB;
					#pragma omp simd
					for (int u = 0; u < CB; u++) loc[u] += vr * row[u];
				}
				#pragma omp critical
				for (int u = 0; u < CB; u++) sums[u] += loc[u];
			}
			for (int k = k0; k < k1; k++) {
				if (k < m) { phi[m][k] = NEG; continue; }  // needs m boundaries before k
				const double sk = sums[k - k0];
				if (sk >= floor) { phi[m][k] = log(sk) + p + c[k]; continue; }
				std::vector<double> buf(k);  // exact column
				for (int r = 0; r < k; r++) buf[r] = prev[r] + iec(r + 1, k) - b[k] + b[r];
				phi[m][k] = logSumExp(buf.data(), k);
			}
		}
	}
	for (int m = 1; m <= mmax; m++)
		for (int k = 0; k < K; k++) fwd[m][k] = b[k] + phi[m][k];
}

/** the last central iteration of the plain evidences, for reuse by forwardBackward:
    valid while spikeCounter::mDataVersion == version */
struct ForwardCache {
	unsigned long version = 0;
	int K = -1, mmax = -1;
	std::vector<std::vector<double> > fwd;
};
extern ForwardCache cache;

}  // namespace fb

/**
\brief The SDF and its second moment at every time index in one pass.

The original code obtains the SDF at time index t as a ratio of evidences, the
numerator computed with one virtual spike added at t (section 4 of the paper).
That is one run of the central iteration per time index and per moment:
O(M T^3) for the whole SDF.

Here a forward iteration (fb::fastForward) and the same iteration on the reversed
time axis (the evidences of the tail t+1..T-1) give the posterior probability of
every candidate bin [a,b] at once:

   P([a,b] is a bin | D, M) = sum_i fwd_{i-1}[a-1] * IEC(a,b) * bwd_{M-i-1}[b] / E[M]

and E[f_t] = sum over bins [a,b] containing t of P([a,b]) * E[f | bin], likewise
E[f_t^2]. The average over M is folded into the backward messages first; the sum
over the bin index is then a scaled product with small inner dimension. Total cost
O(M T^2), with the same underflow bound as fb::fastForward (1e-14 absolute in a bin
probability; entries that could exceed it are recomputed with log-sum-exp).
*/
class forwardBackward : public spikeCounter
{
public:
    /** compute E[f_t] and E[f_t^2] for every time index of the current interval.
	\param logEvidences: log P(D|M) for M=0..(at least) m2, e.g. from evidenceComputer<evifunc>.
	\param m1,m2: range of bin boundary numbers to average over, weighted by P(M|D).
	\param sdf: output, E[f_t].
	\param sdf2: output, E[f_t^2].
	Requires the plain Beta prior (no upper bound on the firing probability).
    */
    void compute(const std::vector<double> &logEvidences, int m1, int m2,
		 std::vector<double> &sdf, std::vector<double> &sdf2);

    /** the central iteration for the plain evidences, fwd[m][K-1] + prior = log P(D|M)
	(used by evidenceComputer<evifunc> when the prior is the plain Beta). */
    void forward(int mmax, std::vector<std::vector<double> > &fwd);
};

#endif
