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

namespace fb {

const double NEG = -std::numeric_limits<double>::infinity();

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
underflow is below DBL_MIN, so a column whose sum is not far above 2*T*DBL_MIN could be off
by more than tol (relative); such columns are recomputed exactly with log-sum-exp.
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

	// E(r,k) = exp(G[r][k] - c[k]) for r < k, packed row by row (upper triangle, no zero fill):
	// row r holds k = r+1..K-1 at off(r) = r*(2K-r-1)/2
	const size_t n = (size_t)K * (K - 1) / 2;
	std::unique_ptr<double[]> E(new double[n]);
	auto off = [K](int r) -> size_t { return (size_t)r * (2 * (size_t)K - r - 1) / 2; };
	vector<double> c(K, NEG);
	#pragma omp parallel
	{
		vector<double> cl(K, NEG);  // this thread's column maxima
		#pragma omp for schedule(guided)
		for (int r = 0; r < K - 1; r++) {
			double *row = E.get() + off(r) - (r + 1);  // row[k], k > r
			for (int k = r + 1; k < K; k++) {
				const double g = iec(r + 1, k) - b[k] + b[r];
				row[k] = g;
				cl[k] = fmax(cl[k], g);
			}
		}
		#pragma omp critical
		for (int k = 0; k < K; k++) c[k] = fmax(c[k], cl[k]);
		#pragma omp barrier
		#pragma omp for schedule(guided)
		for (int r = 0; r < K - 1; r++) {
			double *row = E.get() + off(r) - (r + 1);
			for (int k = r + 1; k < K; k++) row[k] = exp(row[k] - (c[k] == NEG ? 0.0 : c[k]));
		}
	}
	for (int k = 0; k < K; k++) if (c[k] == NEG) c[k] = 0.0;

	const double floor = 2.0 * K * DBL_MIN / tol;
	const int BLOCK = 256;
	vector<double> phi(K, 0.0), v(K), sums(K), next(K);
	for (int m = 1; m <= mmax; m++) {
		double p = NEG;
		for (int r = 0; r < K; r++) p = fmax(p, phi[r]);
		if (p == NEG) p = 0.0;
		for (int r = 0; r < K; r++) v[r] = exp(phi[r] - p);
		std::fill(sums.begin(), sums.end(), 0.0);
		#pragma omp parallel for schedule(dynamic)
		for (int k0 = 0; k0 < K; k0 += BLOCK) {
			const int k1 = std::min(K, k0 + BLOCK);
			for (int r = 0; r < k1 - 1; r++) {
				const double vr = v[r];
				if (vr == 0.0) continue;
				const double *row = E.get() + off(r) - (r + 1);
				for (int k = std::max(r + 1, k0); k < k1; k++) sums[k] += vr * row[k];
			}
		}
		#pragma omp parallel for schedule(guided)
		for (int k = 0; k < K; k++) {
			if (k < m) { next[k] = NEG; continue; }  // needs m boundaries before k
			if (sums[k] >= floor) { next[k] = log(sums[k]) + p + c[k]; continue; }
			std::vector<double> buf(k);  // exact column
			for (int r = 0; r < k; r++) buf[r] = phi[r] + iec(r + 1, k) - b[k] + b[r];
			next[k] = logSumExp(buf.data(), k);
		}
		phi.swap(next);
		for (int k = 0; k < K; k++) fwd[m][k] = b[k] + phi[k];
	}
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
