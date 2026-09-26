/***************************************************************************
 *   Forward-backward computation of the SDF for binsdfc.                  *
 *   Added 2026 to binsdfc 0.1 (Copyright (C) 2007 by Dominik Endres).     *
 *   Copyright (C) 2026 by Peter Földiák                                   *
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

	// The steps are blocked over columns: block [k0,k1) at step m needs phi_{m-1}[r] for r < k1 only,
	// which is final (earlier blocks: all m; this block: m-1). So each block's slice of
	// E(r,k) = exp(G[r][k] - c[k]) is built just before its m-loop and discarded after it: memory
	// O(K * CB), not O(K^2), and for moderate K the slice stays in cache for all m. The scale p is the maximum of phi_{m-1} over r < k1 (per block and m: consistent
	// within each column's sum). phi_m = fwd[m] - b is kept in fwd until the end.
	const int CBMAX = 64, CB = CBMAX;  // (narrower blocks, to fit a slice in cache, measured slower)
	std::unique_ptr<double[]> Eb(new double[(size_t)K * CB]);
	for (int k = 0; k < K; k++) fwd[0][k] = 0.0;  // phi_0 = 0
	// row groups of RG rows (a multiple of CB): once a group's phi_m is complete it gets a fixed scale
	// Sg[m][g] and V[m][r] = exp(phi_m[r] - Sg[m][g]), exponentiated once; a block then combines groups
	// with one factor exp(Sg - p) each (a multiply per row, not an exp)
	const int RG = 256, ng = (K + RG - 1) / RG;
	vector<vector<double> > V(mmax + 1, vector<double>((size_t)ng * RG, 0.0)), Sg(mmax + 1, vector<double>(ng, NEG));
	const double floor = 2.0 * K * LOST / tol;  // a lost term (underflow, exp cut-off) is < LOST
	vector<double> v((size_t)ng * RG), c(CB);
	double sums[CBMAX];
	for (int k0 = 0; k0 < K; k0 += CB) {
		const int k1 = std::min(K, k0 + CB), w = k1 - k0, nr = k1 - 1;  // rows r < k1 - 1 reach it
		// this block's slice: G and its column maxima, then E = exp(G - c) (vectorised, vmath.cpp)
		for (int u = 0; u < CB; u++) c[u] = NEG;
		#pragma omp parallel
		{
			double cl[CBMAX];
			for (int u = 0; u < CB; u++) cl[u] = NEG;
			#pragma omp for schedule(static)
			for (int r = 0; r < nr; r++) {
				double *row = Eb.get() + (size_t)r * CB;
				for (int u = 0; u < CB; u++) {
					const int k = k0 + u;
					if (u >= w || k <= r) { row[u] = NEG; continue; }
					const double g = iec(r + 1, k) - b[k] + b[r];
					row[u] = g;
					cl[u] = fmax(cl[u], g);
				}
			}
			#pragma omp critical
			for (int u = 0; u < CB; u++) c[u] = fmax(c[u], cl[u]);
			#pragma omp barrier
			#pragma omp for schedule(static)
			for (int r = 0; r < nr; r++) {
				double *row = Eb.get() + (size_t)r * CB;
				for (int u = 0; u < CB; u++)
					row[u] = row[u] == NEG ? -1000.0 : fmax(row[u] - (c[u] == NEG ? 0.0 : c[u]), -1000.0);
				vexpInPlace(row, CB);  // -1000 -> exactly 0
			}
		}
		for (int u = 0; u < CB; u++) if (c[u] == NEG) c[u] = 0.0;

		for (int m = 1; m <= mmax; m++) {
			const double *prev = fwd[m - 1].data();  // phi_{m-1}
			// scale: completed row groups carry a fixed scale Sg[m-1][g] and V[m-1][r] =
			// exp(phi - Sg), made once; the current (incomplete) group is exponentiated here
			const int gcur = (nr > 0 ? (nr - 1) : 0) / RG, rcur = gcur * RG;
			double p = NEG;
			for (int g = 0; g < gcur; g++) p = fmax(p, Sg[m - 1][g]);
			for (int r = rcur; r < nr; r++) p = fmax(p, prev[r]);
			if (p == NEG) { for (int k = k0; k < k1; k++) fwd[m][k] = NEG; continue; }
			for (int g = 0; g < gcur; g++) {  // one exp per completed group, a multiply per row
				const double lf = Sg[m - 1][g] - p;
				const double f = lf < -700.0 ? 0.0 : exp(lf);
				const double *Vg = V[m - 1].data() + (size_t)g * RG;
				double *vg = v.data() + (size_t)g * RG;
				#pragma omp simd
				for (int r = 0; r < RG; r++) vg[r] = f * Vg[r];
			}
			for (int r = rcur; r < nr; r++) v[r] = prev[r] == NEG ? -1000.0 : fmax(prev[r] - p, -1000.0);
			if (nr > rcur) vexpInPlace(v.data() + rcur, nr - rcur);
			for (int u = 0; u < CB; u++) sums[u] = 0.0;
			#pragma omp parallel
			{
				FlushDenormals ftz;
				double loc[CBMAX];
				for (int u = 0; u < CB; u++) loc[u] = 0.0;
				#pragma omp for schedule(static) nowait
				for (int r = 0; r < nr; r++) {
					const double vr = v[r];
					const double *row = Eb.get() + (size_t)r * CB;
					#pragma omp simd
					for (int u = 0; u < CB; u++) loc[u] += vr * row[u];
				}
				#pragma omp critical
				for (int u = 0; u < CB; u++) sums[u] += loc[u];
			}
			for (int k = k0; k < k1; k++) {
				if (k < m) { fwd[m][k] = NEG; continue; }  // needs m boundaries before k
				const double sk = sums[k - k0];
				if (sk >= floor) { fwd[m][k] = log(sk) + p + c[k - k0]; continue; }
				std::vector<double> buf(k);  // exact column
				for (int r = 0; r < k; r++) buf[r] = prev[r] + iec(r + 1, k) - b[k] + b[r];
				fwd[m][k] = logSumExp(buf.data(), k);
			}
			// a row group completed at step m: fix its scale and exponentiate it once
			if (k1 % RG == 0 || k1 == K) {
				const int g = (k1 - 1) / RG, g0 = g * RG, g1 = std::min(K, g0 + RG);
				double sg = NEG;
				for (int r = g0; r < g1; r++) sg = fmax(sg, fwd[m][r]);
				Sg[m][g] = sg;
				double *Vg = V[m].data() + g0;
				for (int r = g0; r < g1; r++) Vg[r - g0] = (sg == NEG || fwd[m][r] == NEG) ? -1000.0 : fmax(fwd[m][r] - sg, -1000.0);
				vexpInPlace(Vg, g1 - g0);
			}
		}
		// step 0 (phi_0 = 0) needs its groups too, for step 1 of later blocks
		if (k1 % RG == 0 || k1 == K) {
			const int g = (k1 - 1) / RG, g0 = g * RG, g1 = std::min(K, g0 + RG);
			Sg[0][g] = 0.0;
			for (int r = g0; r < g1; r++) V[0][r] = 1.0;
		}
	}
	for (int m = 0; m <= mmax; m++)
		for (int k = 0; k < K; k++) fwd[m][k] += b[k];
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
