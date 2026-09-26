/***************************************************************************
 *   Forward-backward computation of the SDF for binsdfc.                  *
 *   Added 2026 to binsdfc 0.1 (Copyright (C) 2007 by Dominik Endres).     *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 ***************************************************************************/
#include "forwardbackward.h"
#include "omp.h"

using namespace std;

void vexpMul(double *w, const double *h, int n);  // vmath.cpp: w[i] *= exp(h[i]), vectorised

bool fb::enabled = true;
fb::ForwardCache fb::cache;
using fb::NEG;
using fb::logSumExp;

void forwardBackward::forward(int mmax, vector<vector<double> > &fwd)
{
	precomputeSubIntervals();
	const int K = mIntervEnd - mIntervStart;
	fb::fastForward(K, min(mmax, K - 1), [](int a, int b) { return iecOf(a, b); }, fwd);
}

void forwardBackward::compute(const vector<double> &logEvidences, int m1, int m2,
			      vector<double> &sdf, vector<double> &sdf2)
{
	precomputeSubIntervals();
	const int K = mIntervEnd - mIntervStart;
	if (m2 > K - 1) m2 = K - 1;
	if (m1 < 0) m1 = 0;
	sdf.assign(K, 0.0);
	sdf2.assign(K, 0.0);
	if (K <= 0 || m1 > m2) return;

	// IEC(a,b): log evidence contribution of the bin a..b (inclusive), prior normalisation excluded,
	// from prefix sums and lgamma tables (no K^2 table)
	auto IEC = [](int a, int b) -> double { return iecOf(a, b); };

	// forward: fwd[m][k] = log evidence of 0..k as m+1 bins, the last ending at k
	// (the plain evidences already ran it for these data: reuse, by reference)
	vector<vector<double> > own;
	const bool cached = fb::cache.version == mDataVersion && fb::cache.K == K && fb::cache.mmax >= m2;
	if (!cached) fb::fastForward(K, m2, IEC, own);
	const vector<vector<double> > &fwd = cached ? fb::cache.fwd : own;

	// backward, as the forward iteration on the reversed time axis:
	// bwd[j][k] = log evidence of k+1..K-1 as j+1 bins = rev[j][K-2-k] (read in place)
	vector<vector<double> > rev;
	fb::fastForward(K, m2, [&](int a, int b) { return IEC(K - 1 - b, K - 1 - a); }, rev);

	// weights of the models: c[M] = log(P(M|D) / sum_{m1..m2} P) - log E_unnormalised[M]
	vector<double> ev(logEvidences.begin() + m1, logEvidences.begin() + m2 + 1);
	const double evSum = logSumExp(ev.data(), (int)ev.size());
	vector<double> c(m2 + 1, NEG);
	for (int M = m1; M <= m2; M++) c[M] = logEvidences[M] - evSum - fwd[M][K - 1];

	const int NI = m2 + 1;
	// messages: left_i[a] (head 0..a-1 as i bins), right_j[b] (tail b+1..K-1 as j bins); the right
	// ones folded with the model weights: Rl[b*NI+i] = log sum_j c[i+j] * right_j[b]
	auto left = [&](int i, int a) -> double {
		if (i == 0) return a == 0 ? 0.0 : NEG;
		return a >= 1 ? fwd[i - 1][a - 1] : NEG;
	};
	auto right = [&](int j, int b) -> double {
		if (j == 0) return b == K - 1 ? 0.0 : NEG;
		return b <= K - 2 ? rev[j - 1][K - 2 - b] : NEG;
	};
	vector<double> Rl((size_t)K * NI, NEG);
	#pragma omp parallel for schedule(static)
	for (int t = 0; t < K; t++) {
		vector<double> buf(NI);
		for (int i = 0; i < NI; i++) {
			int n = 0;
			for (int j = 0; i + j <= m2; j++) buf[n++] = c[i + j] + right(j, t);
			Rl[(size_t)t * NI + i] = logSumExp(buf.data(), n);
		}
	}
	// scaled once: Bt[i*K+b] = exp(R - mb[b]), i-major so that the sum over the bin index runs as
	// contiguous multiply-adds along b; the left factors exp(left - ma[a]) are made per row
	vector<double> Bt((size_t)NI * K), mb(K);
	#pragma omp parallel for schedule(static)
	for (int t = 0; t < K; t++) {
		double y = NEG;
		for (int i = 0; i < NI; i++) y = fmax(y, Rl[(size_t)t * NI + i]);
		mb[t] = y == NEG ? 0.0 : y;
		for (int i = 0; i < NI; i++) Bt[(size_t)i * K + t] = exp(Rl[(size_t)t * NI + i] - mb[t]);
	}
	const double p1 = mPrior1, p01 = mPrior1 + mPrior0;
	// every interval holds one spike or gap per trial, so a bin's total count n depends only on its
	// length: 1/n and 1/(n+1) from tables instead of two divisions per bin (else divide)
	bool uniform = true;
	const int perInterval = mSpikeTrain[0].first + mSpikeTrain[0].second;
	for (int t = 1; t < K; t++)
		if (mSpikeTrain[t].first + mSpikeTrain[t].second != perInterval) { uniform = false; break; }
	vector<double> inv1(K + 1), inv2(K + 1);
	for (int len = 1; len <= K; len++) {
		const double n = (double)perInterval * len + p01;
		inv1[len] = 1.0 / n;
		inv2[len] = 1.0 / (n + 1.0);
	}

	// bins, row by row: W[a,b] = (sum_i A[a,i] Bt[i,b]) * exp(ma[a] + mb[b] + IEC(a,b)). Every term the
	// scaled sum loses is < LOST, so W is off by at most NI*LOST*exp(ma+mb+IEC): where that could
	// exceed 1e-14, the entry is redone exactly (a separate scalar pass over the flagged b).
	const double logRisk = log(NI * fb::LOST) - log(1e-14);
	const int nthreads = omp_get_max_threads();
	vector<vector<double> > d1(nthreads, vector<double>(K + 1, 0.0)), d2(nthreads, vector<double>(K + 1, 0.0));
	// tiled: bin starts a in groups of AB, ends b in chunks of BC, so that a chunk of Bt (NI x BC)
	// stays in cache for all AB rows of a group instead of NI rows of length K-a streaming per row
	const int AB = 64, BC = 512;
	const int nga = (K + AB - 1) / AB;
	#pragma omp parallel
	{
		fb::FlushDenormals ftz;
		const int tid = omp_get_thread_num();
		vector<double> W(BC), sh(BC), iec(BC), m1v(BC), m2v(BC), q1(BC), q2(BC), buf(NI);
		vector<double> Aall((size_t)AB * NI), Lall((size_t)AB * NI), mall(AB), row1(AB), row2(AB);
		vector<int> risky;
		double *D1 = d1[tid].data(), *D2 = d2[tid].data();
		const int *C1 = mCum1.data(), *C0 = mCum0.data();
		#pragma omp for schedule(dynamic)
		for (int ga = 0; ga < nga; ga++) {
			const int a0 = ga * AB, a1 = min(K, a0 + AB);
			// the left factors of this group's rows: exp(left_i[a] - ma[a])
			for (int a = a0; a < a1; a++) {
				double *La = &Lall[(size_t)(a - a0) * NI], *Aa = &Aall[(size_t)(a - a0) * NI];
				double ma = NEG;
				for (int i = 0; i < NI; i++) { La[i] = left(i, a); ma = fmax(ma, La[i]); }
				if (ma == NEG) ma = 0.0;
				mall[a - a0] = ma;
				for (int i = 0; i < NI; i++) Aa[i] = exp(La[i] - ma);
				row1[a - a0] = row2[a - a0] = 0.0;
			}
			for (int b0 = a0; b0 < K; b0 += BC) {
				const int b1 = min(K, b0 + BC);
				for (int a = a0; a < a1; a++) {
					const int bs = max(a, b0), len = b1 - bs;  // this row's bins b = bs..b1-1, u = b - bs
					if (len <= 0) continue;
					const double *Aa = &Aall[(size_t)(a - a0) * NI], *La = &Lall[(size_t)(a - a0) * NI];
					const double ma = mall[a - a0];
					const double *mbA = mb.data() + bs;
					double *w = W.data(), *h = sh.data(), *ie = iec.data(), *e1 = m1v.data(), *e2 = m2v.data();
					double *r1 = q1.data(), *r2 = q2.data();
					// per bin a..b: counts, evidence, posterior moments, and the scale shift of W
					for (int u = 0; u < len; u++) {
						const int b = bs + u;
						const int sp = C1[b + 1] - C1[a], g = C0[b + 1] - C0[a];
						ie[u] = mLgA[sp] + mLgB[g] - mLgC[sp + g];
						const double s1 = sp + p1;
						if (uniform) {
							e1[u] = s1 * inv1[b - a + 1];
							e2[u] = e1[u] * (s1 + 1.0) * inv2[b - a + 1];
						} else {
							const double n1 = sp + g + p01;
							e1[u] = s1 / n1;
							e2[u] = e1[u] * (s1 + 1.0) / (n1 + 1.0);
						}
					}
					#pragma omp simd
					for (int u = 0; u < len; u++) { h[u] = ma + mbA[u] + ie[u]; w[u] = 0.0; }
					// sum over the bin index: contiguous multiply-adds along b, Bt chunk from cache
					for (int i = 0; i < NI; i++) {
						const double ai = Aa[i];
						if (ai == 0.0) continue;
						const double *Bi = &Bt[(size_t)i * K + bs];
						#pragma omp simd
						for (int u = 0; u < len; u++) w[u] += ai * Bi[u];
					}
					vexpMul(w, h, len);
					risky.clear();
					for (int u = 0; u < len; u++)
						if (h[u] + logRisk > 0.0) risky.push_back(u);
					for (int u : risky) {  // underflow could matter: exact, in log space
						for (int i = 0; i < NI; i++) buf[i] = La[i] + Rl[(size_t)(bs + u) * NI + i];
						const double lw = logSumExp(buf.data(), NI);
						w[u] = lw == NEG ? 0.0 : exp(lw + ie[u]);
					}
					double s1r = 0.0, s2r = 0.0;
					#pragma omp simd reduction(+:s1r,s2r)
					for (int u = 0; u < len; u++) {
						r1[u] = w[u] * e1[u];
						r2[u] = w[u] * e2[u];
						s1r += r1[u];
						s2r += r2[u];
					}
					row1[a - a0] += s1r;
					row2[a - a0] += s2r;
					// range-add over a..b: -q at b+1 (the +q at a is the row sum, added below)
					#pragma omp simd
					for (int u = 0; u < len; u++) { D1[bs + u + 1] -= r1[u]; D2[bs + u + 1] -= r2[u]; }
				}
			}
			for (int a = a0; a < a1; a++) { D1[a] += row1[a - a0]; D2[a] += row2[a - a0]; }
		}
	}
	double acc1 = 0.0, acc2 = 0.0;
	for (int t = 0; t < K; t++) {
		for (int th = 0; th < nthreads; th++) { acc1 += d1[th][t]; acc2 += d2[th][t]; }
		sdf[t] = acc1;
		sdf2[t] = acc2;
	}
}
