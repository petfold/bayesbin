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
	fb::fastForward(K, min(mmax, K - 1),
			[this](int a, int b) { return mIntervalEvidences[a][b - a]; }, fwd);
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

	// IEC(a,b): log evidence contribution of the bin a..b (inclusive), prior normalisation excluded
	auto IEC = [this](int a, int b) -> double { return mIntervalEvidences[a][b - a]; };

	// forward: fwd[m][k] = log evidence of 0..k as m+1 bins, the last ending at k
	// (the plain evidences already ran it for these data: reuse)
	vector<vector<double> > fwd;
	if (fb::cache.version == mDataVersion && fb::cache.K == K && fb::cache.mmax >= m2)
		fwd.assign(fb::cache.fwd.begin(), fb::cache.fwd.begin() + m2 + 1);
	else
		fb::fastForward(K, m2, IEC, fwd);

	// backward, as the forward iteration on the reversed time axis:
	// bwd[j][k] = log evidence of k+1..K-1 as j+1 bins = rev[j][K-2-k]
	vector<vector<double> > rev;
	fb::fastForward(K, m2, [&](int a, int b) { return IEC(K - 1 - b, K - 1 - a); }, rev);
	vector<vector<double> > bwd(m2 + 1, vector<double>(K, NEG));
	for (int j = 0; j <= m2; j++)
		for (int k = 0; k <= K - 2; k++) bwd[j][k] = rev[j][K - 2 - k];

	// weights of the models: c[M] = log(P(M|D) / sum_{m1..m2} P) - log E_unnormalised[M]
	vector<double> ev(logEvidences.begin() + m1, logEvidences.begin() + m2 + 1);
	const double evSum = logSumExp(ev.data(), (int)ev.size());
	vector<double> c(m2 + 1, NEG);
	for (int M = m1; M <= m2; M++) c[M] = logEvidences[M] - evSum - fwd[M][K - 1];

	const int NI = m2 + 1;
	// right messages right_j[b] (tail b+1..K-1 as j bins), folded with the model weights:
	// R[b*NI+i] = log sum_j c[i+j] * right_j[b]; left messages L[a*NI+i] (head 0..a-1 as i bins)
	auto right = [&](int j, int b) -> double {
		if (j == 0) return b == K - 1 ? 0.0 : NEG;
		return b <= K - 2 ? bwd[j - 1][b] : NEG;
	};
	vector<double> Rl((size_t)K * NI, NEG), Ll((size_t)K * NI, NEG);
	#pragma omp parallel for schedule(static)
	for (int t = 0; t < K; t++) {
		vector<double> buf(NI);
		for (int i = 0; i < NI; i++) {
			int n = 0;
			for (int j = 0; i + j <= m2; j++) buf[n++] = c[i + j] + right(j, t);
			Rl[(size_t)t * NI + i] = logSumExp(buf.data(), n);
			Ll[(size_t)t * NI + i] = i == 0 ? (t == 0 ? 0.0 : NEG) : (t >= 1 ? fwd[i - 1][t - 1] : NEG);
		}
	}
	// scaled once: A[a*NI+i] = exp(L - ma[a]); Bt[i*K+b] = exp(R - mb[b]), i-major so that the
	// sum over the bin index runs as contiguous multiply-adds along b
	vector<double> A((size_t)K * NI), Bt((size_t)NI * K), ma(K), mb(K);
	#pragma omp parallel for schedule(static)
	for (int t = 0; t < K; t++) {
		double x = NEG, y = NEG;
		for (int i = 0; i < NI; i++) { x = fmax(x, Ll[(size_t)t * NI + i]); y = fmax(y, Rl[(size_t)t * NI + i]); }
		ma[t] = x == NEG ? 0.0 : x;
		mb[t] = y == NEG ? 0.0 : y;
		for (int i = 0; i < NI; i++) {
			A[(size_t)t * NI + i] = exp(Ll[(size_t)t * NI + i] - ma[t]);
			Bt[(size_t)i * K + t] = exp(Rl[(size_t)t * NI + i] - mb[t]);
		}
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
	// scaled sum loses is < DBL_MIN, so W is off by at most NI*DBL_MIN*exp(ma+mb+IEC): where that could
	// exceed 1e-14, the entry is redone exactly (a separate scalar pass over the flagged b).
	const double logRisk = log(NI * DBL_MIN) - log(1e-14);
	const int nthreads = omp_get_max_threads();
	vector<vector<double> > d1(nthreads, vector<double>(K + 1, 0.0)), d2(nthreads, vector<double>(K + 1, 0.0));
	#pragma omp parallel
	{
		const int tid = omp_get_thread_num();
		vector<double> W(K), sh(K), m1(K), m2(K), q1(K), q2(K), buf(NI);
		vector<int> risky;
		double *D1 = d1[tid].data(), *D2 = d2[tid].data();
		#pragma omp for schedule(guided)
		for (int a = 0; a < K; a++) {
			const int len = K - a;  // b = a..K-1 at index u = b - a
			const double *Aa = &A[(size_t)a * NI];
			const double *iec = mIntervalEvidences[a].data();
			const pair<int, int> *cnt = mIntervalCounts[a].data();
			const double *mbA = mb.data() + a;
			double *w = W.data(), *h = sh.data(), *e1 = m1.data(), *e2 = m2.data(), *r1 = q1.data(), *r2 = q2.data();
			// posterior moments of each bin a..b, and the scale shift of W
			if (uniform) {
				const double *i1 = inv1.data() + 1, *i2 = inv2.data() + 1;  // index by u = len-1
				for (int u = 0; u < len; u++) {
					const double s1 = cnt[u].first + p1;
					e1[u] = s1 * i1[u];
					e2[u] = e1[u] * (s1 + 1.0) * i2[u];
				}
			} else {
				for (int u = 0; u < len; u++) {
					const double s1 = cnt[u].first + p1, n1 = cnt[u].first + cnt[u].second + p01;
					e1[u] = s1 / n1;
					e2[u] = e1[u] * (s1 + 1.0) / (n1 + 1.0);
				}
			}
			#pragma omp simd
			for (int u = 0; u < len; u++) { h[u] = ma[a] + mbA[u] + iec[u]; w[u] = 0.0; }
			// sum over the bin index: contiguous multiply-adds along b
			for (int i = 0; i < NI; i++) {
				const double ai = Aa[i];
				if (ai == 0.0) continue;
				const double *Bi = &Bt[(size_t)i * K + a];
				#pragma omp simd
				for (int u = 0; u < len; u++) w[u] += ai * Bi[u];
			}
			vexpMul(w, h, len);
			risky.clear();
			for (int u = 0; u < len; u++)
				if (h[u] + logRisk > 0.0) risky.push_back(u);
			for (int u : risky) {  // underflow could matter: exact, in log space
				for (int i = 0; i < NI; i++) buf[i] = Ll[(size_t)a * NI + i] + Rl[(size_t)(a + u) * NI + i];
				const double lw = logSumExp(buf.data(), NI);
				w[u] = lw == NEG ? 0.0 : exp(lw + iec[u]);
			}
			#pragma omp simd
			for (int u = 0; u < len; u++) { r1[u] = w[u] * e1[u]; r2[u] = w[u] * e2[u]; }
			double row1 = 0.0, row2 = 0.0;
			#pragma omp simd reduction(+:row1,row2)
			for (int u = 0; u < len; u++) { row1 += r1[u]; row2 += r2[u]; }
			// range-add over a..b: +q at a (the row sum), -q at b+1
			D1[a] += row1;
			D2[a] += row2;
			#pragma omp simd
			for (int u = 0; u < len; u++) { D1[a + u + 1] -= r1[u]; D2[a + u + 1] -= r2[u]; }
		}
	}
	double acc1 = 0.0, acc2 = 0.0;
	for (int t = 0; t < K; t++) {
		for (int th = 0; th < nthreads; th++) { acc1 += d1[th][t]; acc2 += d2[th][t]; }
		sdf[t] = acc1;
		sdf2[t] = acc2;
	}
}
