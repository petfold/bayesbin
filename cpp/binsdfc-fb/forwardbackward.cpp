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

bool fb::enabled = true;
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
	vector<vector<double> > fwd;
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
	// scaled once: A[a*NI+i] = exp(L - ma[a]), B[b*NI+i] = exp(R - mb[b])
	vector<double> A((size_t)K * NI), B((size_t)K * NI), ma(K), mb(K);
	#pragma omp parallel for schedule(static)
	for (int t = 0; t < K; t++) {
		double x = NEG, y = NEG;
		for (int i = 0; i < NI; i++) { x = fmax(x, Ll[(size_t)t * NI + i]); y = fmax(y, Rl[(size_t)t * NI + i]); }
		ma[t] = x == NEG ? 0.0 : x;
		mb[t] = y == NEG ? 0.0 : y;
		for (int i = 0; i < NI; i++) {
			A[(size_t)t * NI + i] = exp(Ll[(size_t)t * NI + i] - ma[t]);
			B[(size_t)t * NI + i] = exp(Rl[(size_t)t * NI + i] - mb[t]);
		}
	}

	// bins: W = sum_i A*B * exp(ma+mb+IEC); every term the scaled sum loses is < DBL_MIN, so
	// W is off by at most NI*DBL_MIN*exp(ma+mb+IEC): where that could exceed tol, redo exactly
	const double logRisk = log(NI * DBL_MIN) - log(1e-14);
	const int nthreads = omp_get_max_threads();
	vector<vector<double> > d1(nthreads, vector<double>(K + 1, 0.0)), d2(nthreads, vector<double>(K + 1, 0.0));
	#pragma omp parallel
	{
		const int tid = omp_get_thread_num();
		vector<double> buf(NI);
		#pragma omp for schedule(guided)
		for (int a = 0; a < K; a++) {
			const double *Aa = &A[(size_t)a * NI];
			for (int b = a; b < K; b++) {
				const double *Bb = &B[(size_t)b * NI];
				double s = 0.0;
				for (int i = 0; i < NI; i++) s += Aa[i] * Bb[i];
				const double shift = ma[a] + mb[b] + IEC(a, b);
				double w;
				if (shift + logRisk > 0.0) {  // underflow could matter here: exact
					for (int i = 0; i < NI; i++) buf[i] = Ll[(size_t)a * NI + i] + Rl[(size_t)b * NI + i];
					const double lw = logSumExp(buf.data(), NI);
					w = lw == NEG ? 0.0 : exp(lw + IEC(a, b));
				} else {
					w = s == 0.0 ? 0.0 : s * exp(shift);
				}
				if (w == 0.0) continue;
				const pair<int, int> &cnt = mIntervalCounts[a][b - a];
				const double sp = cnt.first + mPrior1, n = cnt.first + cnt.second + mPrior1 + mPrior0;
				const double mean = sp / n, second = mean * (sp + 1.0) / (n + 1.0);
				d1[tid][a] += w * mean;
				d1[tid][b + 1] -= w * mean;
				d2[tid][a] += w * second;
				d2[tid][b + 1] -= w * second;
			}
		}
	}
	double acc1 = 0.0, acc2 = 0.0;
	for (int t = 0; t < K; t++) {
		for (int th = 0; th < nthreads; th++) { acc1 += d1[th][t]; acc2 += d2[th][t]; }
		sdf[t] = acc1;
		sdf2[t] = acc2;
	}
}
