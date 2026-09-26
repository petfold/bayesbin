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
#include <cmath>
#include <limits>
#include <algorithm>
#include "omp.h"

using namespace std;

static const double NEG = -numeric_limits<double>::infinity();

/** log(sum_i exp(x_i)) of a small buffer, max-shifted; -inf if all are -inf */
static inline double logSumExp(const double *x, int n)
{
	double mx = NEG;
	for (int i = 0; i < n; i++) mx = fmax(mx, x[i]);
	if (mx == NEG) return NEG;
	double s = 0.0;
	for (int i = 0; i < n; i++) s += exp(x[i] - mx);
	return mx + log(s);
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
	auto IEC = [&](int a, int b) -> double { return mIntervalEvidences[a][b - a]; };

	// forward: fwd[m][k] = log evidence of 0..k as m+1 bins, the last ending at k (the paper's subE_m[k])
	vector<vector<double> > fwd(m2 + 1, vector<double>(K, NEG));
	for (int k = 0; k < K; k++) fwd[0][k] = IEC(0, k);
	for (int m = 1; m <= m2; m++) {
		#pragma omp parallel for schedule(guided)
		for (int k = m; k < K; k++) {
			vector<double> buf(k - m + 1);
			for (int r = m - 1; r <= k - 1; r++) buf[r - m + 1] = fwd[m - 1][r] + IEC(r + 1, k);
			fwd[m][k] = logSumExp(buf.data(), (int)buf.size());
		}
	}

	// backward: bwd[j][k] = log evidence of k+1..K-1 as j+1 bins (k <= K-2)
	vector<vector<double> > bwd(m2 + 1, vector<double>(K, NEG));
	for (int k = 0; k <= K - 2; k++) bwd[0][k] = IEC(k + 1, K - 1);
	for (int j = 1; j <= m2; j++) {
		#pragma omp parallel for schedule(guided)
		for (int k = 0; k <= K - 2; k++) {
			int rmax = K - 2;
			if (rmax < k + 1) continue;
			vector<double> buf(rmax - k);
			for (int r = k + 1; r <= rmax; r++) buf[r - k - 1] = IEC(k + 1, r) + bwd[j - 1][r];
			bwd[j][k] = logSumExp(buf.data(), (int)buf.size());
		}
	}

	// weights of the models: c[M] = log(P(M|D) / sum_{m1..m2} P) - log E_unnormalised[M]
	vector<double> ev(logEvidences.begin() + m1, logEvidences.begin() + m2 + 1);
	const double evSum = logSumExp(ev.data(), (int)ev.size());
	vector<double> c(m2 + 1, NEG);
	for (int M = m1; M <= m2; M++) c[M] = logEvidences[M] - evSum - fwd[M][K - 1];

	// right messages right_j[b] (tail b+1..K-1 as j bins), folded with the model weights:
	// R[i][b] = log sum_j c[i+j] * right_j[b]
	auto right = [&](int j, int b) -> double {
		if (j == 0) return b == K - 1 ? 0.0 : NEG;
		return b <= K - 2 ? bwd[j - 1][b] : NEG;
	};
	vector<vector<double> > R(m2 + 1, vector<double>(K, NEG));
	#pragma omp parallel for schedule(static)
	for (int b = 0; b < K; b++) {
		vector<double> buf(m2 + 1);
		for (int i = 0; i <= m2; i++) {
			int n = 0;
			for (int j = 0; i + j <= m2; j++) buf[n++] = c[i + j] + right(j, b);
			R[i][b] = logSumExp(buf.data(), n);
		}
	}
	auto left = [&](int i, int a) -> double {  // head 0..a-1 as i bins
		if (i == 0) return a == 0 ? 0.0 : NEG;
		return a >= 1 ? fwd[i - 1][a - 1] : NEG;
	};

	// bins: P([a,b]) * E[f|bin] and * E[f^2|bin], range-added over a..b via difference arrays
	const int nthreads = omp_get_max_threads();
	vector<vector<double> > d1(nthreads, vector<double>(K + 1, 0.0)), d2(nthreads, vector<double>(K + 1, 0.0));
	#pragma omp parallel
	{
		const int tid = omp_get_thread_num();
		vector<double> lf(m2 + 1), buf(m2 + 1);
		#pragma omp for schedule(guided)
		for (int a = 0; a < K; a++) {
			for (int i = 0; i <= m2; i++) lf[i] = left(i, a);
			for (int b = a; b < K; b++) {
				for (int i = 0; i <= m2; i++) buf[i] = lf[i] + R[i][b];
				const double lw = logSumExp(buf.data(), m2 + 1);
				if (lw == NEG) continue;
				const double w = exp(lw + IEC(a, b));
				const pair<int, int> &cnt = mIntervalCounts[a][b - a];
				const double s = cnt.first + mPrior1, n = cnt.first + cnt.second + mPrior1 + mPrior0;
				const double mean = s / n, second = mean * (s + 1.0) / (n + 1.0);
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
