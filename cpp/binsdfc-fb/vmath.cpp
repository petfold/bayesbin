/***************************************************************************
 *   Vectorised exp for forwardbackward.cpp. Added 2026-09-26 to binsdfc.  *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 ***************************************************************************/
// Compile THIS FILE ONLY with -ffast-math: glibc declares its vector exp (libmvec, <= 4 ulp) only
// then. Link without it, so no flush-to-zero is switched on. Inputs must be finite (the callers
// replace -inf). Below -700 the result is exactly 0: libmvec leaves its fast path for large negative
// arguments (and exp(-700) ~ 1e-304 is below anything the callers keep); above 700 it is clamped
// (such entries are recomputed exactly by the caller).
#include <cmath>

void vexpMul(double *w, const double *h, int n)
{
	#pragma omp simd
	for (int i = 0; i < n; i++) {
		const double e = std::exp(std::fmin(std::fmax(h[i], -700.0), 700.0));
		w[i] *= h[i] < -700.0 ? 0.0 : e;
	}
}

// x[i] = exp(x[i]); exactly 0 below -700
void vexpInPlace(double *x, int n)
{
	#pragma omp simd
	for (int i = 0; i < n; i++) {
		const double e = std::exp(std::fmin(std::fmax(x[i], -700.0), 700.0));
		x[i] = x[i] < -700.0 ? 0.0 : e;
	}
}
