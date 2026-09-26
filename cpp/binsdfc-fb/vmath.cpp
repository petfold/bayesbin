/***************************************************************************
 *   Vectorised exp for forwardbackward.cpp. Added 2026-09-26 to binsdfc.  *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 ***************************************************************************/
// Compile THIS FILE ONLY with -ffast-math: glibc declares its vector exp (libmvec, <= 4 ulp) only
// then. Link without it, so no flush-to-zero is switched on. Inputs are clamped (finite, <= 700):
// entries that large are recomputed exactly by the caller anyway.
#include <cmath>

void vexpMul(double *w, const double *h, int n)
{
	#pragma omp simd
	for (int i = 0; i < n; i++) w[i] *= std::exp(std::fmin(h[i], 700.0));
}
