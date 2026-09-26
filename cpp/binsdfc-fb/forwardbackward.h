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

/**
\brief The SDF and its second moment at every time index in one pass.

The original code obtains the SDF at time index t as a ratio of evidences, the
numerator computed with one virtual spike added at t (section 4 of the paper).
That is one run of the central iteration per time index and per moment:
O(M T^3) for the whole SDF.

Here a forward iteration (the paper's subE[]) and a matching backward one
(evidences of the tail t+1..T-1) give the posterior probability of every
candidate bin [a,b] at once:

   P([a,b] is a bin | D, M) = sum_i fwd_{i-1}[a-1] * IEC(a,b) * bwd_{M-i-1}[b] / E[M]

and E[f_t] = sum over bins [a,b] containing t of P([a,b]) * E[f | bin], likewise
E[f_t^2]. The average over M is folded into the backward messages first. Total
cost O(M T^2). All sums are exact log-sum-exps (max-shifted), not table lookups.
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
};

#endif
