/***************************************************************************
 *   Copyright (C) 2006 by Dominik Endres   *
 *   research@itas-sys.com   *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/
#ifndef SPIKECOUNTER_H
#define SPIKECOUNTER_H

using namespace std;

#include <vector>
#include <utility>
#include "math.h"
#include "float.h"
#include "logadd.h"

/**
\brief Counts spike trains and stores results. 

A spike train is represented as a std::vector<int>, each int being the time index when a spike was observed. Most members are static, so that data and precomputed priors and interval evidences can be shared across instances of objects which inherit from this object, without having to duplicate data. spikeCounter::mIntervalEvidences are marginal probabilities of each interval under a binomial observation model with a beta prior on the success (i.e. spike) probability of the binomial. If you want to change the observation model, derive from this object.

@author Dominik Endres
*/

class spikeCounter{
public:
    spikeCounter();

    ~spikeCounter();
    /** set maximal number of bin boundaries 
	\param m: maximal number of bin boundaries
    */
    void setMaxM(int m);

    /** set interval within which SDF is computed */
    void setInterval(int start, int end);

    /** prior exponents for Beta distribution \propto p^p1*(1-p)^p0 in every bin. Must be > 1e-5 
	\param p0: exponent for "no spike"
	\param p1: exponent for "spike"
    */
    void setIntervalPrior(double p0,double p1,double pub=1.0);
    /** clears the spiketrain */

    void clearData();
    /** add a run (from dengkeData) to the current spiketrain
	\param begin: iterator to beginning of spike train to be added.
	\param end: iterator past its end.
    */
    void addData(vector<int>::iterator begin,vector<int>::iterator end);

    /** \return total number of spikes */
    int getSpikes();

    /** \return total number of gaps (i.e. observed time indexes with no spikes) */
    int getGaps();

    /** test code. Generate some spike trains and count
	\param pfire0: baseline firing probability
	\param pfire1: transient firing probability
	\param pfire2: sustained firing probability
	\param numtrains: number of spike trains to generate
    */
    void test(double pfire0,double pfire1,double pfire2,int numtrains);


protected:
    /** Evidences of sub-intervals. The return values of the function getIEC() in <a href="http://books.nips.cc/nips20.html">Endres, Oram, Schindelin, Foldiak (2007): Bayesian binning beats approximate alternatives: estimating peri-stimulus time histograms </a> */
    static vector<vector<double> > mIntervalEvidences;
    /** counts of spikes and gaps in the sub-intervals */
    static vector<vector<pair<int,int> > > mIntervalCounts;
    /** priors for a given bin number */
    static vector<double> mPriors;
    /** priors of the beta distributions in the sub-intervals */
    static double mPrior1,mPrior0;
    /** beginning and end (exclusive) of time interval within which SDF is computed */
    static int mIntervStart,mIntervEnd;
    /** maximal number of bin boundaries (bins-1) in the time interval */
    static int mMMax;
    /** the marginal spiketrain. Stores pairs of <spikes,gaps> for every time index in the interval */
    static vector<pair<int,int> > mSpikeTrain;
    /** the current spiketrain */
    vector<int> mCurSp;
    /** autocorrelation counts of the spiketrain. mAutoCorr[i][j<i]=single spike train autocorrelation for time indexes i,j */
    static vector<vector<int> > mAutoCorr;
    /** true if data have been changed since last call to precomputeSubIntervals */
    static bool mbDataChanged;
    /** upper bound on firing prob. */
    static double mPUB;
    
protected:
    /** allocate storage for mIntervalEvidences and mIntervalCounts */
    bool allocArrays();
    /** precompute counts and sub-evidences from current spiketrain. */
    virtual void precomputeSubIntervals();
    /** compute M-priors. Return false if impossible (i.e. mMMax>mIntervEnd-mIntervStart-1) */
    virtual bool computePrior();
    
};


#endif
