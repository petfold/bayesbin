// Modified 2026-09-26: SDF by forward-backward (forwardbackward.h); see README.fb.md.
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
#ifndef SPIKEDENSITYFUNCTION_H
#define SPIKEDENSITYFUNCTION_H


#include "logadd.h"
#include "spikecounter.h"
#include "evidencecomputer.h"
#include "forwardbackward.h"
#include <list>


using namespace std;

/**
\brief compute bin spike density function and related quantities, e.g. latency, window end, signal probability.

@author Dominik Endres
*/
class spikeDensityFunction{
public:
    	spikeDensityFunction();

    	~spikeDensityFunction();
	/** set maximal number of bin boundaries */
    	void setMaxM(int mmax);

	/** set prior exponents of the Beta distributions for each interval */
    	void setIntervalPriors(double p0,double p1,double pub=1.0);

	/** time interval within which sdf etc. is computed */
    	void setTimeInterval(int start,int end);

	/** clears all spike data and precomputed arrays */
    	void clearData();

	/** add a stimulus run to the data */
	void addData(vector<int>::iterator begin,vector<int>::iterator end);

	/** find an interval of bin boundary numbers such that the probability that the correct model is
	 * included is at least pmin. \return the actual probability */
    	double findMInterval(double pmin,int &mmin,int &mmax);

	/** set interval of bin boundary numbers to be used in subsequent inferences. */
	void setMInterval(int m1,int m2);

	/** \return posterior probability of a model with m bin boundaries. Assumes uniform M prior. */
    	double getMProb(int m);

	/** \return expected SDF at time index tind */
    	double getSDF(int tind);

	/** \return expected SDF standard deviation. 
	\param tind: time index.
	\param sdfval: expexted sdf at tind.*/
    	double getSDFStdDev(int tind,double sdfval);
	/** SDF (and optionally its standard deviation) at every time index of the interval in one
	forward-backward pass, O(M T^2); same values as getSDF()/getSDFStdDev() for every tind.
	\return false if not applicable (upper bound on the firing probability set): use getSDF() then. */
	bool getSDFForwardBackward(vector<double> &sdf,vector<double> &sdfstd,bool doStdDev);

	/** \return probability that bin boundary m is at time index tind*/
    	double getBBProb(int m,int tind);

	/** \return expectation of bin boundary m */
    	double getBBExpectation(int m);

	/** \return standard deviation of bin boundary m, given its expectation */
	double getBBStdDev(int m,double bbexp);

	/** \return probability that signal is above noiselevel (==noise firing probability) at time index tind */
    	double getPAbNoise(int tind,double noiselevel);

	/** \return probability that latency is at time index tind given noiselevel (i.e. prob that signal is below noiselevel before tind, and above noiselevel for at least one bin starting at tind). 
	\param tind: time index
	\param noiselevel: signal separation level, i.e. firing probability below which there's no signal (i.e. noise) and above which there is a signal.
	*/
    	double getLatencyProb(int tind,double noiselevel);

	/** compute probabilities that latency is at tind given M, for all M in the interval set by getLatencyProb().
	\param tind: time index
	\param noiselevel: signal separation level
	\param res: vector of latency probabilities for each M
	*/
	void getLatencyProbs(int tind,double noiselevel,vector<double> &res);

	/** \return probability that the signal in the spiketrain starts at tstart and ends at tend (i.e. is below the supplied noise level prior to tstart, above it between tstart and tend, and below after tend), given that there is a signal in the spiketrain */
	double getBABProb(int tstart,int tend,double noiselevel);

	/** \return probability that there is no signal in the spiketrain */
	double getNOSIG(double noiselevel);

    	/** probability that separation between below/above noiselevel is possible */
	double getSeparationProb(double noiselevel,int tstart=INT_MIN,int tend=INT_MAX);

	/** find best separation level. 
	\param lbound: lower bound on search interval for separation level.
	\param ubound. upper bound on search interval for separation level.
	\param noiselevel: best separation level
	\param prob: probability of signal at best separation level.
	\return list of pairs (level,prob) that have been tried by golden section search. noiselevel will be the best level, prob the probability that there's a signal at that noiselevel. */
    	list<pair<double,double> > findBestSeparator(double lbound,double ubound,double &noiselevel,double &prob,int tstart=INT_MIN,int tend=INT_MAX);

	/** \return probability that firing rate is pfire, given latency at tind and noiselevel*/
	double getFiringRateGivenLatencyProb(int tind,double pfire,double noiselevel);

	/** \return probability that firing rate is pfire,  given noiselevel*/
        double getFiringRateProb(double pfire,double noiselevel,int tstart=INT_MIN,int tend=INT_MAX);

	/** \return total evidence P(D), assuming a uniform prior over M */
    	double getTotalEvidence();

	/** \return log sum P(D|M), summing over the interval set in getLatencyProb(). */
        double getIntervalEvidenceSum();

	/** \return log P(D|m) */
        double getEvidence(int m);

	/** \param tind: time index.
	\param inhibitory_noiselevel: inhibitory signal separation level. Inverse of (excitatory) signal separation level.
	\return probability that inhibitory latency is at tind.
	*/
	double getInhibitoryLatencyProb(int tind,double inhibitory_noiselevel);


	/** \param inhibitory_noiselevel: inhibitory signal separation level.
	\param tstart: beginning of tomer i
	\return probability that separation between above/below inhibitory noiselevel is possible */
	double getInhibitorySeparationProb(double inhibitory_noiselevel,int tstart=INT_MIN,int tend=INT_MAX);

	/** find best separation level between lbound and ubound. Returns list of values (level,prob) that
	have been tried by golden section search. noiselevel will be the best level, prob its probability */
    	list<pair<double,double> > findBestInhibitorySeparator(double lbound,double ubound,double &noiselevel,double &prob,int tstart=INT_MIN,int tend=INT_MAX);
	void setExcludedTimeInterval(int ts,int te) {mTex1=ts;mTex2=te;};
	
	/** max number of bin boundaries */
	int mMaxM;
	
protected:
	/** template instances of the central iteration with different averaging functors. See documentation of evidenceComputer */

	evidenceComputer<evifunc> mEV;
	evidenceComputer<binboundary_expectation> mBBE;
	evidenceComputer<binboundary_expectation2> mBBE2;
	evidenceComputer<binboundary_probability> mBBP;
	evidenceComputer<SDF> mSDF;
	evidenceComputer<SDFp> mSDFp;
	evidenceComputer<SDF2> mSDF2;
	evidenceComputer<SDF2p> mSDF2p;
	evidenceComputer<aboveNoise> mABNOI;
	evidenceComputer<latency> mLAT;
	evidenceComputer<nssns> mBAB;
	evidenceComputer<nssns_bin> mBABBin;
	evidenceComputer<nosig> mNOS;
	evidenceComputer<latsum> mLatSum;
	evidenceComputer<inhib_latsum> mInhibitoryLatSum;
	evidenceComputer<inhib_latency> mInhibitoryLatency;
	evidenceComputer<p_firing_rate_and_latency> mPFRAL;
	evidenceComputer<p_firing_rate> mPFR;
	spikeCounter mSP;
	/** one-pass SDF, see forwardBackward */
	forwardBackward mFB;
	
	/** smallest/largest number of bin boundaries used in inference */
	int mM1,mM2;
	/** prior exponents of Beta distribution for each interval, and upper bound on firing prob */
	double mP0,mP1,mPUB;
	/** time interval within which SDF etc. is computed */
	int mIntervStart,mIntervEnd;
	/** true if data changed since last evidence computation */
	bool mbDataChanged;
	/** true if M-interval (i.e. mM1 or mM2) changed since last computation of evidence sums */
	bool mbMIntervalChanged;
	/** total evidence sum */
	double mEvSum;
	/** evidence sum for mM1<=m<=mM2. Same for p_wndstart functor */
	double mEvIntervSum,mWndstartIntervSum;
	/** mWndstartIntervSum needs only be recomputed if one of those differs from the last call to getFiringRateProb */
	int mWndstartLastTime;
	double mNoiselevelLastTime;
	/** excluded time interval for latency calculations (e.g. because it contains stimulation artefacts */
	int mTex1,mTex2;
	
protected:
	/** golden section maximum search.
	\param lbound: lower bound of search interval
	\param ubound: upper bound of search interval
	\param maxpos: position of maximum
	\param max: value at maximum
	\param maxfunc: function to be maximised.
	\return list of pairs of (search point, value at search point). */
	list<pair<double,double> > goldenSectionSearch(double lbound,double ubound,double &maxpos,double &max,int tstart,int tend,double (spikeDensityFunction::*maxfunc)(double,int,int));

	/** compute evidences. mbDataChanged=false if successful */
    	void computeEvidences();

        double getBABBinProb(int bin1,int bin2,double noiselevel);
};

#endif
