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
#ifndef EVIDENCECOMPUTER_H
#define EVIDENCECOMPUTER_H

#include "spikecounter.h"
#include <iostream>

#include "limits.h"

#define DBL_LARGE_NEG (-DBL_MAX*1e-10)

/** \brief Evidence calculation, i.e. when this functor is used, the algorithm as described in the paper is implemented.
*/
class evifunc {
public:
	/** Possibly necesseary initialisations. Call before starting to average. */
	void init(){};
	/**
	\param lb: lower bound (exclusive!) of current bin
	\param ub: upper bound (inclusive!) of current bin
	\param binno: current bin number
	\param numpts: <spikes,gaps> observed in current bin.
	\param evcont: evidence contribution (getIEC() in the paper) of the current bin.
	\return log function value to be averaged. Here: just the evidence contribution of this interval.
	*/
	inline double operator()(const int &lb,const int &ub,const int &binno,const pair<int,int> &numpts,const double &evcont) {return evcont;};

	/** bin numbers */
	int mBin,mBin2;
	/** time indexes */	
	int mT,mT2;
	int mTex1,mTex2;
	/** firing probabilities and their logarithms */
	double mProb,mLP,mL1mP,mProb2;
	/** prior exponents of the beta distributions in the current interval */
	double mIP0,mIP1;
	/** upper bound on firing probability */
	double mPUB;
};

/** \brief functor for computing the expected position of a bin boundary */
class binboundary_expectation : public evifunc {
public:
	/** before starting the iteration, set binboundary_expectation::mBin to the number of the bin whose expected upper bound you want to compute */
	inline double operator()(const int &lb,const int &ub,const int &binno,const pair<int,int> &numpts,const double &evcont) 
	{if(binno==mBin) return evcont+(ub==0?DBL_LARGE_NEG:log(ub)); else return evcont;};	
};

/** \brief functor for bin boundary^2 expectation, to compute posterior variance of bin boundary positions */
class binboundary_expectation2 : public evifunc {
public:
	/** before starting the iteration, set binboundary_expectation::mBin to the number of the bin whose expected upper bound you want to compute */
	inline double operator()(const int &lb,const int &ub,const int &binno,const pair<int,int> &numpts,const double &evcont) 
	{if(binno==mBin) return evcont+(ub==0?DBL_LARGE_NEG:2.0*log(ub)); else return evcont;};	
};

/** \brief functor for probability that a upper bin boundary is at a given time index */
class binboundary_probability : public evifunc {
public:
	/** before starting the iteration, set binboundary_probability::mBin to the number of the bin whose probability of having its upper bound at binboundary_probability::mT you want to compute. */
	inline double operator()(const int &lb,const int &ub,const int &binno,const pair<int,int> &numpts,const double &evcont) 
	{if((binno!=mBin) || ((binno==mBin) && (ub==mT))) return evcont; else return DBL_LARGE_NEG;};	
};



/** \brief functor for spike density function. */
class SDF : public evifunc {
	public:
		/** before starting the iteration, set SDF::mT to the time index at which you want to compute the expected PSTH/SDF, and SDF::mIP0, SDF::mIP1 to the exponents of the beta prior. */
		inline double operator()(const int &lb,const int &ub,const int &binno,const pair<int,int> &numpts,const double &evcont) 
		{if((lb<mT) && (mT<=ub)) return evcont+log(mIP1+numpts.first)-log(mIP1+mIP0+numpts.first+numpts.second); else return evcont;};	
};

/** \brief functor for spike density function with variable upper bound on pfire. 

Only use if SDFp::mPUB<1, else you will waste time. */
class SDFp : public evifunc {
	public:
		/** before starting the iteration, set SDFp::mT to the time index at which you want to compute the expected PSTH/SDF, and SDFp::mPUB to the upper bound on the firing prob and and SDFp::mIP0, SDF::mIP1 to the exponents of the beta prior. */
		inline double operator()(const int &lb,const int &ub,const int &binno,const pair<int,int> &numpts,const double &evcont) 
		{if((lb<mT) && (mT<=ub)) return logAddInstance.getLogIncompleteBeta(mPUB,mIP1+numpts.first+1,mIP0+numpts.second); else return evcont;};	
};


/** \brief functor for spike density function^2 (to compute variance of SDF) */
class SDF2 : public evifunc {
	public:
		/** before starting the iteration, set SDF::mT to the time index at which you want to compute the expected PSTH/SDF^2, and SDF2::mIP0, SDF2::mIP1 to the exponents of the beta prior. */
		inline double operator()(const int &lb,const int &ub,const int &binno,const pair<int,int> &numpts,const double &evcont) 
		{if((lb<mT) && (mT<=ub)) 
			return evcont+log(mIP1+numpts.first)+log(mIP1+numpts.first+1)
				-log(mIP1+mIP0+numpts.first+numpts.second)-log(mIP1+mIP0+numpts.first+numpts.second+1); 
			else return evcont;};	
};

/** \brief functor for spike density function^2 (to compute variance of SDF) with variable upper bound on pfire. 

Only use if SDF2p::mPUB<1, else you will waste time. */
class SDF2p : public evifunc {
	public:
		/** before starting the iteration, set SDF2p::mT to the time index at which you want to compute the expected PSTH/SDF, SDF2p::mPUB to the upper bound on the firing prob , and SDF2p::mIP0, SDF2p::mIP1 to the exponents of the beta prior.*/
		inline double operator()(const int &lb,const int &ub,const int &binno,const pair<int,int> &numpts,const double &evcont) 
		{if((lb<mT) && (mT<=ub)) return logAddInstance.getLogIncompleteBeta(mPUB,mIP1+numpts.first+2,mIP0+numpts.second); else return evcont;};	
};

/** \brief compute probability that firing probability is above aboveNoise::mProb, which might e.g. represent a noise level, or a background firing rate, or some such. */
class aboveNoise : public evifunc {
	public:
		/** set aboveNoise::mProb to the noise level before iterating, and aboveNoise::mIP0, aboveNoise::mIP1 to the exponents of the beta prior. */
		inline double operator()(const int &lb,const int &ub,const int &binno,pair<int,int> &numpts,const double &evcont) 
		{if((lb<mT) && (mT<=ub)) return evcont+log(1.0-logAddInstance.getIncompleteBeta(mProb,numpts.first+mIP1,numpts.second+mIP0)); else return evcont;};
};


/** \brief functor for probability density that ( firing probability==log(p_firing_rate_and_latency::mProb)==p_firing_rate_and_latency::mLP at p_firing_rate_and_latency::mT and firing probability prior to that was below p_firing_rate_and_latency::mProb.

 This can only be sensibly evaluated if a bin starts at p_firing_rate_and_latency::mT. */
class p_firing_rate_and_latency : public evifunc {
	public:

		double mIP1m1,mIP0m1;
		int mTm1;
		void init() {
			mIP1m1=mIP1-1.0;
			mIP0m1=mIP0-1.0;
			mTm1=mT-1;
		}
		/** before averaging, set p_firing_rate_and_latency::mT, p_firing_rate_and_latency::mProb, p_firing_rate_and_latency::mLP,p_firing_rate_and_latency::mL1mP and p_firing_rate_and_latency::mIP1,p_firing_rate_and_latency::mIP0 */
		inline double operator()(const int &lb,const int &ub,const int &binno,const pair<int,int> &numpts,const double &evcont) {
			if(mT<=lb) return evcont;
			else if(mT>ub) return evcont+log(logAddInstance.getIncompleteBeta(mProb,numpts.first+mIP1,numpts.second+mIP0));
			else if(mTm1==lb) return (mIP1m1+numpts.first)*mLP+(mIP0m1+numpts.second)*mL1mP;
			else return DBL_LARGE_NEG;
		}
	
};

/** \brief functor for the sum of p_firing_rate_and_latency across all time indexes in the interval of interest. 

Instead of having to run the central iteration T times (once for each time index), we can carry out the sum over the time indexes prior to running the iteration, if we fix the bin number instead. That means that the central iteration has to be run only M times, rather than T. This yields the probability density of (the firing rate and the firing rate is above p_firing_rate::mProb).
*/
class p_firing_rate: public evifunc {
	public:

		double mIP1m1,mIP0m1;
		int mTm1;
		void init() {
			mIP1m1=mIP1-1.0;
			mIP0m1=mIP0-1.0;
		}
		/** before averaging, set p_firing_rate::mProb, p_firing_rate::mLP,p_firing_rate::mL1mP and p_firing_rate::mIP1,p_firing_rate::mIP0 */
		inline double operator()(const int &lb,const int &ub,const int &binno,const pair<int,int> &numpts,const double &evcont) {
			if(binno>mBin) return evcont;
			else if(binno<mBin) return evcont+log(logAddInstance.getIncompleteBeta(mProb,numpts.first+mIP1,numpts.second+mIP0));
			else return (mIP1m1+numpts.first)*mLP+(mIP0m1+numpts.second)*mL1mP;
		}
	
};

/** \brief probability that latency is at latency::mT, i.e. that the firing probability was below latency::mProb (this is the "signal separation level" or noise level) prior to latency::mT, and is above latency::mProb for at least 1 bin after latency::mT. 

This statement can only be true if a bin starts at latency::mT. Moreover, this statement can only be true for at most 1 time index.
*/
class latency : public evifunc {
	public:
		/** You need to set latency::mT, latency::mProb and the beta prior exponents.
		*/
		inline double operator()(const int &lb,const int &ub,const int &binno,const pair<int,int> &numpts,const double &evcont) {
			if(mT<=lb) return evcont;
			else if(mT>ub) return evcont+log(logAddInstance.getIncompleteBeta(mProb,numpts.first+mIP1,numpts.second+mIP0));
			else if(mT==lb+1) return evcont+log(1.0-logAddInstance.getIncompleteBeta(mProb,numpts.first+mIP1,numpts.second+mIP0));
			else return DBL_LARGE_NEG;
		}
	
};

/** \brief to compute the probability that there is a signal somewhere in the spiketrain, the averages of the latecy functor need to be summed over all time indexes in the interval of interest. 

This can potentially take long. To speed things up, we can compute this sum a priori given that the latency is in bin latsum::mBin, and then sum over all bins (which will only require the main iteration to be run M times, instead of T times).
*/

class latsum : public evifunc {
	public:
		/** You need to set latency::mProb and the beta prior exponents.
		*/
		inline double operator()(int &lb,int &ub,int &binno,pair<int,int> &numpts,double &evcont) {
			if(binno>mBin) return evcont;
			if(binno<mBin) return evcont+log(logAddInstance.getIncompleteBeta(mProb,numpts.first+mIP1,numpts.second+mIP0));
			if(lb+1<mT || lb+1>=mT2) return DBL_LARGE_NEG;
			return evcont+log(1.0-logAddInstance.getIncompleteBeta(mProb,numpts.first+mIP1,numpts.second+mIP0));
		}

};

/** \brief probability that inhibitory latency is at inhib_latency::mT, i.e. that the firing probability was above inhib_latency::mProb (this is the "inhibitory signal separation level") prior to inhib_latency::mT, and is below latency::mProb for at least 1 bin after inhib_latency::mT. 

This statement can only be true if a bin starts at inhib_latency::mT. Moreover, this statement can only be true for at most 1 time index. Data in the interval (inhib_latency::mTex1,inhib_latency::mTex2] are ignored.
*/
class inhib_latency : public evifunc {
	public:
		inline double operator()(int &lb,int &ub,int &binno,pair<int,int> &numpts,double &evcont) {
			if(mT<=lb || (ub<mT && mTex1<lb+1 && ub<mTex2)) return evcont;
			if (ub<mT) return evcont+log(1.0-logAddInstance.getIncompleteBeta(mProb,numpts.first+mIP1,numpts.second+mIP0));
			if(mT==lb+1) return evcont+log(logAddInstance.getIncompleteBeta(mProb,numpts.first+mIP1,numpts.second+mIP0));
			return DBL_LARGE_NEG;
		}

};

/** functor for fast(er) summation of inhib_latency across all time indexes of interest, to determine the probability of the presence of an inhibitory response. 

Same idea as latsum vs. latency. */
class inhib_latsum : public evifunc {
	public:
		inline double operator()(int &lb,int &ub,int &binno,pair<int,int> &numpts,double &evcont) {
			if(binno>mBin) return evcont;
			if(binno<mBin) {
				if(mTex1<lb+1 && ub<mTex2) return evcont;
				return evcont+log(1.0-logAddInstance.getIncompleteBeta(mProb,numpts.first+mIP1,numpts.second+mIP0));
			}
			if(lb+1<mT || lb+1>=mT2) return DBL_LARGE_NEG;
			return evcont+log(logAddInstance.getIncompleteBeta(mProb,numpts.first+mIP1,numpts.second+mIP0));
		}

};



// functor for no signal,signal,no signal
class nssns : public evifunc {
	public:
		inline double operator()(int &lb,int &ub,int &binno,pair<int,int> &numpts,double &evcont) {
			if((lb+1<mT && mT<ub) || (lb+1<mT2 && mT2<ub)) return DBL_LARGE_NEG;
			else if(mT2<=lb || mT>ub) return evcont+log(logAddInstance.getIncompleteBeta(mProb,numpts.first+mIP1,numpts.second+mIP0));
			else if(mT>lb && ub<=mT2) return evcont+log(1.0-logAddInstance.getIncompleteBeta(mProb,numpts.first+mIP1,numpts.second+mIP0));
			else return DBL_LARGE_NEG;
		}
	
};

// functor for no signal,signal,no signal PER BIN
class nssns_bin : public evifunc {
	public:
		inline double operator()(int &lb,int &ub,int &binno,pair<int,int> &numpts,double &evcont) {
			if(binno<mBin || binno>mBin2) return evcont+log(logAddInstance.getIncompleteBeta(mProb,numpts.first+mIP1,numpts.second+mIP0));
			else return evcont+log(1.0-logAddInstance.getIncompleteBeta(mProb,numpts.first+mIP1,numpts.second+mIP0));
		}
	
};


// functor for no signal
class nosig : public evifunc {
	public:
		inline double operator()(int &lb,int &ub,int &binno,pair<int,int> &numpts,double &evcont) {
			return evcont+log(logAddInstance.getIncompleteBeta(mProb,numpts.first+mIP1,numpts.second+mIP0));
		}
	
};


/**
\brief The template for the central iteration of the algorithm, as described in <a href="http://books.nips.cc/nips20.html">Endres, Oram, Schindelin, Foldiak (2007): Bayesian binning beats approximate alternatives: estimating peri-stimulus time histograms </a>. 

The template parameter is the function to be iterated over, which is in the innermost loop. Hence, using a compile-time resolvable parameter should speed up the averaging process. All computations in this object are done logarithmically, thus the averaging functor has to return the log of the values to be averaged.


@author Dominik Endres
*/

template<class AVGFUNC>
class evidenceComputer : public spikeCounter
{
public:
    evidenceComputer();

    ~evidenceComputer();
    /** do precomputation, compute priors and compute evidences.
	\return true on success.
    */
    bool compute();
    /** \return log evidence for a model with m bin boundaries */
    double getEvidence(int m);
    /** \return log sum of evidences for models with m1<=m<=m2 */
    double getEvidenceSum(int m1,int m2);
    /** set parameters of the averaging functor. Exact meanings will depend on which functor is used. */
    void setFunctorParams(int bin1,int bin2,int t1,int t2,double prob,double prob2,int tex1=INT_MAX,int tex2=-INT_MAX);
    /** test this object */
    void test();
    

protected:
    /** the averaging functor */
    AVGFUNC mAF;
    /** Evidences/averages for models with a given number of bins. The array E[] in the paper. */
    vector<double> mEvidences;
    /** sub-evidence array for the calculation. for details, see paper. The array subE[] in the paper. */
    vector<double> mSubEvidences;
    /** compute Evidences for models with up to mMMax bin boundaries */
    void computeEvidences();
};



#endif
