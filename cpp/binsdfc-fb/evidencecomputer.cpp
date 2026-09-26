// Modified 2026-09-26 by Peter Földiák: plain evidences via forwardbackward.h (see README.fb.md).
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
#include "evidencecomputer.h"
#include "forwardbackward.h"
#include "logadd.h"
#include <iostream>
#include "omp.h"

template<class AVGFUNC>
evidenceComputer<AVGFUNC>::evidenceComputer()
 : spikeCounter()
{
}

template<class AVGFUNC>
evidenceComputer<AVGFUNC>::~evidenceComputer()
{
}




/*!
    \fn evidenceComputer::computeEvidences()
 */
template<class AVGFUNC>
void evidenceComputer<AVGFUNC>::computeEvidences()
{
	ensureTables();  // the functors read the per-bin tables (lean mode builds them on first need)
	int k,K,kk,m,lb;
	mAF.mIP0=mPrior0;
	mAF.mIP1=mPrior1;
	mAF.mPUB=mPUB;
	mAF.init();
	K=mIntervEnd-mIntervStart;
	mSubEvidences.resize(K);
	mEvidences.resize(mMMax+1);
	m=0;
	kk=-1;
	for(k=0;k<K;k++) mSubEvidences[k]=mAF(kk,k,m,mIntervalCounts[0][k],mIntervalEvidences[0][k]);
	mEvidences[0]=mSubEvidences[K-1]+mPriors[0];	
	// Modified 2026-09-26: with the new paths on, exact log-additions (logAddR) instead of the table
	// lookup, so that these averages are as exact as the plain evidences they are divided by
	const bool exact=fb::enabled;
	for(m=1;m<=mMMax;m++) {
		lb=(m==mMMax?K-1:m);
		for(k=K-1;k>=lb;k--) {
			mSubEvidences[k]=DBL_LARGE_NEG;
			if(exact)
			for(int kk=m-1;kk<=k-1;kk++) {
				logAddInstance.logAddR(mSubEvidences[k],mSubEvidences[kk]+mAF(kk,k,m,mIntervalCounts[kk+1][k-kk-1],mIntervalEvidences[kk+1][k-kk-1]));
			}
			else
			for(int kk=m-1;kk<=k-1;kk++) {
				logAddInstance.add(mSubEvidences[k],mSubEvidences[kk]+mAF(kk,k,m,mIntervalCounts[kk+1][k-kk-1],mIntervalEvidences[kk+1][k-kk-1]));
			}
		}
		mEvidences[m]=mSubEvidences[K-1]+mPriors[m];
	}
	
}


/*!
    \fn evidenceComputer::compute()
 */
template<class AVGFUNC>
bool evidenceComputer<AVGFUNC>::compute()
{
	if(!computePrior()) return false;
	precomputeSubIntervals();
	computeEvidences();
	return true;
}


/*!
    \fn evidenceComputer::getEvidence(int m)
 */
template<class AVGFUNC>
double evidenceComputer<AVGFUNC>::getEvidence(int m)
{
	if(m<mEvidences.size()) return mEvidences[m];
	else return DBL_LARGE_NEG;
}


/*!
    \fn evidenceComputer::setFunctorParams(int bin,int t,int pos,double prob,double prob2)
 */
template<class AVGFUNC>
void evidenceComputer<AVGFUNC>::setFunctorParams(int bin1,int bin2,int t1,int t2,double prob,double prob2,int tex1,int tex2) {
	mAF.mBin=bin1;
	mAF.mBin2=bin2;
	mAF.mT=t1;
	mAF.mT2=t2;
	mAF.mProb=prob;	
	mAF.mProb2=prob2;
	mAF.mLP=log(prob2);
	mAF.mL1mP=log(1.0-prob2);
	mAF.mTex1=tex1;
	mAF.mTex2=tex2;
}

/*!
    \fn evidenceComputer::getEvidenceSum(int m1,int m2)
 */
template<class AVGFUNC>
double evidenceComputer<AVGFUNC>::getEvidenceSum(int m1,int m2) {
	if(m1>m2) return DBL_LARGE_NEG;
	if(m1<0) m1=0;
	if(m2>=mEvidences.size()) m2=mEvidences.size()-1;
	double retval=DBL_LARGE_NEG;
	for(int i=m1;i<=m2;i++) {  // exact log-addition with the new paths on (2026-09-26)
		if(fb::enabled) logAddInstance.logAddR(retval,mEvidences[i]);
		else logAddInstance.add(retval,mEvidences[i]);
	}
	return retval;	
}



/*!
    \fn evidenceComputer::test()
 */
template<class AVGFUNC>
void evidenceComputer<AVGFUNC>::test()
{
	// test if iteration works with no data
	spikeCounter::test(0.5,0.5,0.5,0);
	cout<<"=== Testing evidencecomputer ==="<<endl;
	cout<<"=== Testing evidencecomputer with no data ==="<<endl;
	compute(); 
	int m;
	for(m=0;m<=mMMax;m++)
		cout<<"Evidence for m="<<m<<": "<<getEvidence(m)<<", should be 0.0"<<endl;
	cout<<"=== Testing evidencecomputer with no data, done ==="<<endl;
	cout<<"=== Testing evidencecomputer with data, 1 interval ==="<<endl;
	spikeCounter::test(0.1,0.1,0.1,100);
	compute();
	for(m=0;m<=mMMax;m++)
		cout<<"Evidence for m="<<m<<": "<<getEvidence(m)<<endl;
	cout<<"=== Testing evidencecomputer with data, 1 interval,done ==="<<endl;
	cout<<"=== Testing evidencecomputer with data, 2 interval ==="<<endl;
	spikeCounter::test(0.015,0.1,0.1,100);
	compute();
	for(m=0;m<=mMMax;m++)
		cout<<"Evidence for m="<<m<<": "<<getEvidence(m)<<endl;
	cout<<"=== Testing evidencecomputer with data, 2 interval,done ==="<<endl;
	cout<<"=== Testing evidencecomputer with data, 3 interval ==="<<endl;
	spikeCounter::test(0.015,0.1,0.020,100);
	compute();
	for(m=0;m<=mMMax;m++)
		cout<<"Evidence for m="<<m<<": "<<getEvidence(m)<<endl;
	cout<<"=== Testing evidencecomputer with data, 3 interval,done ==="<<endl;
	cout<<"=== Testing evidencecomputer done==="<<endl;
}

// Modified 2026-09-26: the plain evidences (evifunc) by the matrix-vector central
// iteration of forwardbackward.h, when the prior is the plain Beta; else as before.
template<>
void evidenceComputer<evifunc>::computeEvidences()
{
	int K=mIntervEnd-mIntervStart;
	mEvidences.resize(mMMax+1);
	if(fb::enabled && mPUB==1.0 && K>0) {
		vector<vector<double> > fwd;
		fb::fastForward(K,mMMax,[](int a,int b){return iecOf(a,b);},fwd);
		for(int m=0;m<=mMMax;m++) mEvidences[m]=fwd[m][K-1]+mPriors[m];
		mSubEvidences=fwd[mMMax];
		fb::cache.version=mDataVersion;
		fb::cache.K=K;
		fb::cache.mmax=mMMax;
		fb::cache.fwd.swap(fwd);
		return;
	}
	ensureTables();
	int k,kk,m,lb;
	mAF.mIP0=mPrior0;
	mAF.mIP1=mPrior1;
	mAF.mPUB=mPUB;
	mAF.init();
	mSubEvidences.resize(K);
	m=0;
	kk=-1;
	for(k=0;k<K;k++) mSubEvidences[k]=mAF(kk,k,m,mIntervalCounts[0][k],mIntervalEvidences[0][k]);
	mEvidences[0]=mSubEvidences[K-1]+mPriors[0];	
	// Modified 2026-09-26: with the new paths on, exact log-additions (logAddR) instead of the table
	// lookup, so that these averages are as exact as the plain evidences they are divided by
	const bool exact=fb::enabled;
	for(m=1;m<=mMMax;m++) {
		lb=(m==mMMax?K-1:m);
		for(k=K-1;k>=lb;k--) {
			mSubEvidences[k]=DBL_LARGE_NEG;
			if(exact)
			for(int kk=m-1;kk<=k-1;kk++) {
				logAddInstance.logAddR(mSubEvidences[k],mSubEvidences[kk]+mAF(kk,k,m,mIntervalCounts[kk+1][k-kk-1],mIntervalEvidences[kk+1][k-kk-1]));
			}
			else
			for(int kk=m-1;kk<=k-1;kk++) {
				logAddInstance.add(mSubEvidences[k],mSubEvidences[kk]+mAF(kk,k,m,mIntervalCounts[kk+1][k-kk-1],mIntervalEvidences[kk+1][k-kk-1]));
			}
		}
		mEvidences[m]=mSubEvidences[K-1]+mPriors[m];
	}
}

// template instances
template class evidenceComputer<evifunc>;
template class evidenceComputer<binboundary_expectation>;
template class evidenceComputer<binboundary_expectation2>;
template class evidenceComputer<binboundary_probability>;
template class evidenceComputer<SDF>;
template class evidenceComputer<SDFp>;
template class evidenceComputer<SDF2>;
template class evidenceComputer<SDF2p>;
template class evidenceComputer<aboveNoise>;
template class evidenceComputer<latency>;
template class evidenceComputer<nssns>;
template class evidenceComputer<nssns_bin>;
template class evidenceComputer<nosig>;
template class evidenceComputer<latsum>;
template class evidenceComputer<p_firing_rate_and_latency>;
template class evidenceComputer<p_firing_rate>;
template class evidenceComputer<inhib_latency>;
template class evidenceComputer<inhib_latsum>;
