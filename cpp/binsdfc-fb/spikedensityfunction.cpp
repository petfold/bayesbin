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
#include "spikedensityfunction.h"
#include "math.h"
#include "time.h"

spikeDensityFunction::spikeDensityFunction()
{
	setMaxM(10);
	setMInterval(0,10);
	setIntervalPriors(0.5,0.5);
	setTimeInterval(0,600);
	clearData();
}


spikeDensityFunction::~spikeDensityFunction()
{
}




/*!
    \fn spikeDensityFunction::setMaxM(int mmax)
 */
void spikeDensityFunction::setMaxM(int mmax) {
	if(mmax<0) return;
	mMaxM=mmax;
	if(mMaxM<mM2) mM2=mMaxM;
	if(mM1>mM2) mM1=mM2;
	mSP.setMaxM(mMaxM);
	mbMIntervalChanged=true;
}

/*!
    \fn spikeDensityFunction::setMInterval(int m1,int m2)
 */
void spikeDensityFunction::setMInterval(int m1,int m2)
{
	if(m1>m2 || m1<0 || m2>mMaxM) return;
	mM1=m1;
	mM2=m2;
	mbMIntervalChanged=true;
	mWndstartLastTime=-1000;
	mNoiselevelLastTime=-1;
}


/*!
    \fn spikeDensityFunction::setIntervalPriors(double p0,double p1)
 */
void spikeDensityFunction::setIntervalPriors(double p0,double p1,double pub) {
	if(p0>1e-5) mP0=p0;
	if(p1>1e-5) mP1=p1;
	if(pub<1e-5) pub=1e-5;
	if(pub>1.0) pub=1.0;
	mPUB=pub;
	mSP.setIntervalPrior(mP0,mP1,pub);
	mbDataChanged=true;
}


/*!
    \fn spikeDensityFunction::setTimeInterval(int start,int end)
 */
void spikeDensityFunction::setTimeInterval(int start,int end) {
	if(end>start) {
		mIntervStart=start;
		mIntervEnd=end;
		mSP.setInterval(mIntervStart,mIntervEnd);
	}
}


/*!
    \fn spikeDensityFunction::clearData()
 */
void spikeDensityFunction::clearData() {
	mSP.clearData();
	mbDataChanged=true;
	mEvSum=DBL_LARGE_NEG;
	mEvIntervSum=DBL_LARGE_NEG;
	mWndstartLastTime=-1000;
	mNoiselevelLastTime=-1;
}


/*!
    \fn spikeDensityFunction::addData()
 */
void spikeDensityFunction::addData(vector<int>::iterator begin,vector<int>::iterator end) {
	mSP.addData(begin,end);
	mbDataChanged=true;
	mEvSum=DBL_LARGE_NEG;
	mEvIntervSum=DBL_LARGE_NEG;
	mWndstartLastTime=-1000;
	mNoiselevelLastTime=-1;
}


/*!
    \fn spikeDensityFunction::findInterval(double pmin,int &mmin,int &mmax)
 */
double spikeDensityFunction::findMInterval(double pmin,int &mmin,int &mmax)
{
	computeEvidences();
	double cev=DBL_LARGE_NEG;
	int i;
	// find maximum
	for(i=0;i<=mMaxM;i++) {
		if(cev<mEV.getEvidence(i)) {
			cev=mEV.getEvidence(i);
			mmin=mmax=i;
		}
	}
	// increase bracket until pmin is reached
	while(exp(cev-mEvSum)<pmin) {
		if((mmin>0) && ((mmax==mMaxM) || (mEV.getEvidence(mmin-1)>mEV.getEvidence(mmax+1)))) {
			mmin--;
			logAddInstance.add(cev,mEV.getEvidence(mmin));
		} else if(mmax<mMaxM) {
			mmax++;
			logAddInstance.add(cev,mEV.getEvidence(mmax));
		} else break;
	}
		
	return exp(cev-mEvSum);	
}


/*!
    \fn spikeDensityFunction::computeEvidences()
 */
void spikeDensityFunction::computeEvidences()
{
	if(mbDataChanged) {
		mbDataChanged=!mEV.compute();
		mEvSum=mEV.getEvidenceSum(0,mMaxM);
		mEvIntervSum=mEV.getEvidenceSum(mM1,mM2);
		mbMIntervalChanged=false;
	}
	if(mbMIntervalChanged) {
		mEvSum=mEV.getEvidenceSum(0,mMaxM);
		mEvIntervSum=mEV.getEvidenceSum(mM1,mM2);
		mbMIntervalChanged=false;
	}
}

/*!
    \fn spikeDensityFunction::getTotalEvidence()
 */
double spikeDensityFunction::getTotalEvidence() {
	computeEvidences();
	return mEvSum-log(mMaxM+1);
}

/*!
    \fn spikeDensityFunction::getIntervalEvidenceSum()
 */
double spikeDensityFunction::getIntervalEvidenceSum() {
	computeEvidences();
	return mEvIntervSum;
  
}

/*!
    \fn spikeDensityFunction::getEvidence(int m)
 */
double spikeDensityFunction::getEvidence(int m) {
    	if(m>mMaxM || m<0) return -DBL_MAX;
	computeEvidences();
	return mEV.getEvidence(m);
}




/*!
    \fn spikeDensityFunction::getMProb(int m)
 */
double spikeDensityFunction::getMProb(int m) {
	if(m>mMaxM || m<0) return 0.0;
	computeEvidences();
	return exp(mEV.getEvidence(m)-mEvSum);
}



/*!
    \fn spikeDensityFunction::getSDF(int tind)
 */
double spikeDensityFunction::getSDF(int tind)
{
	computeEvidences();
	if(mPUB==1.0) {
		mSDF.setFunctorParams(0,0,tind-mIntervStart,0,0.0,0.0);
		mSDF.compute();
		return exp(mSDF.getEvidenceSum(mM1,mM2)-mEvIntervSum);
	} else {
		mSDFp.setFunctorParams(0,0,tind-mIntervStart,0,0.0,0.0);
		mSDFp.compute();
		return exp(mSDFp.getEvidenceSum(mM1,mM2)-mEvIntervSum);
	}
}



/*!
    \fn spikeDensityFunction::getSDFForwardBackward(vector<double> &sdf,vector<double> &sdfstd,bool doStdDev)
 */
bool spikeDensityFunction::getSDFForwardBackward(vector<double> &sdf,vector<double> &sdfstd,bool doStdDev)
{
	computeEvidences();
	if(mPUB!=1.0) return false;
	vector<double> ev(mMaxM+1);
	for(int m=0;m<=mMaxM;m++) ev[m]=mEV.getEvidence(m);
	vector<double> sdf2;
	mFB.compute(ev,mM1,mM2,sdf,sdf2);
	sdfstd.assign(sdf.size(),0.0);
	if(doStdDev)
		for(unsigned t=0;t<sdf.size();t++) {
			double v=sdf2[t]-sdf[t]*sdf[t];
			sdfstd[t]=v>0.0?sqrt(v):0.0;
		}
	return true;
}


/*!
    \fn spikeDensityFunction::getSDFStdDev(int tind,double sdfval)
 */
double spikeDensityFunction::getSDFStdDev(int tind,double sdfval)
{
	computeEvidences();
	if(mPUB==1.0) {
		mSDF2.setFunctorParams(0,0,tind-mIntervStart,0,0.0,0.0);
		mSDF2.compute();
		sdfval=exp(mSDF2.getEvidenceSum(mM1,mM2)-mEvIntervSum)-pow(sdfval,2);    
	} else {
		mSDF2p.setFunctorParams(0,0,tind-mIntervStart,0,0.0,0.0);
		mSDF2p.compute();
		sdfval=exp(mSDF2p.getEvidenceSum(mM1,mM2)-mEvIntervSum)-pow(sdfval,2);
	}
	if(sdfval<=0.0) return 0.0;
	else return sqrt(sdfval);
}



/*!
    \fn spikeDensityFunction::getBBProb(int m,int tind)
 */
double spikeDensityFunction::getBBProb(int m,int tind)
{
	computeEvidences();
	mBBP.setFunctorParams(m,0,tind-mIntervStart,0,0.0,0.0);
	mBBP.compute();
	return exp(mBBP.getEvidenceSum(mM1,mM2)-mEvIntervSum); 
}


/*!
    \fn spikeDensityFunction::getBBExpectation(int m)
 */
double spikeDensityFunction::getBBExpectation(int m)
{
	computeEvidences();
	mBBE.setFunctorParams(m,0,0,0,0.0,0.0);
	mBBE.compute();
	return exp(mBBE.getEvidenceSum(mM1,mM2)-mEvIntervSum)+mIntervStart; 
}

/*!
    \fn spikeDensityFunction::getBBStdDev(int m,double bbexp)
 */
double spikeDensityFunction::getBBStdDev(int m,double bbexp)
{
	computeEvidences();
	mBBE2.setFunctorParams(m,0,0,0,0.0,0.0);
	mBBE2.compute();
	bbexp=exp(mBBE2.getEvidenceSum(mM1,mM2)-mEvIntervSum)-pow(bbexp-mIntervStart,2); 
	if(bbexp<=0.0) return 0.0;
	else return sqrt(bbexp);
}


/*!
    \fn spikeDensityFunction::getPAbNoise(int tind,double noiselevel)
 */
double spikeDensityFunction::getPAbNoise(int tind,double noiselevel)
{
	computeEvidences();
	mABNOI.setFunctorParams(0,0,tind-mIntervStart,0,noiselevel,0.0);
	mABNOI.compute();
	return exp(mABNOI.getEvidenceSum(mM1,mM2)-mEvIntervSum); 
}


/*!
    \fn spikeDensityFunction::getLatencyProb(int tind,double noiselevel)
 */
double spikeDensityFunction::getLatencyProb(int tind,double noiselevel)
{
	computeEvidences();
	mLAT.setFunctorParams(0,0,tind-mIntervStart,0,noiselevel,0.0);
	mLAT.compute();
	return exp(mLAT.getEvidenceSum(mM1,mM2)-mEvIntervSum); 
}

/*!
    \fn spikeDensityFunction::getInhibitoryLatencyProb(int tind,double noiselevel)
 */
double spikeDensityFunction::getInhibitoryLatencyProb(int tind,double inhibitory_noiselevel)
{
	computeEvidences();
	mInhibitoryLatency.setFunctorParams(0,0,tind-mIntervStart,0,inhibitory_noiselevel,0.0,mTex1-mIntervStart,mTex2-mIntervStart);
	mInhibitoryLatency.compute();
	return exp(mInhibitoryLatency.getEvidenceSum(mM1,mM2)-mEvIntervSum); 
}


/*!
    \fn spikeDensityFunction::getLatencyProbs(int tind,double noiselevel,vector<double> &res)
 */
void spikeDensityFunction::getLatencyProbs(int tind,double noiselevel,vector<double> &res)
{
	computeEvidences();
	mLAT.setFunctorParams(0,0,tind-mIntervStart,0,noiselevel,0.0);
	mLAT.compute();
	if(res.size()<=mM2) res.resize(mM2+1);
	for(int m=mM1;m<=mM2;m++)
		res[m]=exp(mLAT.getEvidence(m)-mEV.getEvidence(m));
}




/*!
    \fn spikeDensityFunction::getFiringRateGivenLatencyProb(int tind,double pfire,double noiselevel)
 */
double spikeDensityFunction::getFiringRateGivenLatencyProb(int tind,double pfire,double noiselevel) {
	computeEvidences();
	if(tind!=mWndstartLastTime || noiselevel!=mNoiselevelLastTime) {
		mWndstartLastTime=tind;
		mNoiselevelLastTime=noiselevel;
		mLAT.setFunctorParams(0,0,tind-mIntervStart,0,noiselevel,0.0);
		mLAT.compute();
		mWndstartIntervSum=mLAT.getEvidenceSum(mM1,mM2);
	}
	if(mWndstartIntervSum<=DBL_LARGE_NEG) return 0.0;
	mPFRAL.setFunctorParams(0,0,tind-mIntervStart,0,noiselevel,pfire<1e-200?1e-200:pfire);
	mPFRAL.compute();
	return exp(mPFRAL.getEvidenceSum(mM1,mM2)-mWndstartIntervSum);
}

/*!
    \fn spikeDensityFunction::getFiringRateProb(double pfire,double noiselevel)
 */
double spikeDensityFunction::getFiringRateProb(double pfire,double noiselevel,int tstart,int tend)
{
    	computeEvidences();
	double retval=DBL_LARGE_NEG;
	int MMin=mM1<1?1:mM1;
	for(int m=1;m<=mM2;m++) {
		int lb=m<MMin?MMin:m;
		mPFR.setFunctorParams(m,0,tstart-mIntervStart,tend-mIntervStart,noiselevel,pfire<1e-200?1e-200:pfire);
		mPFR.compute();
		logAddInstance.add(retval,mPFR.getEvidenceSum(lb,mM2));
	}
	return exp(retval-mEV.getEvidenceSum(MMin,mM2));
}



/*!
    \fn spikeDensityFunction::getBABProb(int tstart,int tend,double noiselevel)
 */
double spikeDensityFunction::getBABProb(int tstart,int tend,double noiselevel) {
	computeEvidences();
	mBAB.setFunctorParams(0,0,tstart-mIntervStart,tend-mIntervStart,noiselevel,0.0);
	mBAB.compute();
	return exp(mBAB.getEvidenceSum(mM1,mM2)-mEvIntervSum);
}

/*!
    \fn spikeDensityFunction::getNOSIG(double noiselevel)
 */
double spikeDensityFunction::getNOSIG(double noiselevel) {
	computeEvidences();
	mNOS.setFunctorParams(0,0,0,0,noiselevel,0.0);
	mNOS.compute();
	return exp(mNOS.getEvidenceSum(mM1,mM2)-mEvIntervSum);
}




/*!
    \fn spikeDensityFunction::getBABBinProb(int bin1,int bin2,double noiselevel)
 */
double spikeDensityFunction::getBABBinProb(int bin1,int bin2,double noiselevel)
{
	computeEvidences();
	mBABBin.setFunctorParams(bin1,bin2,0,0,noiselevel,0.0);
	mBABBin.compute();
	return exp(mBABBin.getEvidenceSum(mM1,mM2)-mEvIntervSum);   
}

/*!
    \fn spikeDensityFunction::getSeparationProb(double noiselevel)
 */
double spikeDensityFunction::getSeparationProb(double noiselevel,int tstart,int tend) {
	computeEvidences();
	double retval=DBL_LARGE_NEG;
	int MMin=mM1<1?1:mM1;
	for(int m=1;m<=mM2;m++) {
		int lb=m<MMin?MMin:m;
		mLatSum.setFunctorParams(m,0,tstart-mIntervStart,tend-mIntervStart,noiselevel,0.0);
		mLatSum.compute();
		logAddInstance.add(retval,mLatSum.getEvidenceSum(lb,mM2));
	}
	return exp(retval-mEV.getEvidenceSum(MMin,mM2));


}

/*!
    \fn spikeDensityFunction::getInhibitorySeparationProb(double noiselevel)
 */
double spikeDensityFunction::getInhibitorySeparationProb(double noiselevel,int tstart,int tend) {
	computeEvidences();
	double retval=DBL_LARGE_NEG;
	int MMin=mM1<1?1:mM1;
	for(int m=1;m<=mM2;m++) {
		int lb=m<MMin?MMin:m;
		mInhibitoryLatSum.setFunctorParams(m,0,tstart-mIntervStart,tend-mIntervStart,noiselevel,0.0,mTex1-mIntervStart,mTex2-mIntervStart);
		mInhibitoryLatSum.compute();
		logAddInstance.add(retval,mInhibitoryLatSum.getEvidenceSum(lb,mM2));
	}
	return exp(retval-mEV.getEvidenceSum(MMin,mM2));


}




list<pair<double,double> > spikeDensityFunction::goldenSectionSearch(double lbound,double ubound,double &maxpos,double &max,int tstart,int tend,double (spikeDensityFunction::*maxfunc)(double,int,int)) {
	list<pair<double,double> > retval;
	const double golden=(3.0-sqrt(5.0))/2.0;
	double a,b,c,x;
	double fa,fb,fc,fx;
	if(ubound<lbound) {
		a=lbound;
		lbound=ubound;
		ubound=a;
	}
	a=lbound;
	c=ubound;
	b=(c-a)*golden;
	fa=(this->*maxfunc)(a,tstart,tend);
	fb=(this->*maxfunc)(b,tstart,tend);
	fc=(this->*maxfunc)(c,tstart,tend);
	retval.push_back(pair<double,double>(a,fa));
	retval.push_back(pair<double,double>(b,fb));
	retval.push_back(pair<double,double>(c,fc));
	for(int steps=0;steps<10;steps++) {
		if((c-b)>(b-a)) {
			x=b+golden*(c-b);
			fx=(this->*maxfunc)(x,tstart,tend);
			retval.push_back(pair<double,double>(x,fx));
			if(fx>fb) {
				a=b;
				fa=fb;
				b=x;
				fb=fx;
			} else {
				c=x;
				fc=fx;
			}
		} else {
			x=b-golden*(b-a);
			fx=(this->*maxfunc)(x,tstart,tend);
			retval.push_back(pair<double,double>(x,fx));
			if(fx>fb) {
				c=b;
				fc=fb;
				b=x;
				fb=fx;
			} else {
				a=x;
				fa=fx;
			}
		}
	}
	maxpos=x;
	max=fx;
	retval.sort();
	return retval;
	
}



/*!
    \fn spikeDensityFunction::findBestSeparator(double lbound,double ubound,double &noiselevel,double &prob)
 */
list<pair<double,double> > spikeDensityFunction::findBestSeparator(double lbound,double ubound,double &noiselevel,double &prob,int tstart,int tend) {
	return goldenSectionSearch(lbound,ubound,noiselevel,prob,tstart,tend,&spikeDensityFunction::getSeparationProb);
}

list<pair<double,double> > spikeDensityFunction::findBestInhibitorySeparator(double lbound,double ubound,double &noiselevel,double &prob,int tstart,int tend) {
	return goldenSectionSearch(lbound,ubound,noiselevel,prob,tstart,tend,&spikeDensityFunction::getInhibitorySeparationProb);
}

