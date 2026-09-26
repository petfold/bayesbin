// Modified 2026-09-26: bin evidences from lgamma tables and prefix sums; co-occurrence counts
// from spike positions (see README.fb.md). The original code runs with --virtual-spike.
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
#ifndef SPIKECOUNTER_CPP
#define SPIKECOUNTER_CPP
#include "spikecounter.h"
#include "forwardbackward.h"
#include <cmath>
#include <iostream>
#include "time.h"
#include "stdlib.h"


spikeCounter::spikeCounter()
{
	setMaxM(10);
	setInterval(0,600);
	setIntervalPrior(1.0,1.0);
	clearData();
}


spikeCounter::~spikeCounter()
{
}




/*!
    \fn spikeCounter::setMaxM(int m)
 */
void spikeCounter::setMaxM(int m)
{
	if(m<0) return;
	mPriors.resize(m+1,0.0);
	mMMax=m;
}


/*!
    \fn spikeCounter::setInterval(int start, int end)
 */
void spikeCounter::setInterval(int start, int end)
{
	if(end<=start) return;
	mIntervStart=start;
	mIntervEnd=end;
	mSpikeTrain.resize(mIntervEnd-mIntervStart);
	mCurSp.resize(mIntervEnd-mIntervStart);
	allocArrays();
}


/*!
    \fn spikeCounter::setIntervalPrior(double p0,double p1)
 */
void spikeCounter::setIntervalPrior(double p0,double p1,double pub)
{
	if(p0<=1e-5) p0=1e-5;
	if(p1<=1e-5) p1=1e-5;
	if(pub<1e-5) pub=1e-5;
	if(pub>1.0) pub=1.0;
	mPrior0=p0;
	mPrior1=p1;
	mPUB=pub;
	mbDataChanged=true;
}


/*!
    \fn spikeCounter::allocArrays()
 */
bool spikeCounter::allocArrays()
{
	int i,j,cs;

	if(mAutoCorr.size()!=mIntervEnd-mIntervStart) {
		mAutoCorr.resize(mIntervEnd-mIntervStart);
		for(i=0;i<mIntervEnd-mIntervStart;i++)
			mAutoCorr[i].resize(i+1);
	}


	if(mIntervalEvidences.size()==mIntervEnd-mIntervStart) return true;
	vector<double> dummy;
	mIntervalEvidences.resize(mIntervEnd-mIntervStart,dummy);
	vector<pair<int,int> > dummy1;
	mIntervalCounts.resize(mIntervEnd-mIntervStart,dummy1);
	pair<int,int> zv;
	zv.first=zv.second=0;
	for(i=0;i<mIntervEnd-mIntervStart;i++) {
		cs=mIntervEnd-mIntervStart-i;
		mIntervalEvidences[i].resize(cs,0.0);
		mIntervalCounts[i].resize(cs,zv);
	}
	return true;
}


/*!
    \fn spikeCounter::clearData()
 */
void spikeCounter::clearData()
{
	vector<pair<int,int> >::iterator i;
	for(i=mSpikeTrain.begin();i!=mSpikeTrain.end();i++) 
		i->first=i->second=0;
	vector<vector<int> >::iterator it1;
	vector<int>::iterator it2;
	for(it1=mAutoCorr.begin();it1!=mAutoCorr.end();it1++)
		for(it2=it1->begin();it2!=it1->end();it2++)
			*it2=0;

	mbDataChanged=true;
}


/*!
    \fn spikeCounter::addData(vector<int>::iterator begin,vector<int>::iterator end)
 */
void spikeCounter::addData(vector<int>::iterator begin,vector<int>::iterator end)
{
	vector<int>::iterator cur=begin,icsp;
       	for(icsp=mCurSp.begin();icsp!=mCurSp.end();*icsp=0,icsp++);
	int i,j;
	while(cur!=end && *cur<mIntervStart) cur++;
	for(i=0;i<mSpikeTrain.size();i++) {
		if(cur!=end && i==*cur-mIntervStart) {
			mSpikeTrain[i].first++;
			mCurSp[i]++;
			if(cur!=end) cur++;
		} else mSpikeTrain[i].second++;
			
	}
	
	if(fb::enabled) {  // the same counts, from the positions that spiked: O(spikes^2), not O(T^2)
		vector<int> idx;
		for(i=0;i<(int)mCurSp.size();i++) if(mCurSp[i]) idx.push_back(i);
		for(unsigned p=0;p<idx.size();p++)
			for(unsigned q=0;q<p;q++) mAutoCorr[idx[p]][idx[q]]++;
	} else
	for(i=0;i<mCurSp.size();i++)
		for(j=0;j<i;j++)
			if(mCurSp[i] && mCurSp[j]) mAutoCorr[i][j]++;



	mbDataChanged=true;
}


/*!
    \fn spikeCounter::precomputeSubIntervals()
 */
void spikeCounter::precomputeSubIntervals()
{
	int i,j,cs;
	if(!mbDataChanged) return;
	allocArrays();
	mDataVersion++;
	if(fb::enabled && mPUB==1.0) {
		// counts from prefix sums and log Beta(s+prior1, g+prior0) from lgamma tables indexed by the
		// integer counts: no lgamma per bin, and every row independent (in parallel)
		const int K=mIntervalCounts.size();
		vector<int> c1(K+1,0),c0(K+1,0);
		for(i=0;i<K;i++) { c1[i+1]=c1[i]+mSpikeTrain[i].first; c0[i+1]=c0[i]+mSpikeTrain[i].second; }
		const int S=c1[K],G=c0[K];
		vector<double> lA(S+1),lB(G+1),lC(S+G+1);  // lgamma is not thread-safe (signgam): build here
		for(int s=0;s<=S;s++) lA[s]=lgamma(s+mPrior1);
		for(int g=0;g<=G;g++) lB[g]=lgamma(g+mPrior0);
		for(int n=0;n<=S+G;n++) lC[n]=lgamma(n+mPrior1+mPrior0);
		#pragma omp parallel for schedule(guided)
		for(int a=0;a<K;a++) {
			const int len=mIntervalCounts[a].size();
			for(int jj=0;jj<len;jj++) {
				const int s=c1[a+jj+1]-c1[a],g=c0[a+jj+1]-c0[a];
				mIntervalCounts[a][jj].first=s;
				mIntervalCounts[a][jj].second=g;
				mIntervalEvidences[a][jj]=lA[s]+lB[g]-lC[s+g];
			}
		}
		mbDataChanged=false;
		return;
	}
	for(i=mIntervalCounts.size()-1;i>=0;i--) {
		mIntervalCounts[i][0]=mSpikeTrain[i];
		if(mPUB==1.0) mIntervalEvidences[i][0]=logAddInstance.getBeta(mIntervalCounts[i][0].first+mPrior1,mIntervalCounts[i][0].second+mPrior0);
		else mIntervalEvidences[i][0]=logAddInstance.getLogIncompleteBeta(mPUB,mIntervalCounts[i][0].first+mPrior1,mIntervalCounts[i][0].second+mPrior0);
		for(j=1;j<mIntervalCounts[i].size();j++) {
			mIntervalCounts[i][j].first=mIntervalCounts[i+1][j-1].first+mIntervalCounts[i][0].first;
			mIntervalCounts[i][j].second=mIntervalCounts[i+1][j-1].second+mIntervalCounts[i][0].second;
			if(mPUB==1.0) mIntervalEvidences[i][j]=logAddInstance.getBeta(mIntervalCounts[i][j].first+mPrior1,mIntervalCounts[i][j].second+mPrior0);
			else mIntervalEvidences[i][j]+=logAddInstance.getLogIncompleteBeta(mPUB,mIntervalCounts[i][j].first+mPrior1,mIntervalCounts[i][j].second+mPrior0);
		}
	}
	mbDataChanged=false;
}


/*!
    \fn spikeCounter::getSpikes()
 */
int spikeCounter::getSpikes()
{
	precomputeSubIntervals();
	return mIntervalCounts[0][mIntervalCounts[0].size()-1].first;
}


/*!
    \fn spikeCounter::getGaps()
 */
int spikeCounter::getGaps()
{
	precomputeSubIntervals();
	return mIntervalCounts[0][mIntervalCounts[0].size()-1].second;
}


/*!
    \fn spikeCounter::computePrior()
 */
bool spikeCounter::computePrior()
{
	int npos=mIntervEnd-mIntervStart-1;
	if(mMMax>npos) return false;
	double bf=logAddInstance.getLogIncompleteBeta(mPUB,mPrior1,mPrior0);
	for(int i=0;i<=mMMax;i++)
		mPriors[i]=-(i+1)*bf-logAddInstance.getKoutofN(npos,i);
	return true;
}

/*!
    \fn spikeCounter::test(double pfire,int numtrains)
 */
void spikeCounter::test(double pfire0,double pfire1,double pfire3,int numtrains)
{
	int istart=20;
	int iend=620;
	cout<<"=== Testing spikecounter ==="<<endl;
	cout<<"Generating "<<numtrains<<" spiketrains with pfires="<<pfire0<<","<<pfire1<<","<<pfire3<<endl;
	cout<<"Using at most 100 bin boundaries "<<endl;
	setMaxM(20);
	cout<<"Setting counting interval to "<<istart<<"-"<<iend<<endl;
	setInterval(istart,iend);
	srand48(time(0));
	vector<int> srun;
	int i,nrun,ns,ng;
	clearData();
	ns=ng=0;
	for(nrun=0;nrun<numtrains;nrun++) {
		srun.clear();
		double pf;
		for(i=0;i<700;i++) {
			if(i<80) pf=pfire0;
			else if(i<180) pf=pfire1;
			else pf=pfire3;
			if(drand48()<=pf) {
				srun.push_back(i);
				if(istart<=i && i<iend) ns++;
			} else if(istart<=i && i<iend) ng++;
		}
		addData(srun.begin(),srun.end());
	}
	precomputeSubIntervals();
	cout<<"found "<<getSpikes()<<" spikes, should be "<<ns<<endl;
	cout<<"found "<<getGaps()<<" gaps, should be "<<ng<<endl; 
	cout<<"Total: "<<getSpikes()+getGaps()<<", should be "<<600*numtrains<<endl;
	cout<<"=== Testing spikecounter done ==="<<endl<<endl;
}

// instantiation of static members
vector<vector<double> > spikeCounter::mIntervalEvidences;
vector<vector<pair<int,int> > > spikeCounter::mIntervalCounts;
vector<double> spikeCounter::mPriors;
unsigned long spikeCounter::mDataVersion=0;
double spikeCounter::mPrior1,spikeCounter::mPrior0;
int spikeCounter::mIntervStart,spikeCounter::mIntervEnd;
int spikeCounter::mMMax;
vector<pair<int,int> > spikeCounter::mSpikeTrain;
vector<vector<int> > spikeCounter::mAutoCorr;
bool spikeCounter::mbDataChanged;
double spikeCounter::mPUB;

#endif
