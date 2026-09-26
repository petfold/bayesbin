/***************************************************************************
 *   Copyright (C) 2007 by Dominik Endres   *
 *   dme2@st-andrews.ac.uk   *
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
#include "simplexminimiser.h"
#include "math.h"
#include "float.h"
#include <iostream>

const double TINY=1e-10;

simplexMinimiser::simplexMinimiser() {
	mTol=1e-7;
	mSteps=0;
}


simplexMinimiser::~simplexMinimiser()
{
}




/*!
    \fn simplexMinimiser::init(int dim)
 */
bool simplexMinimiser::init(int dim) {
	if(dim>0) {
		mTestpoint.clear();
		// component dim+1 holds the value at the point
		mTestpoint.resize(dim+1,0.0);
	} else {
		dim=mTestpoint.size()-1;
	}

	mSimplex.clear();
	mCenter.clear();
	mSearchDir.resize(dim,0.0);
	
	if(!checkConstraints(mTestpoint.begin(),mTestpoint.end()-1)) return false;

	// the first point: origin
	mTestpoint[dim]=evaluate(mTestpoint.begin(),mTestpoint.end()-1);
	mSimplex.push_back(mTestpoint);
	// other points of simplex: along the axes. search for max elongation<1
	mCenter.insert(mCenter.begin(),mTestpoint.begin(),mTestpoint.end()-1);
	int i,j;
	double ms;
	for(i=0;i<dim;i++) {
		mSearchDir[i]=1.0;
		if(i>0) mSearchDir[i-1]=0.0;
		ms=findMaxStretch(1.0);
		if(ms==0.0)
			ms=findMaxStretch(-1.0);
		if(ms==0.0) continue; // no elongation in this direction possible. 
		mTestpoint[dim]=evaluate(mTestpoint.begin(),mTestpoint.end()-1);
		mSimplex.push_back(mTestpoint);
	}
	return true;
}

/*!
    \fn simplexMinimiser::init(tPoint::iterator begin,tPoint::iterator end)
 */
bool simplexMinimiser::init(tPoint::iterator begin,tPoint::iterator end) {
	mTestpoint=tPoint(begin,end);
	mTestpoint.push_back(0);
	return init(0);
}


/*!
    \fn simplexMinimiser::evaluate(tpoint::iterator begin,tpoint::iterator end)
 */
double simplexMinimiser::evaluate(tPoint::iterator begin,tPoint::iterator end) {
	tPoint::iterator cp;
	int nd=1;
	double retval=0.0;
	for(cp=begin;cp!=end;cp++,nd++)
		retval+=pow(*cp-(nd%2==0?1:-1)*nd,2.0);
	return retval+1.0;
}


/*!
    \fn simplexMinimiser::findHiLo()
 */
void simplexMinimiser::findHiLo() {
	// find highest, 2nd highest and lowest point of simplex.
	tSimplex::iterator cp;
	int dim=mTestpoint.size()-1;
	if(mSimplex[0][dim]>mSimplex[1][dim]) {
		mHigh=mSimplex.begin();
		m2ndHigh=mHigh+1;
	} else {
		m2ndHigh=mSimplex.begin();
		mHigh=m2ndHigh+1;
	}
	mLow=mSimplex.begin();
	for(cp=mSimplex.begin();cp!=mSimplex.end();cp++) {
		if((*cp)[dim] <= (*mLow)[dim]) mLow=cp;
		if((*cp)[dim] > (*mHigh)[dim]) {
			m2ndHigh=mHigh;
			mHigh=cp;
		} else if(((*cp)[dim] > (*m2ndHigh)[dim]) && (cp!=mHigh)) m2ndHigh=cp;
	}
}


/*!
    \fn simplexMinimiser::hasConverged()
 */
bool simplexMinimiser::hasConverged() {
	int dim=mTestpoint.size()-1;
	findHiLo();
	// check if function values of high and low points are close enough
	if(fabs(((*mHigh)[dim]-(*mLow)[dim])/((*mHigh)[dim]+(*mLow)[dim]+TINY))>=mTol) return false;
	double dist=0.0;
	// check if simplex is small enough (to avoid getting stuck on an iso-surface)
	for(int i=0;i<dim;i++) dist+=pow((*mHigh)[i]-(*mLow)[i],2);
	if(dist>mTol) return false;
	
	mTestpoint=*mLow;
	return true;
}


/*!
    \fn simplexMinimiser::extrapolate()
 */
void simplexMinimiser::extrapolate(double factor) {
	// extrapolate high point through center of rest of simplex by factor (factor=1: simplex is unchanged)
	int i,dim=mTestpoint.size()-1;
	tSimplex::iterator cp;
	for(i=0;i<dim;i++) {
		mCenter[i]=0.0;
		for(cp=mSimplex.begin();cp!=mSimplex.end();cp++) {
			if(cp==mHigh) continue;
			mCenter[i]+=(*cp)[i];
		}
		mCenter[i]/=dim;
		mSearchDir[i]=(*mHigh)[i]-mCenter[i];
	}
	double ms=findMaxStretch(factor);
	if(ms==-DBL_MAX) cerr<<"ERROR in extrapolate(): invalid startpoint. This shouldn't happen. A bug?."<<endl;
	mTestpoint[dim]=evaluate(mTestpoint.begin(),mTestpoint.end()-1);
}

/*!
    \fn simplexMinimiser::computeTestpoint(const double &factor)
 */
void simplexMinimiser::computeTestpoint(const double &factor) {
	for(int i=0;i<mCenter.size();i++)
		mTestpoint[i]=mCenter[i]+factor*mSearchDir[i];
}



/*!
    \fn simplexMinimiser::findMaxStretch(double factor)
 */
double simplexMinimiser::findMaxStretch(double factor) {
	double pf;
	if(factor>0.0) pf=1.0;
	else pf=-1.0;
	double a,b,c;
	c=factor;
	computeTestpoint(c);
	if(checkConstraints(mTestpoint.begin(),mTestpoint.end()-1)) return factor;
	a=0.0;
	computeTestpoint(a);
	if(!checkConstraints(mTestpoint.begin(),mTestpoint.end()-1)) return -DBL_MAX; // error: interval doesn't start at valid point
	while(fabs(c-a)>1e-4) {
		b=(a+c)/2.0;
		computeTestpoint(b);
		if(checkConstraints(mTestpoint.begin(),mTestpoint.end()-1))
			a=b;
		else
			c=b;
	}
	if(a!=b) computeTestpoint(a);
	return a;
}

/*!
    \fn simplexMinimiser::checkConstraints(double &fac,tPoint &start,tPoint &direction)
 */
bool simplexMinimiser::checkConstraints(tPoint::iterator begin,tPoint::iterator end) {
    	// check if supplied point fulfills constraints
	return true;
}





/*!
    \fn simplexMinimiser::shrink()
 */
void simplexMinimiser::shrink() {
	// contract simplex by factor 0.5 around low point
	tSimplex::iterator cp;
	int i,dim=mTestpoint.size()-1;
	for(cp=mSimplex.begin();cp!=mSimplex.end();cp++) {
		if(cp==mLow) continue;
		for(i=0;i<dim;i++) 
			(*cp)[i]=((*cp)[i]+(*mLow)[i])/2.0;
		(*cp)[dim]=evaluate(cp->begin(),cp->end()-1);
	}
}


/*!
    \fn simplexMinimiser::findMinimum(double tol)
 */
tPoint simplexMinimiser::findMinimum(double tol,int maxsteps) {
	if(tol>=1e-7) mTol=tol;
	mSteps=0;
	iterate(maxsteps);
	init();
	iterate(maxsteps);
	return mTestpoint;
}


/*!
    \fn simplexMinimiser::iterate()
 */
void simplexMinimiser::iterate(int maxsteps) {
	int dim=mTestpoint.size()-1,i,j;
	while(!hasConverged()) {
		// try flipping the simplex away from the high point
		extrapolate(-1.0);
		if(mTestpoint[dim]<(*mHigh)[dim]) {
			*mHigh=mTestpoint;
		}
		// if it's even better than the current low point, do it again
		if(mTestpoint[dim]<(*mLow)[dim]) {
			extrapolate(2.0);
			if(mTestpoint[dim]<(*mHigh)[dim]) {
				*mHigh=mTestpoint;
			}
				
		} else 
			// higher than 2nd highest -- try contraction of high point
			if(mTestpoint[dim]>=(*m2ndHigh)[dim]) {
				extrapolate(0.5);
				if(mTestpoint[dim]<(*mHigh)[dim]) {
					*mHigh=mTestpoint;
				}
				else {
					shrink(); // contract around low point
				}
			}
		//printSimplex();
		mSteps++;	
		if(mSteps>maxsteps) {
			break;
		}
	}
	
}

/*!
    \fn simplexMinimiser::printSimplex()
 */
void simplexMinimiser::printSimplex() {
	int i,j;
	for(i=0;i<mSimplex.size();i++) {
		cerr<<i<<":";
		for(j=0;j<mSimplex[i].size()-1;j++)
			cerr<<mSimplex[i][j]<<", ";
		cerr<<"value: "<<mSimplex[i][j]<<endl;
	} 
}


/*!
    \fn simplexMinimiser::test()
 */
void simplexMinimiser::test() {
	const int dim=20;
	if(init(dim)) cout<<"Init ok"<<endl;
	findMinimum(1e-3,10000);
	cout<<"steps: "<<mSteps<<endl;
	cout<<"Lowest point: "<<endl;
	for(int i=1;i<dim+1;i++) {
		double tval=(i%2==0?1:-1)*i;
		cout<<i<<" "<<mTestpoint[i-1]<<" should be "<<tval<<" error "<<mTestpoint[i-1]-tval<<endl;
	}
	cout<<"Function value "<<mTestpoint[dim]<<" should be 1.0"<<endl;
}


void constraintTest::test() {
	mE1=3.0;
	mE2=30.0;
	tPoint tp;
	tp.push_back(20.0);
	tp.push_back(20.0);
	if(init(tp.begin(),tp.end())) cout<<"Init ok"<<endl;
	findMinimum(1e-3,100);
	cout<<"Lowest point: ("<<mTestpoint[0]<<","<<mTestpoint[1]<<") should be (3.0,30.0) "<<mSteps<<" steps"<<endl;
	mE1=0.5;
	mE2=30.0;
	init(tp.begin(),tp.end());
	findMinimum(1e-5);
	cout<<"Lowest point: ("<<mTestpoint[0]<<","<<mTestpoint[1]<<") for unconstrained minimum at (0.5,30.0) "<<mSteps<<" steps"<<endl;
	mE1=0.3;
	mE2=110.0;
	init(tp.begin(),tp.end());
	findMinimum(1e-5);
	cout<<"Lowest point: ("<<mTestpoint[0]<<","<<mTestpoint[1]<<") for unconstrained minimum (0.3,110.0) "<<mSteps<<" steps"<<endl;


};









