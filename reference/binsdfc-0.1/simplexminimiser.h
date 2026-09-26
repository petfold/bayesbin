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
#ifndef SIMPLEXMINIMISER_H
#define SIMPLEXMINIMISER_H

#include <vector>
#include <iostream>
#include "math.h"

using namespace std;



/** points of the simplex. Highest component holds value at point. */
typedef vector<double> tPoint;
typedef vector<tPoint> tSimplex;

/**
\brief Perform minimum search in N-dim by simplex method (see Numerical Recipes, chapter 10)

	@author Dominik Endres <dme2@st-andrews.ac.uk>
*/
class simplexMinimiser{
public:
	simplexMinimiser();

	~simplexMinimiser();
	/** initialize the simplex to origin and unit vectors along the axes. If called with dim==0: restart simplex at current mTestPoint (==result from previous call to  simplexMinimiser::findMinimum() )
	\param dim: dimensionality of space in which the search is done
	\return  false: the start point was not valid (according to checkConstraints())
	*/
    	bool init(int dim=0);


	/** init the simplex.
	\param begin: iterator to first component of start point
	\param end: iterator past last component of start point
	\return: false, the point was not valid (according to checkConstraints()
	*/
	bool init(tPoint::iterator begin,tPoint::iterator end);

	/** 
	\param begin: iterator to first component of evaluation point
	\param begin: iterator past last component of evaluation point
	\return value of the function to be minimised 
	*/
    	virtual double evaluate(tPoint::iterator begin,tPoint::iterator end);
	
	/** \return true if supplied point fulfills constraints */
    	virtual bool checkConstraints(tPoint::iterator begin,tPoint::iterator end);

	/** find the minimum.
	\param tol: tolerance of simplex.
	\param maxsteps: max number of minimisation steps
	*/
    	tPoint findMinimum(double tol=1e-7,int maxsteps=1000);


	/** \return true if converged */
	bool hasConverged();

	/** \return number of steps */
	int getNumSteps() {return mSteps;};
    	
	/** call to test this object */
	void test();

    	void printSimplex();
    
	

protected:
	/** find highest, 2nd highest and lowest point, store in mHigh,m2ndHigh,mLow. */
	void findHiLo();

	/** extrapolate high-point through the hypersurface spanned by the rest of the simplex.
	Store resulting point in mTestpoint. If factor==1, simplex remains unchanged (i.e. mTestpoint=*mHigh)
	\param factor: extrapolation factor
	*/
	void extrapolate(double factor=-1);

	/** shrink simplex around mLow by factor 0.5 */
	void shrink();

	/** simplex iteration until function does not decrease any more.
	\param maxsteps: maximal number of minimisationb steps
	*/
    	void iterate(int maxsteps);

	/** 
	\return x such that |x|<=|factor|, |x| maximal and mCenter+x*mSearchDir fulfills constraints. set mTestpoint to this point
	*/
	double findMaxStretch(double factor);

	/** compute testpoint by going factor times mSearchDir past mCenter */
    	void computeTestpoint(const double &factor);

	/** the simplex, a vector of double vectors. */
	tSimplex mSimplex;
	/** test point, center of simplex w/o high point and search direction */
	tPoint mTestpoint,mCenter,mSearchDir;
	/** pointers to highest,2nd highest and lowest point */
	tSimplex::iterator mHigh,m2ndHigh,mLow;
	/** fractional tolerance for termination */
	double mTol;
	/** number of steps so far. */
	int mSteps;
	
};


class constraintTest : public simplexMinimiser {
public:
	constraintTest() : simplexMinimiser(),mE1(1.0),mE2(1.0) {};
	virtual double evaluate(tPoint::iterator begin,tPoint::iterator end) {
		double t1=*begin;
		double t2=*(begin+1);
		return pow(mE1-t1,4)+pow(mE2-t2,4)+1.0;
	};
	virtual bool checkConstraints(tPoint::iterator begin,tPoint::iterator end) {
		if(*begin<1.0 || *begin>100.0) return false;
		if(*(begin+1)<1.0 || *(begin+1)>100.0) return false;
		return true;
	};
	void test();
	double mE1,mE2;

};

#endif
