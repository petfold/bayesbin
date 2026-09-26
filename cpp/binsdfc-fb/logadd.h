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
#ifndef LOGADD_H
#define LOGADD_H

#include "math.h"
#include "float.h"
#include "time.h"

#define POINTS_PER_UNIT_INTERVAL 400

/**
perform logarithmic addition per lookup-table. seems to be about 3 times faster than direct evaluation

@author Dominik Endres
*/

const double LDEPS=-log(DBL_EPSILON);
//CUTOFF should be >= ceil(-log(DBL_EPSILON))
template<int NPPUI,int CUTOFF=40,int MAXIBITER=200>
class logAdd{
public:
	logAdd();
	/** logarithmic addition. use lookup in pre-computed interpolation table. seems to be a bit faster than direct calculation (e.g. via logAdd::logAddR()
	 \return log(exp(a)+exp(b)) */
	inline void add(double &a,double b) const {
		double ind,rem;
		int i;
		rem=modf(fabs(a-b)*NPPUI,&ind);
		if(ind>=CUTOFF*NPPUI) a=fmax(a,b);
		else { 
			i=2*(int)ind;
			a=fmax(a,b)+mInterpols[i]+mInterpols[++i]*rem;
		}
	};
	/** a=log(exp(a)+exp(b)) */
	inline void logAddR(double &a,double b) const {a=fabs(a-b)>LDEPS?fmax(a,b):fmax(a,b)+log1p(exp(-fabs(a-b)));}

	/** a=log(exp(a)+exp(b)) */
	inline void logAddRR(double &a,double &b) const {a=fabs(a-b)>LDEPS?fmax(a,b):fmax(a,b)+log1p(exp(-fabs(a-b)));}

	/** log beta function */
	double getBeta(double a,double b) const {return lgamma(a)+lgamma(b)-lgamma(a+b);};

	/** log (k out of n) */
	double getKoutofN(int n,int k) const {return lgamma(n+1)-lgamma(k+1)-lgamma(n-k+1);};

	/** normalized incomplete beta function. */
	inline double getIncompleteBeta(double p,double a,double b) const;

	/** incomplete beta function. Includes B(a,b) factor */
	inline double getLogIncompleteBeta(double p,double a,double b) const;

        /** Digamma function */
	inline double getDLogGamma(double &x);
	void test() const;

	~logAdd(){};
protected:
	/** interpolation values of log(1+x) */
	static double mInterpols[2*(CUTOFF*NPPUI+1)];
	/** continued fraction routine for incomplete beta */
	inline double betacf(double a, double b, double  p) const;
};

const logAdd<POINTS_PER_UNIT_INTERVAL> logAddInstance;

#endif
