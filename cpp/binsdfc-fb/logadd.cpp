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
#include "logadd.h"
using namespace std;
#include <iostream>

static double logAdd_GammaCof[6]={76.18009172947146,-86.50532032941677,
24.01409824083091,-1.231739572450155,
0.1208650973866179e-2,-0.5395239384953e-5};

template<int NPPUI,int CUTOFF,int MAXIBITER>
logAdd<NPPUI,CUTOFF,MAXIBITER>::logAdd() {
	int x;
	for(x=CUTOFF*NPPUI;x>=0;x--) {
		mInterpols[2*x]=log1p(exp(-(double)x/(double)NPPUI));
		if(x<CUTOFF*NPPUI) {
			mInterpols[2*x+1]=mInterpols[2*(x+1)]-mInterpols[2*x];
		}
		else mInterpols[2*x+1]=0.0;
	}
	
}

template<int NPPUI,int CUTOFF,int MAXIBITER>
inline double logAdd<NPPUI,CUTOFF,MAXIBITER>::getIncompleteBeta(double p,double a,double b) const {
	double bt;

	if (p < 0.0 || p > 1.0) {
		cout<<"Improper argument for getIncompleteBeta! p="<<p<<endl;
		return 0.0;
	}
	if(p==0.0 || p==1.0) return p;
			
	bt=exp(lgamma(a+b)-lgamma(a)-lgamma(b)+a*log(p)+b*log(1.0-p));
	if (p < (a+1.0)/(a+b+2.0))
		return bt*betacf(a,b,p)/a;
	else
		return 1.0-bt*betacf(b,a,1.0-p)/b;					
}


template<int NPPUI,int CUTOFF,int MAXIBITER>
inline double logAdd<NPPUI,CUTOFF,MAXIBITER>::getLogIncompleteBeta(double p,double a,double b) const {
	return log(getIncompleteBeta(p,a,b))+lgamma(a)+lgamma(b)-lgamma(a+b);
}


template<int NPPUI,int CUTOFF,int MAXIBITER>
inline double logAdd<NPPUI,CUTOFF,MAXIBITER>::betacf(double a, double b, double  p) const {
	int m,m2;
	double aa,c,d,del,h,qab,qam,qap;

	qab=a+b;
	qap=a+1.0;
	qam=a-1.0;
	c=1.0;
	d=1.0-qab*p/qap;
	if (fabs(d) < DBL_MIN) d=DBL_MIN;
	d=1.0/d;
	h=d;
	for (m=1;m<=MAXIBITER;m++) {
		m2=2*m;
		aa=m*(b-m)*p/((qam+m2)*(a+m2));
		d=1.0+aa*d;
		if (fabs(d) < DBL_MIN) d=DBL_MIN;
		c=1.0+aa/c;
		if (fabs(c) < DBL_MIN) c=DBL_MIN;
		d=1.0/d;
		h *= d*c;
		aa = -(a+m)*(qab+m)*p/((a+m2)*(qap+m2));
		d=1.0+aa*d;
		if (fabs(d) < DBL_MIN) d=DBL_MIN;
		c=1.0+aa/c;
		if (fabs(c) < DBL_MIN) c=DBL_MIN;
		d=1.0/d;
		del=d*c;
		h *= del;
		if (fabs(del-1.0) < DBL_EPSILON*1e2) break;
	}
	if (m > MAXIBITER) cout<<"a or b too big, or MAXIBITER too small in betacf"<<endl;
	return h;
}

template<int NPPUI,int CUTOFF,int MAXIBITER>
void logAdd<NPPUI,CUTOFF,MAXIBITER>::test() const {
	double ssize=0.001/NPPUI;
	double maxerr=0.0,maxx=0.0,avgerr=0.0,cerr,a,x;
	cout<<"=== logAdd::test() starts ==="<<endl;
	for(x=0.0;x>=-40.0;x-=ssize) {
		a=0.0;
		add(a,x);
		cerr=fabs(log1p(exp(x))-a);
		avgerr+=cerr;
		if(cerr>maxerr) {
			maxerr=cerr;
			maxx=x;
		}
	}
	cout<<"=== logAdd::test() : maximum error: "<<maxerr<<", at "<<maxx<<endl;	
	cout<<"=== logAdd::test() : average error: "<<avgerr/(40.0/ssize)<<endl<<endl;
	cout<<"=== timing test starts ===="<<endl;
	clock_t ts,tl,tlu,td;
	ts=clock();
	double res1=0.0,res2=0.0;
	for(x=0.0;x>=-40.0;x-=ssize) {
		a=x+1.0;;
		res1+=a+x;
	}
	tl=clock()-ts;
	cout<<"Just a dummy to keep the compiler from optimizing this loop away:"<<res1<<endl;
	cout<<"=== loop runtime in s: "<<(double)tl/CLOCKS_PER_SEC<<endl;
	ts=clock();
	res1=0.0;
	for(x=0.0;x>=-40.0;x-=ssize) {
		a=x+1.0;
		add(a,x);
		res1+=a+x;
	}
	tlu=clock()-ts;
	cout<<"=== lookup runtime in s: "<<(double)tlu/CLOCKS_PER_SEC<<endl;
	ts=clock();
	for(x=0.0;x>=-40.0;x-=ssize) {
		a=x+1.0;
		a=fabs(a-x)>40.0?fmax(a,x):fmax(a,x)+log1p(exp(-fabs(a-x)));
		res2+=a+x;
	}
	td=clock()-ts;
	cout<<"lookup: "<<res1<<", exact:"<<res2<<endl; // to force the compiler to compile the above loops
	cout<<"=== direct runtime in s:"<<(double)td/CLOCKS_PER_SEC<<endl;
	cout<<"=== factor lookup/direct: "<<(double)(tlu-tl)/(td-tl)<<endl;
}

/** compute d(ln(gamma(x))/dx */
template<int NPPUI,int CUTOFF,int MAXIBITER>
inline double logAdd<NPPUI,CUTOFF,MAXIBITER>::getDLogGamma(double &x){
  double y,tmp,serd,sern;
  unsigned short int j;
  y=x;
  tmp=x+5.5;
  tmp=log(tmp)-5.0/tmp-1.0/x;
  serd=1.000000000190015;
  sern=0.0;
  for (j=0;j<=5;j++) {
    serd += logAdd_GammaCof[j]/++y;
    sern -= logAdd_GammaCof[j]/(y*y);
  }
  return tmp+sern/serd;
}

template<int NPPUI,int CUTOFF,int MAXIBITER>
double logAdd<NPPUI,CUTOFF,MAXIBITER>::mInterpols[2*(CUTOFF*NPPUI+1)];


template class logAdd<POINTS_PER_UNIT_INTERVAL>;
