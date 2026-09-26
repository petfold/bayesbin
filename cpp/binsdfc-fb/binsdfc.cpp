/***************************************************************************
 *   Copyright (C) 2007 by Dominik Endres   *
 *   dominik.endres@gmail.com   *
 *   Modified 2026-09-26 by Peter Foldiak: SDF by forward-backward        *
 *   (forwardbackward.h),                                                  *
 *   the original per-time-index path kept behind --virtual-spike (-V).    *
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

/** \mainpage

\author <a href="mailto:dominik.endres@gmail.com">Dominik M Endres</a>

<b>binsdfc</b> is a command line implementation of the algorithm described in <a href="http://books.nips.cc/nips20.html">Endres, Oram, Schindelin, Foldiak (2007): Bayesian binning beats approximate alternatives: estimating peri-stimulus time histograms </a>. It computes spike density functions (SDF) or peri-stimulus time histograms (PSTH). Given that it performs exact Bayesian averaging, the result is somewhere in the middle between the two: while the underlying model is comprised of bins like a PSTH, the averaging process gives rise to a more "continuous" prediction like a SDF. If you haven't read the paper, do it now.<br>
<b>binsdfc</b> accepts input from stdin or a supplied file, and computes the expected SDF, its variance, latencies (as described in  <a href="http://dx.doi.org/10.1016/j.jphysparis.2009.11.015">Endres, Schindelin, Foldiak, Oram (2010): Modelling spike trains and extracting response latency with Bayesian binning</a>), and various other averages. The central iteration is implemented in the template class evidenceComputer, the functions to be averaged implement the interface evifunc. The way these two interact follow the formalism developed in <a href="http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=1522639">Endres and Foldiak (2005), Bayesian bin distribution inference and mutual information</a>.<br>

<pre>
Usage: binsdf [options] filename.
Compute spike density functions. Input either from file or stdin (if filename='-').
Input file format: 1 trial per line, spike times aligned to stimulus onset, line format:
\<stimulus name\> \<spike time 1\> <spike time 2\> .... \<spike time N\>"<<endl
--help: this text
--start,-s: start time index for sdf calculations 
--end,-e: end time index 
--efire,-f: prior exponent of the Beta distribution of each bin for firing
--egap,-g: prior exponent of the Beta distribution of each bin for not firing
--max-num-bins,-m: maximal number of bin boundaries for sdf inside the time interval
--sdfvar,-v: calculate variance of spike density function.
--m-posterior,-o: calculate number-of-bin-boundaries posterior instead of sdf
--p-post-lbound,-l: lower bound of posterior probability that chosen number-of-bin-boundaries interval contains correct value
--bin-boundaries,-b \<number of bin boundaries\>: compute expectations and std.devs of bin boundary positions for a model with supplied \<number of bin boundaries\>
--bin-boundary-posterior,-p \<number of bin boundaries\>: compute posterior distributions of bin boundary positions for a model with supplied \<number of bin boundaries\>
--find-prior-exponents,-P: find most probable prior exponents
--find-signal-separation-level,-y: find best signal separation level
--find-latency,-L \<signal separation level\>: find latency based on \<signal separation level\>
--find-inhibitory-signal-separation-level,-i: find best inhibitory signal separation level"
--find-inhibitory-latency,-I \<inhibitory signal sep. level\>: find inhibitory latency based on \<inhibitory singal sep. level\>
--latency-start,-S: start time index for latency calculations
--latency-end,-E: end time index for latency calculations
--no-sdf, -n: don't compute sdf
--marginal-likelihood, -M: return list with log(P(data|M)), for M between 0 and --max-num-bins.
</pre>
<br>
<br>

<h1>Very short and incomplete tutorial </h1>
<br>
Building the sources: we use the cmake build system:
<pre>
cd binsdfc
cmake .
make
</pre>
and feel free to add more optimisation options to CMakeLists.txt .<br>
If you would like to rebuild this documentation, install doxygen and call it from the source directory.<br>
The <tt>bindsdfc/src/</tt> directory contains a python script called <b>testdata.py</b> which create some data in the required format and print them to stdout. Just pipe them into binsdfc, like
<pre>
./testdata.py | ./binsdfc -s -100 -e 500
</pre>
and you should get the SDF to stdout, in the interval from -100ms pre-stimulus (or whatever unit you use) to 500ms post-stimulus. Now let's see how certain we can be of the values we just got by adding the -v option for the standard deviation of the SDF:
<pre>
./testdata.py | ./binsdfc -s -100 -e 500 -v
</pre>
You might have noticed that the first line of the output (actually sent to stderr) says something like
<pre>
Including models with 3<=M<=3 at probability 0.292397
</pre>
This means that the most probable M (number of bin boundaries) is 3, and the posterior probability P(M|D) is about 0.29. The prior over M is assumed to be uniform, and within 0..10. If you'd like to consider a larger range of bin numbers, use the <tt>--max-num-bins</tt> option.<br>
You might feel that a probabiltiy of 0.29 that the model (M==3) is the correct one (within the model class, as usual in Bayesian analyses) is not high enough. So let's raise the lower bound on the M posterior probability by using <tt>-l</tt>:
<pre>
./testdata.py | ./binsdfc -s -100 -e 500 -v -l 0.9
</pre>
Now the first line of the output will look something like
<pre>
Including models with 3<=M<=5 at probability 0.932465
</pre>
i.e. to be more than 90\% sure that you include the correct model, you have to include models with 3\<=5\<=M in all subsequent calculations. If you compare the SDF computed with this setting with the previous one, you should notice slight differences.<br>
You can also tweak the hyperparameters of the Beta prior on the firing rate probabilities, using the options <tt>-f</tt> for the exponent of the firing probability and <tt>-g</tt> for the exponent of the not-firing probability. The default setting is <tt>-f 1 -g 32</tt>, which reproduces the average firing rate of an STSa neuron under experimental condition. Alternatively, you can try to optimise these hyperparameters with the <tt>-P</tt> options (caution: this feature is still experimental, and the optimisation might get stuck. If you end up with a hyperparameter less than 0.001 or more than 300.0, something's gone wrong. Please let me know.):
<pre>
./testdata.py | ./binsdfc -s -100 -e 500 -v -l 0.9 -P
</pre>
which should give you an output like
<pre>
Best values for efire,egap = 0.170484,2.07251, firing rate 0.0760071+-0.147159
</pre>
at the beginning. This minimisation might take a while, but if nothing happens for a minute, it might have gotten stuck.<br>
Now let's find the latency. First, we need to determine the signal separation level, i.e. that firing probability below which we'll classify the cell as "not responding":
<pre>
./testdata.py | ./binsdfc -s -100 -e 500 -l 0.9 -P -y
</pre>
(of course you could have supplied the previously found values for the beta hyperparameters with <tt>-f, -g</tt> instead of running the optimisation again). You should get an output like:
<pre>
Best signal separation level: 0.0441759 at probability 0.999398
</pre>
This means that at the signal separation level of 0.0441759, the probability for the existence of a latency (i.e. of a signal) is 0.999398 -- we can be reasonably confident that this cell is responding. Let's see where exactly by computing the latency posterior distribution with the <tt>-L</tt> option:
<pre>
./testdata.py | ./binsdfc -s -100 -e 500 -l 0.9 -P  -L 0.0441759
</pre>
You will hopefully see that the latency posterior has a sharp peak at 80ms, which is exatly where the test data are simulating a response.




*/
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif


#include <cstdlib>

using namespace std;

#include "unistd.h"
#include "getopt.h"
#include "spikedensityfunction.h"
#include "forwardbackward.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include "simplexminimiser.h"
#include "string.h"
#include "omp.h"



using namespace std;

/** uses simplex minimiser to find the best prior hyperparameters for the beta prior of the firing rate
within the bins. TODO: can get stuck at constraints! */
class findPriorExp : public simplexMinimiser {
	public:
		spikeDensityFunction *mpSDF;
		double mShapeF,mScaleF;
		double mShapeG,mScaleG;

		virtual double evaluate(tPoint::iterator begin,tPoint::iterator end) {
			double egap=*begin;
			double efire=*(begin+1),f,g;
			if(egap<1e-3) egap=1e-3;
			if(efire<1e-3) efire=1e-3;
			mpSDF->setIntervalPriors(egap,efire);
			// gamma prior in f=efire/(egap+efire) and g=.5*(egap^2+efire^2)
			f=efire/(egap+efire)/mScaleF;
			g=0.5*(efire*efire+egap*egap)/mScaleG;
			return -mpSDF->getTotalEvidence()
				//-log((egap*egap+efire*efire)/pow(egap+efire,2.0))
				-(mShapeF-1.0)*log(f)+f-(mShapeG-1.0)*log(g)+g;
		}
	virtual bool checkConstraints(tPoint::iterator begin,tPoint::iterator end) {
		if(*begin<1.0 || *begin>300.0) return false;
		if(*(begin+1)<0.001 || *(begin+1)>300.0) return false;
		return true;
	};


};


/** explode string by whitespaces (spaces and tabs) */
void explode(string thestr,vector<string> &parts) {
	int bpos;
	string subs;
	while(thestr.length()>0) {
		bpos=thestr.find_first_of(" \t");
		if(bpos>=0) {
			subs=thestr.substr(0,bpos);
			thestr=thestr.substr(bpos+1);
		} else {
			subs=thestr;
			thestr="";
		}
		if(subs.length()>0) parts.push_back(subs);
	}
}

struct option lopts[]={
	{"help",no_argument,0,'h'},
	{"start",required_argument,0,'s'},
	{"end",required_argument,0,'e'},
	{"efire",required_argument,0,'f'},
	{"egap",required_argument,0,'g'},
	{"p-post-lbound,",required_argument,0,'l'},
	{"max-num-bins",required_argument,0,'m'},
	{"sdfvar",no_argument,0,'v'},
	{"m-posterior",no_argument,0,'o'},
	{"bin-boundaries",required_argument,0,'b'},
	{"bin-boundary-posterior",required_argument,0,'p'},
	{"find-prior-exponents",no_argument,0,'P'},
	{"find-signal-separation-level",no_argument,0,'y'},
	{"find-latency",required_argument,0,'L'},
	{"latency-start",required_argument,0,'S'},
	{"latency-end",required_argument,0,'E'},
	{"no-sdf",no_argument,0,'n'},
	{"virtual-spike",no_argument,0,'V'},
	{"learn-from",required_argument,0,'r'},
	{"marginal-likelihood",no_argument,0,'M'},
	{"find-inhibitory-signal-separation-level",no_argument,0,'i'},
	{"find-inhibiotry-latency",required_argument,0,'I'},
	{"start-exclude",required_argument,0,'x'},
	{"end-exclude",required_argument,0,'X'},
	{"fano-factor",required_argument,0,'N'},
	{"output-file",required_argument,0,'O'}
	

};

int main(int argc, char *argv[])
{
	
	int start=0,end=600;
	double efire=1.0,egap=32.0;
	double ppostlb=0.001;
	double pfireub=1.0;
	int maxbins=10;
	int doBinBoundaries=-1;
	int doBinBoundaryPosterior=-1;
	bool doSdfVar=false;
	bool doMPost=false;
	bool doMargLike=false;
	bool doSDF=true;
	bool virtualSpike=false;
	bool findSeplev=false;
	bool findPriorExponents=false;
	bool findLatency=false;
	bool findInhibitorySeplev=false;
	bool findInhibitoryLatency=false;
	double inhibitory_noiselevel=-1.0;
	int latStart=30,latEnd=150;
	int learnnum=INT_MAX;
	double noiselevel=-1.0;
	int nopt,*iptr=0;
	string fname;
	string ofname="";
	int exStart=INT_MAX;
	int exEnd=-INT_MAX;
	int fanoFactorLength=-1;

	
	while((nopt=getopt_long(argc,argv,"s:e:f:g:l:m:vohb:p:PyL:S:E:nr:MiI:x:X:N:O:V",lopts,iptr))>=0) {
		switch(nopt) {
			case 'x': 	exStart=atol(optarg);
			break;
			case 'X': 	exEnd=atol(optarg);
			break;
			case 's': 	start=atol(optarg);
			break;
			case 'e': 	end=atol(optarg);
			break;
			case 'S': 	latStart=atol(optarg);
			break;
			case 'E': 	latEnd=atol(optarg);
			break;
			case 'r': 	learnnum=atol(optarg);
			break;
			case 'N': 	fanoFactorLength=atol(optarg);
			break;
			case 'f': 	efire=atof(optarg);
			break;
			case 'g': 	egap=atof(optarg);
			break;
			case 'l': 	ppostlb=atof(optarg);
			break;
			case 'm': 	maxbins=atol(optarg);
			break;
			case 'b': 	doBinBoundaries=atol(optarg);
			break;
			case 'p': 	doBinBoundaryPosterior=atol(optarg);
			break;
			case 'v': 	doSdfVar=true;
			break;
			case 'o': 	doMPost=true;
			break;
			case 'M': 	doMargLike=true;
			break;
			case 'P': 	findPriorExponents=true;
			break;
			case 'V': 	virtualSpike=true;
					fb::enabled=false;  // everything as in the original
					break;
			case 'n': 	doSDF=false;
			break;
			case 'y': 	findSeplev=true;
			break;
			case 'L': 	findLatency=true;
					if(optarg!=NULL) noiselevel=atof(optarg);
			break;
			case 'i': 	findInhibitorySeplev=true;
			break;
			case 'I': 	findInhibitoryLatency=true;
					if(optarg!=NULL) inhibitory_noiselevel=atof(optarg);
			break;
			case 'O':	ofname=optarg;
			break;
			
			default:
				cout<<"Usage: binsdf [options] filename"<<endl;
				cout<<"Compute spike density functions. Input either from file or stdin (if filename='-')."<<endl;
				cout<<"Input file format: 1 trial per line, line format:"<<endl;
				cout<<"<stimulus name> <spike time 1> <spike time 2> .... <spike time N>"<<endl<<endl;
				cout<<"--help: this text "<<endl;
				cout<<"--start,-s: start time index for sdf calculations "<<endl;		
				cout<<"--end,-e: end time index "<<endl;
				cout<<"--efire,-f: prior exponent of the Beta distribution of each bin for firing"<<endl;
				cout<<"--egap,-g: prior exponent of the Beta distribution of each bin for not firing"<<endl;
				cout<<"--max-num-bins,-m: maximal number of bin boundaries for sdf inside the time interval"<<endl;
				cout<<"--sdfvar,-v: calculate variance of spike density function."<<endl;
				cout<<"--m-posterior,-o: calculate number-of-bin-boundaries posterior instead of sdf"<<endl;
				cout<<"--p-post-lbound,-l: lower bound of posterior probability that chosen number-of-bin-boundaries interval contains correct value"<<endl;
				cout<<"--bin-boundaries,-b <number of bin boundaries>: compute expectations and std.devs of bin boundary positions for a model with supplied <number of bin boundaries>"<<endl;
				cout<<"--bin-boundary-posterior,-p <number of bin boundaries>: compute posterior distributions of bin boundary positions for a model with supplied <number of bin boundaries>"<<endl;
				cout<<"--find-prior-exponents,-P: find most probable prior exponents"<<endl;
				cout<<"--find-signal-separation-level,-y: find best signal separation level"<<endl;
				cout<<"--find-latency,-L <signal separation level>: find latency based on <signal separation level>"<<endl;
				cout<<"--find-inhibitory-signal-separation-level,-i: find best inhibitory signal separation level"<<endl;
				cout<<"--find-inhibitory-latency,-I <inhibitory signal sep. level>: find inhibitory latency based on <inhibitory singal sep. level>"<<endl;
				cout<<"--latency-start,-S: start time index for latency calculations "<<endl;		
				cout<<"--latency-end,-E: end time index for latency calculations "<<endl;
				cout<<"--no-sdf, -n: don't compute sdf "<<endl;
				cout<<"--virtual-spike, -V: compute everything as the original binsdfc 0.1 does (sdf: one evidence computation per time index; slow, for comparison)"<<endl;
				//cout<<"--learn-from, -r <num>: learn from the first <num> spiketrains in the data, compute log predictive prob. for the rest "<<endl;
				cout<<"--marginal-likelihood, -M: return list with log(P(data|M)), for M between 0 and --max-num-bins."<<endl;
				cout<<"--output-file, -O <filename> write to file instead of stdout."<<endl;
				return EXIT_SUCCESS;
		}
	}

	if(maxbins>=end-start)
		maxbins=end-start-1;
	if(doBinBoundaries>maxbins)
		doBinBoundaries=maxbins;
	if(doBinBoundaryPosterior>maxbins)
		doBinBoundaryPosterior=maxbins;
	if(fanoFactorLength>0) doSdfVar=true;
	if(doSdfVar) doSDF=true;
	



	istream *instr;
	ifstream istr;
	ostream *outstr;
	ofstream ostr;
	bool fromStdin=false;
	bool toStdout=false;
	if(optind>=argc || strcmp(argv[optind],"-")==0) {
		instr=&cin;
		fromStdin=true;
	} else {
		istr.open(argv[optind]);
		instr=&istr;
	}
	if(ofname!="") {
		ostr.open(ofname.c_str());
		outstr=&ostr;
	} else {
		outstr=&cout;
		toStdout=true;
	}

	spikeDensityFunction sdf;
	sdf.setMaxM(maxbins);
	sdf.setTimeInterval(start,end);
	if(latStart<=exEnd) {
		cerr<<"ERROR: latency start "<<latStart<<" must be AFTER end of excluded time interval "<<exEnd<<endl;
		abort();
	}
	sdf.setExcludedTimeInterval(exStart,exEnd);
	sdf.setIntervalPriors(egap,efire);
	
	
	vector<vector<int> > predictdata;
	int numtrains=0;
	while(!instr->eof()) {
		string cline;
		getline(*instr,cline);
		vector<string> spikes;
		explode(cline,spikes);
		if(spikes.size()>1) {
			vector<int> spt;
			for(vector<string>::iterator curs=spikes.begin()+1;curs!=spikes.end();curs++)
				spt.push_back(atol(curs->c_str()));
			if(numtrains<learnnum)	sdf.addData(spt.begin(),spt.end());
			else predictdata.push_back(spt);
			numtrains++;
		}	
	}

	if(findPriorExponents) {
		findPriorExp fpe;
		fpe.mpSDF=&sdf;
		fpe.mShapeF=2.0;
		fpe.mScaleF=0.030;
		fpe.mShapeG=2.0;
		fpe.mScaleG=1.0;
		tPoint startpoint;
		startpoint.push_back(egap);
		startpoint.push_back(efire);
		fpe.init(startpoint.begin(),startpoint.end());
		startpoint=fpe.findMinimum(1e-3);
		egap=startpoint[0];
		efire=startpoint[1];
		double fra,sddev;
		fra=efire/(efire+egap);
		sddev=sqrt(egap/(egap+efire+1)/(egap+efire)*fra);
		*outstr<<"# Best values for efire,egap = "<<efire<<","<<egap<<", firing prob. "<<fra<<"+-"<<sddev<<endl;
		sdf.setIntervalPriors(egap,efire);
	}

	if(doMPost) {
		*outstr<<"# number-of-binboundaries posterior-prob. "<<endl;
		for(int m=0;m<=maxbins;m++)
			*outstr<<m<<" "<<sdf.getMProb(m)<<endl;
	}

	if(doMargLike) {
		*outstr<<"# number-of-binboundaries evidence "<<endl;
		*outstr<<"# marginal likelihood of dataset :"<<sdf.getTotalEvidence()<<endl;
		for(int m=0;m<=maxbins;m++)
			*outstr<<m<<" "<<sdf.getEvidence(m)<<endl;
	}

	if(doBinBoundaries>0) {
		sdf.setMaxM(doBinBoundaries);
		sdf.setMInterval(doBinBoundaries,doBinBoundaries);
		double bpos,bstddev;
		for(int m=0;m<doBinBoundaries;m++) {
			bpos=sdf.getBBExpectation(m);
			bstddev=sdf.getBBStdDev(m,bpos);
			*outstr<<m<<" "<<bpos<<" "<<bstddev<<endl;
		}
		return EXIT_SUCCESS;
	}

	if(doBinBoundaryPosterior>0) {
		sdf.setMaxM(doBinBoundaryPosterior);
		sdf.setMInterval(doBinBoundaryPosterior,doBinBoundaryPosterior);
		double bpos,bstddev;
		for(int t=start;t<end;t++) {
			*outstr<<t;
			for(int m=0;m<doBinBoundaryPosterior;m++)
				*outstr<<" "<<sdf.getBBProb(m,t);
			*outstr<<endl;
			
		}
		return EXIT_SUCCESS;
	}
	
	int mmin,mmax;
	double ppost=sdf.findMInterval(ppostlb,mmin,mmax);
	*outstr<<"# Including models with "<<mmin<<"<=M<="<<mmax<<" at probability "<<ppost<<endl;
	sdf.setMaxM(mmax);
	if(findLatency || findSeplev || findInhibitoryLatency) {
		if(mmin<1) mmin=1;
		if(mmax<mmin) mmax=mmin;
	}
	sdf.setMInterval(mmin,mmax);

	if(findSeplev || (findLatency && (noiselevel<=0.0))) {
		double prob;
		sdf.findBestSeparator(0.0,0.1,noiselevel,prob,latStart,latEnd);
		*outstr<<"# Best signal separation level: "<<noiselevel<<" at probability "<<prob<<endl;
		if(!findLatency) return EXIT_SUCCESS;
	}

	if(findInhibitorySeplev || (findInhibitoryLatency && (inhibitory_noiselevel<=0.0))) {
		double prob;
		list<pair<double,double> > ts=sdf.findBestInhibitorySeparator(0.0,0.01,inhibitory_noiselevel,prob,latStart,latEnd);
		list<pair<double,double> >::iterator pts;
		for(pts=ts.begin();pts!=ts.end();pts++)
			*outstr<<"# iseplev="<<(*pts).first<<" at p="<<(*pts).second<<endl;
		*outstr<<"# Best inhibitory signal separation level: "<<inhibitory_noiselevel<<" at probability "<<prob<<endl;
		return EXIT_SUCCESS;
	}

	if(findLatency) {
		double maxp=0.0,curp,totp=0.0;
		double ex=0.0,ex2=0.0;
		int latpos=0;
		for(int l=latStart;l<latEnd;l++) {
			curp=sdf.getLatencyProb(l,noiselevel);
			*outstr<<l<<" "<<curp<<endl;
			ex+=l*curp;
			ex2+=l*l*curp;
			totp+=curp;
			if(curp>maxp) {
				maxp=curp;
				latpos=l;
			}
		}
		*outstr<<"# Probability that latency exists "<<totp<<endl;
		*outstr<<"# Latency mode at "<<latpos<<" with probability "<<maxp<<endl;
		ex/=totp;
		ex2/=totp;
		ex2-=ex*ex;
		ex2=ex2>=0.0?sqrt(ex2):0.0;
		*outstr<<"# Latency expectation at "<<ex<<"+-"<<ex2<<endl;
		*outstr<<"# CSV: "<<totp<<","<<latpos<<","<<maxp<<","<<ex<<","<<ex2<<endl;
		return EXIT_SUCCESS;
	}

	if(findInhibitoryLatency) {
		double maxp=0.0,curp;
		int latpos=0;
		for(int l=latStart;l<latEnd;l++) {
			curp=sdf.getInhibitoryLatencyProb(l,inhibitory_noiselevel);
			*outstr<<l<<" "<<curp<<endl;
			if(curp>maxp) {
				maxp=curp;
				latpos=l;
			}
		}
		*outstr<<"# Inhibitory latency mode at "<<latpos<<" with probability "<<maxp<<endl;
		return EXIT_SUCCESS;
	}


	if(doSDF) {
		*outstr<<"# time sdf sdfstd sdf+sdfstd sdf-sdfstd fano-factor(wndlen="<<fanoFactorLength<<")"<<endl;
		double stime=omp_get_wtime();
		vector<double> sdfval(end-start,0.0);
		vector<double> sdfstdval(end-start,0.0);
		// forward-backward: every time index in one pass (threads: OMP_NUM_THREADS, else all);
		// the original per-index path with -V, on the original's 4 threads
		if(virtualSpike || !sdf.getSDFForwardBackward(sdfval,sdfstdval,doSdfVar)) {
		omp_set_num_threads(4);
		#pragma omp parallel 
		{
			spikeDensityFunction mysdf=sdf;
			#pragma omp for schedule(guided)
			for(int tind=start;tind<end;tind++) {
				sdfval[tind-start]=mysdf.getSDF(tind);
				if(doSdfVar) {
					sdfstdval[tind-start]=mysdf.getSDFStdDev(tind,sdfval[tind-start]);
				}
			}
		}
		}
		stime=omp_get_wtime()-stime;
		for(int tind=0;tind<end-start;tind++) {
			*outstr<<tind+start<<" "<<sdfval[tind];
			if(doSdfVar) {
				*outstr<<" "<<sdfstdval[tind]<<" "<<sdfval[tind]+sdfstdval[tind]<<" "<<sdfval[tind]-sdfstdval[tind];
				if(fanoFactorLength>0) {
						*outstr<<" "<<1.0-sdfval[tind];
				}
			}
			*outstr<<endl;

		}
		*outstr<<"# this took "<<stime<<" seconds"<<endl;


	}

	if(predictdata.size()>0) {
		double devi=sdf.getIntervalEvidenceSum();
		for(int sn=0;sn<predictdata.size();sn++)
			sdf.addData(predictdata[sn].begin(),predictdata[sn].end());
		devi=sdf.getIntervalEvidenceSum()-devi;
		*outstr<<"# Predictive log probability in testset per datapoint :"<<devi/(end-start)/predictdata.size()<<endl;
	}
	
	
	
	if(!fromStdin) {
		istr.close();
	}

	if(!toStdout) {
		ostr.close();
	}


	return EXIT_SUCCESS;
}
