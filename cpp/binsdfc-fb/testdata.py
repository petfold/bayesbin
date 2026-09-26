#!/usr/bin/python

import random

# number of trials per stimulus
NUMTRIALS=30
# number of stimuli
NUMSTIM=1

# firing probabilities per ms
BACKGROUND=0.01
TRANSIENT=0.08
SUSTAINED=0.05

# latency and durations in ms
LATENCY=80
TRANSIENT_DURATION=50
SUSTAINED_DURATION=250

random.seed()

genf=open("generator.dat","w")
for stim in range(NUMSTIM):
	for trial in range(NUMTRIALS):
		print "Stimulus%d" % stim,
		# this cell responds to stimulus 0 and not to the others
		if stim==0:
			bg=BACKGROUND
			tr=TRANSIENT
			su=SUSTAINED
		else:
			tr=su=BACKGROUND
			if random.random()>1.0/NUMSTIM:
				bg=SUSTAINED
			else:
				bg=BACKGROUND
		
		for timeindex in range(-100,500):
			rval=random.random()
			if timeindex<LATENCY:
				print >> genf,timeindex,bg
				if rval<bg:
					print timeindex,
			elif timeindex<LATENCY+TRANSIENT_DURATION:
				print >> genf,timeindex,tr
				if rval<tr:
					print timeindex,
			elif timeindex<LATENCY+TRANSIENT_DURATION+SUSTAINED_DURATION:
				print >> genf,timeindex,su
				if rval<su:
					print timeindex,
			else:
				print >> genf,timeindex,bg
				if rval<bg:
					print timeindex,
		print
	
genf.close()
		
