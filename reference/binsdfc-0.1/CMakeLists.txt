
PROJECT(binsdfc)

#if you don't want the full compiler output, remove the following line
SET(CMAKE_VERBOSE_MAKEFILE ON)

#add definitions, compiler switches, etc.
ADD_DEFINITIONS(-O2 -march=native -ffast-math -fopenmp)


#list all source files here
ADD_EXECUTABLE(binsdfc binsdfc.cpp evidencecomputer.cpp  logadd.cpp  simplexminimiser.cpp  spikecounter.cpp  spikedensityfunction.cpp)


#need to link to some other libraries ? just add them here
TARGET_LINK_LIBRARIES(binsdfc gomp)
 
