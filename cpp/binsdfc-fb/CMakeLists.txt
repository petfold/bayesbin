# Modified 2026-09-26: forwardbackward.cpp and vmath.cpp added (SDF by forward-backward)

PROJECT(binsdfc)

#if you don't want the full compiler output, remove the following line
SET(CMAKE_VERBOSE_MAKEFILE ON)

#add definitions, compiler switches, etc.
# -ffast-math removed from the global flags (2026-09-26): the forward-backward code relies on
# -inf arithmetic, which -ffinite-math-only lets the compiler assume away. -fno-math-errno is safe.
ADD_DEFINITIONS(-O2 -march=native -fno-math-errno -fopenmp)


#list all source files here
ADD_EXECUTABLE(binsdfc binsdfc.cpp evidencecomputer.cpp  logadd.cpp  simplexminimiser.cpp  spikecounter.cpp  spikedensityfunction.cpp forwardbackward.cpp vmath.cpp)
# vmath.cpp only: glibc's vector exp is declared under -ffast-math (see the file)
SET_SOURCE_FILES_PROPERTIES(vmath.cpp PROPERTIES COMPILE_FLAGS -ffast-math)


#need to link to some other libraries ? just add them here
TARGET_LINK_LIBRARIES(binsdfc gomp)
# linking without -ffast-math keeps flush-to-zero off
 
