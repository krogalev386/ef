### Allows better regexp support.
SHELL:=/bin/bash -O extglob


##### Compilers
#CC=clang++
CC=mpic++
HDF5FLAGS=-isystem${HOME}/hdf5/usr/include/openmpi-x86_64 -D_LARGEFILE_SOURCE -D_LARGEFILE64_SOURCE -D_BSD_SOURCE
WARNINGS=-Wall
CUSPFLAGS=-I/zfs/hybrilit.jinr.ru/user/k/krogalev/ -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP
CFLAGS = -O2 ${HDF5FLAGS} -std=c++11 ${WARNINGS} -fopenmp ${CUSPFLAGS}
LDFLAGS = 

### Libraries
COMMONLIBS=-lm -lgomp
BOOSTLIBS=-lboost_program_options
HDF5LIBS=-L${HOME}/hdf5/usr/lib64/openmpi/lib -lhdf5_hl -lhdf5 -Wl,-z,relro -lpthread -lz -ldl -lm -Wl,-rpath -Wl,${HOME}/hdf5/usr/lib64/openmpi/lib
LIBS=${COMMONLIBS} ${BOOSTLIBS} ${HDF5LIBS} 

### Sources and executable
CPPSOURCES=$(wildcard *.cpp)
CPPHEADERS=$(wildcard *.h)
OBJECTS=$(CPPSOURCES:%.cpp=%.o)
EXECUTABLE=ef.out
MAKE=make
TINYEXPR=./lib/tinyexpr
TINYEXPR_OBJ=./lib/tinyexpr/tinyexpr.o
SUBDIRS=doc

$(EXECUTABLE): $(OBJECTS) $(TINYEXPR)
	$(CC) $(LDFLAGS) $(OBJECTS) $(TINYEXPR_OBJ) -o $@ $(LIBS)

$(OBJECTS):%.o:%.cpp $(CPPHEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: allsubdirs $(SUBDIRS) $(TINYEXPR) clean cleansubdirs cleanall

allsubdirs: $(SUBDIRS)

$(TINYEXPR):
	$(MAKE) -C $@

$(SUBDIRS):
	$(MAKE) -C $@

all: $(EXECUTABLE) doc

clean: cleansublibs
	rm -f *.o *.out *.mod *.zip

cleansublibs:
	for X in $(TINYEXPR); do $(MAKE) clean -C $$X; done 

cleansubdirs:
	for X in $(SUBDIRS); do $(MAKE) clean -C $$X; done 

cleanall: clean cleansubdirs

