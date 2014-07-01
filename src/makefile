# Makefile
# Built from the amazing template by Ayekat

# Compiler:
CC = gcc
NVCC = nvcc

# Application name:
APPNAME = dabrain

# Flags:
CFLAGS  = -W -Wall -Wextra -pedantic -g
CFLAGS += -Wcast-align -Wcast-qual -Wconversion -Wwrite-strings -Wfloat-equal
CFLAGS += -Wlogical-op -Wpointer-arith -Wformat=2
CFLAGS += -Winit-self -Wuninitialized
CFLAGS += -Wstrict-prototypes -Wmissing-declarations -Wmissing-prototypes
CFLAGS += -Wpadded -Wshadow
CFLAGS += -std=c99
CFLAGS += -O2
CFLAGS += -g

# Specify model
CFLAGS += -DMODEL_IZHIKEVICH

NVCCFLAGS  = -arch=sm_20
NVCCFLAGS += -g

# Libraries
LIBS = -lm -lcudart -lcublas -lcsv

# C files
CSRC = $(wildcard *.c)
# Nvidia CU files
NVCSRC = $(wildcard *.cu)
# Object files
OBJECTS = $(CSRC:%.c=build/%.o) $(NVCSRC:%.cu=build/%.o)
DEPENDS = $(OBJECTS:%.o=%.d)
BUILDDIR = build

# Default: Build application
all: $(APPNAME)

# Handy actions:
clean:
	rm -rf ${BUILDDIR}
	rm -f ${APPNAME}
run:
	./${APPNAME}

# Build dependencies:
-include ${DEPENDS}

# Compile & Link:
$(BUILDDIR)/%.o: %.c
	@if [ ! -d ${BUILDDIR} ]; then mkdir ${BUILDDIR}; fi
	@printf "compiling \033[1m%s\033[0m ...\n" $@
	$(CC) ${CFLAGS} -c $< -o $@
	$(CC) ${CFLAGS} -MM -MT $@ $< > ${BUILDDIR}/$*.d
$(BUILDDIR)/%.o: %.cu
	@if [ ! -d ${BUILDDIR} ]; then mkdir ${BUILDDIR}; fi
	@printf "compiling \033[1m%s\033[0m ...\n" $@
	$(NVCC) ${NVCCFLAGS} -c $< -o $@
	$(NVCC) ${NVCCFLAGS} -M $@ $< > ${BUILDDIR}/$*.d
$(APPNAME): $(OBJECTS)
	@printf "linking \033[1m%s\033[0m ...\n" $@
	$(NVCC) ${NVCCFLAGS} ${LIBS} ${OBJECTS} -o $@

# Phony targets:
.PHONY: all
.PHONY: clean
.PHONY: run
.PHONY: all
