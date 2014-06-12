# Makefile
# Built from the amazing template by Ayekat

# Compiler:
CC ?= gcc

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

# File names:
SOURCES = $(wildcard *.c)
OBJECTS = $(SOURCES:%.c=build/%.o)
DEPENDS = $(OBJECTS:%.o=%.d)
BUILDDIR = build

# Default: Build application
all: $(APPNAME)

# Handy actions:
clean:
	rm -rf ${BUILDDIR}
mrproper: clean
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
$(APPNAME): $(OBJECTS)
	@printf "linking \033[1m%s\033[0m ...\n" $@
	$(CC) ${OBJECTS} -o $@

# Phony targets:
.PHONY: all
.PHONY: clean
.PHONY: mrproper
.PHONY: run
.PHONY: all
