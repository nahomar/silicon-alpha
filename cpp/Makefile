# Build the Heston MC engine. Auto-detects Homebrew libomp for OpenMP;
# falls back to a serial build (still correct, just single-threaded).
CXX      ?= clang++
CXXSTD   ?= -std=c++17
OPT      ?= -O3 -march=native -funroll-loops
WARN     ?= -Wall -Wextra

# Detect Homebrew libomp (Apple clang ships no OpenMP runtime by default).
OMP_PREFIX := $(shell brew --prefix libomp 2>/dev/null)
ifneq ($(OMP_PREFIX),)
  OMP_FLAGS := -Xpreprocessor -fopenmp -I$(OMP_PREFIX)/include
  OMP_LIBS  := -L$(OMP_PREFIX)/lib -lomp
else
  OMP_FLAGS :=
  OMP_LIBS  :=
endif

heston_mc: heston_mc.cpp
	$(CXX) $(CXXSTD) $(OPT) $(WARN) $(OMP_FLAGS) -o $@ $< $(OMP_LIBS)

.PHONY: test bench clean
test: heston_mc
	./heston_mc --selftest

bench: heston_mc
	./heston_mc --benchmark

clean:
	rm -f heston_mc
