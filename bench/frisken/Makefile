# Build Frisken's reference SurfaceNets on a Unix toolchain.
#
# The upstream repo ships SurfaceNets.sln, which builds the Qt/OpenGL sample
# application as well as the library. We only want the library, and the library
# is portable ISO C++ -- so the solution file is not ported, it is bypassed.
#
#   make stock            build against the clone as it comes
#   make patched          build with the two efficiency patches applied
#   make FRISKEN=/path/to/surfaceNets-multi-material stock
#
# The clone is never modified. `patched` copies the sources into build/patched
# and applies patches/*.patch to the copy.

FRISKEN ?= ../../../surfaceNets-multi-material
SNLIB   := $(FRISKEN)/Source/SNLib

# -O3 -DNDEBUG matters. The shipped .vcxproj configuration inspected here has
# Optimization=Disabled, and timings from an unoptimised build would say
# nothing about the algorithm.
CXXFLAGS := -std=c++17 -O3 -DNDEBUG -Wall -Wno-unused-variable

SOURCES := MMCellFlag.cpp MMCellMap.cpp MMGeometryGL.cpp MMGeometryOBJ.cpp \
           MMSurfaceNet.cpp
PATCHES := $(sort $(wildcard patches/*.patch))

.PHONY: all stock patched clean check
all: stock

check:
	@test -d "$(SNLIB)" || { \
	  echo "Frisken sources not found at $(SNLIB)"; \
	  echo "Clone https://github.com/…/surfaceNets-multi-material and pass"; \
	  echo "  make FRISKEN=/path/to/surfaceNets-multi-material"; \
	  exit 1; }

stock: check
	@mkdir -p build/stock
	$(CXX) $(CXXFLAGS) -I$(SNLIB) -o build/stock/sn_bench \
	    sn_bench.cpp $(addprefix $(SNLIB)/,$(SOURCES))
	@echo "built build/stock/sn_bench"

# The copy is refreshed from the clone every time so a stale patched tree can
# never be mistaken for a current one.
patched: check
	@rm -rf build/patched && mkdir -p build/patched/src
	@cp $(SNLIB)/*.h $(SNLIB)/*.cpp build/patched/src/
	@for p in $(PATCHES); do \
	    echo "applying $$p"; \
	    patch -s -d build/patched/src -p1 < $$p || exit 1; \
	done
	$(CXX) $(CXXFLAGS) -Ibuild/patched/src -o build/patched/sn_bench \
	    sn_bench.cpp $(addprefix build/patched/src/,$(SOURCES))
	@echo "built build/patched/sn_bench"

clean:
	rm -rf build
