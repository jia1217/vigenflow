CXX := g++-13
MODEL_NAME ?= flux.2-klein-4B
TARGET_DIR ?= ../../exe_models/$(MODEL_NAME)
TARGET ?= $(TARGET_DIR)/run.exe
HOST_SRC ?= host.cpp
HOST_OBJ ?= .build/host.o
HOST_LIB_DIR ?= ../../lib
HOST_LIB ?= $(HOST_LIB_DIR)/libhost_flux.so

CXXFLAGS ?= -std=c++23 -O2
LDFLAGS ?=
LDLIBS ?=

.PHONY: all clean check run

all: $(TARGET)

$(HOST_LIB):
	$(error Missing $(HOST_LIB). Copy or build libhost_flux.so before building $(TARGET))

$(HOST_OBJ): $(HOST_SRC)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c "$<" -o "$@"

$(TARGET): $(HOST_OBJ) $(HOST_LIB)
	@mkdir -p "$(@D)"
	$(CXX) -o "$@" "$(HOST_OBJ)" -L"$(HOST_LIB_DIR)" -lhost_flux -Wl,-rpath,'$$ORIGIN/../../lib' $(LDFLAGS) $(LDLIBS)

check: $(TARGET)
	"$(TARGET)" --help

run: $(TARGET)
	"$(TARGET)"

clean:
	rm -rf .build "$(TARGET)"
