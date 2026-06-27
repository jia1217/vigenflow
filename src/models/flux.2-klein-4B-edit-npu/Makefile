CXX := g++-13
MODEL_NAME ?= flux.2-klein-4B-edit
TARGET_DIR ?= ../../exe_models/$(MODEL_NAME)
TARGET ?= $(TARGET_DIR)/run.exe
HOST_SRC ?= host.cpp
HOST_OBJ ?= .build/host.o
HOST_LIB_DIR ?= ../../lib
HOST_LIB_NAME ?= host_flux_edit
HOST_LIB ?= $(HOST_LIB_DIR)/lib$(HOST_LIB_NAME).so

CXXFLAGS ?= -std=c++23 -O2
LDFLAGS ?=
LDLIBS ?=

.PHONY: all clean check run

all: $(TARGET)

$(HOST_LIB):
	$(error Missing $(HOST_LIB). Copy or build lib$(HOST_LIB_NAME).so before building $(TARGET))

$(HOST_OBJ): $(HOST_SRC)
	@mkdir -p $(@D)
	$(CXX) $(CXXFLAGS) -c "$<" -o "$@"

$(TARGET): $(HOST_OBJ) $(HOST_LIB)
	@mkdir -p "$(@D)"
	$(CXX) -o "$@" "$(HOST_OBJ)" -L"$(HOST_LIB_DIR)" -l$(HOST_LIB_NAME) -Wl,-rpath,'$$ORIGIN/../../lib' $(LDFLAGS) $(LDLIBS)

check: $(TARGET)
	"$(TARGET)" --help

run: $(TARGET)
	"$(TARGET)"

clean:
	rm -rf .build "$(TARGET)"
