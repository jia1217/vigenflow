# Embed the prebuilt Linux host into run.exe. System runtime dependencies (XRT,
# Boost, etc.) remain external; libhost_<model>.so is only a build input.
WORKER_SUPPORT_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))
CXX := g++-13
TARGET_DIR ?= ../../exe_models/$(MODEL_NAME)
TARGET ?= $(TARGET_DIR)/run.exe
HOST_SRC ?= host.cpp
HOST_OBJ ?= .build/host.o
HOST_LIB_DIR ?= ../../lib
HOST_LIB_NAME ?= host_$(MODEL_NAME)
HOST_LIB ?= $(HOST_LIB_DIR)/lib$(HOST_LIB_NAME).so
EMBEDDED_HOST_OBJ := .build/embedded_host.o
EMBEDDED_LOADER_OBJ := .build/embedded_loader.o

CXXFLAGS ?= -std=c++23 -O2
LDFLAGS ?=
LDLIBS ?=

.PHONY: all clean check run
all: $(TARGET)

$(HOST_LIB):
	$(error Missing $(HOST_LIB). Copy or build it before building $(TARGET))

$(HOST_OBJ): $(HOST_SRC) $(WORKER_SUPPORT_DIR)embedded_host.h
	@mkdir -p "$(@D)"
	$(CXX) $(CXXFLAGS) -c "$<" -o "$@"

$(EMBEDDED_LOADER_OBJ): $(WORKER_SUPPORT_DIR)embedded_host.cpp $(WORKER_SUPPORT_DIR)embedded_host.h
	@mkdir -p "$(@D)"
	$(CXX) $(CXXFLAGS) -c "$<" -o "$@"

$(EMBEDDED_HOST_OBJ): $(HOST_LIB) $(WORKER_SUPPORT_DIR)embedded_host.S
	@mkdir -p "$(@D)"
	$(CXX) -DVIGENFLOW_HOST_LIBRARY='"$(abspath $(HOST_LIB))"' -c "$(WORKER_SUPPORT_DIR)embedded_host.S" -o "$@"

$(TARGET): $(HOST_OBJ) $(EMBEDDED_LOADER_OBJ) $(EMBEDDED_HOST_OBJ) $(WORKER_SUPPORT_DIR)standalone_worker.mk Makefile
	@mkdir -p "$(@D)"
	$(CXX) -o "$@" "$(HOST_OBJ)" "$(EMBEDDED_LOADER_OBJ)" "$(EMBEDDED_HOST_OBJ)" $(LDFLAGS) -ldl $(LDLIBS)

check: $(TARGET)
	"$(TARGET)" --help

run: $(TARGET)
	"$(TARGET)"

clean:
	rm -rf .build "$(TARGET)"
