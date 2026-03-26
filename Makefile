#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
# Modified by Alfred

include ../makefiles/common.mk

DEVICE ?= npu2
TRACE_SIZE ?= 0
HOME_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))/../
HOST_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))

# do you want chesscc? compiles slower
ENABLE_CHESSCC ?= 1

IRON_ARGS := ${DEVICE}
IRON_ARGS += ${TRACE_SIZE}
##############################################add addition link package###########################################
CXXFLAGS += -I/usr/local/include -I/home/kelsey/NPU_projects/NPU_new/new_test/AUser_host/external_libs/tokenizers-cpp/include
# Add the path to where the compiled library (.a or .so file) lives
LDFLAGS += -L/home/kelsey/NPU_projects/NPU_new/new_test/AUser_host/external_libs/tokenizers-cpp/build

# Tell the linker the name of the library to link
LDLIBS += -ltokenizers_cpp

LDLIBS += -ltokenizers_cpp -ltokenizers_c
##################################################################################################################
# Kernel makefile
include ../makefiles/kernel.mk

# Bitstream makefile
include ../makefiles/bitstream.mk
include ../makefiles/mlir_bitstream.mk

# Host makefile
include ../makefiles/host.mk

.PHONY: run all kernel link bitstream host clean instructions
all: ${XCLBIN_TARGETS} ${INSTS_TARGETS} ${HOST_C_TARGET}

clean:
	-@rm -rf build 
	-@rm -rf log
	-@rm -rf *.exe
	-@rm -rf trace*
	-@rm -rf routes

test:
	echo "test"
	echo ${AIEOPT_DIR}


kernel: ${KERNEL_OBJS}


instructions: ${INSTS_TARGETS}



bitstream: ${XCLBIN_TARGETS}


host: ${HOST_C_TARGET}


clean_host:
	-@rm -rf build/host


run: ${HOST_C_TARGET} ${XCLBIN_TARGETS} ${INSTS_TARGETS}
	./${HOST_C_TARGET} | tee out.log

route: ${IRON_BOTH_MLIR_TARGET}
	mkdir -p routes
	aie-opt --aie-create-pathfinder-flows --aie-find-flows ${<} | aie-translate --aie-flows-to-json > ./routes/route.json
	python ../visualize.py -j ./routes/route.json
