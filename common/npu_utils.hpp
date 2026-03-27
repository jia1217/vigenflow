#ifndef __NPU_UTILS_HPP__
#define __NPU_UTILS_HPP__

#include <cstdlib>
#include <iostream>
#include <string>
#include <fstream>
#include <boost/program_options.hpp>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdfloat>
#include <stdio.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/types.h>
#include <drm/drm.h>
#include <stdfloat>
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"
#include "amdxdna_accel.h"
#include "buffer.hpp"
#include "debug_utils.hpp"
#include "experimental/xrt_kernel.h"
#include "experimental/xrt_ext.h"
#include "experimental/xrt_module.h"
#include "experimental/xrt_elf.h"
#include "xrt/xrt_graph.h"

#include "npu_instr_utils.hpp"

///@brief accel_user_desc
///@param xclbin_name name of the xclbin file
///@param instr_seq instruction sequence, an object of npu_sequence
///@see npu_sequence
typedef struct {
    std::string xclbin_name;
    std::string app_name;
} npu_app_desc;

///@brief accel_xclbin_desc
///@param xclbin xclbin object
///@param kernel kernel object
///@param context hardware context
///@see xrt::xclbin, xrt::kernel, xrt::hw_context
typedef struct {
    xrt::xclbin xclbin;
    xrt::kernel kernel;
    xrt::hw_context context;
} accel_xclbin_desc;

///@brief accel_kernel_desc
///@param app_name name of the application
///@param kernel_desc kernel descriptor
///@param instr_seq instruction sequence
typedef struct {
    std::string app_name;
    std::unique_ptr<accel_xclbin_desc> xclbin_desc;
    npu_sequence instr_seq;
} accel_kernel_desc;

class npu_app {
public:
    int app_id;
    npu_sequence* instr_seq;
private:
    xrt::kernel* kernel;
    xrt::device* device;
public:
    npu_app() {
        this->instr_seq = nullptr;
        this->kernel = nullptr;
        this->device = nullptr;
        this->app_id = -1;
    }
    npu_app(int app_id, npu_sequence* instr_seq, xrt::kernel* kernel, xrt::device* device):
        app_id(app_id), instr_seq(instr_seq), kernel(kernel), device(device){}

    template<typename... BoArgs>
    void operator()(BoArgs&&... args){
        auto run = this->kernel->operator()(3, this->instr_seq->bo(), this->instr_seq->size(), args.bo()...);
         ert_cmd_state r = run.wait();

        if (r != ERT_CMD_STATE_COMPLETED) {
            std::cout << "Kernel did not complete. Returned status: " << r << "\n";
            }
    }

    template<typename... BoArgs>
    xrt::run create_run(BoArgs&&... args){
        xrt::run run = xrt::run(*this->kernel);
        run.set_arg(0, 3);
        run.set_arg(1, this->instr_seq->bo());
        run.set_arg(2, this->instr_seq->size());
        bytes* bo_args[] = {&args...};
        for (int i = 0; i < sizeof...(args); i++){
            run.set_arg(3 + i, bo_args[i]->bo());
        }
        return run;
    }

    template<typename T>
    buffer<T> create_bo_buffer(size_t size, int group_id){
        assert(size > 0);
        assert(group_id >= 3);
        assert(group_id < 8);
        LOG_VERBOSE(2, "Creating buffer buffer with size: " << size << " and group_id: " << group_id);
        return buffer<T>(size, *this->device, *this->kernel, group_id);
    }
    template<typename T>
    buffer<T> create_bo_buffer(T*user_allocated_ptr,  size_t size, int group_id){
        assert(size > 0);
        assert(group_id >= 3);
        assert(group_id < 8);
        LOG_VERBOSE(2, "Creating user allocated  buffer with size: " << size << " and group_id: " << group_id);
        return buffer<T>(size,  user_allocated_ptr, *this->device, *this->kernel, group_id);
    }
};



///@brief npu_manager
///@note There should be only one npu_manager inside main.
///@note It handles all xclbins and instr_sequences.
///@note Each xclbin may have multiple instr_sequences.
///@note Each xclbin and instr_sequence has a unique id.
///@note Both id shall be provided to run an accelerator.
///@note Therefore, the xclbin_name between different accel_descriptions may overlap, but the instr_name is unique.
class npu_manager{
public:
    constexpr static int max_xclbins = 16; // This is hard constraint from the XRT driver
    constexpr static int max_kernels = 64; // This is hard constraint from the XRT driver
    
    npu_manager(npu_device device = device_npu2, unsigned int device_id = 0U);

    npu_app create_app(npu_app_desc& desc);
    // ~npu_manager();
    int _load_xclbin(std::string xclbin_name);

    xrt::runlist create_runlist(npu_app& app);
    
    void list_kernels();
    void write_out_trace(char *traceOutPtr, size_t trace_size, std::string path);
    void print_npu_info();
    float get_npu_power(bool print = true);
    int get_bo_info(uint32_t handle, bool print = true);


private:
    std::vector<accel_xclbin_desc> xclbin_descs;
    std::vector<accel_kernel_desc> kernel_descs;
    std::vector<std::string> registered_xclbin_names;

    int kernel_desc_count;
    int xclbin_desc_count;

    // the only device instance
    xrt::device device;

    npu_device npu_gen;
};

#endif
