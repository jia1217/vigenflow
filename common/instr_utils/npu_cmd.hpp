#ifndef __NPU_CMD_HPP__
#define __NPU_CMD_HPP__

#include <stdlib.h>
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <stdio.h>
#include "buffer.hpp"
#include "debug_utils.hpp"
#include "xrt/xrt_bo.h"

const int INSTR_PRINT_WIDTH = 80;

typedef enum: uint32_t{
    queue_write = 0x00,
    dma_block_write = 0x01,
    dma_mask_rite = 0x03,
    dma_sync_write = 0x80,
    dma_ddr_patch_write = 0x81,
} op_headers;


typedef enum{
    e_npu_cmd_ddr,
    e_npu_cmd_issue_token,
    e_npu_cmd_wait,
    e_npu_cmd_write_dma,
    e_npu_cmd_write
} npu_cmd_type;

typedef enum {
    S2MM,
    MM2S
} dma_direction;

inline void instr_print(int line_number, uint32_t word, std::string msg){
    if (line_number == -1){ // -1 for the case when one line has multiple messages
        MSG_BOX_LINE(INSTR_PRINT_WIDTH, std::dec << std::setw(7) << " | " << std::setw(11) << " | " << msg);
    }
    else{
        MSG_BOX_LINE(INSTR_PRINT_WIDTH, std::dec << std::setw(4) << line_number << " | " << std::hex << std::setfill('0') << std::setw(8) << word << " | " << msg);
    }
}

///@brief npu command
///@note This is an interface (pure virtual class) for all npu commands
///@note The class is used to print the command, convert the command to the npu format and dump the command to the buffer
///@warning The class is not used directly, but is used as a base class for all npu commands
struct npu_cmd{
    ///@brief print the command
    ///@param bd the buffer to dump the command
    ///@param line_number the line number of the command
    ///@param op_count the operation count of the command
    ///@return the number of lines of the command
    virtual int print_cmd(uint32_t *bd, int line_number, int op_count) = 0;

    ///@brief convert the command to the npu format
    ///@param npu_seq the npu sequence to dump the command
    ///@note The function will convert the command to the npu format and dump the command to the buffer
    virtual void to_npu(std::vector<uint32_t>& npu_seq) = 0;

    // virtual npu_cmd_type get_type() = 0;
    ///@brief dump the command to the buffer
    ///@param bd the buffer to dump the command
    virtual void dump_cmd(uint32_t *bd) = 0;

    ///@brief get the number of lines of the command
    ///@return the number of lines of the command
    virtual int get_op_lines() = 0;
};



#endif
