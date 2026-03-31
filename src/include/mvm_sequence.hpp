#ifndef __MVM_SEQUENCE_HPP__
#define __MVM_SEQUENCE_HPP__

#include "../common/npu_utils.hpp"
#include "../common/npu_instr_utils.hpp"
///@brief generate the mvm sequence
///@param M the number of rows of the matrix A
///@param K the number of columns of the matrix A
///@param m the row tile size
///@param k the column tile size
///@param rows the number of CT rows
///@param cols the number of CT columns
///@return the npu sequence
///@note The function will generate the mvm sequence for the matrix multiplication
///@warning The function is only used for the npu2
///@warning This function will write the sequence to a file named "generated.txt"
///@warning The sequence name is "mvm_i8"
void generate_mvm_sequence(npu_sequence& seq, uint32_t M, uint32_t K, uint32_t m, uint32_t k, uint32_t rows, uint32_t cols){
    int cores = rows * cols;
    const int Arg_A = 0;
    const int Arg_B = 1;
    const int Arg_C = 2;
    uint32_t col_offset = 3;
    seq.clear_cmds();
    std::vector<npu_tiles> shim_tiles;
    for (int i = 0; i < cols; i++){
        shim_tiles.push_back(get_tile(0, i + col_offset));
    }
    /*
    npu_dma_memcpy_nd(
        metadata=memB_fifo,
        bd_id=2,
        mem=B,
        offsets=[0, 0, 0, 0],
        sizes=[M // m // n_cores, 1, 1, K],
        strides=[0, 0, 0, 1],
    )
    aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][4, 1, 1, 512][0, 0, 0, 1]) {id = 2 : i64, metadata = @memB} : memref<512xi8>
    */
    seq.npu_dma_memcpy_nd(
        sizeof(int8_t),
        Arg_B, 
        MM2S,
        shim_tiles[0],
        bd_2,
        it_channel_1,
        {0, 0, 0, 0},
        {M / m / cores, 1, 1, K},
        {0, 0, 0, 1},
        -1, 0, false
    );

    for (int i = 0; i < cols; i++){
        uint32_t A_offset = i * M * K / cols;
        uint32_t C_offset = i * M / cols;
        // npu_dma_memcpy_nd(
        //     metadata=memA_fifos[i],
        //     bd_id=1,
        //     mem=A,
        //     offsets=[0, 0, 0, A_offset],
        //     sizes=[M // mvm_cols // (m * mvm_rows), K_div_k, mvm_rows * m, k],
        //     strides=[m_x_K * mvm_rows, k, K, 1],
        // )
                    
        seq.npu_dma_memcpy_nd(
            sizeof(int8_t),
            Arg_A, 
            MM2S,
            shim_tiles[i],
            bd_1,
            it_channel_0,
            {0, 0, 0, A_offset},
            {M / cores / m, K / k, rows * m, k},
            {m * K * rows, k, K, 1},
            -1, 0, false
        );
        // npu_dma_memcpy_nd(
        //     metadata=memC_fifos[i],
        //     bd_id=0,
        //     mem=C,
        //     offsets=[0, 0, 0, C_offset],
        //     sizes=[1, 1, M // m // mvm_cols // mvm_rows, mvm_rows * m],
        //     strides=[0, 0, mvm_rows * m, 1],
        // )
        seq.npu_dma_memcpy_nd(
            sizeof(int32_t),
            Arg_C,
            S2MM,
            shim_tiles[i],
            bd_0,
            it_channel_0,
            {0, 0, 0, C_offset},
            {1, 1, M / cores / m, rows * m},
            {0, 0, rows * m, 1},
            -1, 0, true
        );
        
    }

    // DMA wait

    for (int i = 0; i < cols; i++){
        seq.npu_dma_wait(
            shim_tiles[i],
            S2MM,
            it_channel_0
        );
    }

    seq.cmds2seq();
    seq.write_out_sequence("generated.txt");
}


void generate_cap_atten_sequence(npu_sequence& seq,
     uint32_t num_head, uint32_t num_q,
      uint32_t num_kv, uint32_t rows, uint32_t cols){
    int cores = rows * cols;
    const int Arg_A = 0;
    const int Arg_B = 1;
    const int Arg_C = 2;
    uint32_t num_iter_kv = 1;
    ///////////////////////////////////////////////////////////////////////////
    constexpr int CT_lock_address_base = 0x000001F000;

    constexpr npu_it_channel it_for_send_vec_in = it_channel_1;
    constexpr npu_it_channel it_for_send_datablock = it_channel_0; 
    constexpr npu_it_channel it_for_recv_vec_out = it_channel_0;

    constexpr int CT_rtp_lock_id = 2;
    constexpr int CT_rtp_address = 15616;

    ///////////////////////////////////////////////////////////////////////////
    uint32_t col_offset = 0;
    seq.clear_cmds();
    std::vector<npu_tiles> shim_tiles;
    npu_tiles CT_tile_0;
    npu_tiles CT_tile_1;
    npu_tiles CT_tile_2;
    npu_tiles CT_tile_3;

    uint32_t q_all_offset;
    uint32_t qo_offset_j;
    uint32_t head_offset;
    for (int i = 0; i < cols; i++)
    {
        shim_tiles.push_back(get_tile(0, i + col_offset));
    }
    for (int i = 0; i < num_head; i++)
    {
        head_offset = i * 128;
        for (int j = 0; j < num_q / rows / cols; j++)
        {
            q_all_offset = 32*4*cols*3840*j + head_offset;
            for(int num_rtp = 0; num_rtp < cols; num_rtp++)
            {
                CT_tile_0 = get_tile(2, num_rtp);
                CT_tile_1 = get_tile(3, num_rtp);
                CT_tile_2 = get_tile(4, num_rtp);
                CT_tile_3 = get_tile(5, num_rtp);
                seq.rtp_write(CT_tile_0, CT_rtp_address, num_iter_kv); // 
                seq.rtp_write(CT_tile_0, CT_lock_address_base+16*(CT_rtp_lock_id), 1);
                seq.rtp_write(CT_tile_1, CT_rtp_address, num_iter_kv); 
                seq.rtp_write(CT_tile_1, CT_lock_address_base+16*(CT_rtp_lock_id), 1);
                seq.rtp_write(CT_tile_2, CT_rtp_address, num_iter_kv); 
                seq.rtp_write(CT_tile_2, CT_lock_address_base+16*(CT_rtp_lock_id), 1); 
                seq.rtp_write(CT_tile_3, CT_rtp_address, num_iter_kv); 
                seq.rtp_write(CT_tile_3, CT_lock_address_base+16*(CT_rtp_lock_id), 1); 
            }
            seq.npu_dma_memcpy_nd(
                sizeof(int16_t),
                Arg_B,
                MM2S,
                shim_tiles[0],
                bd_0,
                it_channel_1,
                {0, 0, 0, head_offset},
                {1, 1, 256, 128},
                {0, 0, 3840, 1},
                -1, 0, false
            );
            for(int i = 0; i < cols; i++)
            {
                qo_offset_j = 32 * 4 * 3840 * i;
                seq.npu_dma_memcpy_nd(
                    sizeof(int16_t),
                    Arg_A, 
                    MM2S,
                    shim_tiles[i],
                    bd_1,
                    it_channel_0,
                    {0, 0, 0, qo_offset_j + q_all_offset},
                    {1, 1, 128, 128},
                    {0, 0, 3840, 1},
                    -1, 0, false
                );
                seq.npu_dma_memcpy_nd(
                    sizeof(int16_t),
                    Arg_C,
                    S2MM,
                    shim_tiles[i],
                    bd_2,
                    it_channel_0,
                    {0, 0, 0, qo_offset_j + q_all_offset},
                    {1, 1, 128, 128},
                    {0, 0, 3840, 1},
                    -1, 0, true
                );
            }
            for(int i = 0; i < cols; i++)
            {
                seq.npu_dma_wait(
                    shim_tiles[i],
                    S2MM,
                    it_channel_0
                );
            }
        }
    }

    seq.cmds2seq();
    seq.write_out_sequence("generated.txt");
}


void generate_all_atten_sequence(npu_sequence& seq,
    uint32_t num_head, uint32_t num_q,
     uint32_t num_kv, uint32_t row_offset, uint32_t rows, uint32_t cols){
   int cores = rows * cols;
   const int Arg_A = 0;
   const int Arg_B = 1;
   const int Arg_C = 2;
   uint32_t num_iter_kv = num_kv/2;
   ///////////////////////////////////////////////////////////////////////////
   constexpr int CT_lock_address_base = 0x000001F000;

   constexpr npu_it_channel it_for_send_vec_in = it_channel_1;
   constexpr npu_it_channel it_for_send_datablock = it_channel_0; 
   constexpr npu_it_channel it_for_recv_vec_out = it_channel_0;

   constexpr int CT_rtp_lock_id = 2;
   constexpr int CT_rtp_address = 64064;

   ///////////////////////////////////////////////////////////////////////////
   uint32_t col_offset = 0;
  
   seq.clear_cmds();
   std::vector<npu_tiles> shim_tiles;
   npu_tiles CT_tile_0;
   npu_tiles CT_tile_1;
   npu_tiles CT_tile_2;
   npu_tiles CT_tile_3;

   uint32_t q_all_offset;
   uint32_t qo_offset_j;
   uint32_t head_offset;
   uint32_t Q_offset;
   for (int i = 0; i < cols; i++)
   {
       shim_tiles.push_back(get_tile(0, i + col_offset));
   }

   for(int num_rtp = 0; num_rtp < cols; num_rtp++)
    {
        CT_tile_0 = get_tile(2, num_rtp);
        CT_tile_1 = get_tile(3, num_rtp);
        CT_tile_2 = get_tile(4, num_rtp);
        CT_tile_3 = get_tile(5, num_rtp);
        seq.rtp_write(CT_tile_0, CT_rtp_address, num_iter_kv); // 
        seq.rtp_write(CT_tile_0, CT_lock_address_base+16*(CT_rtp_lock_id), 1);
        seq.rtp_write(CT_tile_1, CT_rtp_address, num_iter_kv); 
        seq.rtp_write(CT_tile_1, CT_lock_address_base+16*(CT_rtp_lock_id), 1);
        seq.rtp_write(CT_tile_2, CT_rtp_address, num_iter_kv); 
        seq.rtp_write(CT_tile_2, CT_lock_address_base+16*(CT_rtp_lock_id), 1); 
        seq.rtp_write(CT_tile_3, CT_rtp_address, num_iter_kv); 
        seq.rtp_write(CT_tile_3, CT_lock_address_base+16*(CT_rtp_lock_id), 1); 
    }

   for (int i = 0; i < num_head; i++)
   {
       head_offset = i * 128;
       for (int j = 0; j < 128 / rows / cols; j++)
       {
           q_all_offset = 32*4*cols*row_offset*j + head_offset;
           
           seq.npu_dma_memcpy_nd(
               sizeof(int16_t),
               Arg_B,
               MM2S,
               shim_tiles[0],
               bd_0,
               it_channel_1,
               {0, 0, 0, head_offset},
               {1, num_iter_kv, 256, 128},
               {0, 256*row_offset, row_offset, 1},
               -1, 0, false
           );
           for(int i = 0; i < cols; i++)
           {
               qo_offset_j = 32 * 4 * row_offset * i;
               seq.npu_dma_memcpy_nd(
                   sizeof(int16_t),
                   Arg_A, 
                   MM2S,
                   shim_tiles[i],
                   bd_1,
                   it_channel_0,
                   {0, 0, 0, qo_offset_j + q_all_offset},
                   {1, 1, 128, 128},
                   {0, 0, row_offset, 1},
                   -1, 0, false
               );
               seq.npu_dma_memcpy_nd(
                   sizeof(int16_t),
                   Arg_C,
                   S2MM,
                   shim_tiles[i],
                   bd_2,
                   it_channel_0,
                   {0, 0, 0, qo_offset_j + q_all_offset},
                   {1, 1, 128, 128},
                   {0, 0, row_offset, 1},
                   -1, 0, true
               );
           }
           for(int i = 0; i < cols; i++)
           {
               seq.npu_dma_wait(
                   shim_tiles[i],
                   S2MM,
                   it_channel_0
               );
           }

       }
       
        Q_offset = 4096*row_offset + head_offset;
        seq.npu_dma_memcpy_nd(
        sizeof(int16_t),
        Arg_B,
        MM2S,
        shim_tiles[0],
        bd_0,
        it_channel_1,
        {0, 0, 0, head_offset},
        {1, num_iter_kv, 256, 128},
        {0, 256*row_offset, row_offset, 1},
        -1, 0, false
        );
        for(int i = 0; i < 4; i++)
        {
            qo_offset_j = 32 * 4 * row_offset * i;
            seq.npu_dma_memcpy_nd(
                sizeof(int16_t),
                Arg_A, 
                MM2S,
                shim_tiles[i],
                bd_1,
                it_channel_0,
                {0, 0, 0, qo_offset_j + Q_offset},
                {1, 1, 128, 128},
                {0, 0, row_offset, 1},
                -1, 0, false
            );
            seq.npu_dma_memcpy_nd(
                sizeof(int16_t),
                Arg_C,
                S2MM,
                shim_tiles[i],
                bd_2,
                it_channel_0,
                {0, 0, 0, qo_offset_j + Q_offset},
                {1, 1, 128, 128},
                {0, 0, row_offset, 1},
                -1, 0, true
            );
        }
        for(int i = 0; i < 4; i++)
        {
            seq.npu_dma_wait(
                shim_tiles[i],
                S2MM,
                it_channel_0
            );
        }
 
    }

   seq.cmds2seq();
   seq.write_out_sequence("generated.txt");
}


void generate_noise_atten_sequence(npu_sequence& seq,
    uint32_t num_head, uint32_t num_q,
     uint32_t num_kv, uint32_t row_offset, uint32_t rows, uint32_t cols){
   int cores = rows * cols;
   const int Arg_A = 0;
   const int Arg_B = 1;
   const int Arg_C = 2;
   uint32_t num_iter_kv = num_kv/2;
   ///////////////////////////////////////////////////////////////////////////
   constexpr int CT_lock_address_base = 0x000001F000;

   constexpr npu_it_channel it_for_send_vec_in = it_channel_1;
   constexpr npu_it_channel it_for_send_datablock = it_channel_0; 
   constexpr npu_it_channel it_for_recv_vec_out = it_channel_0;

   constexpr int CT_rtp_lock_id = 2;
   constexpr int CT_rtp_address = 15616;

   ///////////////////////////////////////////////////////////////////////////
   uint32_t col_offset = 0;
  
   seq.clear_cmds();
   std::vector<npu_tiles> shim_tiles;
   npu_tiles CT_tile_0;
   npu_tiles CT_tile_1;
   npu_tiles CT_tile_2;
   npu_tiles CT_tile_3;

   uint32_t q_all_offset;
   uint32_t qo_offset_j;
   uint32_t head_offset;
   uint32_t Q_offset;
   for (int i = 0; i < cols; i++)
   {
       shim_tiles.push_back(get_tile(0, i + col_offset));
   }

   

   for (int i = 0; i < num_head; i++)
   {
       head_offset = i * 128;
       for (int j = 0; j < 128 / rows / cols; j++)
       {
           q_all_offset = 32*4*cols*row_offset*j + head_offset;
           for(int num_rtp = 0; num_rtp < cols; num_rtp++)
            {
                CT_tile_0 = get_tile(2, num_rtp);
                CT_tile_1 = get_tile(3, num_rtp);
                CT_tile_2 = get_tile(4, num_rtp);
                CT_tile_3 = get_tile(5, num_rtp);
                seq.rtp_write(CT_tile_0, CT_rtp_address, num_iter_kv); // 
                seq.rtp_write(CT_tile_0, CT_lock_address_base+16*(CT_rtp_lock_id), 1);
                seq.rtp_write(CT_tile_1, CT_rtp_address, num_iter_kv); 
                seq.rtp_write(CT_tile_1, CT_lock_address_base+16*(CT_rtp_lock_id), 1);
                seq.rtp_write(CT_tile_2, CT_rtp_address, num_iter_kv); 
                seq.rtp_write(CT_tile_2, CT_lock_address_base+16*(CT_rtp_lock_id), 1); 
                seq.rtp_write(CT_tile_3, CT_rtp_address, num_iter_kv); 
                seq.rtp_write(CT_tile_3, CT_lock_address_base+16*(CT_rtp_lock_id), 1); 
            }
           seq.npu_dma_memcpy_nd(
               sizeof(int16_t),
               Arg_B,
               MM2S,
               shim_tiles[0],
               bd_0,
               it_channel_1,
               {0, 0, 0, head_offset},
               {1, num_iter_kv, 256, 128},
               {0, 256*row_offset, row_offset, 1},
               -1, 0, false
           );
           for(int i = 0; i < cols; i++)
           {
               qo_offset_j = 32 * 4 * row_offset * i;
               seq.npu_dma_memcpy_nd(
                   sizeof(int16_t),
                   Arg_A, 
                   MM2S,
                   shim_tiles[i],
                   bd_1,
                   it_channel_0,
                   {0, 0, 0, qo_offset_j + q_all_offset},
                   {1, 1, 128, 128},
                   {0, 0, row_offset, 1},
                   -1, 0, false
               );
               seq.npu_dma_memcpy_nd(
                   sizeof(int16_t),
                   Arg_C,
                   S2MM,
                   shim_tiles[i],
                   bd_2,
                   it_channel_0,
                   {0, 0, 0, qo_offset_j + q_all_offset},
                   {1, 1, 128, 128},
                   {0, 0, row_offset, 1},
                   -1, 0, true
               );
           }
           for(int i = 0; i < cols; i++)
           {
               seq.npu_dma_wait(
                   shim_tiles[i],
                   S2MM,
                   it_channel_0
               );
           }

       }
    }

   seq.cmds2seq();
   seq.write_out_sequence("generated.txt");
}


void generate_mm_128_double_sequence(npu_sequence& seq,
    uint32_t M_size, uint32_t K_size,
     uint32_t N_size, uint32_t k_size, uint32_t rows, uint32_t cols){
   int cores = rows * cols;
   seq.clear_cmds();
   const int Arg_A = 0;
   const int Arg_B = 1;
   const int Arg_C = 2;
   const int m_size = 128;
   const int n_size = 64;
   uint32_t K_div_k = K_size / k_size /2;
   uint32_t num_N = N_size / 64 / 8;
   uint32_t num_M = M_size / 128 / 4;

   ///////////////////////////////////////////////////////////////////////////
   constexpr int CT_lock_address_base = 0x000001F000;
   constexpr int CT_rtp_lock_id = 6;
   constexpr int CT_rtp_address = 54528;

   ///////////////////////////////////////////////////////////////////////////
   uint32_t col_offset = 0;
   uint32_t n_aie_rows = rows;
   uint32_t n_aie_cols = cols;
   uint32_t whole_row_size = m_size * n_aie_rows;
   uint32_t whole_col_size = n_size * n_aie_cols;
   uint32_t M_num = M_size / whole_row_size;
   uint32_t repeat_A = N_size / whole_col_size;
   uint32_t one_A_size = m_size*K_size;
   uint32_t one_C_size = m_size*N_size;
   
   uint32_t MT_K_size = 512;


   std::vector<npu_tiles> shim_tiles;
   npu_tiles CT_tile_0;
   npu_tiles CT_tile_1;
   npu_tiles CT_tile_2;
   npu_tiles CT_tile_3;

   for (int i = 0; i < cols; i++)
   {
       shim_tiles.push_back(get_tile(0, i + col_offset));
   }
   for(int num_rtp = 0; num_rtp < cols; num_rtp++)
       {
           CT_tile_0 = get_tile(2, num_rtp);
           CT_tile_1 = get_tile(3, num_rtp);
           CT_tile_2 = get_tile(4, num_rtp);
           CT_tile_3 = get_tile(5, num_rtp);
           seq.rtp_write(CT_tile_0, CT_rtp_address, K_div_k); 
           seq.rtp_write(CT_tile_0, CT_rtp_address+4, num_N); 
           seq.rtp_write(CT_tile_0, CT_rtp_address+8, num_M); 
           seq.rtp_write(CT_tile_0, CT_lock_address_base+16*(CT_rtp_lock_id), 1);
   
           seq.rtp_write(CT_tile_1, CT_rtp_address, K_div_k); 
           seq.rtp_write(CT_tile_1, CT_rtp_address+4, num_N); 
           seq.rtp_write(CT_tile_1, CT_rtp_address+8, num_M);
           seq.rtp_write(CT_tile_1, CT_lock_address_base+16*(CT_rtp_lock_id), 1);
   
           seq.rtp_write(CT_tile_2, CT_rtp_address, K_div_k); 
           seq.rtp_write(CT_tile_2, CT_rtp_address+4, num_N); 
           seq.rtp_write(CT_tile_2, CT_rtp_address+8, num_M);
           seq.rtp_write(CT_tile_2, CT_lock_address_base+16*(CT_rtp_lock_id), 1); 
   
           seq.rtp_write(CT_tile_3, CT_rtp_address, K_div_k); 
           seq.rtp_write(CT_tile_3, CT_rtp_address+4, num_N); 
           seq.rtp_write(CT_tile_3, CT_rtp_address+8, num_M);
           seq.rtp_write(CT_tile_3, CT_lock_address_base+16*(CT_rtp_lock_id), 1); 
       }
   for (int i = 0; i < M_num; i++)
   {
        uint32_t A_offset = i*one_A_size*n_aie_rows*1;
        uint32_t C_offset = i*one_C_size*n_aie_rows*1;

        for(int num_A = 0; num_A < repeat_A/2; num_A++)
        {

        for(int j = 0; j < 4; j++)
        {   
            uint32_t B_offset = j *n_size;
            uint32_t B_offset_1 = (j+4)*n_size;
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_A, 
                MM2S,
                shim_tiles[j],
                bd_0,
                it_channel_0,
                {0, 0, 0, A_offset + j * m_size * K_size},
                {1, K_size / MT_K_size, m_size, MT_K_size},
                {0, MT_K_size, K_size, 1},
                -1, 0, false
                );
            seq.npu_dma_memcpy_nd(
                    sizeof(uint16_t),
                    Arg_B, 
                    MM2S,
                    shim_tiles[j],
                    bd_1,
                    it_channel_1,
                    {0, 0, 0, B_offset + num_A * 512 * 2},
                    {1, K_size / k_size, k_size, n_size},
                    {0, k_size * N_size, N_size, 1},
                    -1, 0, false
                );  
            seq.npu_dma_memcpy_nd(
                    sizeof(uint16_t),
                    Arg_C,
                    S2MM,
                    shim_tiles[j],
                    bd_2,
                    it_channel_0,
                    {0, 0, 0, C_offset + B_offset + num_A * 512 * 2},
                    {1, n_aie_rows, m_size, n_size},
                    {0, m_size * N_size, N_size, 1},
                    -1, 0, true
                ); 
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_B, 
                MM2S,
                shim_tiles[j+4],
                bd_3,
                it_channel_1,
                {0, 0, 0, B_offset_1 + num_A * 512 * 2},
                {1, K_size / k_size, k_size, n_size},
                {0, k_size * N_size, N_size, 1},
                -1, 0, false
                );  
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_C,
                S2MM,
                shim_tiles[j+4],
                bd_4,
                it_channel_0,
                {0, 0, 0, C_offset + B_offset_1 + num_A * 512 * 2},
                {1, n_aie_rows, m_size, n_size},
                {0, m_size * N_size, N_size, 1},
                -1, 0, true
                );      
        }
        for(int k = 0; k < cols; k++)
        {
            seq.npu_dma_wait(
                shim_tiles[k],
                S2MM,
                it_channel_0
            );
        }

        for(int j = 0; j < 4; j++)
        {   
            uint32_t B_offset = j *n_size;
            uint32_t B_offset_1 = (j+4)*n_size;
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_A, 
                MM2S,
                shim_tiles[j],
                bd_5,
                it_channel_0,
                {0, 0, 0, A_offset + j*m_size*K_size},
                {1, K_size/MT_K_size, m_size, MT_K_size},
                {0, MT_K_size, K_size, 1},
                -1, 0, false
                );
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_B, 
                MM2S,
                shim_tiles[j],
                bd_6,
                it_channel_1,
                {0, 0, 0, B_offset + num_A * 512 * 2 +  512},
                {1, K_size/k_size, k_size, n_size},
                {0, k_size*N_size, N_size, 1},
                -1, 0, false
                );  
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_C,
                S2MM,
                shim_tiles[j],
                bd_7,
                it_channel_0,
                {0, 0, 0, C_offset + B_offset + num_A * 512 * 2 +  512},
                {1, n_aie_rows, m_size, n_size},
                {0, m_size*N_size, N_size, 1},
                -1, 0, true
                ); 
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_B, 
                MM2S,
                shim_tiles[j+4],
                bd_8,
                it_channel_1,
                {0, 0, 0, B_offset_1 + num_A * 512 * 2 +  512},
                {1, K_size/k_size, k_size, n_size},
                {0, k_size*N_size, N_size, 1},
                -1, 0, false
                );  
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_C,
                S2MM,
                shim_tiles[j+4],
                bd_9,
                it_channel_0,
                {0, 0, 0, C_offset + B_offset_1 + num_A * 512 * 2 + 512},
                {1, n_aie_rows, m_size, n_size},
                {0, m_size*N_size, N_size, 1},
                -1, 0, true
                );      
        }
        if (num_A > 0)
        {
            for(int k = 0; k < cols; k++)
            {
                seq.npu_dma_wait(
                    shim_tiles[k],
                    S2MM,
                    it_channel_0
                );
            }
        }
            
        }
        for(int k = 0; k < cols; k++)
        {
            seq.npu_dma_wait(
                shim_tiles[k],
                S2MM,
                it_channel_0
            );
        }
    }    
   
    seq.cmds2seq();
    seq.write_out_sequence("generated.txt");
}


void generate_mm_64_double_sequence(npu_sequence& seq,
    uint32_t M_size, uint32_t K_size,
     uint32_t N_size, uint32_t k_size, uint32_t rows, uint32_t cols){
   int cores = rows * cols;
   seq.clear_cmds();
   const int Arg_A = 0;
   const int Arg_B = 1;
   const int Arg_C = 2;
   const int m_size = 64;
   const int n_size = 128;
   uint32_t K_div_k = K_size / k_size /2;
   uint32_t num_N = N_size / 128 / 8;
   uint32_t num_M = M_size / 64 / 4;

   ///////////////////////////////////////////////////////////////////////////
   constexpr int CT_lock_address_base = 0x000001F000;
   constexpr int CT_rtp_lock_id = 6;
   constexpr int CT_rtp_address = 62976;

   ///////////////////////////////////////////////////////////////////////////
   uint32_t col_offset = 0;
   uint32_t n_aie_rows = rows;
   uint32_t n_aie_cols = cols;
   uint32_t whole_row_size = m_size * n_aie_rows;
   uint32_t whole_col_size = n_size * n_aie_cols;
   uint32_t M_num = M_size / whole_row_size;
   uint32_t repeat_A = N_size / whole_col_size;
   uint32_t one_A_size = m_size*K_size;
   uint32_t one_C_size = m_size*N_size;
   
   uint32_t MT_K_size = 512;

//    seq.clear_cmds();
   std::vector<npu_tiles> shim_tiles;
   npu_tiles CT_tile_0;
   npu_tiles CT_tile_1;
   npu_tiles CT_tile_2;
   npu_tiles CT_tile_3;

   for (int i = 0; i < cols; i++)
   {
       shim_tiles.push_back(get_tile(0, i + col_offset));
   }
   for(int num_rtp = 0; num_rtp < cols; num_rtp++)
       {
           CT_tile_0 = get_tile(2, num_rtp);
           CT_tile_1 = get_tile(3, num_rtp);
           CT_tile_2 = get_tile(4, num_rtp);
           CT_tile_3 = get_tile(5, num_rtp);
           seq.rtp_write(CT_tile_0, CT_rtp_address, K_div_k); 
           seq.rtp_write(CT_tile_0, CT_rtp_address+4, num_N); 
           seq.rtp_write(CT_tile_0, CT_rtp_address+8, num_M); 
           seq.rtp_write(CT_tile_0, CT_lock_address_base+16*(CT_rtp_lock_id), 1);
   
           seq.rtp_write(CT_tile_1, CT_rtp_address, K_div_k); 
           seq.rtp_write(CT_tile_1, CT_rtp_address+4, num_N); 
           seq.rtp_write(CT_tile_1, CT_rtp_address+8, num_M);
           seq.rtp_write(CT_tile_1, CT_lock_address_base+16*(CT_rtp_lock_id), 1);
   
           seq.rtp_write(CT_tile_2, CT_rtp_address, K_div_k); 
           seq.rtp_write(CT_tile_2, CT_rtp_address+4, num_N); 
           seq.rtp_write(CT_tile_2, CT_rtp_address+8, num_M);
           seq.rtp_write(CT_tile_2, CT_lock_address_base+16*(CT_rtp_lock_id), 1); 
   
           seq.rtp_write(CT_tile_3, CT_rtp_address, K_div_k); 
           seq.rtp_write(CT_tile_3, CT_rtp_address+4, num_N); 
           seq.rtp_write(CT_tile_3, CT_rtp_address+8, num_M);
           seq.rtp_write(CT_tile_3, CT_lock_address_base+16*(CT_rtp_lock_id), 1); 
       }
   for (int i = 0; i < M_num; i++)
   {
        uint32_t A_offset = i*one_A_size*n_aie_rows*1;
        uint32_t C_offset = i*one_C_size*n_aie_rows*1;

        for(int num_A = 0; num_A < repeat_A/2; num_A++)
        {

        for(int j = 0; j < 4; j++)
        {   
            uint32_t B_offset = j *n_size;
            uint32_t B_offset_1 = (j+4)*n_size;
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_A, 
                MM2S,
                shim_tiles[j],
                bd_0,
                it_channel_0,
                {0, 0, 0, A_offset + j * m_size * K_size},
                {1, K_size / MT_K_size, m_size, MT_K_size},
                {0, MT_K_size, K_size, 1},
                -1, 0, false
                );
            seq.npu_dma_memcpy_nd(
                    sizeof(uint16_t),
                    Arg_B, 
                    MM2S,
                    shim_tiles[j],
                    bd_1,
                    it_channel_1,
                    {0, 0, 0, B_offset + num_A * 1024 * 2},
                    {1, K_size / k_size, k_size, n_size},
                    {0, k_size * N_size, N_size, 1},
                    -1, 0, false
                );  
            seq.npu_dma_memcpy_nd(
                    sizeof(uint16_t),
                    Arg_C,
                    S2MM,
                    shim_tiles[j],
                    bd_2,
                    it_channel_0,
                    {0, 0, 0, C_offset + B_offset + num_A * 1024 * 2},
                    {1, n_aie_rows, m_size, n_size},
                    {0, m_size * N_size, N_size, 1},
                    -1, 0, true
                ); 
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_B, 
                MM2S,
                shim_tiles[j+4],
                bd_3,
                it_channel_1,
                {0, 0, 0, B_offset_1 + num_A * 1024 * 2},
                {1, K_size / k_size, k_size, n_size},
                {0, k_size * N_size, N_size, 1},
                -1, 0, false
                );  
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_C,
                S2MM,
                shim_tiles[j+4],
                bd_4,
                it_channel_0,
                {0, 0, 0, C_offset + B_offset_1 + num_A * 1024 * 2},
                {1, n_aie_rows, m_size, n_size},
                {0, m_size * N_size, N_size, 1},
                -1, 0, true
                );      
        }
        for(int k = 0; k < cols; k++)
        {
            seq.npu_dma_wait(
                shim_tiles[k],
                S2MM,
                it_channel_0
            );
        }

        for(int j = 0; j < 4; j++)
        {   
            uint32_t B_offset = j *n_size;
            uint32_t B_offset_1 = (j+4)*n_size;
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_A, 
                MM2S,
                shim_tiles[j],
                bd_5,
                it_channel_0,
                {0, 0, 0, A_offset + j*m_size*K_size},
                {1, K_size/MT_K_size, m_size, MT_K_size},
                {0, MT_K_size, K_size, 1},
                -1, 0, false
                );
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_B, 
                MM2S,
                shim_tiles[j],
                bd_6,
                it_channel_1,
                {0, 0, 0, B_offset + num_A * 1024 * 2 +  1024},
                {1, K_size/k_size, k_size, n_size},
                {0, k_size*N_size, N_size, 1},
                -1, 0, false
                );  
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_C,
                S2MM,
                shim_tiles[j],
                bd_7,
                it_channel_0,
                {0, 0, 0, C_offset + B_offset + num_A * 1024 * 2 +  1024},
                {1, n_aie_rows, m_size, n_size},
                {0, m_size*N_size, N_size, 1},
                -1, 0, true
                ); 
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_B, 
                MM2S,
                shim_tiles[j+4],
                bd_8,
                it_channel_1,
                {0, 0, 0, B_offset_1 + num_A * 1024 * 2 +  1024},
                {1, K_size/k_size, k_size, n_size},
                {0, k_size*N_size, N_size, 1},
                -1, 0, false
                );  
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_C,
                S2MM,
                shim_tiles[j+4],
                bd_9,
                it_channel_0,
                {0, 0, 0, C_offset + B_offset_1 + num_A * 1024 * 2 + 1024},
                {1, n_aie_rows, m_size, n_size},
                {0, m_size*N_size, N_size, 1},
                -1, 0, true
                );      
        }
        if (num_A > 0)
        {
            for(int k = 0; k < cols; k++)
            {
                seq.npu_dma_wait(
                    shim_tiles[k],
                    S2MM,
                    it_channel_0
                );
            }
        }
            
        }
        for(int k = 0; k < cols; k++)
        {
            seq.npu_dma_wait(
                shim_tiles[k],
                S2MM,
                it_channel_0
            );
        }
    }    
   
    seq.cmds2seq();
    seq.write_out_sequence("generated.txt");
}

// void generate_mm_128silu_double_sequence(npu_sequence& seq,
//     uint32_t M_size, uint32_t K_size,
//      uint32_t N_size, uint32_t k_size, uint32_t add_silu, uint32_t rows, uint32_t cols){
//    int cores = rows * cols;
//    seq.clear_cmds();
//    const int Arg_A = 0;
//    const int Arg_B = 1;
//    const int Arg_C = 2;
//    const int m_size = 128;
//    const int n_size = 64;
//    uint32_t K_div_k = K_size / k_size /2;
//    uint32_t num_N = N_size / 64 / 8;
//    uint32_t num_M = M_size / 128 / 4;
   

//    ///////////////////////////////////////////////////////////////////////////
//    constexpr int CT_lock_address_base = 0x000001F000;
//    constexpr int CT_rtp_lock_id = 6;
//    constexpr int CT_rtp_address = 54656;

//    ///////////////////////////////////////////////////////////////////////////
//    uint32_t col_offset = 0;
//    uint32_t n_aie_rows = rows;
//    uint32_t n_aie_cols = cols;
//    uint32_t whole_row_size = m_size * n_aie_rows;
//    uint32_t whole_col_size = n_size * n_aie_cols;
//    uint32_t M_num = M_size / whole_row_size;
//    uint32_t repeat_A = N_size / whole_col_size;
//    uint32_t one_A_size = m_size*K_size;
//    uint32_t one_C_size = m_size*N_size;
   
//    uint32_t MT_K_size = 512;

// //    seq.clear_cmds();
//    std::vector<npu_tiles> shim_tiles;
//    npu_tiles CT_tile_0;
//    npu_tiles CT_tile_1;
//    npu_tiles CT_tile_2;
//    npu_tiles CT_tile_3;

//    for (int i = 0; i < cols; i++)
//    {
//        shim_tiles.push_back(get_tile(0, i + col_offset));
//    }
//    for(int num_rtp = 0; num_rtp < cols; num_rtp++)
//        {
//            CT_tile_0 = get_tile(2, num_rtp);
//            CT_tile_1 = get_tile(3, num_rtp);
//            CT_tile_2 = get_tile(4, num_rtp);
//            CT_tile_3 = get_tile(5, num_rtp);
//            seq.rtp_write(CT_tile_0, CT_rtp_address, K_div_k); 
//            seq.rtp_write(CT_tile_0, CT_rtp_address+4, num_N); 
//            seq.rtp_write(CT_tile_0, CT_rtp_address+8, num_M); 
//            seq.rtp_write(CT_tile_0, CT_rtp_address+12, add_silu); 
//            seq.rtp_write(CT_tile_0, CT_lock_address_base+16*(CT_rtp_lock_id), 1);
   
//            seq.rtp_write(CT_tile_1, CT_rtp_address, K_div_k); 
//            seq.rtp_write(CT_tile_1, CT_rtp_address+4, num_N); 
//            seq.rtp_write(CT_tile_1, CT_rtp_address+8, num_M);
//            seq.rtp_write(CT_tile_1, CT_rtp_address+12, add_silu); 
//            seq.rtp_write(CT_tile_1, CT_lock_address_base+16*(CT_rtp_lock_id), 1);
   
//            seq.rtp_write(CT_tile_2, CT_rtp_address, K_div_k); 
//            seq.rtp_write(CT_tile_2, CT_rtp_address+4, num_N); 
//            seq.rtp_write(CT_tile_2, CT_rtp_address+8, num_M);
//            seq.rtp_write(CT_tile_2, CT_rtp_address+12, add_silu); 
//            seq.rtp_write(CT_tile_2, CT_lock_address_base+16*(CT_rtp_lock_id), 1); 
   
//            seq.rtp_write(CT_tile_3, CT_rtp_address, K_div_k); 
//            seq.rtp_write(CT_tile_3, CT_rtp_address+4, num_N); 
//            seq.rtp_write(CT_tile_3, CT_rtp_address+8, num_M);
//            seq.rtp_write(CT_tile_3, CT_rtp_address+12, add_silu); 
//            seq.rtp_write(CT_tile_3, CT_lock_address_base+16*(CT_rtp_lock_id), 1); 
//        }
//    for (int i = 0; i < M_num; i++)
//    {
//         uint32_t A_offset = i*one_A_size*n_aie_rows*1;
//         uint32_t C_offset = i*one_C_size*n_aie_rows*1;

//         for(int num_A = 0; num_A < repeat_A/2; num_A++)
//         {

//         for(int j = 0; j < 4; j++)
//         {   
//             uint32_t B_offset = j *n_size;
//             uint32_t B_offset_1 = (j+4)*n_size;
//             seq.npu_dma_memcpy_nd(
//                 sizeof(uint16_t),
//                 Arg_A, 
//                 MM2S,
//                 shim_tiles[j],
//                 bd_0,
//                 it_channel_0,
//                 {0, 0, 0, A_offset + j * m_size * K_size},
//                 {1, K_size / MT_K_size, m_size, MT_K_size},
//                 {0, MT_K_size, K_size, 1},
//                 -1, 0, false
//                 );
//             seq.npu_dma_memcpy_nd(
//                     sizeof(uint16_t),
//                     Arg_B, 
//                     MM2S,
//                     shim_tiles[j],
//                     bd_1,
//                     it_channel_1,
//                     {0, 0, 0, B_offset + num_A * 512 * 2},
//                     {1, K_size / k_size, k_size, n_size},
//                     {0, k_size * N_size, N_size, 1},
//                     -1, 0, false
//                 );  
//             seq.npu_dma_memcpy_nd(
//                     sizeof(uint16_t),
//                     Arg_C,
//                     S2MM,
//                     shim_tiles[j],
//                     bd_2,
//                     it_channel_0,
//                     {0, 0, 0, C_offset + B_offset + num_A * 512 * 2},
//                     {1, n_aie_rows, m_size, n_size},
//                     {0, m_size * N_size, N_size, 1},
//                     -1, 0, true
//                 ); 
//             seq.npu_dma_memcpy_nd(
//                 sizeof(uint16_t),
//                 Arg_B, 
//                 MM2S,
//                 shim_tiles[j+4],
//                 bd_3,
//                 it_channel_1,
//                 {0, 0, 0, B_offset_1 + num_A * 512 * 2},
//                 {1, K_size / k_size, k_size, n_size},
//                 {0, k_size * N_size, N_size, 1},
//                 -1, 0, false
//                 );  
//             seq.npu_dma_memcpy_nd(
//                 sizeof(uint16_t),
//                 Arg_C,
//                 S2MM,
//                 shim_tiles[j+4],
//                 bd_4,
//                 it_channel_0,
//                 {0, 0, 0, C_offset + B_offset_1 + num_A * 512 * 2},
//                 {1, n_aie_rows, m_size, n_size},
//                 {0, m_size * N_size, N_size, 1},
//                 -1, 0, true
//                 );      
//         }
//         for(int k = 0; k < cols; k++)
//         {
//             seq.npu_dma_wait(
//                 shim_tiles[k],
//                 S2MM,
//                 it_channel_0
//             );
//         }

//         for(int j = 0; j < 4; j++)
//         {   
//             uint32_t B_offset = j *n_size;
//             uint32_t B_offset_1 = (j+4)*n_size;
//             seq.npu_dma_memcpy_nd(
//                 sizeof(uint16_t),
//                 Arg_A, 
//                 MM2S,
//                 shim_tiles[j],
//                 bd_5,
//                 it_channel_0,
//                 {0, 0, 0, A_offset + j*m_size*K_size},
//                 {1, K_size/MT_K_size, m_size, MT_K_size},
//                 {0, MT_K_size, K_size, 1},
//                 -1, 0, false
//                 );
//             seq.npu_dma_memcpy_nd(
//                 sizeof(uint16_t),
//                 Arg_B, 
//                 MM2S,
//                 shim_tiles[j],
//                 bd_6,
//                 it_channel_1,
//                 {0, 0, 0, B_offset + num_A * 512 * 2 +  512},
//                 {1, K_size/k_size, k_size, n_size},
//                 {0, k_size*N_size, N_size, 1},
//                 -1, 0, false
//                 );  
//             seq.npu_dma_memcpy_nd(
//                 sizeof(uint16_t),
//                 Arg_C,
//                 S2MM,
//                 shim_tiles[j],
//                 bd_7,
//                 it_channel_0,
//                 {0, 0, 0, C_offset + B_offset + num_A * 512 * 2 +  512},
//                 {1, n_aie_rows, m_size, n_size},
//                 {0, m_size*N_size, N_size, 1},
//                 -1, 0, true
//                 ); 
//             seq.npu_dma_memcpy_nd(
//                 sizeof(uint16_t),
//                 Arg_B, 
//                 MM2S,
//                 shim_tiles[j+4],
//                 bd_8,
//                 it_channel_1,
//                 {0, 0, 0, B_offset_1 + num_A * 512 * 2 +  512},
//                 {1, K_size/k_size, k_size, n_size},
//                 {0, k_size*N_size, N_size, 1},
//                 -1, 0, false
//                 );  
//             seq.npu_dma_memcpy_nd(
//                 sizeof(uint16_t),
//                 Arg_C,
//                 S2MM,
//                 shim_tiles[j+4],
//                 bd_9,
//                 it_channel_0,
//                 {0, 0, 0, C_offset + B_offset_1 + num_A * 512 * 2 + 512},
//                 {1, n_aie_rows, m_size, n_size},
//                 {0, m_size*N_size, N_size, 1},
//                 -1, 0, true
//                 );      
//         }
//         if (num_A > 0)
//         {
//             for(int k = 0; k < cols; k++)
//             {
//                 seq.npu_dma_wait(
//                     shim_tiles[k],
//                     S2MM,
//                     it_channel_0
//                 );
//             }
//         }
            
//         }
//         for(int k = 0; k < cols; k++)
//         {
//             seq.npu_dma_wait(
//                 shim_tiles[k],
//                 S2MM,
//                 it_channel_0
//             );
//         }
//     }    
   
//     seq.cmds2seq();
//     seq.write_out_sequence("generated.txt");
// }

void generate_mm_128silu_double_sequence(npu_sequence& seq,
    uint32_t M_size, uint32_t K_size,
     uint32_t N_size, uint32_t k_size, uint32_t rows, uint32_t cols){
   int cores = rows * cols;
   seq.clear_cmds();
   const int Arg_A = 0;
   const int Arg_B = 1;
   const int Arg_C = 2;
   const int m_size = 128;
   const int n_size = 64;
   uint32_t K_div_k = K_size / k_size /2;
   uint32_t num_N = N_size / 64 / 8;
   uint32_t num_M = M_size / 128 / 4;

   ///////////////////////////////////////////////////////////////////////////
   constexpr int CT_lock_address_base = 0x000001F000;
   constexpr int CT_rtp_lock_id = 6;
   constexpr int CT_rtp_address = 54656;

   ///////////////////////////////////////////////////////////////////////////
   uint32_t col_offset = 0;
   uint32_t n_aie_rows = rows;
   uint32_t n_aie_cols = cols;
   uint32_t whole_row_size = m_size * n_aie_rows;
   uint32_t whole_col_size = n_size * n_aie_cols;
   uint32_t M_num = M_size / whole_row_size;
   uint32_t repeat_A = N_size / whole_col_size;
   uint32_t one_A_size = m_size*K_size;
   uint32_t one_C_size = m_size*N_size;
   
   uint32_t MT_K_size = 512;

//    seq.clear_cmds();
   std::vector<npu_tiles> shim_tiles;
   npu_tiles CT_tile_0;
   npu_tiles CT_tile_1;
   npu_tiles CT_tile_2;
   npu_tiles CT_tile_3;

   for (int i = 0; i < cols; i++)
   {
       shim_tiles.push_back(get_tile(0, i + col_offset));
   }
   for(int num_rtp = 0; num_rtp < cols; num_rtp++)
       {
           CT_tile_0 = get_tile(2, num_rtp);
           CT_tile_1 = get_tile(3, num_rtp);
           CT_tile_2 = get_tile(4, num_rtp);
           CT_tile_3 = get_tile(5, num_rtp);
           seq.rtp_write(CT_tile_0, CT_rtp_address, K_div_k); 
           seq.rtp_write(CT_tile_0, CT_rtp_address+4, num_N); 
           seq.rtp_write(CT_tile_0, CT_rtp_address+8, num_M); 
        //    seq.rtp_write(CT_tile_0, CT_rtp_address+12, add_silu); 
           seq.rtp_write(CT_tile_0, CT_lock_address_base+16*(CT_rtp_lock_id), 1);
   
           seq.rtp_write(CT_tile_1, CT_rtp_address, K_div_k); 
           seq.rtp_write(CT_tile_1, CT_rtp_address+4, num_N); 
           seq.rtp_write(CT_tile_1, CT_rtp_address+8, num_M);
        //    seq.rtp_write(CT_tile_1, CT_rtp_address+12, add_silu); 
           seq.rtp_write(CT_tile_1, CT_lock_address_base+16*(CT_rtp_lock_id), 1);
   
           seq.rtp_write(CT_tile_2, CT_rtp_address, K_div_k); 
           seq.rtp_write(CT_tile_2, CT_rtp_address+4, num_N); 
           seq.rtp_write(CT_tile_2, CT_rtp_address+8, num_M);
        //    seq.rtp_write(CT_tile_2, CT_rtp_address+12, add_silu); 
           seq.rtp_write(CT_tile_2, CT_lock_address_base+16*(CT_rtp_lock_id), 1); 
   
           seq.rtp_write(CT_tile_3, CT_rtp_address, K_div_k); 
           seq.rtp_write(CT_tile_3, CT_rtp_address+4, num_N); 
           seq.rtp_write(CT_tile_3, CT_rtp_address+8, num_M);
        //    seq.rtp_write(CT_tile_3, CT_rtp_address+12, add_silu); 
           seq.rtp_write(CT_tile_3, CT_lock_address_base+16*(CT_rtp_lock_id), 1);  
       }
   for (int i = 0; i < M_num; i++)
   {
        uint32_t A_offset = i*one_A_size*n_aie_rows*1;
        uint32_t C_offset = i*one_C_size*n_aie_rows*1;

        for(int num_A = 0; num_A < repeat_A/2; num_A++)
        {

        for(int j = 0; j < 4; j++)
        {   
            uint32_t B_offset = j *n_size;
            uint32_t B_offset_1 = (j+4)*n_size;
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_A, 
                MM2S,
                shim_tiles[j],
                bd_0,
                it_channel_0,
                {0, 0, 0, A_offset + j * m_size * K_size},
                {1, K_size / MT_K_size, m_size, MT_K_size},
                {0, MT_K_size, K_size, 1},
                -1, 0, false
                );
            seq.npu_dma_memcpy_nd(
                    sizeof(uint16_t),
                    Arg_B, 
                    MM2S,
                    shim_tiles[j],
                    bd_1,
                    it_channel_1,
                    {0, 0, 0, B_offset + num_A * 512 * 2},
                    {1, K_size / k_size, k_size, n_size},
                    {0, k_size * N_size, N_size, 1},
                    -1, 0, false
                );  
            seq.npu_dma_memcpy_nd(
                    sizeof(uint16_t),
                    Arg_C,
                    S2MM,
                    shim_tiles[j],
                    bd_2,
                    it_channel_0,
                    {0, 0, 0, C_offset + B_offset + num_A * 512 * 2},
                    {1, n_aie_rows, m_size, n_size},
                    {0, m_size * N_size, N_size, 1},
                    -1, 0, true
                ); 
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_B, 
                MM2S,
                shim_tiles[j+4],
                bd_3,
                it_channel_1,
                {0, 0, 0, B_offset_1 + num_A * 512 * 2},
                {1, K_size / k_size, k_size, n_size},
                {0, k_size * N_size, N_size, 1},
                -1, 0, false
                );  
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_C,
                S2MM,
                shim_tiles[j+4],
                bd_4,
                it_channel_0,
                {0, 0, 0, C_offset + B_offset_1 + num_A * 512 * 2},
                {1, n_aie_rows, m_size, n_size},
                {0, m_size * N_size, N_size, 1},
                -1, 0, true
                );      
        }
        for(int k = 0; k < cols; k++)
        {
            seq.npu_dma_wait(
                shim_tiles[k],
                S2MM,
                it_channel_0
            );
        }

        for(int j = 0; j < 4; j++)
        {   
            uint32_t B_offset = j *n_size;
            uint32_t B_offset_1 = (j+4)*n_size;
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_A, 
                MM2S,
                shim_tiles[j],
                bd_5,
                it_channel_0,
                {0, 0, 0, A_offset + j*m_size*K_size},
                {1, K_size/MT_K_size, m_size, MT_K_size},
                {0, MT_K_size, K_size, 1},
                -1, 0, false
                );
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_B, 
                MM2S,
                shim_tiles[j],
                bd_6,
                it_channel_1,
                {0, 0, 0, B_offset + num_A * 512 * 2 +  512},
                {1, K_size/k_size, k_size, n_size},
                {0, k_size*N_size, N_size, 1},
                -1, 0, false
                );  
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_C,
                S2MM,
                shim_tiles[j],
                bd_7,
                it_channel_0,
                {0, 0, 0, C_offset + B_offset + num_A * 512 * 2 +  512},
                {1, n_aie_rows, m_size, n_size},
                {0, m_size*N_size, N_size, 1},
                -1, 0, true
                ); 
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_B, 
                MM2S,
                shim_tiles[j+4],
                bd_8,
                it_channel_1,
                {0, 0, 0, B_offset_1 + num_A * 512 * 2 +  512},
                {1, K_size/k_size, k_size, n_size},
                {0, k_size*N_size, N_size, 1},
                -1, 0, false
                );  
            seq.npu_dma_memcpy_nd(
                sizeof(uint16_t),
                Arg_C,
                S2MM,
                shim_tiles[j+4],
                bd_9,
                it_channel_0,
                {0, 0, 0, C_offset + B_offset_1 + num_A * 512 * 2 + 512},
                {1, n_aie_rows, m_size, n_size},
                {0, m_size*N_size, N_size, 1},
                -1, 0, true
                );      
        }
        if (num_A > 0)
        {
            for(int k = 0; k < cols; k++)
            {
                seq.npu_dma_wait(
                    shim_tiles[k],
                    S2MM,
                    it_channel_0
                );
            }
        }
            
        }
        for(int k = 0; k < cols; k++)
        {
            seq.npu_dma_wait(
                shim_tiles[k],
                S2MM,
                it_channel_0
            );
        }
    }    
   
    seq.cmds2seq();
    seq.write_out_sequence("generated.txt");
}

#endif