/*
* Copyright (c) Huawei Technologies Co., Ltd. 2025-2025. All rights reserved.
*/

#ifndef DISETANGLE_ATTENTION_KERNEL_H
#define DISETANGLE_ATTENTION_KERNEL_H

#include "kernel_operator.h"
#include "lib/matmul_intf.h"

using namespace AscendC;

constexpr uint32_t UB_BUFFER_NUM = 1;
constexpr uint32_t TRANS_ADDR_NUM = 16;
constexpr uint32_t UB_LOOP_PROC_N = 48 / UB_BUFFER_NUM;
constexpr uint32_t UB_REV_PROC_N = 28 / UB_BUFFER_NUM;
constexpr uint32_t KBYTES = 1024;
constexpr uint32_t UB_BUFFER_SIZE = UB_LOOP_PROC_N * KBYTES;

struct TaskArgs {
    __aicore__ inline TaskArgs(uint32_t acc_s_offset, uint32_t batch_size, uint32_t head_num, uint32_t acc_s)
    {
        ASCENDC_ASSERT((acc_s != 0), "acc_s can't be zeros");
        ASCENDC_ASSERT((head_num != 0), "head_num can't be zeros");
        this->acc_s_offset = acc_s_offset;
        this->b_idx = acc_s_offset / (head_num * acc_s);
        this->n_idx = (acc_s_offset - (this->b_idx * head_num * acc_s)) / acc_s;
    }

    __aicore__ inline TaskArgs()
    {
        this->b_idx = 0;
        this->n_idx = 0;
        this->acc_s_offset = 0;
    }

    uint32_t b_idx{ 0 };
    uint32_t n_idx{ 0 };
    uint32_t acc_s_offset{ 0 };
};

struct KernelArgs {
    GM_ADDR query_layer;
    GM_ADDR key_layer;
    GM_ADDR value_layer;
    GM_ADDR pos_key_layer;
    GM_ADDR pos_query_layer;
    GM_ADDR relative_pos;
    GM_ADDR atten_mask;
    GM_ADDR atten_outputs;
    GM_ADDR atten_probs;
    GM_ADDR atten_weights;
    GM_ADDR workspace;
    GM_ADDR tiling;
};

template <typename T, bool enableC2P, bool enableP2C>
class DisetangleAttention {
public:
    __aicore__ inline DisetangleAttention() = default;

    __aicore__ inline void init(const KernelArgs &kernel_args, const DisetangleAttentionTilingData &tiling_data,
                                TPipe *_pipe)
    {
        this->batch_size = tiling_data.batchSize;
        this->head_num = tiling_data.headNum;
        this->head_dim = tiling_data.headDim;
        this->seq_lens = tiling_data.seqLen;
        this->acc_s = tiling_data.accS;
        this->long_acc_s = this->acc_s * 2;  // 2 means long acc_s double acc_s
        this->aiv_core_num = tiling_data.aivCoreNum;
        this->score_scale = (T)(tiling_data.scoreScale);
        this->pipe = _pipe;

        auto core_idx = GetBlockIdx();
        this->c2p_acc_long_block_offset =
            core_idx * 2 * this->acc_s * this->long_acc_s;  // 2 means long acc_s double acc_s
        this->p2c_acc_long_block_offset = this->c2p_acc_long_block_offset + this->acc_s * this->long_acc_s;

        this->SFT = tiling_data.SFT;

        // 计算本core 的计算量和全局偏移量
        this->init_core_task(tiling_data);

        // 初始化gm内存
        this->init_gm_buffer(kernel_args);

        // 初始化UB资源
        this->init_ub_buffer();
    }

    __aicore__ inline void process()
    {
        // step1: 构建l2_cache
        this->build_l2_cache();

        // step2: 开启并行计算流水
        if (this->core_acc_s_n != 0) {
            this->run_pipe_line_on_core();
        }
    }

    __aicore__ inline void init_gm_buffer(const KernelArgs &kernel_args)
    {
        query_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(kernel_args.query_layer));
        key_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(kernel_args.key_layer));
        value_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(kernel_args.value_layer));
        pos_key_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(kernel_args.pos_key_layer));
        pos_query_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(kernel_args.pos_query_layer));
        relative_pos_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t *>(kernel_args.relative_pos));
        atten_mask_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(kernel_args.atten_mask));

        atten_outputs_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(kernel_args.atten_outputs));
        atten_probs_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(kernel_args.atten_probs));
        atten_weights_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(kernel_args.atten_weights));

        auto user_workspace = GetUserWorkspace(kernel_args.workspace);
        __gm__ uint8_t *addr = reinterpret_cast<__gm__ uint8_t *>(user_workspace);
        att_pos_gm.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t *>(addr));
        acc_block_gm.SetGlobalBuffer(reinterpret_cast<__gm__ T *>(addr + this->acc_s * this->acc_s * sizeof(int32_t)));
    }

    __aicore__ inline void init_ub_buffer()
    {
        pipe->InitBuffer(this->ub_in_a_que, UB_BUFFER_NUM, UB_BUFFER_SIZE);  // 48K
        pipe->InitBuffer(this->ub_in_b_que, UB_BUFFER_NUM, UB_BUFFER_SIZE);  // 48K
        pipe->InitBuffer(this->ub_out_que, UB_BUFFER_NUM, UB_BUFFER_SIZE);   // 48K
        pipe->InitBuffer(this->ub_tmp_que, UB_BUFFER_NUM, UB_REV_PROC_N * KBYTES / UB_BUFFER_NUM);
    }

    __aicore__ inline void caculate_core_task(uint32_t split_core_idx, uint32_t split_prev_core,
                                            uint32_t split_next_core, uint32_t use_core_num, uint32_t &core_task,
                                            uint32_t &core_offset)
    {
        auto core_idx = GetBlockIdx();
        // 计算本core 的计算量和全局偏移量
        if (core_idx < split_core_idx) {
            core_task = split_prev_core;
            core_offset = core_idx * split_prev_core;
        } else if (core_idx < use_core_num) {
            core_task = split_next_core;
            core_offset = split_core_idx * split_prev_core + (core_idx - split_core_idx) * split_next_core;
        } else {
            core_task = 0;
            core_offset = 0;
        }
    }

    __aicore__ inline void init_core_task(const DisetangleAttentionTilingData &tiling_data)
    {
        caculate_core_task(tiling_data.splitCoreIdx, tiling_data.splitPrevCoreAccSN, tiling_data.splitNextCoreAccSN,
            tiling_data.useCoreNum, this->core_acc_s_n, this->core_acc_s_offset);
        this->core_acc_s_offset *= this->acc_s;
    }

    __aicore__ inline void run_pipe_line_use_relative_pos()
    {
        TaskArgs task_info;
        TaskArgs prev_task_info;
        for (auto task_id = 0; task_id < this->core_acc_s_n; task_id++) {
            auto acc_s_offset = this->core_acc_s_offset + task_id * this->acc_s;
            task_info = TaskArgs(acc_s_offset, this->batch_size, this->head_num, this->acc_s);

            this->p2c_att_mm(task_info);
            this->c2p_att_mm(task_info);
            if (task_id >= 1) {
                this->apply_softmax(prev_task_info);
            }
            this->query_mul_scores(task_info);

            if (task_id >= 1) {
                this->sv_mm(prev_task_info);
            }
            this->qk_mm(task_info);
            this->p2c_gather(task_info);
            this->c2p_gather(task_info);

            prev_task_info = task_info;
        }

        this->apply_softmax(prev_task_info);
        this->sv_mm(prev_task_info);
    }

    __aicore__ inline void run_pipe_line_on_core()
    {
        this->run_pipe_line_use_relative_pos();
    }

    // Matmul
    using MatmulTypeA = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
    using MatmulTypeB = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
    using MatmulTypeB1 = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, true>;
    using MatmulTypeC = matmul::MatmulType<TPosition::GM, CubeFormat::ND, T, false>;
    using MatmulType = matmul::Matmul<MatmulTypeA, MatmulTypeB, MatmulTypeC>;
    using MatmulTypeQk = matmul::Matmul<MatmulTypeA, MatmulTypeB1, MatmulTypeC>;
    MatmulType SV_MM;
    MatmulTypeQk P2C_MM, C2P_MM, QK_MM;

private:
    template <typename MM_T>
    __aicore__ inline void call_async_matmul(MM_T &MM, const GlobalTensor<T> &left_matrix,
                                            const GlobalTensor<T> &right_matrix, const GlobalTensor<T> &result_matrix,
                                            uint8_t is_atmoic = 0, bool is_transpose_b = false)
    {
        MM.SetTensorA(left_matrix);
        MM.SetTensorB(right_matrix, is_transpose_b);
        MM.template IterateAll<false>(result_matrix, is_atmoic, false, true);
    }

    template <typename MM_T>
    __aicore__ inline void wait_async_matmul(MM_T &MM)
    {
        MM.WaitIterateAll();
        MM.End();
    }

    __aicore__ inline void ub_transpose_n_rows(const LocalTensor<T> &trans_out, const LocalTensor<T> &trans_in,
                                            const uint32_t trans_row)
    {
        uint64_t dst_local_list[TRANS_ADDR_NUM];
        uint64_t src_local_list[TRANS_ADDR_NUM];

        TransDataTo5HDParams trans_params = { false, false, static_cast<uint8_t>(this->acc_s / TRANS_ADDR_NUM),
            static_cast<uint16_t>(TRANS_ADDR_NUM * trans_row * sizeof(T) / 32),
            static_cast<uint16_t>(TRANS_ADDR_NUM * sizeof(T) / 32) };

        for (int i = 0; i < trans_row / TRANS_ADDR_NUM; i++) {
            // reload addr
            for (int j = 0; j < TRANS_ADDR_NUM; j++) {
                int64_t src_offset = (i * TRANS_ADDR_NUM + j) * this->acc_s;
                int64_t dst_offset = (i * TRANS_ADDR_NUM + j * trans_row);
                src_local_list[j] = (uint64_t)(trans_in[src_offset].GetPhyAddr());
                dst_local_list[j] = (uint64_t)(trans_out[dst_offset].GetPhyAddr());
            }

            TransDataTo5HD<T>(dst_local_list, src_local_list, trans_params);
        }
    }

    __aicore__ inline void ub_gather_one_loop(const GlobalTensor<T> &input, const GlobalTensor<int32_t> &pos,
                                            const GlobalTensor<T> &output, int64_t loop_offset, uint32_t loop_cnt,
                                            bool is_transpose = false)
    {
        auto pos_lt = this->ub_in_b_que.template AllocTensor<int32_t>();
        auto in_lt = this->ub_in_a_que.template AllocTensor<T>();

        DataCopy(in_lt, input[loop_offset * this->long_acc_s], loop_cnt * this->long_acc_s);
        DataCopy(pos_lt, pos[loop_offset * this->acc_s], loop_cnt * this->acc_s);

        this->ub_in_b_que.template EnQue(pos_lt);
        this->ub_in_a_que.template EnQue(in_lt);

        this->ub_in_b_que.template DeQue<int32_t>();
        this->ub_in_a_que.template DeQue<T>();

        Adds(pos_lt, pos_lt, static_cast<int32_t>(-1 * loop_offset * this->long_acc_s * sizeof(T)),
            loop_cnt * this->acc_s);

        auto pos_uint32_lt = pos_lt.template ReinterpretCast<uint32_t>();
        auto out_lt = this->ub_out_que.template AllocTensor<T>();
        Gather(out_lt, in_lt, pos_uint32_lt, 0, loop_cnt * this->acc_s);

        if (is_transpose) {
            DataCopy(in_lt, out_lt, loop_cnt * this->acc_s);
            this->ub_transpose_n_rows(out_lt, in_lt, loop_cnt);

            this->ub_out_que.template EnQue(out_lt);
            this->ub_out_que.template DeQue<T>();

            DataCopyParams copy_params = { static_cast<uint16_t>(this->acc_s),
                static_cast<uint16_t>(loop_cnt * sizeof(T) / 32), 0,
                static_cast<uint16_t>((this->acc_s - loop_cnt) * sizeof(T) / 32) };

            DataCopy(output[loop_offset], out_lt, copy_params);
        } else {
            this->ub_out_que.template EnQue(out_lt);
            this->ub_out_que.template DeQue<T>();

            DataCopy(output[loop_offset * this->acc_s], out_lt, loop_cnt * this->acc_s);
        }

        this->ub_out_que.template FreeTensor<T>(out_lt);
        this->ub_in_a_que.template FreeTensor<T>(in_lt);
        this->ub_in_b_que.template FreeTensor<int32_t>(pos_lt);
    }

    __aicore__ inline void ub_gather(const GlobalTensor<T> &input, const GlobalTensor<int32_t> &pos,
                                    const GlobalTensor<T> &output, bool is_transpose = false)
    {
        auto loop_cnt = this->acc_s / UB_LOOP_PROC_N;
        auto loop_tail = this->acc_s % UB_LOOP_PROC_N;

        for (int i = 0; i < loop_cnt; i++) {
            auto offset = i * UB_LOOP_PROC_N;
            this->ub_gather_one_loop(input, pos, output, offset, UB_LOOP_PROC_N, is_transpose);
        }

        if (loop_tail != 0) {
            auto offset = loop_cnt * UB_LOOP_PROC_N;
            this->ub_gather_one_loop(input, pos, output, offset, loop_tail, is_transpose);
        }
    }

    __aicore__ inline void ub_softmax_one_loop(const GlobalTensor<T> &atten_weights, const GlobalTensor<T> &atten_probs,
                                            const GlobalTensor<T> &atten_mask, int64_t offset, int64_t proc_cnt,
                                            int64_t loop_rows)
    {
        auto atten_weight_lt = this->ub_in_a_que.template AllocTensor<T>();
        auto atten_bias_lt = this->ub_in_b_que.template AllocTensor<T>();
        auto out_lt = this->ub_out_que.template AllocTensor<T>();

        DataCopy(atten_weight_lt, atten_weights[offset], proc_cnt);
        this->ub_in_a_que.template EnQue(atten_weight_lt);

        DataCopy(atten_bias_lt, atten_probs[offset], proc_cnt);
        this->ub_in_b_que.template EnQue(atten_bias_lt);

        this->ub_in_a_que.template DeQue<T>();
        this->ub_in_b_que.template DeQue<T>();

        // atten_weight = atten_weight + atten_bias
        Muls(atten_bias_lt, atten_bias_lt, this->score_scale, proc_cnt);

        Add(atten_weight_lt, atten_weight_lt, atten_bias_lt, proc_cnt);

        this->ub_in_b_que.template FreeTensor<T>(atten_bias_lt);

        auto mask_in_lt = this->ub_in_b_que.template AllocTensor<T>();
        auto mask_in_lt_u8 = mask_in_lt.template ReinterpretCast<uint8_t>();

        DataCopy(mask_in_lt, atten_mask[offset], proc_cnt);

        this->ub_in_b_que.template EnQue(mask_in_lt);
        this->ub_in_b_que.template DeQue<T>();

        // atten_weight = atten_weight + mask
        Add(atten_weight_lt, atten_weight_lt, mask_in_lt, proc_cnt);

        this->ub_out_que.template EnQue(out_lt);
        this->ub_out_que.template DeQue<T>();

        DataCopy(atten_weights[offset], atten_weight_lt, proc_cnt);
        this->ub_out_que.template FreeTensor<T>(out_lt);

        out_lt = this->ub_out_que.template AllocTensor<T>();
        // softmax
        auto nrows = static_cast<uint32_t>(loop_rows);
        SoftMaxShapeInfo src_shape = { nrows, this->acc_s, nrows, this->acc_s };

        SoftMax<T, false>(out_lt, atten_weight_lt, mask_in_lt_u8, this->SFT, src_shape);

        this->ub_in_b_que.template FreeTensor<T>(mask_in_lt);
        this->ub_in_a_que.template FreeTensor<T>(atten_weight_lt);
        this->ub_out_que.template EnQue(out_lt);
        this->ub_out_que.template DeQue<T>();

        DataCopy(atten_probs[offset], out_lt, proc_cnt);

        this->ub_out_que.template FreeTensor<T>(out_lt);
    }

    __aicore__ inline void ub_softmax(const GlobalTensor<T> &atten_weights, const GlobalTensor<T> &atten_probs,
                                    const GlobalTensor<T> &atten_mask)
    {
        PipeBarrier<PIPE_ALL>();
        auto loop_cnt = this->acc_s / UB_LOOP_PROC_N;
        auto loop_tail = this->acc_s % UB_LOOP_PROC_N;

        for (int i = 0; i < loop_cnt; i++) {
            int64_t offset = i * UB_LOOP_PROC_N * this->acc_s;
            auto proc_cnt = UB_LOOP_PROC_N * this->acc_s;

            this->ub_softmax_one_loop(atten_weights, atten_probs, atten_mask, offset, proc_cnt, UB_LOOP_PROC_N);
        }

        if (loop_tail != 0) {
            int64_t offset = loop_cnt * UB_LOOP_PROC_N * this->acc_s;
            auto proc_cnt = loop_tail * this->acc_s;

            this->ub_softmax_one_loop(atten_weights, atten_probs, atten_mask, offset, proc_cnt, loop_tail);
        }
    }

    __aicore__ inline void qk_mm(const TaskArgs &task_info)
    {
        this->call_async_matmul(this->QK_MM, this->atten_outputs_gm[task_info.acc_s_offset * this->head_dim],
                                this->key_gm[task_info.acc_s_offset * this->head_dim],
                                this->atten_weights_gm[task_info.acc_s_offset * this->acc_s], 0, true);
    }

    __aicore__ inline void apply_softmax(const TaskArgs &task_info)
    {
        this->wait_async_matmul(this->QK_MM);

        this->ub_softmax(this->atten_weights_gm[task_info.acc_s_offset * this->acc_s],
            this->atten_probs_gm[task_info.acc_s_offset * this->acc_s],
            this->atten_mask_gm[task_info.b_idx * this->acc_s * this->acc_s]);
    }

    __aicore__ inline void sv_mm(const TaskArgs &task_info)
    {
        this->call_async_matmul(this->SV_MM, this->atten_probs_gm[task_info.acc_s_offset * this->acc_s],
                                this->value_gm[task_info.acc_s_offset * this->head_dim],
                                this->atten_outputs_gm[task_info.acc_s_offset * this->head_dim]);

        this->wait_async_matmul(this->SV_MM);
    }

    __aicore__ inline void build_l2_cache()
    {
        uint32_t use_core_num = this->aiv_core_num;
        uint32_t split_next_core = this->acc_s / use_core_num;
        uint32_t split_prev_core = split_next_core + 1;
        uint32_t split_core_idx = this->acc_s % use_core_num;

        uint32_t core_task = 0;
        uint32_t core_offset = 0;
        caculate_core_task(split_core_idx, split_prev_core, split_next_core, use_core_num, core_task, core_offset);

        this->build_att_pos(core_offset, core_task);

        SyncAll();
    }

    __aicore__ inline void build_att_pos(uint32_t core_offset, uint32_t core_task)
    {
        auto tmp_lt = this->ub_tmp_que.template AllocTensor<uint8_t>();
        auto in_lt = this->ub_in_a_que.template AllocTensor<int64_t>();
        auto in_fp32_lt = in_lt[core_task * this->acc_s].template ReinterpretCast<float>();
        auto out_lt = this->ub_out_que.template AllocTensor<int32_t>();

        DataCopy(in_lt, this->relative_pos_gm[core_offset * this->acc_s], core_task * this->acc_s);

        this->ub_in_a_que.template EnQue(in_lt);
        this->ub_in_a_que.template DeQue<int64_t>();

        int32_t acc_s_int = static_cast<int32_t>(this->acc_s);
        float add_bias = static_cast<float>(acc_s_int - 1);
        float min_clamp = static_cast<float>(0.0f);
        float max_clamp = static_cast<float>(acc_s_int * 2 - 1);

        Cast(in_fp32_lt, in_lt, RoundMode::CAST_TRUNC, core_task * this->acc_s);
        Adds(in_fp32_lt, in_fp32_lt, add_bias, core_task * this->acc_s);
        ClampMin(in_fp32_lt, in_fp32_lt, tmp_lt, min_clamp, core_task * this->acc_s);
        ClampMax(in_fp32_lt, in_fp32_lt, tmp_lt, max_clamp, core_task * this->acc_s);
        Cast(out_lt, in_fp32_lt, RoundMode::CAST_TRUNC, core_task * this->acc_s);

        for (int i = core_offset; i < core_offset + core_task; ++i) {
            Adds(out_lt[(i - core_offset) * this->acc_s], out_lt[(i - core_offset) * this->acc_s],
                static_cast<int32_t>(i * this->long_acc_s), this->acc_s);

            Muls(out_lt[(i - core_offset) * this->acc_s], out_lt[(i - core_offset) * this->acc_s],
                static_cast<int32_t>(sizeof(T)), this->acc_s);
        }

        this->ub_out_que.template EnQue(out_lt);
        this->ub_out_que.template DeQue<int32_t>();

        DataCopy(att_pos_gm[core_offset * this->acc_s], out_lt, core_task * this->acc_s);

        this->ub_tmp_que.template FreeTensor<uint8_t>(tmp_lt);
        this->ub_in_a_que.template FreeTensor<int64_t>(in_lt);
        this->ub_out_que.template FreeTensor<int32_t>(out_lt);
    }

    __aicore__ inline void c2p_att_mm(const TaskArgs &task_info)
    {
        if constexpr (enableC2P) {
            this->call_async_matmul(this->C2P_MM, this->query_gm[task_info.acc_s_offset * this->head_dim],
                                    this->pos_key_gm[task_info.n_idx * this->head_dim],
                                    this->acc_block_gm[this->c2p_acc_long_block_offset], 0, true);
        }
    }

    __aicore__ inline void p2c_att_mm(const TaskArgs &task_info)
    {
        if constexpr (enableP2C) {
            this->call_async_matmul(this->P2C_MM, this->key_gm[task_info.acc_s_offset * this->head_dim],
                                    this->pos_query_gm[task_info.n_idx * this->head_dim],
                                    this->acc_block_gm[this->p2c_acc_long_block_offset], 0, true);
        }
    }

    __aicore__ inline void c2p_gather(const TaskArgs &task_info)
    {
        if constexpr (enableC2P) {
            this->wait_async_matmul(this->C2P_MM);

            if constexpr (enableP2C) {
                SetAtomicAdd<T>();
            }
            this->ub_gather(this->acc_block_gm[this->c2p_acc_long_block_offset], this->att_pos_gm,
                            this->atten_probs_gm[task_info.acc_s_offset * this->acc_s]);

            if constexpr (enableP2C) {
                SetAtomicNone();
            }
        }
    }

    __aicore__ inline void p2c_gather(const TaskArgs &task_info)
    {
        if constexpr (enableP2C) {
            this->wait_async_matmul(this->P2C_MM);

            this->ub_gather(this->acc_block_gm[this->p2c_acc_long_block_offset], this->att_pos_gm,
                            this->atten_probs_gm[task_info.acc_s_offset * this->acc_s], true);
        }
    }

    __aicore__ inline void ub_muls(const GlobalTensor<T> &input, T scale, const GlobalTensor<T> &output)
    {
        auto loop_cnt = this->acc_s / UB_LOOP_PROC_N;
        auto loop_tail = this->acc_s % UB_LOOP_PROC_N;

        for (int i = 0; i < loop_cnt; i++) {
            int64_t offset = i * UB_LOOP_PROC_N * this->head_dim;
            auto proc_cnt = UB_LOOP_PROC_N * this->head_dim;

            auto lt = this->ub_in_a_que.template AllocTensor<T>();
            DataCopy(lt, input[offset], proc_cnt);

            this->ub_in_a_que.template EnQue(lt);
            this->ub_in_a_que.template DeQue<T>();

            auto out_lt = this->ub_out_que.template AllocTensor<T>();
            Muls(out_lt, lt, scale, proc_cnt);

            this->ub_in_a_que.template FreeTensor<T>(lt);
            this->ub_out_que.template EnQue(out_lt);
            this->ub_out_que.template DeQue<T>();

            DataCopy(output[offset], out_lt, proc_cnt);

            this->ub_out_que.template FreeTensor<T>(out_lt);
        }

        if (loop_tail != 0) {
            int64_t offset = loop_cnt * UB_LOOP_PROC_N * this->head_dim;
            auto proc_cnt = loop_tail * this->head_dim;

            auto lt = this->ub_in_a_que.template AllocTensor<T>();
            DataCopy(lt, input[offset], proc_cnt);

            this->ub_in_a_que.template EnQue(lt);
            this->ub_in_a_que.template DeQue<T>();

            auto out_lt = this->ub_out_que.template AllocTensor<T>();
            Muls(out_lt, lt, scale, proc_cnt);

            this->ub_in_a_que.template FreeTensor<T>(lt);
            this->ub_out_que.template EnQue(out_lt);
            this->ub_out_que.template DeQue<T>();

            DataCopy(output[offset], out_lt, proc_cnt);

            this->ub_out_que.template FreeTensor<T>(out_lt);
        }
    }

    __aicore__ inline void query_mul_scores(const TaskArgs &task_info)
    {
        this->ub_muls(this->query_gm[task_info.acc_s_offset * this->head_dim], this->score_scale,
            this->atten_outputs_gm[task_info.acc_s_offset * this->head_dim]);
    }

    uint32_t batch_size{ 0 };
    uint32_t seq_lens{ 0 };
    uint32_t head_num{ 0 };
    uint32_t head_dim{ 0 };
    uint32_t aiv_core_num{ 0 };

    uint32_t core_acc_s_n{ 0 };       // 每个核的处理acc_s的个数
    uint32_t core_acc_s_offset{ 0 };  // 每个核的acc_s的偏移量
    uint32_t acc_s{ 0 };
    uint32_t long_acc_s{ 0 };

    T score_scale{ 0 };

    TPipe *pipe{ nullptr };
    TQue<TPosition::VECCALC, UB_BUFFER_NUM> ub_tmp_que;

    TQue<TPosition::VECIN, UB_BUFFER_NUM> ub_in_a_que;
    TQue<TPosition::VECIN, UB_BUFFER_NUM> ub_in_b_que;
    TQue<TPosition::VECOUT, UB_BUFFER_NUM> ub_out_que;

    GlobalTensor<T> query_gm;
    GlobalTensor<T> key_gm;
    GlobalTensor<T> value_gm;
    GlobalTensor<T> pos_key_gm;
    GlobalTensor<T> pos_query_gm;
    GlobalTensor<int64_t> relative_pos_gm;
    GlobalTensor<T> atten_mask_gm;

    GlobalTensor<T> atten_outputs_gm;
    GlobalTensor<T> atten_probs_gm;
    GlobalTensor<T> atten_weights_gm;

    GlobalTensor<T> acc_block_gm;
    GlobalTensor<int32_t> att_pos_gm;

    int64_t c2p_acc_long_block_offset{ 0 };
    int64_t p2c_acc_long_block_offset{ 0 };

    SoftMaxTiling SFT;
};

#endif