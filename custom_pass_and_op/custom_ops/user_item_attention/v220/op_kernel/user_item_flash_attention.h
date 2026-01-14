/* Copyright 2026. Huawei Technologies Co.,Ltd. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#pragma once

#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "lib/matrix/matmul/matmul.h"

namespace kernels {

constexpr uint32_t UB_BLOCK_BYTES = 32;
constexpr uint32_t ELM_NUM_PER_BLOCK_FLOAT = 8;
constexpr uint32_t ELM_NUM_PER_BLOCK_FP16 = 16;
constexpr uint32_t VEC_MAX_REPEAT_BYTES = 256;

using namespace AscendC;

template <class InputT, class CalcT, class OutputT>
class UserItemFlashAttentionKernel {
public:
    using QueryType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, InputT>;
    using KeyType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, InputT, true>;
    using ValueType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, InputT, false>;
    using ScoreType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, InputT>;
    using GemmOutputType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CalcT>;
    using BiasType = matmul::MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CalcT>;

    UserItemFlashAttentionTilingData tiling_;
    AscendC::TPipe* pipe_ = nullptr;
    using mm_qk = matmul::Matmul<QueryType, KeyType, GemmOutputType, BiasType>;
    using mm_pv = matmul::Matmul<ScoreType, ValueType, GemmOutputType, BiasType>;
    mm_qk gemm_qk_;
    mm_pv gemm_pv_;

    GlobalTensor<InputT> query_gm_;
    GlobalTensor<InputT> key_item_gm_;
    GlobalTensor<InputT> key_user_gm_;
    GlobalTensor<InputT> value_item_gm_;
    GlobalTensor<InputT> value_user_gm_;
    GlobalTensor<OutputT> attn_out_gm_;

    GlobalTensor<CalcT> gemm_qk_res_ping_gm_;
    GlobalTensor<CalcT> gemm_qk_res_pong_gm_;
    GlobalTensor<InputT> softmax_res_ping_gm_;
    GlobalTensor<InputT> softmax_res_pong_gm_;
    GlobalTensor<CalcT> gemm_pv_res_ping_gm_;
    GlobalTensor<CalcT> gemm_pv_res_pong_gm_;
    GlobalTensor<CalcT> gemm_aggr_gm_;

    TQue<TPosition::VECIN, 1> comm_in_que_;
    TQue<TPosition::VECIN, 1> aggr_in_que_;
    TQue<TPosition::VECOUT, 1> comm_out_que_;

    TBuf<TPosition::VECCALC> softmax_tmp_buf_;
    TBuf<TPosition::VECCALC> vec_tmp_buf_;
    TBuf<TPosition::VECCALC> max_buf_;
    TBuf<TPosition::VECCALC> exp_sum_ping_buf_;
    TBuf<TPosition::VECCALC> exp_sum_pong_buf_;
    TBuf<TPosition::VECCALC> exp_max_ping_buf_;
    TBuf<TPosition::VECCALC> exp_max_pong_buf_;

    GlobalTensor<CalcT> gemm_qk_res_gm_;
    GlobalTensor<CalcT> softmax_in_gm_;
    GlobalTensor<InputT> softmax_out_gm_;
    LocalTensor<CalcT> softmax_in_exp_sum_buf_;
    LocalTensor<CalcT> softmax_exp_sum_buf_;
    LocalTensor<CalcT> softmax_exp_max_buf_;
    GlobalTensor<InputT> gemm_pv_p_gm_;
    GlobalTensor<CalcT> gemm_pv_res_gm_;

    GlobalTensor<CalcT> aggr_in_gm_;
    LocalTensor<CalcT> aggr_exp_max_buf_;
    LocalTensor<CalcT> aggr_exp_sum_buf_;

    uint32_t core_id;
    uint32_t task_start_idx;
    uint32_t task_count;
    int32_t valid_user_seq_len;

    __aicore__ inline UserItemFlashAttentionKernel(AscendC::TPipe* pipe)
    {
        pipe_ = pipe;
    }

    __aicore__ inline void init(GM_ADDR query, GM_ADDR key_user, GM_ADDR value_user, GM_ADDR mask_len, GM_ADDR key_item,
                                GM_ADDR value_item, GM_ADDR workspace, const UserItemFlashAttentionTilingData* tiling,
                                GM_ADDR attn_out)
    {
        tiling_ = *tiling;
        core_id = GetBlockIdx();
        task_start_idx = core_id * tiling_.task_num_per_core;
        task_count = min(tiling_.task_num_per_core, tiling_.total_task_num - task_start_idx);
        query_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(query));
        key_user_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(key_user));
        value_user_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(value_user));

        GlobalTensor<int32_t> mask_len_gm_;
        mask_len_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(mask_len));
        valid_user_seq_len = mask_len_gm_.GetValue(0);

        if (tiling_.has_item_kv == 1) {
            key_item_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(key_item));
            value_item_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(value_item));
        }

        attn_out_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(attn_out));
        uint32_t per_core_workspace_size =
            tiling_.gemm_qk_res_size * 2 + tiling_.gemm_pv_res_size * 2 + tiling_.gemm_aggr_res_size;
        auto* core_workspace_base_addr = workspace + core_id * per_core_workspace_size;

        gemm_qk_res_ping_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ CalcT*>(core_workspace_base_addr));
        gemm_qk_res_pong_gm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ CalcT*>(core_workspace_base_addr + tiling_.gemm_qk_res_size));

        softmax_res_ping_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ InputT*>(core_workspace_base_addr));
        softmax_res_pong_gm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ InputT*>(core_workspace_base_addr + tiling_.gemm_qk_res_size));

        gemm_pv_res_ping_gm_.SetGlobalBuffer(
            reinterpret_cast<__gm__ CalcT*>(core_workspace_base_addr + tiling_.gemm_qk_res_size * 2));
        gemm_pv_res_pong_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ CalcT*>(
            core_workspace_base_addr + tiling_.gemm_qk_res_size * 2 + tiling_.gemm_pv_res_size));
        gemm_aggr_gm_.SetGlobalBuffer(reinterpret_cast<__gm__ CalcT*>(
            core_workspace_base_addr + tiling_.gemm_qk_res_size * 2 + tiling_.gemm_pv_res_size * 2));
        // 初始化UB
        pipe_->InitBuffer(comm_in_que_, 1, tiling_.vec_softmax_batch_block * tiling_.user_kv_seq_block * sizeof(CalcT));
        pipe_->InitBuffer(aggr_in_que_, 1, tiling_.vec_aggr_batch_block * tiling_.shape_info.head_dim * sizeof(CalcT));
        pipe_->InitBuffer(comm_out_que_, 1,
                          tiling_.vec_softmax_batch_block * tiling_.user_kv_seq_block * sizeof(CalcT));
        pipe_->InitBuffer(softmax_tmp_buf_, tiling_.softmax_buf_size);
        pipe_->InitBuffer(vec_tmp_buf_, tiling_.vec_softmax_batch_block * tiling_.user_kv_seq_block * sizeof(CalcT));
        pipe_->InitBuffer(max_buf_, tiling_.item_batch_block * UB_BLOCK_BYTES);
        pipe_->InitBuffer(exp_sum_ping_buf_, tiling_.item_batch_block * UB_BLOCK_BYTES);
        pipe_->InitBuffer(exp_sum_pong_buf_, tiling_.item_batch_block * UB_BLOCK_BYTES);
        pipe_->InitBuffer(exp_max_ping_buf_, tiling_.item_batch_block * UB_BLOCK_BYTES);
        pipe_->InitBuffer(exp_max_pong_buf_, tiling_.item_batch_block * UB_BLOCK_BYTES);

        gemm_pv_.SetOrgShape(tiling_.item_batch_block, tiling_.shape_info.user_seq_stride, tiling_.user_kv_seq_block,
                             tiling_.user_kv_seq_block, tiling_.shape_info.head_dim);
    }

    __aicore__ inline void process()
    {
        auto event_mte32 = static_cast<event_t>(GetTPipePtr()->FetchEventID(AscendC::HardEvent::MTE3_MTE2));
        uint32_t task_real_count = task_count += 2;
        uint32_t loop_cnt = 0;
        uint32_t item_batch_idx_que[3] = {0, 0, 0};
        uint32_t head_num_idx_que[3] = {0, 0, 0};
        uint32_t user_kv_seq_idx_que[3] = {0, 0, 0};
        uint32_t valid_item_batch_size_que[3] = {tiling_.item_batch_block, tiling_.item_batch_block,
                                                 tiling_.item_batch_block};
        uint32_t valid_user_kv_seq_size_que[3] = {tiling_.user_kv_seq_block, tiling_.user_kv_seq_block,
                                                  tiling_.user_kv_seq_block};
        bool is_kv_first_block_que[3] = {false, false, false};
        bool is_kv_last_block_que[3] = {false, false, false};
        bool is_item_kv_block_que[3] = {false, false, false};

        uint32_t i_task;
        bool not_last = true;
        bool not_penultimate = true;
        auto max_ub = max_buf_.Get<CalcT>();

        for (int32_t task_offset = 0; task_offset < task_real_count; task_offset++) {
            if (task_offset == task_real_count - 2) {
                not_penultimate = false;
            } else if (task_offset == task_real_count - 1) {
                not_last = false;
            }
            bool not_last_two_loops = not_penultimate && not_last;
            i_task = task_start_idx + task_offset;
            uint32_t item_batch_idx = 0;
            uint32_t head_num_idx = 0;
            calcIndexIndices(i_task, item_batch_idx, head_num_idx);
            uint32_t user_kv_loop_cnt = DivCeil(valid_user_seq_len, tiling_.user_kv_seq_block);
            uint32_t item_kv_loop_cnt = 0;
            if (tiling_.has_item_kv == 1) {
                item_kv_loop_cnt = 1;
            }
            uint32_t total_kv_loop_cnt = item_kv_loop_cnt + user_kv_loop_cnt;

            if (!not_last_two_loops) {
                total_kv_loop_cnt = 1;
            }

            for (int32_t kv_loop = 0; kv_loop < total_kv_loop_cnt; kv_loop++) {
                uint32_t idx_0 = loop_cnt % 3;
                uint32_t idx_1 = (loop_cnt + 2) % 3;
                uint32_t idx_2 = (loop_cnt + 1) % 3;
                flipPingPong(loop_cnt);
                item_batch_idx_que[idx_0] = item_batch_idx;
                head_num_idx_que[idx_0] = head_num_idx;
                valid_item_batch_size_que[idx_0] =
                    min(tiling_.item_batch_block, tiling_.shape_info.item_batch - item_batch_idx);
                valid_user_kv_seq_size_que[idx_0] = tiling_.user_kv_seq_block;
                is_kv_first_block_que[idx_0] = kv_loop == 0;
                is_kv_last_block_que[idx_0] = kv_loop == (total_kv_loop_cnt - 1);
                is_item_kv_block_que[idx_0] = (kv_loop == 0) && (tiling_.has_item_kv == 1);
                if (kv_loop == total_kv_loop_cnt - 1) {
                    valid_user_kv_seq_size_que[idx_0] =
                        valid_user_seq_len - (kv_loop - item_kv_loop_cnt) * tiling_.user_kv_seq_block;
                }
                user_kv_seq_idx_que[idx_0] = (kv_loop - item_kv_loop_cnt) * tiling_.user_kv_seq_block;
                if (loop_cnt >= 1 && not_last) {
                    if (!is_item_kv_block_que[idx_1]) {
                        waitGemmQK();
                    }
                }
                if (not_last_two_loops) {
                    if (is_item_kv_block_que[idx_0]) {
                        processItemAttentionQK(valid_item_batch_size_que[idx_0], head_num_idx_que[idx_0],
                                               item_batch_idx_que[idx_0], gemm_qk_res_gm_);
                    } else {
                        launchGemmQK(valid_item_batch_size_que[idx_0], valid_user_kv_seq_size_que[idx_0],
                                     item_batch_idx_que[idx_0], head_num_idx_que[idx_0], user_kv_seq_idx_que[idx_0],
                                     gemm_qk_res_gm_);
                    }
                }

                if (loop_cnt > 0 && not_last) {
                    if (is_item_kv_block_que[idx_1]) {
                        processItemAttentionSoftmaxAndSV(valid_item_batch_size_que[idx_1], head_num_idx_que[idx_1],
                                                         item_batch_idx_que[idx_1], softmax_in_gm_, gemm_pv_res_gm_,
                                                         max_ub, softmax_exp_max_buf_, softmax_exp_sum_buf_);
                    } else {
                        processSoftmax(valid_item_batch_size_que[idx_1], valid_user_kv_seq_size_que[idx_1],
                                       softmax_in_gm_, softmax_out_gm_, softmax_exp_max_buf_, max_ub,
                                       softmax_in_exp_sum_buf_, softmax_exp_sum_buf_, !is_kv_first_block_que[idx_1]);
                        SetFlag<HardEvent::MTE3_MTE2>(event_mte32);
                    }
                }
                if (loop_cnt > 1 && !is_item_kv_block_que[idx_2]) {
                    waitGemmPV();
                }

                if (loop_cnt > 0 && not_last && !is_item_kv_block_que[idx_1]) {
                    WaitFlag<HardEvent::MTE3_MTE2>(event_mte32);
                    launchGemmPV(valid_item_batch_size_que[idx_1], valid_user_kv_seq_size_que[idx_1],
                                 item_batch_idx_que[idx_1], head_num_idx_que[idx_1], user_kv_seq_idx_que[idx_1],
                                 gemm_pv_p_gm_, gemm_pv_res_gm_);
                }

                if (loop_cnt > 1) {
                    processAggr(valid_item_batch_size_que[idx_2], head_num_idx_que[idx_2], item_batch_idx_que[idx_2],
                                aggr_in_gm_, aggr_exp_max_buf_, aggr_exp_sum_buf_, is_kv_first_block_que[idx_2],
                                is_kv_last_block_que[idx_2]);
                }
                loop_cnt++;
            }
        }
    }

private:
    __aicore__ inline void launchGemmQK(const uint32_t& item_batch_size, const uint32_t& user_seq_size,
                                        const uint32_t& item_batch_idx, const uint32_t& head_num_idx,
                                        const uint32_t& kv_seq_idx, GlobalTensor<CalcT> gemm_qk_res_gm)
    {
        uint32_t query_item_offset = item_batch_idx * tiling_.shape_info.item_batch_stride +
                                     head_num_idx * tiling_.shape_info.item_head_num_stride;
        uint32_t key_user_offset =
            kv_seq_idx * tiling_.shape_info.user_seq_stride + head_num_idx * tiling_.shape_info.user_head_num_stride;
        gemm_qk_.SetTail(item_batch_size, user_seq_size, tiling_.shape_info.head_dim);
        gemm_qk_.SetTensorA(query_gm_[query_item_offset]);
        gemm_qk_.SetTensorB(key_user_gm_[key_user_offset], true);
        gemm_qk_.template IterateAll<false>(gemm_qk_res_gm, 0, false, true);
    }

    __aicore__ inline void waitGemmQK()
    {
        gemm_qk_.WaitIterateAll();
        gemm_qk_.End();
    }

    __aicore__ inline void launchGemmPV(const uint32_t& item_batch_size, const uint32_t& user_seq_size,
                                        const uint32_t& item_batch_idx, const uint32_t& head_num_idx,
                                        const uint32_t& kv_seq_idx, GlobalTensor<InputT> gemm_score_gm,
                                        GlobalTensor<CalcT> gemm_pv_res_gm)
    {
        uint32_t value_user_offset =
            kv_seq_idx * tiling_.shape_info.user_seq_stride + head_num_idx * tiling_.shape_info.user_head_num_stride;
        gemm_pv_.SetTail(item_batch_size, tiling_.shape_info.head_dim, user_seq_size);
        gemm_pv_.SetTensorA(gemm_score_gm);
        gemm_pv_.SetTensorB(value_user_gm_[value_user_offset]);
        gemm_pv_.template IterateAll<false>(gemm_pv_res_gm, 0, false, true);
    }

    __aicore__ inline void waitGemmPV()
    {
        gemm_pv_.WaitIterateAll();
        gemm_pv_.End();
    }

    __aicore__ inline void processItemAttentionQK(const uint32_t& item_batch_size, const uint32_t& head_num_idx,
                                                  const uint32_t& item_batch_idx, GlobalTensor<CalcT> gemm_qk_res_gm)
    {
        uint32_t item_offset = item_batch_idx * tiling_.shape_info.item_batch_stride +
                               head_num_idx * tiling_.shape_info.item_head_num_stride;
        LocalTensor<InputT> query_item_local = comm_in_que_.AllocTensor<InputT>();
        LocalTensor<InputT> kv_item_local = aggr_in_que_.AllocTensor<InputT>();
        uint32_t qk_input_size = item_batch_size * tiling_.shape_info.head_dim;
        LocalTensor<CalcT> query_item_calc;
        LocalTensor<CalcT> key_item_calc;
        if constexpr (!IsSameType<InputT, float>::value) {
            uint32_t src_type_size = item_batch_size * tiling_.shape_info.head_dim * sizeof(InputT);
            copyGmToUb(query_item_local[src_type_size], query_gm_[item_offset], item_batch_size,
                       tiling_.shape_info.head_dim, tiling_.shape_info.item_batch_stride - tiling_.shape_info.head_dim,
                       0);
            comm_in_que_.EnQue(query_item_local);
            query_item_local = comm_in_que_.DeQue<InputT>();
            query_item_calc = query_item_local.template ReinterpretCast<CalcT>();
            Cast(query_item_calc, query_item_local[src_type_size], RoundMode::CAST_NONE, qk_input_size);
            PipeBarrier<PIPE_V>();
            copyGmToUb(kv_item_local[src_type_size], key_item_gm_[item_offset], item_batch_size,
                       tiling_.shape_info.head_dim, tiling_.shape_info.item_batch_stride - tiling_.shape_info.head_dim,
                       0);
            aggr_in_que_.EnQue(kv_item_local);
            kv_item_local = aggr_in_que_.DeQue<InputT>();
            key_item_calc = kv_item_local.template ReinterpretCast<CalcT>();
            Cast(key_item_calc, kv_item_local[src_type_size], RoundMode::CAST_NONE, qk_input_size);
            PipeBarrier<PIPE_V>();
        } else {
            copyGmToUb(query_item_local, query_gm_[item_offset], item_batch_size, tiling_.shape_info.head_dim,
                       tiling_.shape_info.item_batch_stride - tiling_.shape_info.head_dim, 0);
            copyGmToUb(kv_item_local, key_item_gm_[item_offset], item_batch_size, tiling_.shape_info.head_dim,
                       tiling_.shape_info.item_batch_stride - tiling_.shape_info.head_dim, 0);
            comm_in_que_.EnQue(query_item_local);
            aggr_in_que_.EnQue(kv_item_local);
            query_item_calc = comm_in_que_.DeQue<CalcT>();
            key_item_calc = aggr_in_que_.DeQue<CalcT>();
        }
        LocalTensor<CalcT> vec_tmp = vec_tmp_buf_.template Get<CalcT>();
        // 使用vector计算qk matmul计算
        Mul(vec_tmp, query_item_calc, key_item_calc, qk_input_size);
        PipeBarrier<PIPE_V>();
        comm_in_que_.FreeTensor(query_item_calc);
        aggr_in_que_.FreeTensor(key_item_calc);
        LocalTensor<CalcT> qk_out = comm_out_que_.AllocTensor<CalcT>();
        // 对行方向执行ReduceSum，完成reduce计算
        WholeReduceSum(qk_out, vec_tmp, tiling_.shape_info.head_dim, item_batch_size, 1, 1,
                       tiling_.shape_info.head_dim * sizeof(CalcT) / UB_BLOCK_BYTES);
        PipeBarrier<PIPE_V>();
        // 拷出
        copyUbToGm(gemm_qk_res_gm, qk_out, 1, item_batch_size, 0, 0);
        comm_out_que_.FreeTensor(qk_out);
    }

    __aicore__ inline void processItemAttentionSoftmaxAndSV(
        const uint32_t& item_batch_size, const uint32_t& head_num_idx, const uint32_t& item_batch_idx,
        const GlobalTensor<CalcT>& softmax_in_gm, const GlobalTensor<CalcT>& gemm_pv_res_gm, LocalTensor<CalcT>& max_ub,
        LocalTensor<CalcT>& exp_max_ub, LocalTensor<CalcT>& exp_sum_ub)
    {
        LocalTensor<CalcT> softmax_in_ub = comm_in_que_.AllocTensor<CalcT>();
        copyGmToUb(softmax_in_ub, softmax_in_gm, 1, item_batch_size, 0, 0);
        comm_in_que_.EnQue(softmax_in_ub);
        softmax_in_ub = comm_in_que_.DeQue<CalcT>();
        LocalTensor<CalcT> vec_cal_ub = vec_tmp_buf_.Get<CalcT>();
        Muls(vec_cal_ub, softmax_in_ub, static_cast<CalcT>(tiling_.score_scale), item_batch_size);
        for (int32_t i = 0; i < item_batch_size; i++) {
            Duplicate(max_ub[i * ELM_NUM_PER_BLOCK_FLOAT], vec_cal_ub.GetValue(i), ELM_NUM_PER_BLOCK_FLOAT);
        }
        Duplicate(exp_sum_ub, float(1.0), item_batch_size * ELM_NUM_PER_BLOCK_FLOAT);
        uint32_t item_offset = item_batch_idx * tiling_.shape_info.item_batch_stride +
                               head_num_idx * tiling_.shape_info.item_head_num_stride;
        LocalTensor<InputT> aggr_in_ub = aggr_in_que_.AllocTensor<InputT>();
        LocalTensor<CalcT> aggr_out_ub = comm_out_que_.AllocTensor<CalcT>();
        copyGmToUb(aggr_in_ub, value_item_gm_[item_offset], item_batch_size, tiling_.shape_info.head_dim,
                   tiling_.shape_info.item_batch_stride - tiling_.shape_info.head_dim, 0);
        aggr_in_que_.EnQue(aggr_in_ub);
        aggr_in_ub = aggr_in_que_.DeQue<InputT>();
        if (IsSameType<InputT, float>::value) {
            auto aggr_float_ub = aggr_out_ub.template ReinterpretCast<InputT>();
            DataCopy(aggr_float_ub, aggr_in_ub, item_batch_size * tiling_.shape_info.head_dim);
        } else {
            Cast(aggr_out_ub, aggr_in_ub, RoundMode::CAST_NONE, item_batch_size * tiling_.shape_info.head_dim);
        }
        comm_out_que_.EnQue(aggr_out_ub);
        aggr_out_ub = comm_out_que_.DeQue<CalcT>();
        DataCopy(gemm_pv_res_gm, aggr_out_ub, item_batch_size * tiling_.shape_info.head_dim);
        comm_in_que_.FreeTensor(softmax_in_ub);
        aggr_in_que_.FreeTensor(aggr_in_ub);
        comm_out_que_.FreeTensor(aggr_out_ub);
    }

    __aicore__ inline void processSoftmax(const uint32_t& item_q_block_size, const uint32_t& user_kv_block_size,
                                          const GlobalTensor<CalcT>& softmax_input_gm,
                                          const GlobalTensor<InputT>& softmax_output_gm, LocalTensor<CalcT>& exp_max_ub,
                                          LocalTensor<CalcT>& max_ub, const LocalTensor<CalcT>& in_exp_sum_ub,
                                          LocalTensor<CalcT>& exp_sum_ub, const bool& is_update)
    {
        uint32_t vec_element_num = UB_BLOCK_BYTES / sizeof(CalcT);
        uint32_t vec_q_block_size = tiling_.vec_softmax_batch_block;
        uint32_t loop_times = DivCeil(item_q_block_size, vec_q_block_size);
        uint32_t align_kv_block_size = user_kv_block_size;
        LocalTensor<CalcT> vec_cal_ub = vec_tmp_buf_.Get<CalcT>();
        if (user_kv_block_size < tiling_.user_kv_seq_block) {
            align_kv_block_size = DivCeil(user_kv_block_size, vec_element_num) * vec_element_num;
        }
        uint32_t gm_offset = vec_q_block_size * tiling_.user_kv_seq_block;
        uint32_t reduce_res_ub_stride = vec_q_block_size * vec_element_num;
        auto softmax_api_tmp_buf = softmax_tmp_buf_.Get<uint8_t>();
        for (int32_t i = 0; i < loop_times; i++) {
            uint32_t vec_q_real_size = min(vec_q_block_size, item_q_block_size - i * vec_q_block_size);
            LocalTensor<CalcT> softmax_in_ub = comm_in_que_.AllocTensor<CalcT>();
            copyGmToUb(softmax_in_ub, softmax_input_gm[i * gm_offset], vec_q_real_size, user_kv_block_size,
                       tiling_.user_kv_seq_block - user_kv_block_size, 0);
            comm_in_que_.EnQue(softmax_in_ub);
            softmax_in_ub = comm_in_que_.DeQue<CalcT>();
            Muls(vec_cal_ub, softmax_in_ub, static_cast<CalcT>(tiling_.score_scale),
                 vec_q_real_size * align_kv_block_size);
            PipeBarrier<PIPE_V>();
            comm_in_que_.FreeTensor(softmax_in_ub);

            uint32_t softmax_shape[2] = {vec_q_real_size, align_kv_block_size};
            uint32_t reduce_shape[2] = {vec_q_real_size, vec_element_num};
            vec_cal_ub.SetShapeInfo(AscendC::ShapeInfo(2, softmax_shape, AscendC::DataFormat::ND));
            auto inner_max_ub = max_ub[reduce_res_ub_stride * i];
            auto inner_exp_max_ub = exp_max_ub[reduce_res_ub_stride * i];
            auto inner_in_exp_sum_ub = in_exp_sum_ub[reduce_res_ub_stride * i];
            auto inner_exp_sum_ub = exp_sum_ub[reduce_res_ub_stride * i];
            inner_max_ub.SetShapeInfo(AscendC::ShapeInfo(2, reduce_shape, AscendC::DataFormat::ND));
            inner_exp_max_ub.SetShapeInfo(AscendC::ShapeInfo(2, reduce_shape, AscendC::DataFormat::ND));
            inner_in_exp_sum_ub.SetShapeInfo(AscendC::ShapeInfo(2, reduce_shape, AscendC::DataFormat::ND));
            inner_exp_sum_ub.SetShapeInfo(AscendC::ShapeInfo(2, reduce_shape, AscendC::DataFormat::ND));
            SoftMaxShapeInfo src_shape = {vec_q_real_size, align_kv_block_size, vec_q_real_size, align_kv_block_size};
            if (vec_q_real_size % 8 == 0 && align_kv_block_size % 64 == 0) {
                if (is_update) {
                    SoftmaxFlashV2<CalcT, true, true, true>(vec_cal_ub, inner_exp_sum_ub, max_ub, vec_cal_ub,
                                                            exp_max_ub, inner_in_exp_sum_ub, max_ub,
                                                            softmax_api_tmp_buf, tiling_.softmax_tiling, src_shape);
                } else {
                    SoftmaxFlashV2<CalcT, false, true, true>(vec_cal_ub, inner_exp_sum_ub, max_ub, vec_cal_ub,
                                                             exp_max_ub, inner_in_exp_sum_ub, max_ub,
                                                             softmax_api_tmp_buf, tiling_.softmax_tiling, src_shape);
                }
            } else {
                if (is_update) {
                    SoftmaxFlashV2<CalcT, true, false, false>(vec_cal_ub, inner_exp_sum_ub, max_ub, vec_cal_ub,
                                                              exp_max_ub, inner_in_exp_sum_ub, max_ub,
                                                              softmax_api_tmp_buf, tiling_.softmax_tiling, src_shape);
                } else {
                    SoftmaxFlashV2<CalcT, false, false, false>(vec_cal_ub, inner_exp_sum_ub, max_ub, vec_cal_ub,
                                                               exp_max_ub, inner_in_exp_sum_ub, max_ub,
                                                               softmax_api_tmp_buf, tiling_.softmax_tiling, src_shape);
                }
            }
            LocalTensor<InputT> softmax_out_ub = comm_out_que_.AllocTensor<InputT>();
            if (IsSameType<InputT, float>::value) {
                auto softmax_res_ub = softmax_out_ub.template ReinterpretCast<CalcT>();
                DataCopy(softmax_res_ub, vec_cal_ub, vec_q_real_size * align_kv_block_size);
            } else {
                Cast(softmax_out_ub, vec_cal_ub, RoundMode::CAST_RINT, vec_q_real_size * align_kv_block_size);
            }
            comm_out_que_.EnQue(softmax_out_ub);
            softmax_out_ub = comm_out_que_.DeQue<InputT>();
            auto curr_softmax_output_gm = softmax_output_gm[gm_offset * i];
            copyUbToGm(curr_softmax_output_gm, softmax_out_ub, vec_q_real_size, user_kv_block_size,
                       align_kv_block_size - user_kv_block_size, tiling_.user_kv_seq_block - user_kv_block_size);
            comm_out_que_.FreeTensor(softmax_out_ub);
        }
    }

    __aicore__ inline void processAggr(const uint32_t& item_q_block_size, const uint32_t& head_num_idx,
                                       const uint32_t& o_batch_idx, const AscendC::GlobalTensor<CalcT>& gemm_sv_gm,
                                       const AscendC::LocalTensor<CalcT>& exp_max_ub,
                                       const AscendC::LocalTensor<CalcT>& exp_sum_ub, const bool& is_first_block,
                                       const bool& is_last_block)
    {
        uint32_t vec_block_elements = UB_BLOCK_BYTES / sizeof(CalcT);
        uint32_t vec_repeat_max_elements = VEC_MAX_REPEAT_BYTES / sizeof(CalcT);
        uint32_t vec_aggr_block_size = tiling_.vec_aggr_batch_block;
        uint32_t sv_res_gm_stride = vec_aggr_block_size * tiling_.shape_info.head_dim;
        uint32_t reduce_res_ub_stride = vec_aggr_block_size * vec_block_elements;
        uint32_t loop_times = DivCeil(item_q_block_size, vec_aggr_block_size);
        uint32_t head_num_repeat_times = tiling_.shape_info.head_dim / vec_repeat_max_elements;
        uint32_t head_num_tail = tiling_.shape_info.head_dim % vec_repeat_max_elements;
        uint32_t tail_offset = head_num_repeat_times * vec_repeat_max_elements;

        for (int32_t i = 0; i < loop_times; i++) {
            uint32_t vec_q_real_size = min(vec_aggr_block_size, item_q_block_size - i * vec_aggr_block_size);
            LocalTensor<CalcT> last_aggr_res_ub;
            if (!is_first_block) {
                // 搬入aggr结果
                last_aggr_res_ub = comm_in_que_.AllocTensor<CalcT>();
                DataCopy(last_aggr_res_ub, gemm_aggr_gm_[sv_res_gm_stride * i],
                         vec_q_real_size * tiling_.shape_info.head_dim);
                comm_in_que_.EnQue(last_aggr_res_ub);
                last_aggr_res_ub = comm_in_que_.DeQue<CalcT>();
            }

            // 搬入当前gemm_sv结果
            LocalTensor<CalcT> gemm_sv_res_ub = aggr_in_que_.AllocTensor<CalcT>();
            DataCopy(gemm_sv_res_ub, gemm_sv_gm[sv_res_gm_stride * i], vec_q_real_size * tiling_.shape_info.head_dim);
            aggr_in_que_.EnQue(gemm_sv_res_ub);
            gemm_sv_res_ub = aggr_in_que_.DeQue<CalcT>();
            auto inner_exp_max_ub = exp_max_ub[i * reduce_res_ub_stride];
            auto inner_exp_sum_ub = exp_sum_ub[i * reduce_res_ub_stride];
            auto aggr_calc_out_ub = comm_out_que_.AllocTensor<CalcT>();  // aggr result
            auto aggr_input_out_ub = aggr_calc_out_ub.template ReinterpretCast<InputT>();
            LocalTensor<CalcT> vec_cal_ub = vec_tmp_buf_.Get<CalcT>();

            if (is_first_block) {
                DataCopy(vec_cal_ub, gemm_sv_res_ub, vec_q_real_size * tiling_.shape_info.head_dim);
            } else {
                // 执行更新操作
                AscendC::BinaryRepeatParams repeat_params;
                repeat_params.src0BlkStride = 1;
                repeat_params.src0RepStride = tiling_.shape_info.head_dim / vec_block_elements;
                repeat_params.src1BlkStride = 0;
                repeat_params.src1RepStride = 1;
                repeat_params.dstRepStride = tiling_.shape_info.head_dim / vec_block_elements;
                for (int32_t ii = 0; ii < head_num_repeat_times; ++ii) {
                    Mul(last_aggr_res_ub[ii * vec_repeat_max_elements], last_aggr_res_ub[ii * vec_repeat_max_elements],
                        exp_max_ub, vec_repeat_max_elements, vec_q_real_size, repeat_params);
                }
                if (head_num_tail != 0) {
                    Mul(last_aggr_res_ub[tail_offset], last_aggr_res_ub[tail_offset], exp_max_ub, head_num_tail,
                        vec_q_real_size, repeat_params);
                }
                PipeBarrier<PIPE_V>();
                Add(vec_cal_ub, last_aggr_res_ub, gemm_sv_res_ub, vec_q_real_size * tiling_.shape_info.head_dim);
                comm_in_que_.FreeTensor(last_aggr_res_ub);
            }
            aggr_in_que_.FreeTensor(gemm_sv_res_ub);

            if (is_last_block) {
                AscendC::BinaryRepeatParams repeat_params;
                repeat_params.src0BlkStride = 1;
                repeat_params.src0RepStride = tiling_.shape_info.head_dim / vec_block_elements;
                repeat_params.src1BlkStride = 0;
                repeat_params.src1RepStride = 1;
                repeat_params.dstRepStride = tiling_.shape_info.head_dim / vec_block_elements;
                for (int32_t ii = 0; ii < head_num_repeat_times; ++ii) {
                    Div(vec_cal_ub[ii * vec_repeat_max_elements], vec_cal_ub[ii * vec_repeat_max_elements],
                        inner_exp_sum_ub, vec_repeat_max_elements, vec_q_real_size, repeat_params);
                }
                if (head_num_tail != 0) {
                    Div(vec_cal_ub[tail_offset], vec_cal_ub[tail_offset], inner_exp_sum_ub, head_num_tail,
                        vec_q_real_size, repeat_params);
                }
                PipeBarrier<PIPE_V>();
                if (IsSameType<InputT, float>::value) {
                    auto aggr_input_res_ub = aggr_input_out_ub.template ReinterpretCast<CalcT>();
                    DataCopy(aggr_input_res_ub, vec_cal_ub, vec_q_real_size * tiling_.shape_info.head_dim);
                } else {
                    Cast(aggr_input_out_ub, vec_cal_ub, RoundMode::CAST_RINT,
                         vec_q_real_size * tiling_.shape_info.head_dim);
                }
                comm_out_que_.EnQue(aggr_input_out_ub);
                aggr_input_out_ub = comm_out_que_.DeQue<InputT>();
                // 搬出到GM

                uint32_t o_gm_offset = o_batch_idx * tiling_.shape_info.item_batch_stride +
                                       head_num_idx * tiling_.shape_info.item_head_num_stride +
                                       i * vec_aggr_block_size * tiling_.shape_info.item_batch_stride;
                auto curr_attn_out_gm = attn_out_gm_[o_gm_offset];
                copyUbToGm(curr_attn_out_gm, aggr_input_out_ub, vec_q_real_size, tiling_.shape_info.head_dim, 0,
                           tiling_.shape_info.item_batch_stride - tiling_.shape_info.head_dim);
                comm_out_que_.FreeTensor(aggr_input_out_ub);
            } else {
                DataCopy(aggr_calc_out_ub, vec_cal_ub, vec_q_real_size * tiling_.shape_info.head_dim);
                comm_out_que_.EnQue(aggr_calc_out_ub);
                aggr_calc_out_ub = comm_out_que_.DeQue<CalcT>();
                DataCopy(gemm_aggr_gm_, aggr_calc_out_ub, vec_q_real_size * tiling_.shape_info.head_dim);
                comm_out_que_.FreeTensor(aggr_calc_out_ub);
            }
        }
    }

    __aicore__ inline void calcIndexIndices(const uint32_t& i_task, uint32_t& item_batch_idx, uint32_t& head_num_idx)
    {
        head_num_idx = i_task / tiling_.item_batch_block_num;
        item_batch_idx = (i_task % tiling_.item_batch_block_num) * tiling_.item_batch_block;
    }

    template <typename T>
    __aicore__ inline void copyGmToUb(LocalTensor<T> ub_dst, GlobalTensor<T> gm_src, uint32_t block_count,
                                      uint32_t block_len, uint32_t src_stride, uint32_t dst_stride)
    {
        uint32_t block_elem_num = UB_BLOCK_BYTES / sizeof(T);
        if (block_len % block_elem_num == 0 && src_stride % block_elem_num == 0 && dst_stride % block_elem_num == 0) {
            DataCopyParams copyInParams;
            copyInParams.blockCount = static_cast<uint16_t>(block_count);
            copyInParams.blockLen = static_cast<uint16_t>(block_len / block_elem_num);
            copyInParams.srcStride = static_cast<uint16_t>(src_stride / block_elem_num);
            copyInParams.dstStride = static_cast<uint16_t>(dst_stride / block_elem_num);
            DataCopy(ub_dst, gm_src, copyInParams);
        } else {
            uint32_t align_block_len = DivCeil(block_len, block_elem_num) * block_elem_num;
            DataCopyExtParams copyInParams;
            DataCopyPadExtParams<T> copyInPadParams;
            copyInParams.blockCount = static_cast<uint16_t>(block_count);
            copyInParams.blockLen = block_len * sizeof(T);
            copyInParams.srcStride = src_stride * sizeof(T);
            copyInParams.dstStride = dst_stride / block_elem_num;
            copyInPadParams.isPad = true;
            copyInPadParams.leftPadding = 0;
            copyInPadParams.rightPadding = align_block_len - block_len;
            copyInPadParams.paddingValue = 0;
            DataCopyPad(ub_dst, gm_src, copyInParams, copyInPadParams);
        }
    }

    template <typename T>
    __aicore__ inline void copyUbToGm(GlobalTensor<T> gm_dst, LocalTensor<T> ub_src, uint32_t block_count,
                                      uint32_t block_len, uint32_t src_stride, uint32_t dst_stride)
    {
        uint32_t block_elem_num = UB_BLOCK_BYTES / sizeof(T);
        if (block_len % block_elem_num == 0 && src_stride % block_elem_num == 0 && dst_stride % block_elem_num == 0) {
            DataCopyParams copyInParams;
            copyInParams.blockCount = static_cast<uint16_t>(block_count);
            copyInParams.blockLen = static_cast<uint16_t>(block_len / block_elem_num);
            copyInParams.srcStride = static_cast<uint16_t>(src_stride / block_elem_num);
            copyInParams.dstStride = static_cast<uint16_t>(dst_stride / block_elem_num);
            DataCopy(gm_dst, ub_src, copyInParams);
        } else {
            uint32_t align_block_len = DivCeil(block_len, block_elem_num) * block_elem_num;
            DataCopyExtParams copyOutParams;
            copyOutParams.blockCount = static_cast<uint16_t>(block_count);
            copyOutParams.blockLen = block_len * sizeof(T);
            copyOutParams.srcStride = src_stride / block_elem_num;
            copyOutParams.dstStride = dst_stride * sizeof(T);
            copyOutParams.rsv = 0;
            DataCopyPad(gm_dst, ub_src, copyOutParams);
        }
    }

    __aicore__ inline void flipPingPong(const uint32_t& loop_cnt)
    {
        if ((loop_cnt & 1) == 0) {
            gemm_qk_res_gm_ = gemm_qk_res_ping_gm_;
            softmax_in_gm_ = gemm_qk_res_pong_gm_;
            softmax_out_gm_ = softmax_res_pong_gm_;
            softmax_exp_max_buf_ = exp_max_pong_buf_.template Get<CalcT>();
            softmax_in_exp_sum_buf_ = exp_sum_ping_buf_.template Get<CalcT>();
            softmax_exp_sum_buf_ = exp_sum_pong_buf_.template Get<CalcT>();
            gemm_pv_p_gm_ = softmax_res_pong_gm_;
            gemm_pv_res_gm_ = gemm_pv_res_pong_gm_;
            aggr_in_gm_ = gemm_pv_res_ping_gm_;
            aggr_exp_max_buf_ = exp_max_ping_buf_.template Get<CalcT>();
            aggr_exp_sum_buf_ = exp_sum_ping_buf_.template Get<CalcT>();
        } else {
            gemm_qk_res_gm_ = gemm_qk_res_pong_gm_;
            softmax_in_gm_ = gemm_qk_res_ping_gm_;
            softmax_out_gm_ = softmax_res_ping_gm_;
            softmax_exp_max_buf_ = exp_max_ping_buf_.template Get<CalcT>();
            softmax_in_exp_sum_buf_ = exp_sum_pong_buf_.template Get<CalcT>();
            softmax_exp_sum_buf_ = exp_sum_ping_buf_.template Get<CalcT>();
            gemm_pv_p_gm_ = softmax_res_ping_gm_;
            gemm_pv_res_gm_ = gemm_pv_res_ping_gm_;
            aggr_in_gm_ = gemm_pv_res_pong_gm_;
            aggr_exp_max_buf_ = exp_max_pong_buf_.template Get<CalcT>();
            aggr_exp_sum_buf_ = exp_sum_pong_buf_.template Get<CalcT>();
        }
    }
};
}  // namespace kernels
