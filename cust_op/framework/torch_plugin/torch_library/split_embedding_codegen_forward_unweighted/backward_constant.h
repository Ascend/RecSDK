/**
* @file backward_constant.h
*
* Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
* Copyright (c) Meta Platforms, Inc. and affiliates.
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/

#ifndef MXREC_BACKWARD_CONSTANT_H
#define MXREC_BACKWARD_CONSTANT_H

// for FGBGEMM 1.2.0 param index and size.
namespace optim_param_idx_fbgemm_120 {
// weight: dev_weights uvm_weights weights_placements weights_offsets lxu_cache_weights
// used by Adam, AdaGrad, SGD.
constexpr size_t WEIGHTS_SIZE = 5;
constexpr int64_t DEV_WEIGHTS_INDEX = 0;
constexpr int64_t UVM_WEIGHTS_INDEX = 1;
constexpr int64_t WEIGHTS_PLACEMENTS_INDEX = 2;
constexpr int64_t WEIGHTS_OFFSETS_INDEX = 3;
constexpr int64_t LXU_CACHE_WEIGHTS_INDEX = 4;

// aux_tensor: B_offsets, vbe_output_offsets_feature_rank, vbe_B_offsets_rank_per_feature, lxu_cache_locations,
//             uvm_cache_stats, prev_iter_dev
// used by Adam, AdaGrad, SGD.
constexpr size_t AUX_TENSOR_SIZE = 6;
constexpr int64_t LXU_CACHE_LOCATIONS_INDEX = 3;
constexpr int64_t UVM_CACHE_STATS_INDEX = 4;

// aux_bool: is_experimental_tbe, use_uniq_cache_locations_bwd, use_homogeneous_placements, apply_global_weight_decay,
//           gradient_clipping, stochastic_rounding, mixed_D
// used by Adam, AdaGrad, SGD.
constexpr size_t AUX_BOOL_SIZE = 7;
constexpr int64_t IS_EXPERIMENTAL_TBE_INDEX = 0;
constexpr int64_t USE_UNIQ_CACHE_LOCATIONS_BWD_INDEX = 1;
constexpr int64_t USE_HOMOGENEOUS_PLACEMENTS_INDEX = 2;
constexpr int64_t GRADIENT_CLIPPING_INDEX = 4;
constexpr int64_t STOCHASTIC_ROUNDING_INDEX = 5;

// aux_float: gwd_lower_bound, max_gradient
// used by Adam, AdaGrad, SGD.
constexpr size_t AUX_FLOAT_SIZE = 2;
constexpr int64_t MAX_GRADIENT_INDEX = 1;

// momentum1: momentum1.dev, momentum1.uvm, momentum1.placements, momentum1.offsets
// used by Adam, AdaGrad.
constexpr size_t MOMENTUM1_SIZE = 4;
constexpr int64_t MOMENTUM1_DEV_INDEX = 0;
constexpr int64_t MOMENTUM1_UVM_INDEX = 1;
constexpr int64_t MOMENTUM1_PLACEMENTS_INDEX = 2;
constexpr int64_t MOMENTUM1_OFFSETS_INDEX = 3;

// optim_float: eps, beta1, beta2, weight_decay
// used by Adam.
constexpr size_t OPTIM_FLOAT_SIZE = 4;
constexpr int64_t EPS_INDEX = 0;
constexpr int64_t BETA1_INDEX = 1;
constexpr int64_t BETA2_INDEX = 2;

// aux_int: iter, info_B_num_bits, info_B_mask
// used by Adam.
constexpr size_t AUX_INT_SIZE = 3;
constexpr int64_t ITER_INDEX = 1;
}

#endif  // MXREC_BACKWARD_CONSTANT_H
