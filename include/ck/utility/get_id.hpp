// SPDX-License-Identifier: MIT
// Copyright (c) 2018-2023, Advanced Micro Devices, Inc. All rights reserved.

#pragma once

#include "ck/ck.hpp"

namespace ck {

__host__ __device__ constexpr index_t get_warp_size()
{
    // warpSize is no longer constexpr as of rocm 7.0.
    // Use preprocessor conditionals so constexpr contexts still work.
    // CDNA (gfx90x/gfx94x/gfx950) uses wave size 64; RDNA (gfx10x/gfx11x/gfx12x) uses 32.
#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || \
    defined(__gfx942__) || defined(__gfx950__)
    return 64;
#else
    return 32;
#endif
}

__device__ index_t get_thread_local_1d_id() { return threadIdx.x; }

__device__ index_t get_thread_global_1d_id() { return blockIdx.x * blockDim.x + threadIdx.x; }

__device__ index_t get_warp_local_1d_id() { return threadIdx.x / get_warp_size(); }

__device__ index_t get_lane_local_1d_id() { return threadIdx.x % get_warp_size(); }

__device__ index_t get_block_1d_id() { return blockIdx.x; }

__device__ index_t get_grid_size() { return gridDim.x; }

__device__ index_t get_block_size() { return blockDim.x; }

} // namespace ck
