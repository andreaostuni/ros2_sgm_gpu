// Copyright 2020 Andrea Ostuni
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <https://www.gnu.org/licenses/>.

#ifndef SGM_GPU__DISPARITY_TO_DEPTH_HPP_
#define SGM_GPU__DISPARITY_TO_DEPTH_HPP_

#include <stdint.h>
// max depth evaluated
// #define bigZ FLT_MAX

namespace sgm_gpu
{

__global__ void disparityToDepth(const uint8_t* __restrict__ d_input, float* __restrict__ d_out, const float Tx,
                                 uint32_t rows, uint32_t cols, const float delta_cx);

}  // namespace sgm_gpu
#endif  // SGM_GPU__MEDIAN_FILTER_HPP_
