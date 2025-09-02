#ifndef __REARRANGE_KUNLUN_KERNEL_H__
#define __REARRANGE_KUNLUN_KERNEL_H__

#include "../../../devices/kunlun/kunlun_kernel_common.h"

using namespace device::kunlun::kernel;

/**
 * @brief rearrange kernel function
 * @tparam BLOCK_SIZE the block size of the kernel
 * @tparam T the data type of the input and output tensor
 * @param x the input tensor
 * @param y the output tensor
 * @param shape the shape of the input tensor
 * @param x_stride the stride of the input tensor
 * @param y_stride the stride of the output tensor
 * @param total_size the total size of the input tensor
 */
template <unsigned int BUFF_SIZE, typename Tdata>
__global__ void rearrangeKernel(
    Tdata *y,
    const Tdata *x,
    const void *shape,
    const void *x_stride,
    const void *y_stride,
    uint32_t ndim,
    uint32_t total_size) {

    int cid = core_id();
    int ncores = core_num();
    if (cid >= ncores) {
        return;
    }
    int thread_id = ncores * cluster_id() + cid;
    int nthreads = ncores * cluster_num();

    __local__ Tdata x_local[BUFF_SIZE];
    __local__ _size_t shape_lm[ndim];
    __local__ _ptrdiff_t x_stride_lm[ndim];
    __local__ _ptrdiff_t y_stride_lm[ndim];

    GM2LM_ASYNC(shape, shape_lm, ndim * sizeof(_size_t));
    GM2LM_ASYNC(x_stride, x_stride_lm, ndim * sizeof(_ptrdiff_t));
    GM2LM_ASYNC(y_stride, y_stride_lm, ndim * sizeof(_ptrdiff_t));
    mfence();

    int len_per_loop = min(BUFF_SIZE, roundup_div(total_size, nthreads));

    for (int start = thread_id * len_per_loop; start < total_size; start += nthreads * len_per_loop) {
        int len = min(len_per_loop, total_size - start);
        for (int idx = start; idx < start + len; ++idx) {
            int in_idx = indexToOffset(idx, ndim, shape_lm, x_stride_lm);
            GM2LM_ASYNC(x + in_idx, x_local + idx - start, sizeof(Tdata));
        }
        mfence();
        for (int idx = start; idx < start + len; ++idx) {
            int out_idx = indexToOffset(idx, ndim, shape_lm, y_stride_lm);
            LM2GM_ASYNC(x_local + idx - start, y + out_idx, sizeof(Tdata));
        }
        sync_cluster();
    }
}

#endif
