extern "C" __global__ void residual_warp_kernel(int n,
                                             const float* val,
                                             const int* rowptr,
                                             const int* colind,
                                             const float* x,
                                             const float* b,
                                             float* r)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    if (warp >= n) return;

    const int row = warp;
    const int start = rowptr[row];
    const int end = rowptr[row + 1];

    float sum = 0.0;

    //each 'lane' computes entries spaced by the warp size
    for (int j = start + lane; j < end; j += 32)
        sum += val[j] * x[colind[j]];

    //warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff,sum,offset);

   // write back r = b - Ax
    if (lane == 0)
        r[row] = b[row] - sum;

}