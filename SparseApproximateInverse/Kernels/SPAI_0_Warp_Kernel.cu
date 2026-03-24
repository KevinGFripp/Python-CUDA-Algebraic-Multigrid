extern "C" __global__ void spai_0_warp_kernel(
                                             int n,
                                             const float* val,
                                             const int* rowptr,
                                             const int* colind,
                                             const float* x,
                                             const float* b,
                                             const float* spai_diagonal,
                                             float* xnew)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;

    if (warp >= n) return;

    const int row = warp;
    const int start = rowptr[row];
    const int end = rowptr[row + 1];

    float A_times_x = 0.0;

    //each 'lane' computes entries spaced by the warp size
    for (int j = start + lane; j < end; j += 32)
        A_times_x += val[j] * x[colind[j]];

    //warp reduce
    #pragma unroll 16
    for (int offset = 16; offset > 0; offset>>=1)
        A_times_x += __shfl_down_sync(0xffffffff,A_times_x,offset);

   // write back xnew = x + M *(b-Ax)
    if (lane == 0)
    {
        float r = b[row] - A_times_x;
        xnew[row] = x[row] + spai_diagonal[row] * r;

    }
}