extern "C" __global__ void prolongation_warp_kernel(int n,
                                             const float* val,
                                             const int* rowptr,
                                             const int* colind,
                                             const float* x_coarse,
                                             float* x)
{
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) % 32;
    const int lane = threadIdx.x & 31;

    if (warp >= n) return;

    const int row = warp;
    const int start = rowptr[row];
    const int end = rowptr[row + 1];

    float I_x = 0.0;

    //each 'lane' computes entries spaced by the warp size
    for (int j = start + lane; j < end; j += 32)
        I_x += val[j] * x_coarse[colind[j]];

    //warp reduce
    #pragma unroll 16
    for (int offset = 16; offset > 0; offset >>= 1)
        I_x += __shfl_down_sync(0xffffffff,I_x,offset);

   // write back
    if (lane == 0)
        x[row] = x[row] + I_x;

}