extern "C" __global__ void prolongation_threads_kernel(int n,
                                             const float* val,
                                             const int* rowptr,
                                             const int* colind,
                                             const float* x_coarse,
                                             float* x)
{
    const int index = (blockIdx.x * blockDim.x + threadIdx.x);

    if (index >= n) return;

    const int start = rowptr[index];
    const int end = rowptr[index + 1];

    float I_times_x = 0.0;

    //thread per row
    for (int j = start; j < end; j++)
        I_times_x += val[j] * x_coarse[colind[j]];

    x[index] = x[index] + I_times_x;
}