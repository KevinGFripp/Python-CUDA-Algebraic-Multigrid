extern "C" __global__ void residual_threads_kernel(int n,
                                             const float* val,
                                             const int* rowptr,
                                             const int* colind,
                                             const float* x,
                                             const float* b,
                                             float* r)
{
    const int index = (blockIdx.x * blockDim.x + threadIdx.x);

    if (index >= n) return;

    const int start = rowptr[index];
    const int end = rowptr[index + 1];

    float A_times_x = 0.0;

    //thread per row
    for (int j = start; j < end; j++)
        A_times_x += val[j] * x[colind[j]];

    r[index] = b[index] - A_times_x;
}