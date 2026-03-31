extern "C" __global__ void spai_0_threads_black_kernel(
                                             int Nx,
                                             int Ny,
                                             const float* __restrict__ val,
                                             const int* __restrict__ rowptr,
                                             const int* __restrict__ colind,
                                             float* __restrict__ x,
                                             const float* __restrict__ b,
                                             const float* __restrict__ spai_diagonal)
{
    // parity (i+j) & 1 == 0
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= Nx || j >= Ny) return;

    if ( ( (i + j) & 1 ) != 0)
    {

        const int index = i * Ny + j;

        const int start = rowptr[index];
        const int end = rowptr[index + 1];

        float A_times_x = 0.0;

        //thread per row
        for (int k = start; k < end; k++)
            A_times_x += val[k] * x[colind[k]];

        float r = b[index] - A_times_x;
        x[index] += 1.2f*spai_diagonal[index] * r;
    }
}