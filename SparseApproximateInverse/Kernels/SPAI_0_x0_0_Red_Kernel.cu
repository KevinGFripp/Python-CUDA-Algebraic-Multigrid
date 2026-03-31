extern "C" __global__ void spai_0_x0_0_threads_red_kernel(
                                             int Nx,
                                             int Ny,
                                             float* x,
                                             const float* __restrict__ b,
                                             const float* __restrict__ spai_diagonal)
{
    // parity (i+j) & 1 == 0
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= Nx || j >= Ny) return;

    if ( ( (i + j) & 1 ) == 0)
    {
        const int index = i * Ny + j;

        //thread per element
        x[index] = 1.2f*spai_diagonal[index] * b[index];

    }

}