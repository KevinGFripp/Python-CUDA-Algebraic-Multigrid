extern "C" __global__ void spai_0_first_iteration_x0_0(
                                             int n,
                                             const float* b,
                                             const float* spai_diagonal,
                                             float* xnew)
{
    const int index = (blockIdx.x * blockDim.x + threadIdx.x);

    if (index >= n) return;

    //thread per element
    xnew[index] = spai_diagonal[index] * b[index];
}