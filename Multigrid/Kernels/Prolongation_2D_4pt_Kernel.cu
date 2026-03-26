extern "C" __global__ void Prolongation_2D_NyNx_4pt(
                                                        float* __restrict__ fine,
                                                        const float* __restrict__ coarse,
                                                        int Nx_2h, int Ny_2h,
                                                        int Nx, int Ny)
{
    int ic = blockIdx.x * blockDim.x + threadIdx.x;
    int jc = blockIdx.y * blockDim.y + threadIdx.y;

    if (ic >= Nx_2h || jc >= Ny_2h) return;

    int i = 2 * ic;
    int j = 2 * jc;

    // kron(stencil_x,stencil_y)
    const float w[16] = {0.0156f,0.0469f,0.0469f,0.0156f,
                         0.0469f,0.1406f,0.1406f,0.0469f,
                         0.0469f,0.1406f,0.1406f,0.0469f,
                         0.0156f,0.0469f,0.0469f,0.0156f};

    //coarse value
    const float val = coarse[ic*Ny_2h + jc];

    #pragma unroll
    for (int dx = 0; dx < 4; dx++)
    {
        #pragma unroll
         for (int dy = 0; dy < 4; dy++)
         {
            // atomic required, race condition
            atomicAdd(&fine[(i+dx)*Ny + j + dy],w[dx*4 +dy] * val);

         }
    }

}