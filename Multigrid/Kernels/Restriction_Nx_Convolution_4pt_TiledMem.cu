extern "C" __global__ void Restriction_Nx_Convolution_4pt_TiledMem(
                                                      const float* __restrict__ fine,
                                                      float* __restrict__ coarse,
                                                      int Nx, int Nx_2h, int Ny_2h)
{
// stencil striding [Rx_1D[n, 2*n : 2*n + len(stencil)] = stencil for n in range(0,Ny)]
// Full size matrix performing 1D convolution -> Rx_2D = kron(Rx_1D,I_y)

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int ic = blockIdx.x * blockDim.x + tx;
    int j = blockIdx.y * blockDim.y + ty;

    // operating on array which has been reduced in Ny to Ny_2h already
    if ( j >= Ny_2h || ic >= Nx_2h) return;

    // shared memory, column major indexing
    extern __shared__ float tile[];

    int tile_width = blockDim.x * 2 + 3; // halo of size 4pt - 1
    int tile_height = blockDim.y;

    // Map shared index, contiguous in x
    int s_idx = ty * tile_width + tx * 2; //  i_coarse -> 2*i

    // fine grid index i to read
    int i = 2 * ic;

    tile[s_idx]     = (i < Nx)     ? fine[i*Ny_2h + j]     : 0.0f;
    tile[s_idx + 1] = (i + 1 < Nx) ? fine[(i + 1)*Ny_2h + j] : 0.0f;
    tile[s_idx + 2] = (i + 2 < Nx) ? fine[(i + 2)*Ny_2h + j] : 0.0f;
    tile[s_idx + 3] = (i + 3 < Nx) ? fine[(i + 3)*Ny_2h + j] : 0.0f;

    __syncthreads();

    // restriction

    float f0 = (i < Nx)     ?  tile[s_idx]     : 0.0f;
    float f1 = (i + 1 < Nx) ?  tile[s_idx + 1] : 0.0f;
    float f2 = (i + 2 < Nx) ?  tile[s_idx + 2] : 0.0f;
    float f3 = (i + 3 < Nx) ?  tile[s_idx + 3] : 0.0f;

    coarse[ic*Ny_2h + j] = 0.125f * f0 + 0.375f * f1 + 0.375f * f2 + 0.125f * f3;

}