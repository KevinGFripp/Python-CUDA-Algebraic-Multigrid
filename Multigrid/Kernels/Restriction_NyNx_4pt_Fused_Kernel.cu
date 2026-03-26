extern "C" __global__ void restriction_2d_convolution_4pt_fused(
    const float* __restrict__ fine,
    float* __restrict__ coarse,
    int Nx, int Ny,
    int Nx_2h, int Ny_2h)
{
    // Nx and Ny directions use 4pt stencil
    int tx = threadIdx.y;
    int ty = threadIdx.x;

    int ic = blockIdx.y * blockDim.y + tx;
    int jc = blockIdx.x * blockDim.x + ty;

    if (ic >= Nx_2h || jc >= Ny_2h) return;

    // Fine indices
    int i = 2 * ic;
    int j = 2 * jc;

    // Shared memory tile
    extern __shared__ float tile[];

    // halo of stencil length - 1
    int tile_w = 2 * blockDim.y + 3;
    int tile_h = 2 * blockDim.x + 3;

    // Map thread to shared memory
    int si = 2 * tx;
    int sj = 2 * ty;

    // Global index helper
    auto idx = [&](int x, int y) { return x * Ny + y; };

    // shared index helper
    auto s_idx = [&](int x, int y) { return x * tile_h + y; };

    // coarse index helper
    auto c_idx = [&](int x, int y) { return x * Ny_2h + y; };

    // Load 4x4 region per coarse point (cooperatively)
    #pragma unroll
    for (int dx = 0; dx < 4; dx++) {
    #pragma unroll
        for (int dy = 0; dy < 4; dy++) {

            // global indexes
            int gx = i + dx;
            int gy = j + dy;
            // shared indexes
            int sx = si + dx;
            int sy = sj + dy;

            // catch out of bounds
            if (gx < Nx && gy < Ny)
                tile[s_idx(sx,sy)] = fine[idx(gx, gy)];
            else
                tile[s_idx(sx,sy)] = 0.0f;
        }
    }

    __syncthreads();

    // Apply 4x4 stencil
    const float w[4] = {0.125f,0.375f,0.375f,0.125f};

    float val = 0.0f;

    #pragma unroll
    for (int dx = 0; dx < 4; dx++) {
        #pragma unroll
        for (int dy = 0; dy < 4; dy++) {
            val += w[dx] * w[dy] * tile[s_idx(si + dx, sj + dy)];
        }
    }

    coarse[c_idx(ic,jc)] = val;
}