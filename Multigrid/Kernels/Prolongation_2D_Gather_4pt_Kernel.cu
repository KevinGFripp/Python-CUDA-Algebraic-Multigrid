__device__ inline int ind(int i, int j, int Ny);

extern "C" __global__
void Prolongation_2D_Gather_4pt_Kernel(
    float* __restrict__ fine,
    const float* __restrict__ coarse,
    int Nx_2h, int Ny_2h,
    int Nx, int Ny)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= Nx || j >= Ny) return;

    int px = i & 1;
    int py = j & 1;

    float sum = 0.0f;

    // even even : even odd : odd even : odd odd
    switch ((px << 1) | py)
    {
        // 0,0 even even
        case 0:
        {
            int ic0 = i >> 1;
            int ic1 = (i - 2) >> 1;

            int jc0 = j >> 1;
            int jc1 = (j - 2) >> 1;

            const float wx0 = 0.125f, wx2 = 0.375f;
            const float wy0 = 0.125f, wy2 = 0.375f;

            if (ic0 < Nx_2h && ic0 >= 0 && jc0 >=0 && jc0 < Ny_2h)
                sum += wx0 * wy0 * coarse[ind(ic0,jc0,Ny_2h)];

            if (ic0 < Nx_2h && ic0 >= 0 && jc1 >= 0 && jc1 < Ny_2h)
                sum += wx0 * wy2 * coarse[ind(ic0,jc1,Ny_2h)];

            if (ic1 >= 0 && ic1 < Nx_2h && jc0 >= 0 && jc0 < Ny_2h)
                sum += wx2 * wy0 * coarse[ind(ic1,jc0,Ny_2h)];

            if (ic1 >= 0 && ic1 < Nx_2h && jc1 >= 0 && jc1 < Ny_2h)
                sum += wx2 * wy2 * coarse[ind(ic1,jc1,Ny_2h)];

            break;
        }

        // 0,1 even odd
        case 1:
        {
            int ic0 = i >> 1;
            int ic1 = (i - 2) >> 1;

            int jc0 = (j - 1) >> 1;
            int jc1 = (j - 3) >> 1;

            const float wx0 = 0.125f, wx2 = 0.375f;
            const float wy1 = 0.375f, wy3 = 0.125f;

            if (ic0 >=0 && ic0 < Nx_2h && jc0 >= 0 && jc0 < Ny_2h)
                sum += wx0 * wy1 * coarse[ind(ic0,jc0,Ny_2h)];

            if (ic0 >= 0 && ic0 < Nx_2h && jc1 >= 0 && jc1 < Ny_2h)
                sum += wx0 * wy3 * coarse[ind(ic0,jc1,Ny_2h)];

            if (ic1 >= 0 && ic1 < Nx_2h && jc0 >= 0 && jc0 < Ny_2h)
                sum += wx2 * wy1 * coarse[ind(ic1,jc0,Ny_2h)];

            if (ic1 >= 0 && ic1 < Nx_2h && jc1 >= 0 && jc1 < Ny_2h)
                sum += wx2 * wy3 * coarse[ind(ic1,jc1,Ny_2h)];

            break;
        }

        // 1,0 odd even
        case 2:
        {
            int ic0 = (i - 1) >> 1;
            int ic1 = (i - 3) >> 1;

            int jc0 = j >> 1;
            int jc1 = (j - 2) >> 1;

            const float wx1 = 0.375f, wx3 = 0.125f;
            const float wy0 = 0.125f, wy2 = 0.375f;

            if (ic0 >= 0 && ic0 < Nx_2h && jc0 >= 0 && jc0 < Ny_2h)
                sum += wx1 * wy0 * coarse[ind(ic0,jc0,Ny_2h)];

            if (ic0 >= 0 && ic0 < Nx_2h && jc1 >= 0 && jc1 < Ny_2h)
                sum += wx1 * wy2 * coarse[ind(ic0,jc1,Ny_2h)];

            if (ic1 >= 0 && ic1 < Nx_2h && jc0 >= 0 && jc0 < Ny_2h)
                sum += wx3 * wy0 * coarse[ind(ic1,jc0,Ny_2h)];

            if (ic1 >= 0 && ic1 < Nx_2h && jc1 >= 0 && jc1 < Ny_2h)
                sum += wx3 * wy2 * coarse[ind(ic1,jc1,Ny_2h)];

            break;
        }

        // 1,1 odd odd
        case 3:
        {
            int ic0 = (i - 1) >> 1;
            int ic1 = (i - 3) >> 1;

            int jc0 = (j - 1) >> 1;
            int jc1 = (j - 3) >> 1;

            const float wx1 = 0.375f, wx3 = 0.125f;
            const float wy1 = 0.375f, wy3 = 0.125f;

            if (ic0 >= 0 && ic0 < Nx_2h && jc0 >= 0 && jc0 < Ny_2h)
                sum += wx1 * wy1 * coarse[ind(ic0,jc0,Ny_2h)];

            if (ic0 >= 0 && ic0 < Nx_2h && jc1 >= 0 && jc1 < Ny_2h)
                sum += wx1 * wy3 * coarse[ind(ic0,jc1,Ny_2h)];

            if (ic1 >= 0 && ic1 < Nx_2h && jc0 >= 0 && jc0 < Ny_2h)
                sum += wx3 * wy1 * coarse[ind(ic1,jc0,Ny_2h)];

            if (ic1 >= 0 && ic1 < Nx_2h && jc1 >= 0 && jc1 < Ny_2h)
                sum += wx3 * wy3 * coarse[ind(ic1,jc1,Ny_2h)];

            break;
        }
    }

    fine[ind(i,j,Ny)] += 4.0f * sum;
}

__device__ inline int ind(int i, int j, int Ny)
{
    return i * Ny + j;
}