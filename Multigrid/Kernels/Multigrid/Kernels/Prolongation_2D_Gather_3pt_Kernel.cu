__device__ inline int ind(int i, int j, int Ny);

extern "C" __global__
void Prolongation_2D_Gather_3pt_Kernel(
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

            const float wxy = 0.25f * 0.25f;

            if (ic0 < Nx_2h && jc0 < Ny_2h)
                sum += wxy * coarse[ic0 * Ny_2h + jc0];

            if (ic0 < Nx_2h && jc1 >= 0 && jc1 < Ny_2h)
                sum += wxy * coarse[ic0 * Ny_2h + jc1];

            if (ic1 >= 0 && ic1 < Nx_2h && jc0 < Ny_2h)
                sum += wxy * coarse[ic1 * Ny_2h + jc0];

            if (ic1 >= 0 && ic1 < Nx_2h && jc1 >= 0 && jc1 < Ny_2h)
                sum += wxy * coarse[ic1 * Ny_2h + jc1];

            break;
        }

        // 0,1 even odd
        case 1:
        {
            int ic0 = i >> 1;
            int ic1 = (i - 2) >> 1;

            int jc1 = (j - 1) >> 1;

            const float wxy = 0.25f * 0.5f;

            if (ic0 >=0 && ic0 < Nx_2h && jc1 >= 0 && jc1 < Ny_2h)
                sum += wxy * coarse[ic0 * Ny_2h + jc1];

            if (ic1 < Nx_2h && ic1 >= 0 && jc1 >= 0 && jc1 < Ny_2h)
                sum += wxy * coarse[ic1 * Ny_2h + jc1];

            break;
        }

        // 1,0 odd even
        case 2:
        {
            int ic1 = (i - 1) >> 1;

            int jc0 = j >> 1;
            int jc1 = (j - 2) >> 1;

            const float wxy = 0.25f * 0.5f;

            if (ic1 >= 0 && ic1 < Nx_2h && jc0 >= 0 && jc0 < Ny_2h)
                sum += wxy * coarse[ic1 * Ny_2h + jc0];

            if (ic1 >= 0 && ic1 < Nx_2h && jc1 >= 0 && jc1 < Ny_2h)
                sum += wxy * coarse[ic1 * Ny_2h + jc1];

            break;
        }

        // 1,1 odd odd
        case 3:
        {
            int ic1 = (i - 1) >> 1;
            int jc1 = (j - 1) >> 1;

            const float wxy = 0.5f * 0.5f;

            if (ic1 >= 0 && ic1 < Nx_2h && jc1 >= 0 && jc1 < Ny_2h)
                sum += wxy * coarse[ic1 * Ny_2h + jc1];

            break;
        }
    }

    fine[i * Ny + j] += 4.0f * sum;
}

__device__ inline int ind(int i, int j, int Ny)
{
    return i * Ny + j;
}