extern "C" __global__ void Restriction_Ny_Convolution_3pt(
                                                      const float* __restrict__ fine,
                                                      float* __restrict__ coarse,
                                                      int Nx, int Ny, int Ny_2h)
{
// stencil striding [Ry_1D[n, 2*n : 2*n + len(stencil)] = stencil for n in range(0,Ny)]
// Full size matrix performing 1D convolution -> Ry_2D = kron(Ix, Ry_1D)

int tid = blockIdx.x * blockDim.x + threadIdx.x;

int jc = tid % Ny_2h;
int i  = tid / Ny_2h;

int N = Nx * Ny_2h;

 if (tid >= N) return;

 int j =  2 * jc;

 int idy0 = i * Ny + j;
 int idy1 = idy0 + 1;
 int idy2 = idy0 + 2;

 float f0 = (j < Ny)     ? fine[idy0] : 0.0f;
 float f1 = (j + 1 < Ny) ? fine[idy1] : 0.0f;
 float f2 = (j + 2 < Ny) ? fine[idy2] : 0.0f;


 coarse[tid] = 0.25f * f0
               + 0.50f * f1
               + 0.25f * f2;


}