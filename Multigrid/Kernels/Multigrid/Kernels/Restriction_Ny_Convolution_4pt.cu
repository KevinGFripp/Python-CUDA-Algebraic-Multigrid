extern "C" __global__ void Restriction_Ny_Convolution_4pt(
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
 int idy3 = idy0 + 3;

 float f0 = (j < Ny)     ? fine[idy0] : 0.0f;
 float f1 = (j + 1 < Ny) ? fine[idy1] : 0.0f;
 float f2 = (j + 2 < Ny) ? fine[idy2] : 0.0f;
 float f3 = (j + 3 < Ny) ? fine[idy3] : 0.0f;

 coarse[tid] = 0.125f * f0
               + 0.375f * f1
               + 0.375f * f2
               + 0.125f * f3;



//  int tid = blockIdx.x * blockDim.x + threadIdx.x;
//
//  int N = Nx * Ny_2h;
//
//  if (tid >= N) return;
//
//
//  int jc = tid % Ny_2h;
//  int i = tid / Ny_2h;
//
//  int j =  2 * jc;
//
//  if (j + 3 >= Ny) return;
//
//  int idy0 = i * Ny + j;
//  int idy1 = idy0 + 1;
//  int idy2 = idy0 + 2;
//  int idy3 = idy0 + 3;
//
//  float val =   1./8. * fine[idy0]
//              + 3./8. * fine[idy1]
//              + 3./8. * fine[idy2]
//              + 1./8. * fine[idy3];
//
//  coarse[tid] = val;

}