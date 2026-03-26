extern "C" __global__ void Restriction_Nx_Convolution_4pt_2d(
                                                      const float* __restrict__ fine,
                                                      float* __restrict__ coarse,
                                                      int Nx, int Nx_2h, int Ny_2h)
{
// stencil striding [Rx_1D[n, 2*n : 2*n + len(stencil)] = stencil for n in range(0,Ny)]
// Full size matrix performing 1D convolution -> Rx_2D = kron(Rx_1D,I_y)

 int ic = blockIdx.x * blockDim.x + threadIdx.x;
 int j = blockIdx.y * blockDim.y + threadIdx.y;


 if (ic >= Nx_2h || j > Ny_2h) return;

  int i =  2 * ic;

 if (i + 3 > Nx) return;

 int base0 = i * Ny_2h;

 int base1 = base0 + Ny_2h;
 int base2 = base1 + Ny_2h;
 int base3 = base2 + Ny_2h;

 float val =  1./8. * fine[base0 +j]
            + 3./8. * fine[base1 +j]
            + 3./8. * fine[base2 +j]
            + 1./8. * fine[base3 +j];

 coarse[ic * Ny_2h + j] = val;

}