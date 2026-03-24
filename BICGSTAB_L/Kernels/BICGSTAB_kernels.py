from cupy import ReductionKernel, ElementwiseKernel


preamble = '''
__device__ float2 operator+(float2 a,float2 b)
{
return make_float2(a.x + b.x, a.y + b.y);
}
'''
'''
Raw CUDA code above the reduction kernel. Defines the addition operation for float2.
'''

omega_dot_ratio = ReductionKernel('float32 s, float32 t',
                                  'float32 output',
                                  'make_float2(s * t, t* t)',
                                  'a+b',
                                  'output = a.y == 0? 0 : a.x / a.y',
                                  'make_float2(0.0,0.0)',
                                  'dot_ratio',
                                  'float2',
                                  preamble=preamble)
'''
Fused reduction cupy kernel computing reduction of omega = dot(s,t) / dot(t,t) of  bicgstab(1).
'''

beta_ratio = ElementwiseKernel('float32 rho_1, float32 rho,float32 alpha,float32 omega',
                               'float32 beta',
                               'beta = (rho_1/rho) *(alpha/omega)',
                               name='beta_ratio')
'''
Fused element-wise cupy kernel computing reduction of beta = (rho_1/ rho) * (alpha/omega) of  bicgstab(1).
'''
# add two vector arrays within one kernel
# a*x + b*y
ax_plus_by = ElementwiseKernel('float32 a, float32 x,float32 b,float32 y',
                               'float32 out',
                               'out = a*x + b*y',
                               name='ax_plus_by')
'''
Add two vector arrays within one kernel, performing a*x + b*y using cupy element-wise kernels.
a and b are scalar values.
'''


