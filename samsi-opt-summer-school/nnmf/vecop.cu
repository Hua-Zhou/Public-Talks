extern "C"
{

__global__ void vmuldiv_dp(const double *a, const double *b, double *c)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  c[idx] *= a[idx] / b[idx];
}

__global__ void vmuldiv_sp(const float *a, const float *b, float *c)
{
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  c[idx] *= a[idx] / b[idx];
}

}
