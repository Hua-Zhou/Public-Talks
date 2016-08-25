#
# This function fits rank-r nonnegative matrix factorization (NNMF) of
# a data matrix X with nonnegative entries by a multiplicative algorithm.
#
function nnmf(X::Matrix{Float64}, r::Int;
              maxiter::Int=1000, tolfun::Float64=1e-4,
              V::Matrix{Float64}=rand(size(X, 1), r),
              W::Matrix{Float64}=rand(r, size(X, 2)),
              device::String="CPU", precision::String="DP")
  # size of data matrix
  m = size(X, 1)
  n = size(X, 2)
  mn = m * n

  if device == "CPU"

    # MM loop on CPU
    Vnum = zeros(V)
    Vden = zeros(V)
    Wnum = zeros(W)
    Wden = zeros(W)
    B = V * W
    XminusB = X - B
    obj = vecnorm(XminusB)^2
    for iter = 1:maxiter

      # multiplicative update of V and W
      # V = V .* (X * W') ./ (B * W')
      BLAS.gemm!('N', 'T', 1.0, X, W, 0.0, Vnum)
      BLAS.gemm!('N', 'T', 1.0, B, W, 0.0, Vden)
      for j = 1:r
        @simd for i = 1:m
          @inbounds V[i, j] *= (Vnum[i, j] / Vden[i, j])
        end
      end
      # B = V * W
      BLAS.gemm!('N', 'N', 1.0, V, W, 0.0, B)
      # W = W .* (V' * X) ./ (V' * B)
      BLAS.gemm!('T', 'N', 1.0, V, X, 0.0, Wnum)
      BLAS.gemm!('T', 'N', 1.0, V, B, 0.0, Wden)
      for j = 1:n
        @simd for i = 1:r
          @inbounds W[i, j] *= (Wnum[i, j] / Wden[i, j])
        end
      end
      # B = V * W
      BLAS.gemm!('N', 'N', 1.0, V, W, 0.0, B)

      # check convergence
      objold = obj
      BLAS.blascopy!(mn, X, 1, XminusB, 1)
      BLAS.axpy!(mn, -1.0, B, 1, XminusB, 1)
      obj = vecnorm(XminusB); obj = obj * obj
      if abs(obj - objold) < tolfun * (abs(obj) + 1.0)
        break
      end
    end

    # output
    return V, W

  elseif device == "GPU"
    # MM loop on CPU

    # transfer data X, V, W to GPU
    # and load kernel function
    md = CUDArt.CuModule("vecop.ptx", false)
    if precision == "DP"
      d_X = CUDArt.CudaArray(X)
      d_V = CUDArt.CudaArray(V)
      d_W = CUDArt.CudaArray(W)
      vmuldiv = CUDArt.CuFunction(md, "vmuldiv_dp")
    elseif precision == "SP"
      d_X = CUDArt.CudaArray(float32(X))
      d_V = CUDArt.CudaArray(float32(V))
      d_W = CUDArt.CudaArray(float32(W))
      vmuldiv = CUDArt.CuFunction(md, "vmuldiv_sp")
    else
      error("unrecognized precision: SP or DP")
    end
    # constants 1 and 0
    oneconst = one(eltype(d_X))
    zeroconst = zero(eltype(d_X))

    # pre-allocate variables on GPU
    d_Vnum = CUDArt.CudaArray(eltype(d_V), size(d_V))
    d_Vden = CUDArt.CudaArray(eltype(d_V), size(d_V))
    d_Wnum = CUDArt.CudaArray(eltype(d_W), size(d_W))
    d_Wden = CUDArt.CudaArray(eltype(d_W), size(d_W))
    d_B = CUDArt.CudaArray(eltype(d_X), size(d_X))
    d_XminusB = CUDArt.CudaArray(eltype(d_X), size(d_X))

    # initial objective value
    CUBLAS.gemm!('N', 'N', oneconst, d_V, d_W, zeroconst, d_B)
    CUBLAS.blascopy!(mn, d_X, 1, d_XminusB, 1)
    CUBLAS.axpy!(mn, -oneconst, d_B, 1, d_XminusB, 1)
    obj = CUBLAS.nrm2(mn, d_XminusB, 1); obj = obj * obj

    for iter = 1:maxiter

      # V = V .* (X * W') ./ (B * W')
      CUBLAS.gemm!('N', 'T', oneconst, d_X, d_W, zeroconst, d_Vnum)
      CUBLAS.gemm!('N', 'T', oneconst, d_B, d_W, zeroconst, d_Vden)
      CUDArt.launch(vmuldiv, m, r, (d_Vnum, d_Vden, d_V))
      # B = V * W
      CUBLAS.gemm!('N', 'N', oneconst, d_V, d_W, zeroconst, d_B)
      # W = W .* (V' * X) ./ (V' * B)
      CUBLAS.gemm!('T', 'N', oneconst, d_V, d_X, zeroconst, d_Wnum)
      CUBLAS.gemm!('T', 'N', oneconst, d_V, d_B, zeroconst, d_Wden)
      CUDArt.launch(vmuldiv, r, n, (d_Wnum, d_Wden, d_W))
      # B = V * W
      CUBLAS.gemm!('N', 'N', oneconst, d_V, d_W, zeroconst, d_B)

      # check convergence
      objold = obj
      CUBLAS.blascopy!(mn, d_X, 1, d_XminusB, 1)
      CUBLAS.axpy!(mn, -oneconst, d_B, 1, d_XminusB, 1)
      obj = CUBLAS.nrm2(mn, d_XminusB, 1); obj = obj * obj
      if abs(obj - objold) < tolfun * (abs(obj) + 1.0)
        break
      end

    end

    # transfer result to host and output
    V = CUDArt.to_host(d_V)
    W = CUDArt.to_host(d_W)
    return V, W

  elseif device == "cvx"

    for iter = 1:maxiter
      if mod(iter, 2) == 1
        # update V
        V = Convex.Variable(m, r)
        problem = Convex.minimize(vecnorm(X - V * W, 2))
        problem.constraints += V >= 0
      else
        # update W
        W = Convex.Variable(r, n)
        problem = Convex.minimize(vecnorm(X - V * W, 2))
        problem.constraints += W >= 0
      end
      Convex.solve!(problem)
    end

  else

    error("unrecognized device: CPU or GPU")

  end

end
