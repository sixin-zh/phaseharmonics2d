-- Author: Sixin Zhang (sixin.zhang@ens.fr)


local ffi = require 'ffi'

local ccuBLAS = {}

-- CUTORCH
--require 'cutorch'
local ok, err = pcall(function () ccuBLAS.CU=ffi.load('cutorch') end)
if(not ok) then
   print(err)
   error('library cutorch not found...')
end
ffi.cdef[[
cublasHandle_t THCState_getCurrentBlasHandle(THCState *state);
float THCudaBlas_Sdot(THCState *state, long n, float *x, long incx, float *y, long incy);
]]

local ok, err = pcall(function () ccuBLAS.C=ffi.load('cublas') end)
if(not ok) then
   print(err)
   error('library cublas not found...')
end
-- defines types
ffi.cdef[[
struct cublasContext;
typedef struct cublasContext *cublasHandle_t;
typedef enum{
    CUBLAS_STATUS_SUCCESS         =0,
    CUBLAS_STATUS_NOT_INITIALIZED =1,
    CUBLAS_STATUS_ALLOC_FAILED    =3,
    CUBLAS_STATUS_INVALID_VALUE   =7,
    CUBLAS_STATUS_ARCH_MISMATCH   =8,
    CUBLAS_STATUS_MAPPING_ERROR   =11,
    CUBLAS_STATUS_EXECUTION_FAILED=13,
    CUBLAS_STATUS_INTERNAL_ERROR  =14,
    CUBLAS_STATUS_NOT_SUPPORTED   =15,
    CUBLAS_STATUS_LICENSE_ERROR   =16
} cublasStatus_t;

typedef enum {
    CUBLAS_SIDE_LEFT =0, 
    CUBLAS_SIDE_RIGHT=1
} cublasSideMode_t; 

extern cublasStatus_t cublasCreate_v2 (cublasHandle_t *handle);
extern cublasStatus_t cublasDestroy_v2 (cublasHandle_t handle);
extern cublasStatus_t cublasSdot_v2 (cublasHandle_t handle, int n,
                           const float           *x, int incx,
                           const float           *y, int incy,
                           float           *result);
extern cublasStatus_t  cublasSscal_v2 (cublasHandle_t handle, int n,
                            const float           *alpha,
                            float           *x, int incx);
extern cublasStatus_t cublasSdgmm(cublasHandle_t handle, cublasSideMode_t mode,
                          int m, int n,
                          const float           *A, int lda,
                          const float           *x, int incx,
                          float           *C, int ldc);

typedef float float2[2];
typedef float2 cuFloatComplex;
typedef cuFloatComplex cuComplex;
extern cublasStatus_t cublasCdgmm(cublasHandle_t handle, cublasSideMode_t mode,
                          int m, int n,
                          const cuComplex       *A, int lda,
                          const cuComplex       *x, int incx,
                          cuComplex       *C, int ldc);

typedef enum {
    CUBLAS_OP_N=0,
    CUBLAS_OP_T=1,
    CUBLAS_OP_C=2
} cublasOperation_t;
extern cublasStatus_t cublasCgemm_v2(cublasHandle_t handle,
                                     cublasOperation_t transa,
                                     cublasOperation_t transb,
                                     int m,
                                     int n,
                                     int k,
                                     const cuComplex *alpha, /* host or device pointer */
                                     const cuComplex *A,
                                     int lda,
                                     const cuComplex *B,
                                     int ldb,
                                     const cuComplex *beta, /* host or device pointer */
                                     cuComplex *C,
                                     int ldc);
]]

--[[
local function destroy_handle(h)
   ccuBLAS.C['cublasDestroy_v2'](h[0])
   --ccuBLAS.C['cublasShutdown']()
   --print('cublas end')
end
--]]
local THCState_ptr = ffi.typeof('THCState*')

function ccuBLAS.getState()
   return THCState_ptr(cutorch.getState());
end
--]]
function ccuBLAS.getHandle()
   if ccuBLAS.handle == nil then
      ccuBLAS.handle = ccuBLAS.CU['THCState_getCurrentBlasHandle'](ccuBLAS.getState())
      --[[ create one
      --print('cublas init',ccuBLAS.C['cublasInit']())
      ccuBLAS.handle = ffi.new('cublasHandle_t[1]',{})
      print('ccuBLAS create handle',ccuBLAS.C['cublasCreate_v2'](ccuBLAS.handle))
      ffi.gc(ccuBLAS.handle,destroy_handle)
      --]]
   end
   --print('ccublas handle',ccuBLAS.handle)
   return ccuBLAS.handle --[0]
end

-- c = <a,b>
function ccuBLAS.dot(a,b,c)
   local n = a:size(1)
   print('dot:'.. 'a=',a,'b=',b,'c=',c)
   local a_data = torch.data(a)
   --print('a_data',a_data,'n',n)
   local b_data = torch.data(b)
   local c_data = torch.data(c)
   print('c_data',c_data)
   return ccuBLAS.C['cublasSdot_v2'](ccuBLAS.getHandle(),n,a_data,1,b_data,1,c_data)
   --
   --return ccuBLAS.CU['THCudaBlas_Sdot'](ccuBLAS.getState(),n,a_data,1,b_data,1)
   --return c
end

function ccuBLAS.mul(alpha,x)
   local n = x:nElement()
   local alpha_data = torch.data(alpha)
   local x_data = torch.data(x)
   print('n of x is',n,'data type',x_data)
   return ccuBLAS.C['cublasSscal_v2'](ccuBLAS.getHandle(),n,alpha_data,x_data,1)
end

-- C=A*diag(x)
function ccuBLAS.cmul(A,x,C)
   -- side RIGHT
   local mode = ffi.new("cublasSideMode_t","CUBLAS_SIDE_RIGHT")
   local m = A:size(1)
   local n = A:size(2)
   assert(x:nElement()==n)
   assert(C:size(1)==m)
   assert(C:size(2)==n)
   assert(m==1)
   local lda = m
   local ldc = m
   local incx = 1
   local A_ = torch.data(A)
   local C_ = torch.data(C)
   local x_ = torch.data(x)
   return ccuBLAS.C['cublasSdgmm'](ccuBLAS.getHandle(),mode,m,n,A_,lda,x_,incx,C_,ldc)
end

local cucomplex_ptr = 'cuComplex*'
local cucomplex_ptr_c = 'const cuComplex*'

-- c = a .* x
function ccuBLAS.ccmul(a,x,c)
   local mode = ffi.new("cublasSideMode_t","CUBLAS_SIDE_RIGHT")
   local m = 1
   local n = a:size(1)
   assert(x:nElement()==2*n and c:size(1)==n)
   local lda = m
   local ldc = m
   local incx = 1
   local A_ = ffi.cast(cucomplex_ptr_c,torch.data(a))
   local C_ = ffi.cast(cucomplex_ptr,torch.data(c))
   local x_ = ffi.cast(cucomplex_ptr_c,torch.data(x))
   return ccuBLAS.C['cublasCdgmm'](ccuBLAS.getHandle(),mode,m,n,A_,lda,x_,incx,C_,ldc)
end

-----------------------------------------
-- ALL MATRIX SHOULD BE IN COLUMN MAJOR!!
-----------------------------------------

-- C = A × d i a g ( x ) 
function ccuBLAS.ccmulR(A,x,C)
   local mode = ffi.new("cublasSideMode_t","CUBLAS_SIDE_RIGHT")
   local m = A:size(1)
   local n = A:size(2)
   assert(A:size(3)==2 and x:size(2)==2 and C:size(3)==2)
   assert(x:nElement()==2*n)
   assert(C:size(1)==m)
   assert(C:size(2)==n)
   local lda = m
   local ldc = m
   local incx = 1
   local A_ = ffi.cast(cucomplex_ptr_c,torch.data(A))
   local C_ = ffi.cast(cucomplex_ptr,torch.data(C))
   local x_ = ffi.cast(cucomplex_ptr_c,torch.data(x))
   return ccuBLAS.C['cublasCdgmm'](ccuBLAS.getHandle(),mode,m,n,A_,lda,x_,incx,C_,ldc)
end

-- C = d i a g ( x ) x A
function ccuBLAS.ccmulL(A,x,C)
   local mode = ffi.new("cublasSideMode_t","CUBLAS_SIDE_LEFT")
   local m = A:size(1)
   local n = A:size(2)
   assert(A:size(3)==2 and x:size(2)==2 and C:size(3)==2 and
			  x:nElement()==2*m and C:size(1)==m and C:size(2)==n)
   local lda = m
   local ldc = m
   local incx = 1
   local A_ = ffi.cast(cucomplex_ptr_c,torch.data(A))
   local C_ = ffi.cast(cucomplex_ptr,torch.data(C))
   local x_ = ffi.cast(cucomplex_ptr_c,torch.data(x))
   return ccuBLAS.C['cublasCdgmm'](ccuBLAS.getHandle(),mode,m,n,A_,lda,x_,incx,C_,ldc)
end

-- C = A:t() x B
function ccuBLAS.cmmT1(A,B,C,alpha,beta)
	local opA = ffi.new("cublasOperation_t","CUBLAS_OP_T")
	local opB = ffi.new("cublasOperation_t","CUBLAS_OP_N")
	assert(A:dim()==3 and B:dim()==3 and C:dim()==3 and alpha:dim()==1 and beta:dim()==1)
	assert(A:size(3)==2 and B:size(3)==2 and C:size(3)==2 and alpha:size(1)==2 and beta:size(1)==2)
	local m = A:size(2)
	local k = A:size(1)
	local n = C:size(2)
	assert(B:size(1)==k and C:size(1)==m and B:size(2)==n)
	local lda = k
	local ldb = k
	local ldc = m
	local A_ = ffi.cast(cucomplex_ptr_c,torch.data(A))
	local B_ = ffi.cast(cucomplex_ptr_c,torch.data(B))
	local C_ = ffi.cast(cucomplex_ptr,torch.data(C))
	local alpha_ = ffi.cast(cucomplex_ptr_c,torch.data(alpha))
	local beta_ = ffi.cast(cucomplex_ptr_c,torch.data(beta))
	return ccuBLAS.C['cublasCgemm_v2'](ccuBLAS.getHandle(),opA,opB,m,n,k,
									   alpha_,A_,lda,B_,ldb,beta_,C_,ldc)
end

-- C = (A:t())* x B
function ccuBLAS.cmmH1(A,B,C,alpha,beta)
	local opA = ffi.new("cublasOperation_t","CUBLAS_OP_C")
	local opB = ffi.new("cublasOperation_t","CUBLAS_OP_N")
	assert(A:dim()==3 and B:dim()==3 and C:dim()==3 and alpha:dim()==1 and beta:dim()==1)
	assert(A:size(3)==2 and B:size(3)==2 and C:size(3)==2 and alpha:size(1)==2 and beta:size(1)==2)
	local m = A:size(2)
	local k = A:size(1)
	local n = C:size(2)
	assert(B:size(1)==k and C:size(1)==m and B:size(2)==n)
	local lda = k
	local ldb = k
	local ldc = m
	local A_ = ffi.cast(cucomplex_ptr_c,torch.data(A))
	local B_ = ffi.cast(cucomplex_ptr_c,torch.data(B))
	local C_ = ffi.cast(cucomplex_ptr,torch.data(C))
	local alpha_ = ffi.cast(cucomplex_ptr_c,torch.data(alpha))
	local beta_ = ffi.cast(cucomplex_ptr_c,torch.data(beta))
	return ccuBLAS.C['cublasCgemm_v2'](ccuBLAS.getHandle(),opA,opB,m,n,k,
									   alpha_,A_,lda,B_,ldb,beta_,C_,ldc)
end

-- C = A x (B:t())*
function ccuBLAS.cmmH2(A,B,C,alpha,beta)
	local opA = ffi.new("cublasOperation_t","CUBLAS_OP_N")
	local opB = ffi.new("cublasOperation_t","CUBLAS_OP_C")
	assert(A:dim()==3 and B:dim()==3 and C:dim()==3 and alpha:dim()==1 and beta:dim()==1)
	assert(A:size(3)==2 and B:size(3)==2 and C:size(3)==2 and alpha:size(1)==2 and beta:size(1)==2)
	local m = A:size(1)
	local k = A:size(2)
	local n = C:size(2)
	assert(B:size(1)==n and C:size(1)==m and B:size(2)==k)
	local lda = m
	local ldb = n
	local ldc = m
	local A_ = ffi.cast(cucomplex_ptr_c,torch.data(A))
	local B_ = ffi.cast(cucomplex_ptr_c,torch.data(B))
	local C_ = ffi.cast(cucomplex_ptr,torch.data(C))
	local alpha_ = ffi.cast(cucomplex_ptr_c,torch.data(alpha))
	local beta_ = ffi.cast(cucomplex_ptr_c,torch.data(beta))
	return ccuBLAS.C['cublasCgemm_v2'](ccuBLAS.getHandle(),opA,opB,m,n,k,
									   alpha_,A_,lda,B_,ldb,beta_,C_,ldc)
end

-- C = A x B
function ccuBLAS.cmm(A,B,C,alpha,beta)
	local opA = ffi.new("cublasOperation_t","CUBLAS_OP_N")
	local opB = ffi.new("cublasOperation_t","CUBLAS_OP_N")
	assert(A:dim()==3 and B:dim()==3 and C:dim()==3 and alpha:dim()==1 and beta:dim()==1)
	assert(A:size(3)==2 and B:size(3)==2 and C:size(3)==2 and alpha:size(1)==2 and beta:size(1)==2)
	local m = A:size(1)
	local k = A:size(2)
	local n = C:size(2)
	assert(B:size(1)==k and C:size(1)==m and B:size(2)==n)
	local lda = m
	local ldb = k
	local ldc = m
	local A_ = ffi.cast(cucomplex_ptr_c,torch.data(A))
	local B_ = ffi.cast(cucomplex_ptr_c,torch.data(B))
	local C_ = ffi.cast(cucomplex_ptr,torch.data(C))
	local alpha_ = ffi.cast(cucomplex_ptr_c,torch.data(alpha))
	local beta_ = ffi.cast(cucomplex_ptr_c,torch.data(beta))
	return ccuBLAS.C['cublasCgemm_v2'](ccuBLAS.getHandle(),opA,opB,m,n,k,
									   alpha_,A_,lda,B_,ldb,beta_,C_,ldc)
end

return ccuBLAS
