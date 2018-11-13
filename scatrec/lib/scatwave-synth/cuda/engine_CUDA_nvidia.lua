--[[
     ScatWave implementation of Scattering Network
     Written by Edouard Oyallon
     Team DATA ENS
     Copyright 2015
]]

local ffi = require 'ffi'
local cuFFT = {}


local ok, err = pcall(function () cuFFT.C=ffi.load('cufft') end)

if(not ok) then
   print(err)
   error('library cufft not found...')
end

-- defines types
ffi.cdef[[
typedef double double2[2];
typedef float float2[2];
typedef float2 cuFloatComplex;
typedef double2 cuDoubleComplex;
typedef cuFloatComplex cuComplex;
typedef float cufftReal;
typedef double cufftDoubleReal;
typedef cuComplex cufftComplex;
typedef cuDoubleComplex cufftDoubleComplex;
typedef int cufftHandle;
typedef int cufftResult;
typedef int cufftType;
]]


-- defines structures
ffi.cdef[[
extern cufftResult cufftPlanMany(cufftHandle *plan,
                                   int rank,
                                   int *n,
                                   int *inembed, int istride, int idist,
                                   int *onembed, int ostride, int odist,
                                   cufftType type,
                                   int batch);

extern cufftResult cufftExecC2C(cufftHandle plan, 
                                  cufftComplex *idata,
                                  cufftComplex *odata,
                                  int direction);

extern cufftResult cufftExecR2C(cufftHandle plan, 
                                  cufftReal *idata,
                                  cufftComplex *odata);

extern cufftResult cufftExecC2R(cufftHandle plan, 
                                  cufftComplex *idata,
                                  cufftReal *odata);

extern cufftResult cufftDestroy(cufftHandle plan);

extern cufftResult cufftGetVersion(int *version);

extern cufftResult cufftPlan2d(cufftHandle *plan, 
                                 int nx, int ny,
                                 cufftType type);
]]

-- defines constant 
cuFFT.FORWARD  = -1
cuFFT.INVERSE =  1
cuFFT.R2C = 0x2a
cuFFT.C2R = 0x2c
cuFFT.C2C = 0x29
cuFFT.D2Z = 0x6a
cuFFT.Z2D = 0x6c
cuFFT.Z2Z = 0x69     
cuFFT.SUCCESS  = 0x0
   cuFFT_INVALID_PLAN   = 0x1
cuFFT.ALLOC_FAILED   = 0x2
cuFFT.INVALID_TYPE   = 0x3
cuFFT.INVALID_VALUE  = 0x4
cuFFT.INTERNAL_ERROR = 0x5
cuFFT.EXEC_FAILED    = 0x6
cuFFT.SETUP_FAILED   = 0x7
cuFFT.INVALID_SIZE   = 0x8
cuFFT.UNALIGNED_DATA = 0x9
cuFFT.INCOMPLETE_PARAMETER_LIST = 0xA
cuFFT.INVALID_DEVICE = 0xB
cuFFT.PARSE_ERROR = 0xC
cuFFT.NO_WORKSPACE = 0xD
cuFFT.NOT_IMPLEMENTED = 0xE
cuFFT.LICENSE_ERROR = 0x0F




return cuFFT
