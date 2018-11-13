-- Author: Sixin Zhang (sixin.zhang@ens.fr)

local nn = require 'nn'
local tools = require 'scatwave.tools'

local opFFT2d, parent = torch.class('nn.opFFT2d', 'nn.Module')

-- input (real), output in fft (complex)
function opFFT2d:__init(mbdim,disp)
	parent.__init(self)
	self.mbdim = mbdim -- dim for the x
	self.disp = disp or 0
	self:type('torch.FloatTensor')
	self.gradInput_c = torch.Tensor()
	-- assert(mbdim == 3) -- as my_2D_ifft_complex_to_real_batch only works in batch
end

function opFFT2d:type(_type, tensorCache)
	parent.type(self, _type, tensorCache)
	if(_type=='torch.CudaTensor') then
		self.fft=require 'scatwave.cuda/wrapper_CUDA_fft_nvidia'
	elseif(_type=='torch.FloatTensor') then
		self.fft=require 'scatwave.wrapper_fft'
	else
		error('This type is not handled by scatwave')
	end
end

function opFFT2d:updateOutput(input)
	if input:nElement()*2 ~= self.output:nElement() then
		--print('opFFT2d forward alloc')
		assert(input:dim()==4)
		self.output:resize(tools.concatenateLongStorage(input:size(),torch.LongStorage({2}))):fill(0)
	end
	if self.disp == 1 then print('opFFT2d forward with input size',input:size()) end
	self.fft.my_2D_fft_real_batch(input,self.mbdim,self.output)
	return self.output
end

function opFFT2d:updateGradInput(input, gradOutput)
	if self.gradInput:nElement()*2 ~= gradOutput:nElement() then
		local dsize = gradOutput:size()
		assert(dsize[self.mbdim+2]==2)
		self.gradInput_c:resize(dsize):fill(0)
		dsize[self.mbdim+2] = 0
		self.gradInput:resize(dsize):fill(0)
	end
	if self.disp == 1 then print('opFFT2d backward with gradOutput size',gradOutput:size()) end
	self.fft.my_2D_fft_complex_batch(gradOutput,self.mbdim,1,self.gradInput_c,0) -- do not normalize with constant
	self.gradInput:copy(self.gradInput_c:narrow(self.mbdim+2,1,1)) -- take the real part of the ifft
	return self.gradInput
end
