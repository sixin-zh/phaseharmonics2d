-- Author: Sixin Zhang (sixin.zhang@ens.fr)

local nn = require 'nn'
local tools = require 'scatwave.tools'

local opIFFT2d, parent = torch.class('nn.opIFFT2d', 'nn.Module')

-- input in fft (complex), output (real): set realout to 1
-- input in fft (complex), output (complex): set realout to 0
function opIFFT2d:__init(mbdim,realout)
	parent.__init(self)
	self.mbdim = mbdim -- dim for the x
	self.realout = realout
	self.output_c = torch.Tensor()
	self:type('torch.FloatTensor')
end

function opIFFT2d:type(_type, tensorCache)
	parent.type(self, _type, tensorCache)
	if(_type=='torch.CudaTensor') then
		self.fft=require 'scatwave.cuda/wrapper_CUDA_fft_nvidia'
	elseif(_type=='torch.FloatTensor') then
		self.fft=require 'scatwave.wrapper_fft'
	else
		error('This type is not handled by scatwave')
	end
end

function opIFFT2d:updateOutput(input)
	if self.realout == 1 then
		if input:nElement() ~= self.output:nElement()*2 then
			--print('opIFFT2d forward alloc real')
			--assert(input:dim()==5)
			local dsize = input:size()
			self.output_c:resize(dsize)
			assert(dsize[self.mbdim+2]==2)
			dsize[self.mbdim+2] = 0
			self.output:resize(dsize):fill(0)
		end
		self.fft.my_2D_fft_complex_batch(input,self.mbdim,1,self.output_c)
		self.output:copy(self.output_c:narrow(self.mbdim+2,1,1))
		--self.input_c:copy(input)
		--self.fft.my_2D_ifft_complex_to_real_batch(self.input_c,self.mbdim,self.output)
	elseif self.realout == 0 then
		if input:nElement() ~= self.output:nElement() then
			--print('opIFFT2d forward alloc complex')
			--assert(input:dim()==5)
			self.output:resize(input:size()):fill(0)
		end
		self.fft.my_2D_fft_complex_batch(input,self.mbdim,1,self.output)
	else
		error('opIFFT2d: realout not set to be 0 or 1.')
	end
	return self.output
end

function opIFFT2d:updateGradInput(input, gradOutput)
	if self.realout == 1 then
		-- fft, real to complex, need normalization
		if self.gradInput:nElement() ~= gradOutput:nElement()*2 then
			self.gradInput:resize(tools.concatenateLongStorage(gradOutput:size(),torch.LongStorage({2}))):fill(0)
		end
		self.fft.my_2D_fft_real_batch(gradOutput,self.mbdim,self.gradInput,1) -- normalize
	elseif self.realout == 0 then
		-- fft, complex to complex, need normalization
		if self.gradInput:nElement() ~= gradOutput:nElement() then
			self.gradInput:resize(gradOutput:size()):fill(0)
		end
		self.fft.my_2D_fft_complex_batch(gradOutput,self.mbdim,0,self.gradInput,1) -- normalize
	else
		error('opIFFT2d: realout not set to be 0 or 1.')
	end
	return self.gradInput
end
