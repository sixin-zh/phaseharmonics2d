-- Author: Sixin Zhang (sixin.zhang@ens.fr)

local nn = require 'nn'
local complex = require 'scatwave.complex'
local tools = require 'scatwave.tools'

local opConvInFFT2d, parent = torch.class('nn.opConvInFFT2d', 'nn.Module')

-- expand
-- channels: L1->L1*L, filters: L
-- filter_c: complex filter in fft2, with L channels,
function opConvInFFT2d:__init(filter_c, mbdim, cmulmode)
	parent.__init(self)
	self.filter_c = filter_c -- the filter in fft2 complex, dim=4: theta,height,width,2
	--print('opConvInFFT2d, filter_c size is',filter_c:size())
	assert(filter_c:dim()==4) -- TODO for dim==4 and remap it for batch
	self.L = self.filter_c:size(1) -- filter number
	-- for debug only: display filters (real part of fourier)
	--require 'image'
	--local tmp = self.filter_c:select(4,1)
	--local inp = image.toDisplayTensor{input=tmp,padding=1,nrow=4,scaleeach=false}
	--image.display({image=inp,legend='filters fft real',zoom=2})
	self.mbdim = mbdim -- input dim for the x
	self.cmulmode = cmulmode or 1 -- 1 if using real mode
	self.cucomplex = nil
end

function opConvInFFT2d:type(_type, tensorCache)
	parent.type(self, _type, tensorCache)
	if(_type=='torch.CudaTensor') then
		--print('opConvINFFT2d switch to complex cublas')
		self.cucomplex=require 'scatwave.cuda/complex_cublas'
		--print(self.cucomplex,self.cucomplex~=nil, self.cmulmode)
		local fc=self.filter_c
		self.filter_cu = fc:view(fc:size(1),fc:size(2)*fc:size(3),2)
		self.filter_cu_1 = fc:view(fc:size(1)*fc:size(2)*fc:size(3),2)
		-- conjugate it ! (doubled memory)
		self.filter_conj_1 = self.filter_cu_1:clone()
		self.filter_conj_1:select(2,2):mul(-1)
	end
end

-- complex input, with >=1 channels, (mb,c,x,y,2)
-- complex output, times L channels, (mb,c*L,x,y,2)
-- filter_c, (L,x,y,2)
function opConvInFFT2d:updateOutput(input)
	if input:nElement()*self.L ~= self.output:nElement() then
		--print('opConvInFFT2d forward, input size is',input:size())
		assert(input:dim()==5 and self.mbdim == 3 and input:size(5) == 2)
		self.fsize = input:size()
		self.fsize[1]=1 --mb=1
		self.fsize[self.mbdim-1]=1 --channel=1
		local osize = input:size()
		osize[self.mbdim-1] = self.L*osize[self.mbdim-1]
		self.output:resize(osize):fill(0)
	end
	-- TODO in parallel (not along mb)
	for midx = 1,input:size(1) do
		for idc = 1,input:size(self.mbdim-1) do -- for each input channel
			if self.cucomplex ~= nil and self.cmulmode == 2 then
				--print('run with ccmul')
				local output_cu = self.output:narrow(self.mbdim-1,(idc-1)*self.L+1,self.L):select(1,midx)
				local output_cu_ = output_cu:viewAs(self.filter_cu):transpose(1,2) -- to col-major
				local input_cu = input:select(self.mbdim-1,idc):select(1,midx)
				local input_cu_ = input_cu:view(input_cu:size(1)*input_cu:size(2),input_cu:size(3))
				local filter_cu_ = self.filter_cu:transpose(1,2) -- to col-major
				self.cucomplex.ccmulL(filter_cu_,input_cu_,output_cu_)
			else
				--print('run with each mul')
				local input_c_ = input:narrow(self.mbdim-1,idc,1):narrow(1,midx,1)
				for theta = 1,self.L do
					local filter_c_ = self.filter_c:select(1,theta):view(self.fsize)
					local ido = theta+(idc-1)*self.L
					local output_c_ = self.output:narrow(self.mbdim-1,ido,1):narrow(1,midx,1)
					-- output_c = input_c .* filter_c
					if self.cmulmode == 1 then
						-- (this filter_c has same real and imag part, though filter is real in fft)
						complex.multiply_complex_tensor_with_real_modified_tensor_in_place(
							input_c_,filter_c_,output_c_)
					elseif self.cmulmode == 2 then
						complex.multiply_complex_tensor_in_place(
							input_c_,filter_c_,output_c_)
					end
				end
			end
		end
	end
	--print('opConvInFFT2d forward, output size is',self.output:size())
	return self.output
end

function opConvInFFT2d:updateGradInput(input, gradOutput)
	if gradOutput:nElement() ~= self.gradInput:nElement()*self.L then
		--print('opConvInFFT2d backward, gradOutput size is',gradOutput:size())
		self.gradInput:resize(input:size())
	end
	self.gradInput:zero()
	for midx = 1,input:size(1) do
		for idc = 1,input:size(self.mbdim-1) do -- for each input channel
			local gradInput_ = self.gradInput:narrow(self.mbdim-1,idc,1):narrow(1,midx,1)
			if self.cucomplex ~= nil and self.cmulmode == 2 then
				local filter_cu = self.filter_conj_1
				local gradOutput_ = gradOutput:narrow(self.mbdim-1,(idc-1)*self.L+1,self.L):narrow(1,midx,1)
				self.gradSlice = self.gradSlice or gradOutput_:clone()
				local gradSlice_cu = self.gradSlice:viewAs(filter_cu)
				local gradOutput_cu = gradOutput_:viewAs(filter_cu)
				--print('size',filter_cu:size(),gradOutput_cu:size(),gradSlice_cu:size())
				self.cucomplex.ccmul(filter_cu,gradOutput_cu,gradSlice_cu)
				local gradSlice_ = gradSlice_cu:viewAs(gradOutput_)
				--print('gradSlice_ size',gradSlice_:size())
				-- sum gradSlice_ along the channels to get gradInput_
				torch.sum(gradInput_,gradSlice_,self.mbdim-1)
			else
				if self.gradSlice_ == nil then
					self.gradSlice_=gradInput_:clone():zero()
				end
				for theta = 1,self.L do
					local ido = theta+(idc-1)*self.L
					local gradOutput_ = gradOutput:narrow(self.mbdim-1,ido,1):narrow(1,midx,1)
					local filter_c_ = self.filter_c:select(1,theta):view(self.fsize)
					if self.cmulmode == 1 then
						complex.multiply_complex_tensor_with_real_modified_tensor_in_place(
							gradOutput_,filter_c_,self.gradSlice_)
					elseif self.cmulmode == 2 then
						complex.multiply_complex_tensor_in_place_conjugate2(
							gradOutput_,filter_c_,self.gradSlice_)
					end
					gradInput_:add(self.gradSlice_)
				end
			end
		end
	end
	return self.gradInput
end
