-- Author: Sixin Zhang (sixin.zhang@ens.fr)

local nn = require 'nn'
local conv_lib = require 'scatwave.conv_lib'

local opDown2d, parent = torch.class('nn.opDown2d', 'nn.Module')

-- input in real or complex
function opDown2d:__init(mbdim,dsj)
	parent.__init(self)
	self.mbdim = mbdim -- dim for the x
	assert(dsj>=0)
	self.dsj = dsj -- downsampling ratio = 2^dsj
	self.dsfactor = (2^self.dsj)^2
end

function opDown2d:updateOutput(input)
	if self.dsj > 0 then
		if input:nElement() ~= self.output:nElement()*self.dsfactor then
			--print('opDown2d forward alloc')
			local dsize = input:size()
			dsize[self.mbdim] = dsize[self.mbdim]/2^self.dsj
			dsize[self.mbdim+1] = dsize[self.mbdim+1]/2^self.dsj
			self.output:resize(dsize):fill(0)
		end
		self.output:copy(conv_lib.downsample_2D_inplace(
							 input,self.dsj,self.mbdim))
	else
		self.output = input
	end
	--print('opDown2d output is',self.output)
	return self.output
end

function opDown2d:updateGradInput(input, gradOutput)
	if self.dsj > 0 then
		if self.gradInput:nElement() ~= gradOutput:nElement()*self.dsfactor then
			local dsize = gradOutput:size()
			dsize[self.mbdim] = dsize[self.mbdim]*(2^self.dsj)
			dsize[self.mbdim+1] = dsize[self.mbdim+1]*(2^self.dsj)
			self.gradInput:resize(dsize):fill(0)
		end
		self.gradInput:fill(0)
		local grad = conv_lib.downsample_2D_inplace(
			self.gradInput,self.dsj,self.mbdim)
		grad:copy(gradOutput)
	else
		self.gradInput = gradOutput
	end
	--print('opDown2d backward gard is',grad)
	return self.gradInput
end
