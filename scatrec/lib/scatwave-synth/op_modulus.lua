-- Author: Sixin Zhang (sixin.zhang@ens.fr)

local nn = require 'nn'
local complex = require 'scatwave.complex'
local tools = require 'scatwave.tools'

local opModulus, parent = torch.class('nn.opModulus', 'nn.Module')

-- input in complex
function opModulus:__init(mbdim,eps,disp)
	parent.__init(self)
	self.mbdim = mbdim
	self.eps = eps or 1e-16 -- change eps from 1e-12
	self.disp = disp or 0
	self.input_c = torch.Tensor()
	self.output_eps = torch.Tensor()
end

function opModulus:updateOutput(input)
	if input:nElement() ~= self.output:nElement()*2 then
		--print('opModulus forward alloc')
		local dsize = input:size()
		self.input_c:resize(dsize)
		dsize[self.mbdim] = dsize[self.mbdim]
		dsize[self.mbdim+1] = dsize[self.mbdim+1]
		assert(dsize[5]==2)
		dsize[5] = 0
		self.output:resize(dsize):fill(0)
	end
	self.input_c:copy(input)
	complex.abs_value_inplace(self.input_c,self.output) -- this destorys self.input_c
	if self.disp>0 then self:viewOutput(self.output) end
	return self.output
end

function opModulus:updateGradInput(input, gradOutput)
	if self.gradInput:nElement() ~= gradOutput:nElement()*2 then
		--print('opModulus backward alloc')
		self.gradInput:resize(
			tools.concatenateLongStorage(gradOutput:size(),torch.LongStorage({2}))):fill(0)
		self.output_eps:resize(self.output:size()):fill(0)
		--print('opModulus output_eps size:',self.output_eps:size())
	end
	self.input_c:copy(input)
	complex.abs_value_inplace(self.input_c,self.output_eps) -- this destorys self.input_c
	self.output_eps:add(self.eps)
	complex.divide_real_and_complex_tensor(input, self.output_eps, self.gradInput) -- compute phase
	-- compute gradModulus = phase (complex) .* gradOutput (real)
	complex.multiply_complex_tensor_with_real_tensor_in_place(self.gradInput,gradOutput,self.gradInput)
	return self.gradInput
end

function opModulus:viewOutput()
	local tmp = self.output
	assert(tmp:dim() == 4)
	local mb = tmp:size(1)
	local c = tmp:size(2)
	local x = tmp:size(3)
	local y = tmp:size(4)
	local nrows = math.ceil(mb*c/8)
	require 'image'
	tmp=tmp:view(mb*c,x,y)
	if self.disp == 1 then
		local inp = image.toDisplayTensor{input=tmp,padding=1,nrow=nrows,scaleeach=true}
		image.display({image=inp,legend='modulus scaleeach',zoom=2})
	elseif self.disp == 2 then
		local inp = image.toDisplayTensor{input=tmp,padding=1,nrow=nrows,scaleeach=false}
		image.display({image=inp,legend='modulus',zoom=2})
	end
end
