-- Author: Sixin Zhang (sixin.zhang@ens.fr)

local nn = require 'nn'
local opSigmoid0, parent = torch.class('nn.opSigmoid0', 'nn.Module')

-- counting number of x larger than a
-- compute sigmoid((x-a)/b) * C,
-- set C=4*b, so that max of derivative = 1 
-- valid range (non-zero grad) is [a-b,a+b]
function opSigmoid0:__init(a,b,C)
	parent.__init(self)
	--print('init opSigmoid0 with a='..a..',b='..b..',C='..C)
	self.a = a
	self.b = b
	self.C = C
	self.sigmoid = nn.Sigmoid()
	self.input2 = torch.Tensor()
end

function opSigmoid0:updateOutput(input)
	if input:nElement() ~= self.input2:nElement() then
		if input:type()=='torch.CudaTensor' then
			self.output=self.output:cuda()
			self.input2=self.input2:cuda()
			self.sigmoid:cuda()
		end
		self.output:resizeAs(input)
		self.input2:resizeAs(input)
	end
	torch.add(self.input2,input,-self.a)
	self.input2:div(self.b)
	local sig0 = self.sigmoid:forward(self.input2)
	torch.mul(self.output,sig0,self.C)
	return self.output
end

function opSigmoid0:updateGradInput(input, gradOutput)
	if input:nElement() ~= self.gradInput:nElement() then
		if input:type()=='torch.CudaTensor' then
			self.gradInput=self.gradInput:cuda()
		end
		self.gradInput:resizeAs(input)
	end
	local gsigmoid = self.sigmoid:backward(self.input2, gradOutput)
	self.gradInput:copy(gsigmoid)
	self.gradInput:mul(self.C/self.b) -- div(self.b)
	return self.gradInput
end
