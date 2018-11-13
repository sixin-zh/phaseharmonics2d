-- Author: Sixin Zhang (sixin.zhang@ens.fr)

local nn = require 'nn'
local tools = require 'scatwave.tools'

local opSubDC, parent = torch.class('nn.opSubDC', 'nn.Module')

-- sub mean in fft domain
-- hat y = hat x at non-zero freq
-- hat y = 0 at zero freq of hat x
-- for each channel
function opSubDC:__init(mbdim)
	assert(mbdim == 3) --  and d>0)
	parent.__init(self)
	--self.alpha=0 -- 1-1/d
end

-- (mb,c,x,y,2) -> (mb,c,x,y,2)
function opSubDC:updateOutput(input)
	self.output:resizeAs(input)
	self.output:copy(input)
	self.output:narrow(3,1,1):narrow(4,1,1):fill(0)
	--self.output:narrow(3,1,1):narrow(4,1,1):mul(self.alpha)
	return self.output
end

-- (mb,c,x,y,2) -> (mb,c,x,y,2)
function opSubDC:updateGradInput(input, gradOutput)
	self.gradInput:resizeAs(gradOutput)
	self.gradInput:copy(gradOutput)
	self.gradInput:narrow(3,1,1):narrow(4,1,1):fill(0) -- mul(self.alpha) -- fill(0)
	return self.gradInput
end
