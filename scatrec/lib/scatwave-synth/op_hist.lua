-- Author: Sixin Zhang (sixin.zhang@ens.fr)
-- Multiscale

local nn = require 'nn'

local opHist, parent = torch.class('nn.opHist', 'nn.Module')

-- estimate the range and number of bins
-- for sigmoid0 of for each channel of input histogram
function opHist:__init()
	parent.__init(self)
	self.nc = 1 -- supports only 1 slicer, i.e. gray-scaled image
	self.nb = 1
	self.slicer = {}
end

function opHist:inithist(input)
	if self.output:nElement() == 0 then
		local mb = input:size(1)
		assert(self.nc == input:size(2))
		self.output:resize(mb,self.nc*self.nb,
						   input:size(3),
						   input:size(4)):fill(0)
	end
end

-- input: (mb,c,x,y)
-- output: (mb,c*nb,x,y)
function opHist:updateOutput(input)
	self:inithist(input)
	self.output:fill(0)
	-- take one channel, apply each slicer to it
	local ic=1
	for c=1,self.nc do
		local inpc = input:narrow(2,c,1)
		for ib=1,self.nb do
			local outcb=self.slicer[c][ib]:forward(inpc)
			self.output:narrow(2,ic,1):copy(outcb)
			ic=ic+1
		end
	end
	return self.output
end

function opHist:updateGradInput(input, gradOutput)
	if input:nElement() ~= self.gradInput:nElement() then
		self.gradInput:resizeAs(input)
	end
	self.gradInput:fill(0)
	local ic=1
	for c=1,self.nc do
		local inpc = input:narrow(2,c,1)
		local ginc = self.gradInput:narrow(2,c,1)
		for ib=1,self.nb do
			local goutcb = gradOutput:narrow(2,ic,1)
			local gincb = self.slicer[c][ib]:backward(inpc,goutcb)
			ginc:add(gincb)
			ic=ic+1
		end
	end
	return self.gradInput
end
