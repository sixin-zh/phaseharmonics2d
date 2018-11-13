-- Author: Sixin Zhang (sixin.zhang@ens.fr)

local nn = require 'nn'

local opHistsig0, parent = torch.class('nn.opHistsig0', 'nn.opHist')

-- estimate the range and number of bins
-- for sigmoid0 of for each channel of input histogram
function opHistsig0:__init(hrange,nbins,mode,amp)
	parent.__init(self)
	self.nc = 1 -- supports only 1 slicer
	self.nb = nbins or 2
	self.hmin = hrange[1] or -1
	self.hmax = hrange[2] or 1
	self.mode = mode or 2 -- mode=0 -> C0=1, mode=1 -> C0 = 4*b
	self.amp = amp or 1
	--self.C = 1 -- overlapping ratio, larger C, less overlap
	print('init hist sig0 with range=['..self.hmin ..
			  ',' .. self.hmax .. '], nbins='..nbins .. 
			  ',mode=' .. self.mode .. ', amp=' .. self.amp)
	self.slicer = {}
end

function opHistsig0:inithist(input)
	if self.output:nElement() == 0 then
		assert(self.nb>=2)
		local mb = input:size(1)
		assert(self.nc == input:size(2))
		for c=1,self.nc do
			self.slicer[c]={}
			local windows = (self.hmax-self.hmin)/(self.nb-1)
			local wstart = self.hmin
			for ib=1,self.nb do
				if self.mode == 0 then
					self.slicer[c][ib]=nn.opSigmoid0(wstart,windows,self.amp)
				elseif self.mode == 1 then
					self.slicer[c][ib]=nn.opSigmoid0(wstart,windows,4*windows*self.amp)
				elseif self.mode == 2 then
					self.slicer[c][ib]=nn.opSigmoid0(wstart,windows,math.sqrt(6*windows)*self.amp)
				else
					assert(false)
				end
				wstart=wstart+windows
			end
		end
		self.output:resize(mb,self.nc*self.nb,
						   input:size(3),
						   input:size(4)):fill(0)
	end
end
