-- compute lp moments
-- Author: Sixin Zhang (sixin.zhang@ens.fr)

local nn = require 'nn'
local conv_lib = require 'scatwave.conv_lib'

local opLP2dg, parent = torch.class('nn.opLP2dg', 'nn.Module')

-- TODO mb>>1

-- input is real
function opLP2dg:__init(lplist,mbdim,label,lptable)
	assert(mbdim == 3 and #lplist > 0) -- assume minibatch case, otherwise modify the index
	parent.__init(self)
	self.lplist = lplist
	self.lpsize = #lplist
	self.eps = eps or 1e-16
	self.label = label or nil
	self.lptable = lptable or nil
	for i,v in pairs(self.lplist) do
		if v==2 or v==22 then
			self.input_v2=torch.Tensor()
			self.gout_1 = torch.Tensor()
			self.gout_2 = torch.Tensor()
		elseif v==1 then
			self.input_abs=torch.Tensor()
			self.input_sign=torch.Tensor()
		elseif v==101 then
			self.input_exp=torch.Tensor()
		elseif v==-1 then
			-- pass
		else
			assert(false)
		end
		--print('add lp2dg with lp v='..v)
	end
	self.mbdim = mbdim -- dim for the x
end

-- (mb,c,N,N) -> (mb,c*lpsize,1,1)
function opLP2dg:updateOutput(input)
	-- self.ftime = self.ftime or 0
	-- self.ftime = self.ftime -sys.clock()
	local mb=input:size(1)
	local c=input:size(2)
	if self.output:nElement() ~= mb*c*self.lpsize then
		assert(input:dim()==4)
		self.output:resize(mb,c*self.lpsize,1,1)
		--self.output=self.output:typeAs(input)
		if self.input_v2 then
			self.input_v2 = input:clone()
		end
		if self.input_abs then
			self.input_abs = input:clone()
			self.input_sign = input:clone()			
		end
		if self.input_exp then
			self.input_exp = input:clone()
		end
	end
	if self.input_v2 then
		torch.cmul(self.input_v2,input,input)
	end
	if self.input_abs then
		torch.abs(self.input_abs,input)
		torch.sign(self.input_sign,input)
	end
	if self.input_exp then
		torch.exp(self.input_exp,input)
	end
	for ip,v in pairs(self.lplist) do
		if v==-1 then
			for m=1,mb do
				local input_1=input:view(mb,c,-1):select(1,m)
				local output_1=self.output[m]:narrow(1,(ip-1)*c+1,c):view(-1)
				torch.mean(output_1,input_1,2)
			end
		elseif v==2 then
			for m=1,mb do
				local input_2=self.input_v2:view(mb,c,-1):select(1,m)
				local output_2=self.output[m]:narrow(1,(ip-1)*c+1,c):view(-1)
				torch.mean(output_2,input_2,2)
				output_2:sqrt()
			end
		elseif v==22 then
			for m=1,mb do
				local input_2=self.input_v2:view(mb,c,-1):select(1,m)
				local output_2=self.output[m]:narrow(1,(ip-1)*c+1,c):view(-1)
				torch.mean(output_2,input_2,2)
			end
		elseif v==1 then
			for m=1,mb do
				local input_1=self.input_abs:view(mb,c,-1):select(1,m)
				local output_1=self.output[m]:narrow(1,(ip-1)*c+1,c):view(-1)
				torch.mean(output_1,input_1,2)
			end
		elseif v==101 then
			for m=1,mb do
				local input_1=self.input_exp:view(mb,c,-1):select(1,m)
				local output_1=self.output[m]:narrow(1,(ip-1)*c+1,c):view(-1)
				torch.mean(output_1,input_1,2)
			end
		end
	end
	--self.ftime = self.ftime + sys.clock()
	--print('opLP2dg elapsed time',self.ftime)
	if self.label then
		if self.lptable then
			if self.lptable.totalid == nil then
				-- init
				self.lptable.totalid = 0
				self.lptable.id2name = {}
				self.lptable.name2id = {}
				self.lptable.id2dim = {}
			end
			if self.lptable.name2id[self.label] == nil then
				local id = self.lptable.totalid + 1
				print('add ' .. self.label .. ' to table at id=' .. id .. ' dim=' .. self.output:nElement())
				self.lptable.id2name[id] = self.label
				self.lptable.name2id[self.label] = id
				self.lptable.id2dim[id] = self.output:nElement()
				self.lptable.totalid = id
			end
		else
			local lpstr = 'lp['
			for ip,v in pairs(self.lplist) do
				lpstr = lpstr ..  v .. ','
			end
			print(lpstr..']>'..
					  string.format("%s:min=%f,max=%f,mean=%f,median=%f",
									self.label,self.output:min(),self.output:max(),
									self.output:mean(),self.output:view(-1):median()[1]))
		end
	end
	return self.output
end

function opLP2dg:updateGradInput(input, gradOutput)
	local mb=input:size(1)
	local c=input:size(2)
	local N=input:size(3)*input:size(4)
	if self.gradInput:nElement() ~= input:nElement() then
		self.gradInput:resize(input:size()):fill(0)
		if self.gout_1 then
			self.gout_1:resize(mb,c)
			self.gout_2:resize(mb,c)
		end
	end
	self.gradInput:zero()
	for ip,v in pairs(self.lplist) do
		if v==-1 then
			for m=1,mb do
				local gout_1=gradOutput[m]:narrow(1,(ip-1)*c+1,c):view(c,1):expand(c,N)
				local gin_1=self.gradInput[m]:view(c,N)
				gin_1:add(1/N,gout_1)
			end
		elseif v==2 then
			for m=1,mb do
				local out_2=self.output[m]:narrow(1,(ip-1)*c+1,c):view(c,1)
				local gout_1=self.gout_1[m]:view(c,1)
				torch.add(gout_1,out_2,self.eps)
				local gout=gradOutput[m]:narrow(1,(ip-1)*c+1,c):view(c,1)
				local gout_2=self.gout_2[m]:view(c,1)
				gout_2:copy(gout)
				gout_2:cdiv(gout_1)
				local gout_=gout_2:expand(c,N)
				local inp_=input[m]:view(c,N)
				local gin_2=self.gradInput[m]:view(c,N)
				gin_2:addcmul(1/N,gout_,inp_)
			end
		elseif v==22 then
			for m=1,mb do
				local gout_1=gradOutput[m]:narrow(1,(ip-1)*c+1,c):view(c,1):expand(c,N)
				local inp_=input[m]:view(c,N)
				local gin_2=self.gradInput[m]:view(c,N)
				gin_2:addcmul(2/N,gout_1,inp_)
			end
		elseif v==1 then
			for m=1,mb do
				local gout_1=gradOutput[m]:narrow(1,(ip-1)*c+1,c):view(c,1):expand(c,N)
				local gin_1=self.gradInput[m]:view(c,N)
				local input_sign=self.input_sign[m]:view(c,N)
				gin_1:addcmul(1/N,gout_1,input_sign)
			end
		elseif v==101 then
			for m=1,mb do
				local gout_1=gradOutput[m]:narrow(1,(ip-1)*c+1,c):view(c,1):expand(c,N)
				local gin_1=self.gradInput[m]:view(c,N)
				local input_exp=self.input_exp[m]:view(c,N)
				gin_1:addcmul(1/N,gout_1,input_exp)
			end
		end
	end
	return self.gradInput
end
