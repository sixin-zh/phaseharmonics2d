-- Author: Sixin Zhang (sixin.zhang@ens.fr)

local complex = require 'scatwave.complex'
local filters_bank = require 'scatwave.filters_bank'
local conv_lib = require 'scatwave.conv_lib'
local tools = require 'scatwave.tools'

local network = torch.class('network_2d_bfs')

function network:__init()
	self.usegpu = 0
	self.filter_root = './filters/'
end

-- This function is pretty much copied from Module.lua in nn @ https://github.com/torch/nn/blob/master/Module.lua
function network:type(_type)
	if(_type=='torch.CudaTensor') then
		self.fft=require 'scatwave.cuda/wrapper_CUDA_fft_nvidia'
	elseif(_type=='torch.FloatTensor') then
		self.fft=require 'scatwave.wrapper_fft'
	else
		error('This type is not handled by scatwave')
	end
	
	for key,param in pairs(self) do
		self[key] = tools.recursiveType(param,_type)
	end	
	return self
end

function network:float()
	self:type('torch.FloatTensor')
	self.usegpu = 0
end

function network:cuda()
	self:type('torch.CudaTensor')
	self.usegpu = 1
end

function network:load_filters(tfft,tcomplex,J,L,U0_dim,haar2d,gxi,gsigma,gext,Q)
	local U0filters = torch.LongStorage({U0_dim[3],U0_dim[4]})
	local filtername = self.filter_root .. '/network_2d_bfs_o_morlet_J'..J..
		'_L'..L..
		'_N'..U0_dim[3]..
		'_haar2d_'..haar2d..
		'.th'
	if (gxi and gsigma and gext) then
		if Q then
			filtername = self.filter_root .. '/network_2d_bfs_o_morlet_J'..J..
				'_L'..L..'_Q'..Q..
				'_N_'..U0_dim[3]..
				'_haar2d_'..haar2d..
				'_gxi_'..gxi ..
				'_gsigma_'..gsigma..
				'_gext_'..gext..
				'.th'
		else
			filtername = self.filter_root .. '/network_2d_bfs_o_morlet_J'..J..
				'_L'..L..
				'_N_'..U0_dim[3]..
				'_haar2d_'..haar2d..
				'_gxi_'..gxi ..
				'_gsigma_'..gsigma..
				'_gext_'..gext..
				'.th'
		end
	end
	local Q = Q or 1
	local gxi = gxi or 3/4
	local gsigma = gsigma or 0.8
	local gext = gext or 4
	local filters
	if tools.file_exists(filtername) then
		print('load filters from ' .. filtername)
		filters = torch.load(filtername)
	else
		local gaborfilters = filters_bank.morlet_filters_bank_2D(U0filters,J,tfft,0,0,L,gxi,gsigma,gext,Q)
		filters = filters_bank.modify_bfs_psi(gaborfilters,J,L,tcomplex,Q)
		if haar2d == 3 then
			print('add haar2d filter with mode=3')
			local haarfilters = filters_bank.haar_filters_bank_2D(U0filters,3,tfft) -- complex==1
			filters.haar2d = haarfilters.haar2d
			filters.haarmode = 2
		end
		print('cache filters to ' .. filtername)
		torch.save(filtername,filters)
	end
	return filters
end

function network:addHaar2d_l2(filters,d,mbdim,sc1fft,lp_all,OPLP,lplisth)
	-- TODO no crop
	assert(filters.haarmode==2)
	local lplisth = lplisth or {2} -- 2}
	local sc1haar_fft = nn.opConvInFFT2d(filters.haar2d,mbdim,filters.haarmode)(sc1fft) -- filmode is always 2
	local sc1haar_absfft = nn.opModulus(mbdim)(sc1haar_fft)
	local sc1haar_nor = nn.MulConstant(1/math.sqrt(d))(sc1haar_absfft)
	local lp_sc1h = OPLP(lplisth,mbdim)(sc1haar_nor)
	table.insert(lp_all,lp_sc1h)
end

function network:addHist0(input,histparams,lplist1,mbdim,OPL1,l2coeff)
	local histrange = histparams.histrange
	local histbins = histparams.histbins
	local histmode = histparams.histmode
	local histamp = histparams.histamp
	assert(histrange and histbins)
	--print('add histogram sig0 moments')
	local inpsig0 = nn.opHistsig0(histrange,histbins,histmode,histamp)(input)
	local lp_hist = OPL1(lplist1,mbdim)(inpsig0)
	table.insert(l2coeff,lp_hist)
end

return network
