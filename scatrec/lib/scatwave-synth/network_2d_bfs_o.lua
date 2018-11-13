-- Author: Sixin Zhang (sixin.zhang@ens.fr)

local complex = require 'scatwave.complex'
local filters_bank = require 'scatwave.filters_bank'
local conv_lib = require 'scatwave.conv_lib'
local tools = require 'scatwave.tools'

local network,parent = torch.class('network_2d_bfs_o','network_2d_bfs')

function network:__init(J,L,U0_dim,haar2d,lap2d,osr,osj,gxi,gsigma,gext,Q)
	if(J==nil) then
		error('Please specify the scale.')
	else
		self.J=J
	end
	if(U0_dim==nil) then
		error('Please specify the size.')
	else
		self.U0_dim=U0_dim
	end
	assert(#self.U0_dim==4) -- input dim = 4: minibatch,color,height,width
	parent.__init(self)
	self.N=self.U0_dim[3]
	self.J=J or 1
	self.L=L or 4
	self.haar2d = haar2d or 3
	--self.lap2d = lap2d or 0
	--self.gxi = gxi or 0.7
	--self.gsigma = gsigma or 0.85
	--self.gext = gext or 4
	--self.Q = Q or 1
	-- oversampling rate
	self.osr = osr or 1
	self.osj = osj or 1
	self.complex = 1 -- 1 means no hack on the imag part of the filter
	self.filmode = 2
	self.fft=require 'scatwave.wrapper_fft'
	self.filters = self:load_filters(self.fft,self.complex,self.J,self.L,self.U0_dim,
									 self.haar2d,self.gxi,self.gsigma,self.gext,self.Q)
end

function network:scat_lp_nngraph_o2(ampoutput,lplist1,lplist2,histparams)
	--assert(self.Q == 1)
	local nngraph = require 'nngraph'
	local lplist1 = lplist1 or {-1}
	local lplist2 = lplist2 or {2}
	local filters = self.filters -- build scatnet with filters.bpsi
	local mbdim = #self.U0_dim-1
	local filmode = self.filmode
	local join_dim = mbdim-1 -- join at the batch channel
	local ndim = #self.U0_dim
	local J = self.J
	local d = self.U0_dim[3]*self.U0_dim[4]
	local ampoutput = ampoutput or 1
	print('build network:scat_lp_nngraph_o2 with parameters:')
	print('\tampoutput='..ampoutput)
	print('\tlplist1=',lplist1[1],lplist1[2])
	print('\tlplist2=',lplist2[1],lplist2[2])
	print('\tosr='..self.osr)
	print('\tosj='..self.osj)
	print('\td='..d)
	local OPL2=nn.opLP2dg
	local OPL1=nn.opLP2dg
	assert(J>=1)
	
	local input = nn.Identity()()
	local l2coeff = {}

	if histparams then
		assert(false,'hist not supported')
	end
	
	local l1_input = OPL1(lplist1,mbdim)(input)
	table.insert(l2coeff,l1_input)
	local sc1fft = nn.opFFT2d(mbdim)(input)
	local sc1phi_fft = nn.opConvInFFT2d(filters.bphi[1],mbdim,filmode)(sc1fft)
	local sc1phi_fftdc = nn.opSubDC(mbdim)(sc1phi_fft)
	local sc1phi_absfft = nn.opModulus(mbdim)(sc1phi_fftdc)
	local sc1phi_nor = nn.MulConstant(1/math.sqrt(d))(sc1phi_absfft)
	local l2_input = OPL2(lplist2,mbdim)(sc1phi_nor)
	table.insert(l2coeff,l2_input)
	
	-- add the haar2d branch
	if self.haar2d > 0 then
		print('addHaar2d filters with lplist2')
		self:addHaar2d_l2(filters,d,mbdim,sc1fft,l2coeff,OPL2,lplist2)
	end
	
	for j1=0,J-1 do
		-- compute x * psi_{j1,q1} 
		local sc1psi_fft = nn.opConvInFFT2d(filters.bpsi[j1][1],mbdim,filmode)(sc1fft)
		local res1=math.max(0,j1-self.osr+1)
		local dj1 = d / ((2^res1)^2)
		local sc1conv = nn.opIFFT2d(mbdim,0)(sc1psi_fft) -- complex-valued output
		local sc1amp = nn.opModulus(mbdim)(sc1conv) -- x_{j1,q1} = | x * psi_{j1,q1} |
		local sc1amp_down = nn.opDown2d(mbdim,res1)(sc1amp) -- downsample x_{j1,q1}
		-- l1 of x_{j1,q1} (real input)
		local l1_y = OPL1(lplist1,mbdim)(sc1amp_down)
		table.insert(l2coeff,l1_y)
		-- hat x_{j1,q1}  .* hat gphi_J - dc
		local sc2fft = nn.opFFT2d(mbdim)(sc1amp_down)
		local sc2phi_fft = nn.opConvInFFT2d(filters.bphi[res1+1],mbdim,filmode)(sc2fft)
		--nn.opConvInFFT2d_s(filters.bphi[res1+1],filters.bphi_conj[res1+1],mbdim)(sc2fft)
		local sc2phi_fftdc = nn.opSubDC(mbdim)(sc2phi_fft)
		local sc2phi_absfft = nn.opModulus(mbdim)(sc2phi_fftdc)
		local sc2phi_nor = nn.MulConstant(1/math.sqrt(dj1))(sc2phi_absfft)
		local lp_sc1 = OPL2(lplist2,mbdim)(sc2phi_nor) -- \| y_j1 * gphi_J - mean y_j1 \|^2
		table.insert(l2coeff,lp_sc1)
		-- 2nd order
		local j2begin = math.max(j1+1-self.osj,0)
		if j2begin <= J-1 then
			for j2=j2begin,J-1 do
				--print('add 2nd order filter: j1='..j1..' j2='..j2)
				local sc2psi_fft = nn.opConvInFFT2d(filters.bpsi[j2][res1+1],mbdim,filmode)(sc2fft)
				local sc2psi_absfft = nn.opModulus(mbdim)(sc2psi_fft)
				local sc2psi_nor = nn.MulConstant(1/math.sqrt(dj1))(sc2psi_absfft) 
				local lp_sc2 = OPL2(lplist2,mbdim)(sc2psi_nor)
				table.insert(l2coeff,lp_sc2)
			end
		end
	end
	
	-- PUT TOGETHER
	local l2_all = nn.MulConstant(ampoutput)(nn.JoinTable(join_dim,ndim)(l2coeff))
	local sc0 = nn.gModule({input}, {l2_all})
	return sc0
end

return network
