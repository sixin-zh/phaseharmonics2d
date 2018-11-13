--[[
	ScatWave implementation of Scattering Network
	Written by Edouard Oyallon
	Team DATA ENS
	Copyright 2015
]]

local conv_lib = require 'scatwave.conv_lib'
local complex = require 'scatwave.complex'
local tools = require 'scatwave.tools'

local filters_bank ={}

function filters_bank.fftshift(A)
	assert(A:size(1)==1 and A:size(2)==1 and A:nDimension()==4)
	local N=A:size(3)
	local M=A:size(4)
	local B=torch.Tensor(torch.LongStorage({1,1,N,M})):fill(0)
	
	-- fftshift to display... 
	for i=1,N do
		for j=1,M do
			i_=(i-(N/2)-1)%N+1
			j_=(j-(M/2)-1)%M+1
			B[1][1][i_][j_]=A[1][1][i][j]
		end
	end
	return B
end

-- f is real in Fourier
function filters_bank.compute_littlewood_paley_real(f,r)
	local r=r or 1
	local psi0=f.psi[1].signal[r]
	assert(psi0:size(1)==1 and psi0:size(2)==1 and psi0:nDimension()==4)
	local N=psi0:size(3)
	local M=psi0:size(4)
	local A=torch.FloatTensor(1,1,N,M):zero()
	for i=1,#f.psi do
		local psi=f.psi[i].signal[r]
		local Asq = torch.cmul(psi,psi)
		--local asq=complex.abs_value_sq(f.psi[i].signal[r])
		A=A+Asq
	end
	return filters_bank.fftshift(A)
end

function filters_bank.display_littlehood_paley(f,r)
	r= r or 1
	local A=torch.FloatTensor(torch.LongStorage({f.size[r][1],f.size[r][2]})):zero()
	
	for i=1,#f.psi do
		if(#f.psi[i].signal>=r) then
			A=A+complex.abs_value(f.psi[i].signal[r])
		end
	end
	local B=torch.Tensor(torch.LongStorage({f.size[r][1],f.size[r][2]})):fill(0)
	
	-- fftshift to display... 
	for i=1,A:size(1) do
		for j=1,A:size(2) do
			i_=(i-(A:size(1)/2)-1)%A:size(1)+1
			j_=(j-(A:size(2)/2)-1)%A:size(2)+1
			B[i_][j_]=A[i][j]
		end
	end
	return B
end

function get_multires_size(s,max_res,pad)
	local J=max_res
	local sz={}
	local pad = pad or 0
	
	for res=0,J do      
		local M_p=2^J*torch.ceil((s[s:size()]+pad*2*2^J)/2^J)/2^res -- TOCHECK
		M_p=torch.max(torch.Tensor({{M_p,1}}))
		local N_p=2^J*torch.ceil((s[s:size()]+pad*2*2^J)/2^J)
		N_p=torch.max(torch.Tensor({{N_p,1}}))/2^res
		local s_=torch.LongStorage(s:size())
		s_:copy(s)
		
		s_[s_:size()-1]=M_p
		s_[s_:size()]=N_p
		sz[res+1]=s_
	end
	
	return sz
end

function get_multires_stride(s,max_res,pad)
	local J=max_res
	local sz={}
	local pad = pad or 0
	
	for res=0,J do      
		local M_p=2^J*torch.ceil((s[s:size()-1]+pad*2*2^J)/2^J)/2^res
		M_p=torch.max(torch.Tensor({{M_p,1}}))
		local N_p=2^J*torch.ceil((s[s:size()]+pad*2*2^J)/2^J)
		N_p=torch.max(torch.Tensor({{N_p,1}}))/2^res
		local s_=torch.LongStorage(s:size())
		for l=1,s:size()-2 do
			s_[l]=0
		end
		
		s_[s_:size()-1]=N_p
		s_[s_:size()]=1
		sz[res+1]=s_
	end
	
	return sz
end

function reduced_freq_res(x,res,k)
	local s=x:stride()
	for l=1,#s do
		if not l==k then
			s[l]=0
		end
	end
	local mask = torch.FloatTensor(x:size(),s):fill(1) -- Tensor
	
	local z=mask:narrow(k,x:size(k)*2^(-res-1)+1,x:size(k)*(1-2^(-res))):fill(0)
	local y=torch.cmul(x,mask)
	return conv_lib.periodize_along_k(y,k,res,1)
end

function filters_bank.haar_filters_bank_2D(U0_dim,mode,fft)
	local filters={}
	if mode == 3 then
		-- no need for multires
		assert(#U0_dim == 2)
		local M=U0_dim[1]
		local N=U0_dim[2]
		filters.haar2d = torch.FloatTensor(3,M,N,2):zero()
		
		local psi = torch.FloatTensor(M,N,2):zero()
		psi[1][1][1] = 1/4
		psi[1][2][1] = -1/4
		psi[2][1][1] = 1/4
		psi[2][2][1] = -1/4
		local fft2psi = fft.my_2D_fft_complex(psi)
		filters.haar2d[1]:copy(fft2psi)
		
		local psi = torch.FloatTensor(M,N,2):zero()
		psi[1][1][1] = 1/4
		psi[1][2][1] = 1/4
		psi[2][1][1] = -1/4
		psi[2][2][1] = -1/4
		local fft2psi = fft.my_2D_fft_complex(psi)
		filters.haar2d[2]:copy(fft2psi)
		
		local psi = torch.FloatTensor(M,N,2):zero()
		psi[1][1][1] = 1/4
		psi[1][2][1] = -1/4
		psi[2][1][1] = -1/4
		psi[2][2][1] = 1/4
		local fft2psi = fft.my_2D_fft_complex(psi)
		filters.haar2d[3]:copy(fft2psi)
	else
		assert(false)
	end
	return filters
end

function filters_bank.morlet_filters_bank_2D(U0_dim,J,fft,pad,dolpal,L,gxi,gsigma,gext,Q)
	local filters={}
	local i=1
	local pad = pad or 0
	local dolpal = dolpal or 0
	local L=L or 8 -- number of angles between [0,pi)
	local Q=Q or 1 -- fractional scale: s_j = 2^(j/Q), j=0..J-1, res_j = floor(j/Q) - osr + 1
	local sigma0 = sigma0 or 0.8
	local extent = extent or 4

	if (J%Q > 0) then
		print('morlet filter bank error: J does not divide Q')
		assert(false)
	end
	size_multi_res=get_multires_size(U0_dim,J/Q,pad) -- min padding should be a power of 2
	filters.psi={}
	filters.size_multi_res=size_multi_res
	
	stride_multi_res=get_multires_stride(U0_dim,J/Q,pad) -- J
	filters.stride_multi_res = stride_multi_res

	filters.bfs_psi = {}
	for j=0,J-1 do
		filters.bfs_psi[j]={}
		local res_MAX = math.floor(j/Q) + 1
		for res=1,res_MAX do --	for res=1,j+1 do
			filters.bfs_psi[j][res]={}
		end
	end

	for j=0,J-1 do
		for theta=1,L do
			filters.psi[i]={}
			filters.psi[i].signal = {}
			local scale = 2^(j/Q)
			local psi  = morlet_2d(size_multi_res[1][U0_dim:size()-1], 
								   size_multi_res[1][U0_dim:size()], 
								   gsigma*scale, 4/L, gxi*math.pi/scale,
								   theta*math.pi/L, 0, 1, pad, gext) --slant=4/L
			
			psi = complex.realize(fft.my_2D_fft_complex(psi))  
			filters.psi[i].signal[1]=
				torch.FloatTensor(psi:storage(),psi:storageOffset(),
								  size_multi_res[1],stride_multi_res[1])
			filters.bfs_psi[j][1][theta]=filters.psi[i].signal[1]
			
			-- multi-res for psi
			local res_MAX = math.floor(j/Q) + 1
			for res=2,res_MAX do
				local tmp_psi=reduced_freq_res(reduced_freq_res(psi,res-1,1),res-1,2)
				filters.psi[i].signal[res]=torch.FloatTensor(
					tmp_psi:storage(), tmp_psi:storageOffset(),
					size_multi_res[res], stride_multi_res[res])
				filters.bfs_psi[j][res][theta]=filters.psi[i].signal[res]
			end
			
			filters.psi[i].j=j/Q
			filters.psi[i].theta=theta
			i=i+1
		end
	end
	
	-- littlewood-paley normalize psi
	if dolpal == 1 then
		local lpal = filters_bank.compute_littlewood_paley_real(filters,1)
		filters.lpal = lpal
		local lmax = lpal:max()
		print('lpal max = ' .. lmax)
		local i=1
		for j=0,J-1 do
			for theta=1,8 do
				for res=1,j+1 do
					filters.psi[i].signal[res]:div(math.sqrt(lmax/2))
				end
				i=i+1
			end
		end
	end
	
	-- phi
	filters.phi={}
	filters.phi.signal={}

	local scale = 2^((J-1)/Q)
	local phi = gabor_2d(size_multi_res[1][U0_dim:size()-1], 
						 size_multi_res[1][U0_dim:size()], 
						 gsigma*scale, 1, 0, 0, 0, 1, pad, extent)
	
	phi=complex.realize(fft.my_2D_fft_complex(phi))
	
	filters.phi.signal[1]=torch.FloatTensor(
		phi:storage(), phi:storageOffset(),
		size_multi_res[1], stride_multi_res[1])

	-- multi-res for phi
	local res_MAX = math.floor(J/Q)+1
	for res=2,res_MAX do
		local tmp_phi=reduced_freq_res(reduced_freq_res(phi,res-1,1),res-1,2)
		filters.phi.signal[res]=torch.FloatTensor(
			tmp_phi:storage(), tmp_phi:storageOffset(),
			size_multi_res[res], stride_multi_res[res])
	end
	
	filters.phi.j=J
	
	return filters 
end

function filters_bank.modify_bfs_psi(filters,J,L,complex,Q)
	local L=L or 8
	local complex=complex or 1 -- 0
	local Q = Q or 1
	assert(filters.bfs_psi)
	assert(filters.phi)
	
	-- change each filter from real to complex
	for j=0,J-1 do
		local res_MAX = math.floor(j/Q) + 1
		for res=1,res_MAX do
			-- for psi(j,theta)[res]
			for theta=1,L do
				local tmp_r = filters.bfs_psi[j][res][theta]
				if tmp_r==nil then
					print('empty filter at j=',j,'res=',res,'theta=',theta)
					assert(false)
				end
				local n=   tmp_r:nDimension()
				local tmp_c = torch.FloatTensor(
					tools.concatenateLongStorage(tmp_r:size(),torch.LongStorage({2}))):fill(0)
				tmp_c:select(n+1,1):copy(tmp_r)
				if complex == 0 then
					tmp_c:select(n+1,2):copy(tmp_r)
				else
					tmp_c:select(n+1,2):fill(0)
				end
				filters.bfs_psi[j][res][theta]=tmp_c
			end
		end
	end

	if complex == 0 then
		filters.psimode = 1
	else
		filters.psimode = 2
	end
	
	-- batch the storage across theta
	filters.bpsi = {}
	for j=0,J-1 do
		filters.bpsi[j] = {}
		local res_MAX = math.floor(j/Q) + 1
		for res=1,res_MAX do --for res=1,j+1 do
			filters.bpsi[j][res] = torch.FloatTensor(
				tools.concatenateLongStorage(
					torch.LongStorage({L}),
					filters.bfs_psi[j][res][1]:size()))
			for theta=1,L do
				filters.bpsi[j][res]:select(1,theta):copy(filters.bfs_psi[j][res][theta])
			end
		end
	end
	
	-- add phi
	filters.bphi = {}
	local n=filters.phi.signal[1]:nDimension()
	local res_MAX = math.floor(J/Q)+1
	for res=1,res_MAX do -- for res=1,J+1 do
		local tmp=torch.FloatTensor(tools.concatenateLongStorage(filters.phi.signal[res]:size(),torch.LongStorage({2}))):fill(0)
		tmp:select(n+1,1):copy(filters.phi.signal[res])
		if complex == 0 then
			tmp:select(n+1,2):copy(filters.phi.signal[res]) -- bphi in real fourier mode 
		elseif complex == 1 then
			tmp:select(n+1,2):fill(0)
		end
		filters.bphi[res]=torch.FloatTensor(
			tools.concatenateLongStorage(torch.LongStorage({1}),tmp:size()))
		filters.bphi[res]:select(1,1):copy(tmp) -- only 1 channel
		--print('bpsi[res] size is',filters.bphi[res]:size())
	end
	
	if complex == 0 then
		filters.phimode = 1
	else
		filters.phimode = 2
	end
	
	-- remove the rest
	filters.bfs_psi = {}
	filters.psi = {}
	filters.phi = {}
	
	collectgarbage()
	return filters -- add bpsi and bphi
end

function filters_bank.modify(filters)
	local n=   filters.phi.signal[1]:nDimension()
	for j=1,#filters.psi do
		for l=1,#filters.psi[j].signal do
			local tmp=torch.FloatTensor(tools.concatenateLongStorage(filters.psi[j].signal[l]:size(),torch.LongStorage({2}))):fill(0)
			tmp:select(n+1,1):copy(filters.psi[j].signal[l])
			tmp:select(n+1,2):copy(filters.psi[j].signal[l])
			filters.psi[j].signal[l]=tmp
		end
	end
	
	for l=1,#filters.phi.signal do
		local tmp=torch.FloatTensor(tools.concatenateLongStorage(filters.phi.signal[l]:size(),torch.LongStorage({2}))):fill(0)
		tmp:select(n+1,1):copy(filters.phi.signal[l])
		tmp:select(n+1,2):copy(filters.phi.signal[l])
		filters.phi.signal[l]=tmp
	end
	
	collectgarbage()
	return filters
end

function meshgrid(x,y) -- identical to MATLAB
	local xx = torch.repeatTensor(x, y:size(1),1)
	local yy = torch.repeatTensor(y:view(-1,1), 1, x:size(1))
	return xx, yy
end

function gabor_2d(M,N,sigma,slant,xi,theta,offset,fft_shift,pad,extent) 
	-- if pad == 0 then peridize the singal
	local wv=torch.FloatTensor(N,M,2):fill(0)
	local R=torch.FloatTensor({{torch.cos(theta),-torch.sin(theta)},{torch.sin(theta),torch.cos(theta)}}) -- conversion to the axis..
	local R_inv=torch.FloatTensor({{torch.cos(theta),torch.sin(theta)},{-torch.sin(theta),torch.cos(theta)}})
	local g_modulus=torch.FloatTensor(M,N)
	local g_phase=torch.FloatTensor(M,N,2)
	local A=torch.FloatTensor({{1,0},{0,slant^2}})
	local tmp=R*A*R_inv/(2*sigma^2)
	local x0=torch.linspace(offset+1-(M/2)-1,offset+M-(M/2)-1,M)
	local y0=torch.linspace(offset+1-(N/2)-1,offset+N-(N/2)-1,N)
	local i_

	local extent = extent or 4
	--if pad == 0 then extent=4 end
	for ex=-extent,extent do
		for ey=-extent,extent do
			local x = x0 + ex*M
			local y = y0 + ey*N
			-- Shift the variable by half of their lenght
			if(fft_shift) then      
				local x_tmp=torch.FloatTensor(M)
				local y_tmp=torch.FloatTensor(N)
				for i=1,M do
					i_=(i-(M/2)-1)%M+1
					x_tmp[i_]=x[i]
				end
				for i=1,N do
					i_=(i-(N/2)-1)%N+1
					y_tmp[i_]=y[i]
				end
				x = x_tmp
				y = y_tmp
			end
			
			xx,yy = meshgrid(x,y)
			-- this is simply the result exp(-x^T*tmp*x)
			g_modulus = torch.exp(-(torch.cmul(xx,xx)*tmp[1][1]+torch.cmul(xx,yy)*tmp[2][1]+torch.cmul(yy,xx)*tmp[1][2]+torch.cmul(yy,yy)*tmp[2][2]))
			g_phase = xx*xi*torch.cos(theta)+yy*xi*torch.sin(theta)
			g_phase = complex.unit_complex(g_phase)
			local wv0 = complex.multiply_real_and_complex_tensor(g_phase,g_modulus)
			wv = wv + wv0
		end
	end
	
	local norm_factor=1/(2*math.pi*sigma^2/slant)
	wv:mul(norm_factor)
	
	return wv
end

function morlet_2d(M,N,sigma,slant,xi,theta,offset,fft_shift,pad,extent)
	print('compute morlet 2d with sigma=' .. sigma .. ' and extent=' .. extent)
	print('M=',M,'N=',N,'slant=',slant,'xi=',xi,'theta=',theta,'offset=',offset)
	local wv=gabor_2d(M,N,sigma,slant,xi,theta,offset,fft_shift,pad,extent)
	local g_modulus=complex.realize(gabor_2d(M,N,sigma,slant,0,theta,offset,fft_shift,pad,extent))
	local K=torch.squeeze(torch.sum(torch.sum(wv,1),2))/(torch.squeeze(torch.sum(torch.sum(g_modulus,1),2)))
	
	local mor=wv-complex.multiply_complex_number_and_real_tensor(K,g_modulus)
	
	return mor
end

return filters_bank
