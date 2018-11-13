
function construct_sc_model(opt,img)
	local wAE = opt.wAE or 'o1'
	local gpu = opt.gpu or 0
	local sc,sc_wmodule,scat

	local scat = scatwave.network_2d_bfs_o.new(
		opt.J,opt.L,img:size(),opt.haar2d,opt.lap2d,
		opt.osr,opt.osj,opt.gxi,opt.gsigma,opt.gext,opt.Q)
	if gpu>0 then scat:cuda() end
	if wAE=='o2' then
		--io.write('o2, 2nd order scattering\n')
		sc = scat:scat_lp_nngraph_o2(opt.ampoutput,opt.lplist1,opt.lplist2,opt.histparams)
		if gpu>0 then sc:cuda() end
	else
		assert(false,'unknown wAE network')
	end
	return sc,sc_wmodule
end

function initrec(opt,img,imgs)
	print('initrec with uniform white noise between -1/2 and 1/2')
	io.write('initrec with uniform white noise between -1/2 and 1/2\n')
	est=torch.rand(img:size())-0.5
	est = est:typeAs(img)
	return est
end

function do_train_o2(opt,dataimg,testimg)
	local K = opt.K
	local N = opt.N
	local J = opt.J
	local L = opt.L
	local gpu = opt.gpu or 0
	local wAE = opt.wAE or 'o1'
	local disp = opt.disp or 0
	local Krec = opt.Krec or K
	
	local sc = construct_sc_model(opt,dataimg)
	
	--io.write('rec from avg mu of samples\n')
	local state = {}
	state['opt']=opt
	local outfile = opt.log .. '/' .. opt.tkt .. '.th'
	
	-- Estimate the mu
	assert(K==1, 'K should be 1 for mirco-model')
	local mu = nil
	if opt.scatfile~='' then
		print('read scat from scatfile')
		-- read the mu from the scatfile
		local lines = lines_from(opt.scatfile)
		local nscat = 0
		local mu_list = {}
		for k,v in pairs(lines) do
			-- skip line 1, it is the 'j1,theta1,j2,theta2,mu,sigma'
			if k==1 then
				assert(v=='j1,theta1,j2,theta2,mu,sigma')
			else
				--print('read line[' .. k .. ']', v)
				local splitline = v:split(',')
				local j1=tonumber(splitline[1])
				local theta1=tonumber(splitline[2])
				local j2=tonumber(splitline[3])
				local theta2=tonumber(splitline[4])
				local mu=tonumber(splitline[5])
				local sigma=tonumber(splitline[6])
				--print(j1,theta1,j2,theta2,mu,sigma)
				
				if j1==0 and theta1==0 and j2==-1 and theta2==-1 then
					table.insert(mu_list,mu)
					assert(sigma>=0)
					table.insert(mu_list,sigma)
				elseif j1==0 and theta1>=1 and theta1<=3 and j2==-1 and theta2==-1 then
					assert(mu==-1)
					assert(sigma>=0)
					table.insert(mu_list,sigma)
				else
					-- the rest of the coeifficient should all be positive
					if mu~=-1 then
						assert(mu>=0)
						table.insert(mu_list,mu)
					end
					if sigma~=-1 then
						assert(sigma>=0)
						table.insert(mu_list,sigma)
					end
				end
			end
		end
		local nscat = #mu_list
		mu = torch.FloatTensor(1,nscat,1,1)
		for i,v in pairs(mu_list) do
			mu[1][i][1][1]=v
		end
		if gpu>0 then
			mu=mu:cuda()
		end
	else
		print('compute scat from data')
		assert(K == dataimg:size(1), 'K is not the same as dataimg')	
		--local mu_k = {}
		--local inp_k = {}
		for k=1,K do
			local img_=dataimg:narrow(1,k,1)
			--print('img ' .. k .. ' mean is ' .. img_:mean() .. '\n')
			local muk = sc:forward(img_)
			--inp_k[k] = img_:double()
			--mu_k[k] = muk:double()
			if mu == nil then
				mu = muk:clone()
			else
				mu:add(muk)
			end
		end
		mu:mul(1/K)
		print('Phi size is',mu:size())
	end
	
	--io.write('avg mu dim is '..mu:nElement()..'\n')
	state.mu_0 = mu:double()

	-- Reconstruction
	-- generate Krec samples from mu
	if Krec == 0 then
		print('output Phi in txt format')
		local Phi = state.mu_0:view(-1)
		-- assume it is o2 model with defaults
		assert(opt.wAE=='o2')
		io.write('j1,theta1,j2,theta2,mu,sigma\n')
		-- mu, sigma_J
		io.write('0,0,-1,-1,' .. Phi[1] .. ',' .. Phi[2] .. '\n')
		-- sigma_haar
		io.write('0,1,-1,-1,' .. -1 .. ',' .. Phi[3] .. '\n')
		io.write('0,2,-1,-1,' .. -1 .. ',' .. Phi[4] .. '\n')
		io.write('0,3,-1,-1,' .. -1 .. ',' .. Phi[5] .. '\n')
		-- for each j1
		local idx=6
		for j1=0,J-1 do
			-- mu_j1
			for q1=1,L do
				io.write((j1+1) .. ',' .. q1 .. ',-1,-1,' ..
						Phi[idx] .. ',' .. -1 .. '\n')
				idx=idx+1
			end
			-- sigma_j1
			for q1=1,L do
				io.write((j1+1) .. ',' .. q1 .. ',-1,-1,' ..
						-1 .. ',' .. Phi[idx]  .. '\n')
				idx=idx+1
			end
			for j2=j1,J-1 do
				-- for each j2
				-- only sigma_(j1,j2)
				for q1=1,L do
					for q2=1,L do
						io.write((j1+1) .. ',' .. q1 .. ',' .. (j2+1) .. ',' .. q2 .. ',' ..
								-1 .. ',' .. Phi[idx] .. '\n')
						idx=idx+1
					end
				end
			end
		end
	else
		print('do reconstructions')
		for k=1,Krec do
			--scattering inference reconstruction
			local img_=dataimg:narrow(1,1,1)
			local est = initrec(opt,img_,testimg)
			local state_
			state_ = do_infer_demo(sc,mu,est,opt)
			state[k]=state_
			state[k].inp_mu_t = state.mu_0
			print('rec k=' .. k.. ' done.')
			torch.save(outfile,state)
		end
	end
	collectgarbage()
end
