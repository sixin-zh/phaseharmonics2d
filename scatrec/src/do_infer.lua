require 'torch'
require 'sys'

function do_feval(net,crit,image_est,target,dL_dimage)
	-- forward
	local output = net:forward( image_est )
	local L = crit:forward( output , target )
	-- backward
	if dL_dimage then
		dL_dimage:zero()
	end
	local dL_do = crit:backward( output, target )
	local dL_dimage = net:backward( image_est , dL_do )
	return L, dL_dimage
end

function do_infer_demo(net,target,image_est,opt,diagscale)
	local crit
	local penalty = opt.penalty or 1
	local gpu = opt.gpu or 0
	local verbose = opt.verbose or 1
	local niter = opt.niter or 1
	local checkpoint = opt.checkpoint or niter
	
	local crit
	if diagscale then
		print('use diag mse metric crit')
		crit = nn.DiagMSECriterion(0,diagscale)
	else
		print('use mse metric crit')
		crit = nn.MSECriterion(0) -- no average loss
	end
	if gpu>0 then
		crit=crit:cuda()
	end
	
	local t0 = target:clone():zero()
	local L0 = crit:forward(t0, target)
	
	local optimState = nil
	if opt.optmethod=='cg2' then
		dofile('optim_cg2.lua')
		optimState = {
			verbose = opt.optim_verbose or 0,
			rho = opt.optim_rho or 0.05, -- 1e-2,
			sig = opt.optim_sig or 0.5,
			maxIter = opt.optim_maxIter or 1000,-- 30,
			maxEval = opt.optim_maxEval or 5000, -- 60,
			targetValue = opt.targetValue or 0,
		}
	else
		assert(false)
	end
	assert(opt.optmethod)
	local optfun=optim[opt.optmethod]
	
	collectgarbage()
	local dL_dimage = nil
	local L = nil
	local gborder = opt.gborder or 0
	local N = image_est:size(3)
	assert(image_est:size(1)==1)
	assert(image_est:size(2)==1)
	assert(image_est:size(4)==N)
	sys.tic()
	local state = {}
	local outfile = nil
	if opt.tkt then
		outfile = opt.log .. '/' .. opt.tkt .. '.th'
	end
	local function feval(x)
		assert(x:data() == image_est:data())
		L, dL_dimage =
			do_feval(net,crit,image_est,target,dL_dimage)
		return L, dL_dimage:viewAs(x)
	end
	
	for ite=1,niter do
		local x,ret=optfun(feval, image_est:view(-1), optimState)
		local loss=ret[1] 
		-- FOR cg, this is the loss for final value
		local stop=ret[2]
		
		if ite% verbose == 0 or ite == 1 then
			local logstr = 'iter ' .. ite .. ' loss is ' .. loss ..
				' gradnorm is ' .. dL_dimage:norm() ..
				' (in ' .. sys.toc() .. 's)'
			io.write(logstr .. '\n')
			print(logstr)
			io.flush()
			sys.tic()
		end
		if stop==1 or ite% checkpoint == 0 then
			state.ite = ite
			--state.loss_best = loss
			--state.gradnorm_best = dL_dimage:norm()
			state.loss = loss
			--state.gradnorm = state.gradnorm_best
			collectgarbage()
		end
		if stop==1 then
			print('optim stop at iter='..ite)
			break
		end
	end
	state.rec_t = image_est:double()
	state.rec_mu_t = net:forward( image_est ):double() 
	return state
end
