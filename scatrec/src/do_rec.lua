
function do_rec(opt)
	local N = opt.N -- size of image side
	local Kdata = opt.K -- total number of dataimg
	local kstart = opt.kstart or 1 -- the start of dataimg
	local Ktrain = opt.Ktrain or opt.K -- total number of trainimg, trainimg start=1
	local Ktest = opt.Ktest or 1 -- total number of testimg
	local ktest = opt.ktest or 1 -- the start of testimg
	
	opt.seed = opt.seed or os.time()
	math.randomseed(opt.seed)
	torch.manualSeed(opt.seed)
	
	-- start logging
	print(opt.tkt)
	os.execute('mkdir -p ./ckpt')
	local logpath = opt.log .. '/' .. opt.tkt .. '.log'
	local logfile = io.open(logpath,'w')
	io.output(logfile)
	--print_r(opt)
	
	-- load cuda
	local gpu = opt.gpu or 0
	if gpu>0 then
		print('load cuda')
		require 'cutorch'
		require 'cunn'
		cutorch.manualSeed(opt.seed)
		if cutorch.getDeviceCount()>1 then
			local gpuid = opt.gpuid or 1 -- cutorch.getDeviceCount()
			print('set gpuid='..gpuid)
			cutorch.setDevice(gpuid)
		end
	end
	
	-- load data
	local dataimg,trainimg,testimg
	assert(opt.datapath ~= nil)
	dofile('load_data.lua')
	dataimg,trainimg,testimg = load_data(opt.datapath,opt.datalabel,
										 N,Kdata,kstart,Ktrain,ktest,Ktest)
	
	if gpu>0 then
		print('convert img to cuda')
		dataimg = dataimg:cuda()
		trainimg = trainimg:cuda()
		testimg = testimg:cuda()
	end
	
	-- start the reconstruction
	dofile('do_train.lua')
	dofile('do_infer.lua')
	
	print('train with model o2')
	do_train_o2(opt,dataimg,testimg)
	
	io.close(logfile)
	print('check log at:',logpath)
end
