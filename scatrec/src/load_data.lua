function load_data(datapath,label,N,Kdata,kstart,Ktrain,ktest,Ktest)
	local Kdata = Kdata or 1
	local Ktrain = Ktrain or 1
	local kstart = kstart or 1
	local ktest = ktest or 1
	local filename = datapath .. '/' .. label  .. '_N' .. N .. '.th'
	local imgs = torch.load(filename)
	local Kk = imgs.imgs:size(1)
	if imgs.imgs:dim()==2 then
		Kk = 1
	else
		assert(imgs.imgs:dim()==3)
	end
	local Ktest = Ktest or Kk-ktest+1
	local dataimg = imgs.imgs:view(Kk,1,N,N):narrow(1,kstart,Kdata):float()
	local trainimg = imgs.imgs:view(Kk,1,N,N):narrow(1,1,Ktrain):float()
	local testimg = imgs.imgs:view(Kk,1,N,N):narrow(1,ktest,Ktest):float()
	print('data loaded from file '..filename)
	print('dataimg kstart='..kstart..',Kdata='..Kdata)
	print('trainimg Ktrain='..Ktrain)
	print('testimg ktest='..ktest, 'Ktest=' .. Ktest)
	return dataimg,trainimg,testimg
end
