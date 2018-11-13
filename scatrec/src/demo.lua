#!../exe/luajit/bin/luajit

require 'torch'
require 'scatwave'
require 'optim'
dofile('env.lua')
dofile('do_rec.lua')

local cmd = torch.CmdLine()
cmd:option('-kstart',1, 'start index of data')
cmd:option('-K',1,'number of data')
cmd:option('-Krec',0, 'number of reconstructions')
cmd:option('-N',64, 'data size')
cmd:option('-J',6,'scattering max scale')
cmd:option('-L',4,'scattering angle number')
cmd:option('-name','wn','demo name')
cmd:option('-datapath','./rawdata/','dataset path')
cmd:option('-datalabel','demo_whitenoise','dataset label')
cmd:option('-optmethod','cg2','opotimzation method')
cmd:option('-niter',1000,'optimization epoches')
cmd:option('-gpu',0,'use gpu or not')
cmd:option('-gpuid',1,'set gpu id')
cmd:option('-wAE','o2','scattering model')
cmd:option('-scatfile','','the input file to do synthsis')
opt = cmd:parse(arg or {})

opt.log = './ckpt/' .. getdate()
os.execute('mkdir -p ' .. opt.log)
opt.ampoutput = opt.N
opt.tkt=get_tkt_wAE(opt)
do_rec(opt)

