--[[
     ScatWave implementation of Scattering Network
     Written by Edouard Oyallon
     Team DATA ENS
     Copyright 2015
]]

local tools={}

function tools.is_complex(x)
   return (x:size(x:dim())==2)
end

function tools.are_equal_dimension(x,y)
   return (torch.sum(torch.abs(torch.LongTensor(x:size())-torch.LongTensor(y:size())))==0)
end
function tools.are_equal(x,y)
   return (torch.sum(torch.abs(x-y))==0)
end

function tools.concatenateLongStorage(x,y)
      if(not x) then
         return y
      else
         local z=torch.LongStorage(#x+#y)
         for i=1,#x do
            z[i]=x[i]
         end   
         for i=1,#y do
            z[i+#x]=y[i]
         end
         return z
      end
   end


-- Very close from utils.lua in nn except that we need to copy the same stride..
function tools.recursiveType(param, type_str)
   if torch.type(param) == 'table' then
      for k, v in pairs(param) do
         param[k] = tools.recursiveType(v, type_str)
      end
   elseif torch.isTensor(param) then
      local st=param:stride()
      local si=param:size()
      local of=param:storageOffset()
      local stor=param:storage()
      local new_stor
      if(type_str=='torch.CudaTensor') then
         new_stor=torch.CudaStorage(stor:size())
      elseif (type_str=='torch.FloatTensor') then
         new_stor=torch.FloatStorage(stor:size())
      else
         error('Storage not supported')
      end
      new_stor:copy(stor)
      local tmp=torch.Tensor():type(type_str)
      param = tmp:set(new_stor,of,si,st)
   end
   return param
end

-- input: the scattering network
function tools.scat_display(net,arg)
	--print('Display the scatnet',net)
	print ('Loading qt ...')
	require 'qtuiloader'
	require 'qtgui'
	require 'qtwidget'
	require 'qttorch'
	require 'image'
	
	local title = arg.title or 'scatnet'	
	local zoom = arg.zoom or 1
	local wmax = arg.wmax or 1280
	local hmax = arg.hmax or 4000
	local w = arg.w or 0
	local h = arg.h or 0
	local win = arg.win or qtwidget.newwindow(wmax, hmax, title)
	--	qtwidget.newpdf(wmax,hmax,'./'..title..'.pdf')
	
	if net.modules then
		local modules = net.modules		
		for i = 1,#modules do
			local m = modules[i]
			arg.win = win
			arg.h = h
			h,win = scat_display(m,arg)
		end
	else
		local ou = net.output
		print('display ou size is',ou:size())
		local dh = 0
		local dw = 0
		if ou:dim() == 4 then
			print('display module (h,w,max,min,dh,dw)',
				  h,w,ou:max(),ou:min(),ou:size(3),ou:size(4))
			--local title = tostring(m)
			--local win = qtwidget.newwindow(wmax, hmax, title)
			for i=1,ou:size(1) do
				for j=1,ou:size(2) do
					local im = ou:select(1,i):select(1,j)
					dh = math.ceil(im:size(1)*zoom)
					dw = math.ceil(im:size(2)*zoom)
					if w + dw > wmax then
						h = h + dh + 1
						w = 0
					end
					image.display{
						image=im, win=win,
						y=h, x=w, zoom=zoom}
					w = w + dw + 1
				end	
			end
			--h = 0
			--w = 0
		end
		h = h + dh + 2
		w = 0
	end
	return h,win
end

function tools.file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

-- input: 3d tensor [C][N][M], margin
-- output: 2d tensor [C/W*(N+2*margin)][W*(M+2*margin)]
--[[
function tools.makegrids(img3,W,margin)
	local C = img3:size(1)
	local N = img3:size(2)
	local M = img3:size(3)
	local W = W or math.floor(1024/M)
	local margin = margin or 1
	local imgs = torch.FloatTensor(C*(N+margin),C*(M+margin)):fill(0)
	for c=1,C do
		local x = (c-1)*(N+2*margin)+1+margin
		local y = (c-1)*(M+2*margin)+1+margin
		imgs:narrow(1,x,
			
end
--]]

return tools
