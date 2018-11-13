require 'torch'
require 'sys'
require 'os'
require 'optim'

torch.setnumthreads(2)
torch.setdefaulttensortype('torch.FloatTensor')
require 'nn'

function trim1(s)
	return (s:gsub("^%s*(.-)%s*$", "%1"))
end
-- from PiL2 20.4

function pometicket(sec)
	local sec = sec or 0
	local t = os.date("*t")
	local ts = string.format('%d%.2d%.2d_%.2d%.2d%.2d',
							 t.year, t.month, t.day, t.hour, t.min, t.sec + sec)
	local hs = io.popen('hostname -s'):read()
	return hs .. '_' .. ts
end

function get_tkt_wAE(opt)
	if opt.J and opt.L then
		if opt.Q then
			return opt.name .. '_' .. opt.wAE .. '_N'..opt.N..
				'_J'..opt.J..'_L'..opt.L..'_Q'..opt.Q..
				'_K'..opt.K..'_'..pometicket()
		else
			return opt.name .. '_' .. opt.wAE .. '_N'..opt.N..'_J'..opt.J..'_L'..opt.L..'_K'..opt.K..'_'..pometicket()
		end
	elseif opt.Jg and opt.Lg then
		return opt.name .. '_' .. opt.wAE .. '_N'..opt.N
			..'_Js'..opt.Js..'Ls'..opt.Ls
			..'_Jg'..opt.Jg..'Lg'..opt.Lg
			..'_K'..opt.K..'_'..pometicket()
	end
end

function gethost()
	local hs = io.popen('hostname -s'):read()
	return hs
end

function getdate()
	local t = os.date("*t")
	local ts = string.format('%d%.2d%.2d',t.year, t.month, t.day)
	return ts
end

function gettime()
	local t = os.date("*t")
	local ts = string.format('%.2d%.2d%.2d',t.hour, t.min, t.sec)
	return ts
end

function print_r ( t )  
    local print_r_cache={}
    local function sub_print_r(t,indent)
        if (print_r_cache[tostring(t)]) then
            print(indent.."*"..tostring(t))
        else
            print_r_cache[tostring(t)]=true
            if (type(t)=="table") then
                for pos,val in pairs(t) do
                    if (type(val)=="table") then
                        io.write(indent.."["..pos.."] => "..tostring(t).." {")
						io.write('\n')
                        sub_print_r(val,indent..string.rep(" ",string.len(pos)+8))
                        io.write(indent..string.rep(" ",string.len(pos)+6).."}")
						io.write('\n')
                    elseif (type(val)=="string") then
                        io.write(indent.."["..pos..'] => "'..val..'"')
						io.write('\n')
                    else
                        io.write(indent.."["..pos.."] => "..tostring(val))
						io.write('\n')
                    end
                end
            else
                io.write(indent..tostring(t))
				io.write('\n')
            end
        end
    end
    if (type(t)=="table") then
        io.write(tostring(t).." {")
		io.write('\n')
        sub_print_r(t,"  ")
        io.write("}")
		io.write('\n')
    else
        sub_print_r(t,"  ")
    end
	io.write('\n')
end

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

-- get all lines from a file, returns an empty 
-- list/table if the file does not exist
function lines_from(file)
  if not file_exists(file) then return {} end
  lines = {}
  for line in io.lines(file) do 
    lines[#lines + 1] = line
  end
  return lines
end

function string:split(sep)
   local sep, fields = sep or ":", {}
   local pattern = string.format("([^%s]+)", sep)
   self:gsub(pattern, function(c) fields[#fields+1] = c end)
   return fields
end
