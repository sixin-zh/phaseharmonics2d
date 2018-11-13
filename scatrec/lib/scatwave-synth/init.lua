--[[
     ScatWave implementation of Scattering Network
     Written by Edouard Oyallon
     Team DATA ENS
     Copyright 2015
]]

require 'torch'

scatwave = {} -- define the global ScatNet table

scatwave.network_2d_bfs = require 'scatwave.network_2d_bfs'
scatwave.network_2d_bfs_o = require 'scatwave.network_2d_bfs_o'

require 'scatwave.op_fft2d'
require 'scatwave.op_ifft2d'
require 'scatwave.op_down2d'
require 'scatwave.op_lp2dg'
require 'scatwave.op_modulus'
require 'scatwave.op_conv2d_infft'
require 'scatwave.op_subdc'
require 'scatwave.op_hist'
require 'scatwave.op_histsig0'
require 'scatwave.op_sigmoid0'

return scatwave
