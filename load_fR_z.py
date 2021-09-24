#This  module  is a hack. For some reason Cosmosis  can not pass
#the z array and  the info needed for the f(R) module


####################### Import the Emulator #######################
import numpy as  np
from cosmosis.datablock import names, option_section
####################### Define the Cosmological Parameters #########

def setup(options):
    #GPm_list, PCAm_list = loadalldata()
    z_arr = options.get_double_array_1d(option_section,'z_arr')
    return z_arr 

def execute(block, config):
    z_arr = config
    block['fR', 'z_arr'] = z_arr
    return 0

def cleanup(config):
    return 0
