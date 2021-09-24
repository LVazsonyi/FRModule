####################### Import the Emulator #######################

from emu import *
import matplotlib.pylab as plt
from cosmosis.datablock import names, option_section
import gc
import scipy
from scipy.interpolate import interp2d
from scipy.interpolate import interp1d

####################### Define the Cosmological Parameters #########

def setup(options):
    GPm_list, PCAm_list = loadalldata()
    return GPm_list, PCAm_list 

def execute(block, config):
    GPm_list, PCAm_list = config
    h = block['cosmological_parameters', 'h0']
    Om = (h**2)*block['cosmological_parameters', 'omega_m']
    ns = block['cosmological_parameters', 'n_s']
    sig8 = block['cosmological_parameters', 'sigma_8']
    z_arr = block['fR', 'z_arr']
    log_fr0 = block['cosmological_parameters', 'log_fr0']
    fr0 = 10. ** log_fr0
    n = block['cosmological_parameters', 'n']
    curr_pk = block['matter_power_nl', 'p_k']
    curr_k = block['matter_power_nl', 'k_h']
    curr_z = block['matter_power_nl', 'z']
    pk_r = [] 
    s8 = sig8 * (Om/0.3)**(1/2)
    print('finding ratios...')
    for z in z_arr:
        pkratio, k = MGemu(Om=Om, ns=ns, s8=s8, fR0=fr0, n=n, z=z, GPm_list=GPm_list, PCAm_list=PCAm_list)
        pk_r.append(pkratio)
        gc.collect()
    f = interp2d(k, z_arr, pk_r)
    pk_r = []
    for i in curr_z:
        pk_r.append(f(k, i))
    func = interp2d(curr_k, curr_z, curr_pk)
    print('scaling...')
    pk = []
    for i, pkratio in zip(curr_z, pk_r):
        curr = []
        for fr, norm in zip(pkratio, func(k, i)):
            curr.append(fr*norm)
        pk.append(curr)
    pk = np.array(pk)
    #block.put_double_array_2d['matter_power_nl', 'p_k_fr', pk]
    block['matter_power_nl', 'p_k'] = pk
    block['matter_power_nl', 'k_h'] = k 
    return 0

def cleanup(config):
    return 0
