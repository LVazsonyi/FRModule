import numpy as np  
import pickle
from sklearn.decomposition import PCA
import gpflow


def loadalldata():
    GPm_list, PCAm_list = [], []
    print ('INITIATING THE FR MODULE')
    print ('THIS MAY TAKE A FEW MINUTES \n BUT WE ONLY NEED TO DO THIS ONCE PER MCMC CHAIN')
    for i in range(100):
        GPm, PCAm = modelLoad(i)
        GPm_list += [GPm]
        PCAm_list += [PCAm]
    return GPm_list, PCAm_list


def modelLoad(snap_ID, nRankMax = 6):
    '''
    Loading pretrained GP and PCA models (saved in ./TrainedModels/) for given snapshot

    nRankMax: Number of basis vectors in truncated PCA, default = 6

    '''
    modelDir = "./TrainedModels/"
    GPmodel = modelDir + 'GP_smooth_rank' + str(nRankMax) + 'snap' + str(snap_ID)  
    PCAmodel = modelDir + 'PCA_smooth_rank' + str(nRankMax) + 'snap' + str(snap_ID) 

    ctx_for_loading = gpflow.saver.SaverContext(autocompile=False)
    saver = gpflow.saver.Saver()
    GPm = saver.load(GPmodel, context=ctx_for_loading)
    GPm.clear()
    GPm.compile()
    #PCAm = pickle.load(open(PCAmodel, 'rb'))
    f = open(PCAmodel, 'rb')
    PCAm = pickle.load(f)
    f.close()
    return GPm, PCAm

def GP_predict(gpmodel, para_array):
    '''
    GP prediction of the latent space variables
    '''
    m1p = gpmodel.predict_f(para_array)  # [0] is the mean and [1] the predictive
    W_predArray = m1p[0]
    W_varArray = m1p[1]
    return W_predArray, W_varArray

def scale01(f, fmean = np.array([ 0.1369883,   0.95245613,  0.80035087, -5.93216374,  1.97894743]), fstd = np.array([0.01013207, 0.05845525, 0.0567193, 1.17136822, 1.21287971])):
    '''
    Normalizing the input parameters by mean and variance of the training scheme (log(fr0))
    fmean, ftd: are the mean and std of the experimental design, given in 'TrainedModels/paralims_nCorr_val_2.txt' 
    '''
    return (f - fmean)/fstd

def Emu(gpmodel, pcamodel, para_array):
    '''
    Combining GP prediction with PCA reconstrution to output p(k) ratio for the given snapshot
    '''
    para_array = np.array(para_array)
    # print(para_array)
    para_array[3] = np.log10(para_array[3])
    para_array_rescaled = scale01(f = para_array)
    if len(para_array.shape) == 1:
        # print(para_array_rescaled)
        W_predArray, _ = GP_predict(gpmodel, np.expand_dims(para_array_rescaled, axis=0))
        x_decoded = pcamodel.inverse_transform(W_predArray)
        return np.squeeze(x_decoded)#[0]

def MGemu(Om, ns, s8, fR0, n, z, GPm_list, PCAm_list):
    '''
    Redshift interpolation for any redshift between 0 < z < 49
    ''' 
    # redshift of all snapshots
    # altertively z_all = np.loadtxt('TrainedModels/timestepsCOLA.txt', skiprows=1)[:, 1]
    a = np.linspace(0.0298, 1.00000, 100).round(decimals=7) 
    z_all = (1/a) - 1

    # k values of summary statistics
    # alternatively kvals = np.loadtxt('TrainedModels/'ratiobins.txt')[:,0]
    kb = np.logspace(np.log(0.03), np.log(3.5), 301, base=np.e).round(decimals=7) 
    k1 = 0.5*(kb[0:-1] + kb[1:]).round(decimals=7) 
    kmask = np.array([  0,   1,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
         14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  25,  26,  27,
         28,  29,  30,  31,  32,  33,  34,  35,  36,  38,  39,  40,  41,
         42,  43,  44,  45,  47,  48,  49,  50,  51,  52,  54,  55,  56,
         57,  58,  60,  61,  62,  63,  64,  65,  66,  67,  69,  70,  71,
         73,  74,  76,  77,  79,  80,  82,  84,  85,  87,  88,  89,  91,
         93,  96,  99, 101, 102, 107, 108, 111, 118])
    kvals = k1[..., [i for i in np.arange(300) if i not in kmask]]

    if (z==0):
        ## No redshift interpolation for z=0
        GPm, PCAm =  GPm_list[99], PCAm_list[99]  #modelLoad(snap_ID = 99)
        Pk_interp = Emu(GPm, PCAm, [Om, ns, s8, fR0, n])

    else:
        ## Linear interpolation between z1 < z < z2 
        snap_idx_nearest = (np.abs(z_all - z)).argmin()
        if (z > z_all[snap_idx_nearest]): 
            snap_ID_z1 = snap_idx_nearest - 1    
        else:
            snap_ID_z1 = snap_idx_nearest 
        snap_ID_z2 = snap_ID_z1 + 1

        GPm1, PCAm1 = GPm_list[snap_ID_z1], PCAm_list[snap_ID_z1] #modelLoad(snap_ID = snap_ID_z1)
        Pk_z1 = Emu(GPm1, PCAm1, [Om, ns, s8, fR0, n])
        z1 = z_all[snap_ID_z1]

        GPm2, PCAm2 = GPm_list[snap_ID_z2], PCAm_list[snap_ID_z2] #modelLoad(snap_ID = snap_ID_z2)
        Pk_z2 = Emu(GPm2, PCAm2, [Om, ns, s8, fR0, n])
        z2 = z_all[snap_ID_z2]
                
        Pk_interp = np.zeros_like(Pk_z1)
        Pk_interp = Pk_z2 + (Pk_z1 - Pk_z2)*(z - z2)/(z1 - z2)
    return Pk_interp, kvals
