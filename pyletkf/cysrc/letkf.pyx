from __future__ import division
import numpy as np
import scipy.linalg as sp_linalg
cimport numpy as np

DTYPE = np.float64
DTYPE_int = np.int64
DTYPE_c = np.complex64
ctypedef np.float64_t DTYPE_t
ctypedef np.int64_t DTYPE_t_int
ctypedef np.complex64_t DTYPE_t_c

#@cython.boundscheck(False)
#@cython.nonecheck(False)
def letkf(np.ndarray[DTYPE_t,ndim=3] allx, np.ndarray[DTYPE_t,ndim=2] observation,
          np.ndarray[DTYPE_t,ndim=2] obserr, list patches, list obsvars,
          list assimReaches, float undef, str guess):
    
    """
    Data Assimilation with Local Ensemble Transformed Kalman Filter
    Args:
        allx (np.ndarray): state vectors ([nvars,eNum,nReach])
        observation (np.ndarray): observation with observed or undef values
                                ([nobsvar, nReach], nobsvar <= nvars)
        obserr (np.ndarray): observation error (std) ([nobsvar, nReach])
        patches (list): local patch ids
        obsvars (list): either 1 or 0 in shape of [nvars] with same order as observation;
                      1: included in observation
                      0: not included in observation
        assimReaches (list): ids where to be assimilated;
                           for partial assimilation or parallellization.
        undef (float): undef value for the observation
        guess (str): if "mean", where observation is not available replace all values
                     with ensemble mean as a single posteroir. if "prior", just use prior ensembles 
                     for posterior (no update at all).
    Notes:                   
        
        nvars: number of model variables assimilated (state vector)
        nobsvar: number of variables observation is available, usually less than nvars.
        obsvars: binary flags which layer of observation is corresponding with variables
                 in state vector.
    """

    # c type declaration
    assert allx.dtype==DTYPE and observation.dtype==DTYPE and obserr.dtype==DTYPE

    cdef int nvars = allx.shape[0]
    cdef int eNum = allx.shape[1]
    cdef int nReach = allx.shape[2]
    cdef np.ndarray[DTYPE_t,ndim=3] globalxa = np.zeros([nvars,eNum,nReach],dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] xt
    cdef np.ndarray[DTYPE_t,ndim=2] xf
    cdef np.ndarray[DTYPE_t,ndim=1] xf_mean
    cdef np.ndarray[DTYPE_t,ndim=2] xf_m
    cdef np.ndarray[DTYPE_t,ndim=2] xf_me
    cdef np.ndarray[DTYPE_t,ndim=2] xa
    cdef np.ndarray[DTYPE_t,ndim=2] dd
    cdef np.ndarray[DTYPE_t,ndim=1] local_obs_line
    cdef np.ndarray[DTYPE_t,ndim=1] local_obsErr_line
    cdef list xfs
    cdef list xts
    cdef list local_obs_lines
    cdef list local_obsErr_lines
    cdef list Ws = []
    cdef int ovs
    cdef int i
    cdef int idx
    cdef int reach
    cdef float ro = 1.0  # covariance inflation factor

    cdef np.ndarray[DTYPE_t,ndim=2] H
    cdef np.ndarray[DTYPE_t,ndim=2] I = np.identity(eNum,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] Rinv
    cdef np.ndarray[DTYPE_t,ndim=2] Ef
    cdef np.ndarray[DTYPE_t,ndim=2] HEft
    cdef np.ndarray[DTYPE_t,ndim=2] HEf
    cdef np.ndarray[DTYPE_t,ndim=2] HEftRinvHEf
    cdef np.ndarray[DTYPE_t,ndim=2] VDVt
    cdef np.ndarray[DTYPE_t,ndim=1] w
    cdef np.ndarray[DTYPE_t,ndim=2] v
    cdef np.ndarray[DTYPE_t,ndim=2] Dinv
    cdef np.ndarray[DTYPE_t,ndim=2] Dsqr
    cdef np.ndarray[DTYPE_t,ndim=2] Ea
    cdef np.ndarray[DTYPE_t,ndim=2] Pa = np.zeros([eNum,eNum],dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] Pa_sqr = np.zeros([eNum,eNum],dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] d
    cdef np.ndarray[DTYPE_t,ndim=2] Wvec
    cdef np.ndarray[DTYPE_t,ndim=2] Warr = np.zeros([eNum,eNum],dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] W
    # type declaration ends

    # main loop
    for reach in assimReaches:
        # local patch
        patch = patches[reach]

        # data augumentation (vectorizing)
        xfs = []
        for i in range(nvars):
            xfs.append(allx[i, :, patch].T)  # [eNum, patch]; need T.
        xf = (np.concatenate(xfs, axis=1)).T  # [eNum, patch*nvars] > [patch*nvars, eNum]
        patch_nums = len(patch)*nvars
        #xf = allx[:,patch].reshape(-1,eNum)  # bug; .T

        # calc. background increments
        xf_mean = xf.mean(axis=1)
        xf_me = np.ones([patch_nums, eNum])
        for i in range(0, eNum):
            xf_me[:,i] = xf_mean
        xf_m = xf_mean.reshape(-1,1)

        # observations
        idx = 0
        xts = []
        local_obs_lines = []
        local_obsErr_lines = []
        for i in obsvars:
            if i == 1:
                xt = observation[idx, patch]
                local_obs_line = np.ones([len(patch)],dtype=DTYPE) # initialize
                local_obs_line[np.where(xt-undef < 1.)] = 0
                local_obsErr_line = obserr[idx, patch].flatten()
                idx += 1
            else:
                xt = np.zeros(len(patch))
                local_obs_line = np.zeros(len(patch))
                local_obsErr_line = np.ones(len(patch))*1e-8
            xts.append(xt)
            local_obs_lines.append(local_obs_line)
            local_obsErr_lines.append(local_obsErr_line)
        xt = np.hstack(xts)
        local_obs_line = np.hstack(local_obs_lines)
        local_obsErr_line = np.hstack(local_obsErr_lines)
        ovs = local_obs_line.sum()
 
        if ovs > 0:
            """
                observation is available in a local patch.
                LETKF is activated.
            """
            # initialize
            H = np.zeros([patch_nums,patch_nums],dtype=DTYPE)
            #
 
            H[np.where(local_obs_line == 1.)[0],np.where(local_obs_line == 1.)[0]] = 1
            Ef = xf - xf_m
            Rinv = np.diag((local_obsErr_line**2)**(-1)).astype(DTYPE)
            
            HEft = np.dot(H,Ef).T # (patch_nums x eNum)T = eNum x patch_nums
            HEf = np.dot(H,Ef) # patch_nums x eNum
            HEftRinvHEf = np.dot(np.dot(HEft,Rinv),HEf) # (eNum x patch_nums) * (patch_nums x patch_nums) *(patch_nums x eNum) = (eNum x eNum)

            # VDVt = I + HEftRinvHEf
            VDVt = (eNum-1)*I/ro + HEftRinvHEf
            w,v = np.linalg.eigh(VDVt,UPLO="U")
            
            Dinv = np.diag((w+1e-20)**(-1))
            Dsqr = sp_linalg.sqrtm(Dinv)
 
            Pa = np.dot(np.dot(v,Dinv),v.T)
            Pa_sqr = np.dot(np.dot(v,Dsqr),v.T)

            d = (np.dot(H,xt) - np.dot(H,xf_m).T).reshape(-1,1)
            Wvec = np.dot(np.dot(np.dot(Pa,np.dot(H,Ef).T),Rinv),d)

            for i in range(0,eNum):
                Warr[:,i] = Wvec.reshape(eNum)
            # W = Pa_sqr + Warr
            W = Pa_sqr*np.sqrt(eNum-1.0) + Warr
            Ea = np.dot(Ef,W)
            # if reach  == 23916:
            #     print("Ef", Ef)
            #     print("Pa_sqr", Pa_sqr)
            #     print("Wvec", Wvec)
            #     print("Warr", Warr)
            #     print("W", W)
            #     print("Ea", Ea)
            # print(Ws)

            xa = xf_me + Ea
            Ws.append(W)
 
        else:
            """
                No observation is available in a local patch.
                No-assimilation. Return a best guess.
            """
            if guess == "mean":
                xa = xf_me
            elif guess == "prior":
                xa = xf
            else:
                raise NotImplementedError(guess)
            W = np.identity(patch_nums) #np.identity?
            Ws.append(W)
        for i in range(nvars):
            dd = xa[i*len(patch):(i+1)*len(patch)]
            globalxa[i,:,reach] = dd[patch.index(reach),:]
 
    return globalxa, Ws

#@cython.boundscheck(False)
#@cython.nonecheck(False)
def noCostSmoother(np.ndarray[DTYPE_t,ndim=4] allx, list patches,
                   list Ws, list assimReaches):
    """
    No cost ensemble Kalman Smoother.
    Args:
        allx (np.ndarray): [nvars,eNum,time,nReach]
        patches (list): local patch ids
        Ws (list): Weight of Ef from LETKF
        assimReaches (list): ids where to be assimilated
    """
    # c type declaration
    assert allx.dtype==DTYPE

    cdef int nvars = allx.shape[0]
    cdef int eNum = allx.shape[1]
    cdef int nT = allx.shape[2]
    cdef int nReach = allx.shape[3]
    cdef int reach
    cdef int patch_nums
    cdef list xfs
    
    cdef np.ndarray[DTYPE_t,ndim=3] globalxa = np.zeros([nvars, eNum, nT, nReach],dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] xf
    cdef np.ndarray[DTYPE_t,ndim=1] xf_mean
    cdef np.ndarray[DTYPE_t,ndim=2] xf_m
    cdef np.ndarray[DTYPE_t,ndim=2] xf_me
    cdef np.ndarray[DTYPE_t,ndim=2] xa
    cdef np.ndarray[DTYPE_t,ndim=2] dd
    cdef np.ndarray[DTYPE_t,ndim=2] Ef
    cdef np.ndarray[DTYPE_t,ndim=2] Ea
    # c type declaration ends

    for reach in assimReaches:
        # local patch
        patch = patches[reach]
        patch_nums = len(patch)
        W = Ws[reach]
        for t in range(0,nT):
            xfs = []
            for i in range(nvars):
                xfs.append(allx[i, :, t, patch].T)  # [eNum, patch]; need T.
            xf = (np.concatenate(xfs, axis=1)).T  # [eNum, patch*nvars] > [patch*nvars, eNum]
            patch_nums = len(patch)*nvars
            #xf = allx[:,patch].reshape(-1,eNum)  # bug; .T

            # calc. background increments
            xf_mean = xf.mean(axis=1)
            xf_me = np.ones([patch_nums, eNum])
            for i in range(0, eNum):
                xf_me[:,i] = xf_mean
            xf_m = xf_mean.reshape(-1,1)

            if type(W) is not int:
                Ef = xf - xf_m
                Ea = np.dot(Ef,W)
                xa = xf_me + Ea
            else:
                xa = xf_me
            for i in range(nvars):
                dd = xa[i*len(patch):(i+1)*len(patch)]
                globalxa[i,:,t,reach] = dd[patch.index(reach),:]

    return globalxa
