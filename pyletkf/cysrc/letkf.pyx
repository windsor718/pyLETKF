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
          list assimReaches, float undef):
    
    """
    Data Assimilation with Local Ensemble Transformed Kalman Filter
    Args:
        allx (np.ndarray): state vectors ([nvar,eNum,nReach])
        observation (np.ndarray): observation with observed or undef values
                                ([nobsvar, nReach], nobsvar <= nvar)
        obserr (np.ndarray): observation error (std) ([nobsvar, nReach])
        patches (list): local patch ids
        obsvars (list): either 1 or 0 in shape of [nvar] with same order as observation;
                      1: included in observation
                      0: not included in observation
        assimReaches (list): ids where to be assimilated;
                           for partial assimilation or parallellization.
        undef (float): undef value for the observation
    Notes:
        nvar: number of model variables assimilated (state vector)
        nobsvar: number of variables observation is available, usually less than nvar.
        obsvars: binary flags which layer of observation is corresponding with variables
                 in state vector.
    """

    # c type declaration
    assert allx.dtype==DTYPE and observation.dtype==DTYPE and obserr.dtype==DTYPE

    cdef int nvar = allx.shape[0]
    cdef int eNum = allx.shape[1]
    cdef int nReach = allx.shape[2]
    cdef np.ndarray[DTYPE_t,ndim=2] globalxa = np.zeros([nvars,eNum,nReach],dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=1] xt
    cdef np.ndarray[DTYPE_t,ndim=2] xf
    cdef np.ndarray[DTYPE_t,ndim=1] xf_mean
    cdef np.ndarray[DTYPE_t,ndim=2] xf_m
    cdef np.ndarray[DTYPE_t,ndim=2] xf_me
    cdef np.ndarray[DTYPE_t,ndim=2] xa
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

    cdef np.ndarray[DTYPE_t,ndim=2] H
    cdef np.ndarray[DTYPE_t,ndim=2] I = np.identity(eNum,dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] Rinv
    cdef np.ndarray[DTYPE_t,ndim=2] Ef
    cdef np.ndarray[DTYPE_t,ndim=2] Ea
    cdef np.ndarray[DTYPE_t,ndim=2] Pa = np.zeros([eNum,eNum],dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] Pa_sqr = np.zeros([eNum,eNum],dtype=DTYPE)
    cdef np.ndarray[DTYPE_t,ndim=2] d
    cdef np.ndarray[DTYPE_t,ndim=2] Warr = np.zeros([eNum,eNum],dtype=DTYPE)
    # type declaration ends

    # main loop
    for reach in assimReaches:
        # local patch
        patch = patches[reach]

        # data augumentation (vectorizing)
        xfs = []
        for i in range(nvar):
            xfs.append(allx[i, :, patch])
        xf = (np.concatenate(xfs, axis=1)).T  # [eNum, patch*nvar] > [patch*nvar, eNum]
        patch_nums = len(patch)*nvar
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
        for i in obsflag:
            if i == 1:
                xt = observation[idx, patch]
                local_obs_line = np.ones([len(patch)],dtype=DTYPE) # initialize
                local_obs_line[np.where(xt-undef < 1.)] = 0
                local_obsErr_line = obserr[patch].flatten()
            else:
                xt = np.zeros(len(patch))
                local_obs_line = np.zeros(len(patch))
                local_obsErr_line = np.zeros(len(patch))
            xts.append(xt)
            local_obs_lines.append(local_obs_line)
            local_obsErr_lines.append(local_obsErr_line)
        xt = np.hstack(xts)
        print(xt.shape)
        local_obs_line = np.hstack(local_obs_lines)
        print(local_obs_lines.shape)
        local_obsErr_line = np.hstack(local_obsErr_lines)
        print(local_obsErr_line)
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
            
            HEft = np.dot(H,Ef).T # (patch_num x eNum)T = eNum x patch_num
            HEf = np.dot(H,Ef) # patch_num x eNum
            HEftRinvHEf = np.dot(np.dot(HEft,Rinv),HEf) # (eNum x patch_num) * (patch_num x patch_num) *(patch_num x eNum) = (eNum x eNum)
 
            VDVt = I + HEftRinvHEf
            w,v = np.linalg.eigh(VDVt,UPLO="U")
            
            Dinv = np.diag((w+1e-20)**(-1))
            Dsqr = sp_linalg.sqrtm(Dinv)
 
            Pa = np.dot(np.dot(v,Dinv),v.T)
            Pa_sqr = np.dot(np.dot(v,Dsqr),v.T)
            
            d = (np.dot(H,xt) - np.dot(H,xf_m).T).reshape(-1,1)
            Wvec = np.dot(np.dot(np.dot(Pa,np.dot(H,Ef).T),Rinv),d)
            for i in range(0,eNum):
                Warr[:,i] = Wvec.reshape(eNum)
            W = Pa_sqr + Warr
            Ea = np.dot(Ef,W)
            
            xa = xf_me + Ea
            Ws.append(W)
 
        else:
            """
                No observation is available in a local patch.
                No-assimilation. Return prior ensemble mean as a best guess.
            """
            xa = xf_me
            W = 1 #np.identity?
            Ws.append(W)
 
        globalxa[:,:,reach] = xa[patch.index(reach),:]  # need to fix
 
    return globalxa, Ws

#@cython.boundscheck(False)
#@cython.nonecheck(False)
def noCostSmoother(np.ndarray[DTYPE_t,ndim=4] allx, list patches,
                   list Ws, list assimReaches):
    """
    No cost ensemble Kalman Smoother.
    Args:
        allx (np.ndarray): [nvar,eNum,time,nReach]
        patches (list): local patch ids
        Ws (list): Weight of Ef from LETKF
        assimReaches (list): ids where to be assimilated
    """
    # c type declaration
    assert allx.dtype==DTYPE

    cdef int eNum = allx.shape[0]
    cdef int nT = allx.shape[1]
    cdef int nReach = allx.shape[2]
    cdef int reach
    
    cdef np.ndarray[DTYPE_t,ndim=3] globalxa = np.zeros([eNum,nT,nReach],dtype=DTYPE)
    # c type declaration ends

    for reach in assimReaches:
        # local patch
        patch = patches[reach]
        patch_nums = len(patch)
        W = Ws[reach]
        for t in range(0,nT):
            xf = allx[:,t,patch].reshape(-1,eNum)
            xf_mean = xf.mean(axis=1)
            xf_me = np.ones([patch_nums,eNum])
            for i in range(0,eNum):
                xf_me[:,i] = xf_mean
            xf_m = xf_mean.reshape(-1,1)
            if type(W) is not int:
                Ef = xf - xf_m
                Ea = np.dot(Ef,W)
                xa = xf_me + Ea
            else:
                xa = xf_me
            globalxa[:,t,reach] = xa[patch.index(reach),:]

    return globalxa
