#!/usr/bin/env python
# -*- conding: utf-8 -*-

import sys
import numpy as np
import pandas as pd
import h5py
from numba import jit


def read_cache(cachepath):
    """
    read cached hdf5 file and returns list
    Args:
        cachepath (str): path to cached file
    Returns:
        list: local patch ids
    """
    with h5py.File(cachepath, "r") as f:
        key = f.keys()[0]
        patches = f[key][:].tolist()
    return patches


# @jit
def constLocalPatch_vector_nextxy(nextx, nexty, unitArea, patchArea, map2vec,
                                  nvec, name="network.hdf5",
                                  undef=[-9, -10, -9999]):
    """
    Args:
        nextx (ndarray-like): nextx 2d array
        nexty (ndarray-like): nexty 2d array
        unitArea (ndarray): 2d array; unit catchment area of each grid points
        patchArea (float): reference value for local patch area
    Returns:
       NoneType
    Notes:
        patchArea is just reference-the unit catchment will be concatenated
        until sum of the cathment areas go beyond patchArea.
    ToDo:
        Optimize numba behavior
    """
    # create hdf5 dataset
    f = h5py.File("%s" % name, "w")
    dt = h5py.vlen_dtype(np.int32)
    nlon = nextx.shape[1]
    nlat = nextx.shape[0]
    dset = f.create_dataset("vlen_int", (nvec,), dtype=dt)
    patches = []
    for ilat in range(nlat):
        for ilon in range(nlon):
            parea = unitArea[ilat, ilon]
            if parea in undef:
                # skip ocean
                continue
            if map2vec[ilat, ilon] < 0:
                # out of vectorizing domain
                continue
            lats = [ilat]
            lons = [ilon]
            if nextx[ilat, ilon] in undef or nexty[ilat, ilon] in undef:
                # skip further concatenation for riv. mouth/inland termination.
                # because -9 and -10 are global values for all basin.
                # it is possible to implement with this algorithm,
                # using basin.bin by getting basin ID
                # at [clat, clon], mask nextxy.
                dset[map2vec[ilat, ilon]] = [map2vec[ilat, ilon]]
                patches.append([map2vec[ilat, ilon]])
                continue
            if parea >= patchArea:
                # parea already exceeds patchArea
                dset[map2vec[ilat, ilon]] = [map2vec[ilat, ilon]]
                patches.append([map2vec[ilat, ilon]])
                continue
            clat = ilat  # current scope
            clon = ilon  # current scope
            flag = "up"
            upgrids = [[clat, clon]]  # initialize
            downgrids = [[clat, clon]]
            while parea < patchArea:
                if flag == "up":
                    upgrids_new = []
                    for ug in upgrids:
                        clat = ug[0]
                        clon = ug[1]
                        # search further upstream
                        ugs_next, parea_up = concatup_nextxy(clon, clat,
                                                             nextx, nexty,
                                                             unitArea, undef)
                        parea += parea_up
                        [lats.append(ug_next[0]) for ug_next in ugs_next]
                        [lons.append(ug_next[1]) for ug_next in ugs_next]
                        upgrids_new.extend(upgrids)
                        if not parea <= patchArea:
                            break
                    upgrids = upgrids_new
                    flag = "down"
                elif flag == "down":
                    for downgrid in downgrids:
                        # Fotran > C
                        print(downgrid)
                        if downgrid[0]+1 in undef or downgrid[1]+1 in undef:
                            flag = "up"
                            continue
                        else:
                            dw_next = [nexty[downgrid[0], downgrid[1]]-1,
                                       nextx[downgrid[0], downgrid[1]]-1]
                    parea_down = unitArea[dw_next[0], dw_next[1]]
                    if parea_down in undef:
                        continue
                    parea += parea_down
                    if not isinstance(dw_next[0], list):
                        lats.append(dw_next[0])
                        lons.append(dw_next[1])
                        downgrids.append(dw_next)
                    else:
                        [lats.append(dg_next[0]) for dg_next in dw_next]
                        [lons.append(dg_next[1]) for dg_next in dw_next]
                        downgrids.extend(dw_next)
                    flag = "up"
            vecids = [map2vec[plat, plon]
                      for plon, plat in zip(lons, lats)]
            dset[map2vec[ilat, ilon]] = vecids
            patches.append(vecids)
            print(vecids)
    f.close()
    return patches


# @jit()
def concatup_nextxy(clon, clat, nextx, nexty, unitArea, undef):
    upgrids = []
    parea = 0
    upgrids_cond = np.where(  # C > Fortran
                        (nextx == clon+1) * (nexty == clat+1))
    upnum = upgrids_cond[0].shape[0]
    for i in range(upnum):  # upstream
        nlat = upgrids_cond[0][i]  # next upstream point
        nlon = upgrids_cond[1][i]  # next upstream point
        assert nextx[nlat, nlon] == clon+1, \
            "%d is required from nextx, but got %d" \
            % (nextx[nlat, nlon], clon+1)
        assert nexty[nlat, nlon] == clat+1, \
            "%d is required from nexty, but got %d" \
            % (nexty[nlat, nlon], clat+1)
        uniarea = unitArea[nlat, nlon]
        if uniarea in undef:
            continue
        parea += uniarea
        upgrids.append([nlat, nlon])
    return upgrids, parea


#@jit(nopython=False)
# @jit
def constLocalPatch_vector_csv(networkFile, patchArea, nReach, localPatchPath,
                               reach_start=1):
    """
    construct local patch and pre-cache it.
    Args:
        networkfile (str): path to the csv-formatted network file
        patchArea (float): threshold value for one single local patch
        nReach (int): number of vector reaches
        localPatchPath: path for cached local patch data without extentions.
        reach_start (int): reach ID starts from
    Returns:
        list: local patches
    ToDo:
        - Optimize Numba behavior
        - rewrite the ifelse statements with recursive functions.
          just for readability priority is low.
    """
    f = h5py.File("%s" % localPatchPath, "w")
    dt = h5py.vlen_dtype(np.int32)
    dset = f.create_dataset("network", (nReach,), dtype=dt)
    PATCHES = []
    if reach_start > 1:
        raise RuntimeWarning("RuntimeWarning: reach_start > 1 might cause " +
                             "serious error. It should be set to 0 (C-style) " +
                             "or 1 (Fortran-style).")
    print("reading river network csv...")
    assert networkFile.split("/")[-1].split(".")[-1] == "csv", "only csv" +\
                                                        "format is supported"
    network = pd.read_csv(networkFile)
    print("constructing local patches...")
    print("Maximum Local patch size: {:.3f}".format(patchArea))
    for reach in range(reach_start, nReach+reach_start):
        PATCH = [reach]
        area = 0  # initialize
        # uArea = network.loc[network.reach == reach]["upArea"].values[0]
        cArea = network.loc[network.reach == reach]["area"].values[0]
        # area = uArea
        area = cArea
        flag = "up"
        upperBoundary = False
        bottomBoundary = False
        currentReaches_up = [reach]
        currentReaches_down = [reach]
        while area < patchArea:
            if upperBoundary and bottomBoundary:
                # no up/downstream catchments. To avoid infinite loop.
                break
            elif flag == "up":
                currentReaches_up_new = []
                for ireach in currentReaches_up:
                    upReaches, area = concatup_csv(network,
                                                   ireach,
                                                   area,
                                                   patchArea)
                    PATCH.extend(upReaches)
                    currentReaches_up_new.extend(upReaches)
                    if area > patchArea:
                        break
                    if len(upReaches) == 0:
                        # no upstream catchments.
                        upperBoundary = True
                flag = "down"
                currentReaches_up = currentReaches_up_new
            elif flag == "down":
                currentReaches_down_new = []
                for ireach in currentReaches_down:
                    downReaches, area = concatdown_csv(network,
                                                       ireach,
                                                       area,
                                                       patchArea)
                    PATCH.extend(downReaches)
                    currentReaches_down_new.extend(downReaches)
                    if area > patchArea:
                        break
                    if len(downReaches) == 0:
                        # no downstream catchments.
                        bottomBoundary = True
                flag = "up"
                currentReaches_down = currentReaches_down_new
            else:
                raise IOError("iteration goes wrong conditional branch." +
                              "Fatal Error, Abort.")
        PATCH = (np.array(PATCH) - reach_start).tolist()  # pythonize
        print("patch length/area at reach {0}".format(reach),
              len(PATCH),
              area)
        dset[reach - reach_start] = PATCH
        print(dset[reach - reach_start])
        PATCHES.append(PATCH)
    f.close()
    return PATCHES


# @jit
def concatup_csv(network, reach, area, patchArea, max_upReachNum=4):
    REACHES = []
    for upNum in range(0, max_upReachNum):
        columnName = "u%d" % upNum
        upReach = int(network.loc[network.reach == reach][columnName])
        if upReach == -1:
            continue
        cArea = network.loc[network.reach == upReach]["area"].values[0]
        area = area + cArea
        REACHES.append(upReach)
        if area > patchArea:
            break
    return REACHES, area


# @jit
def concatdown_csv(network, reach, area, patchArea, max_downReachNum=4):

    REACHES = []
    for downNum in range(0, max_downReachNum):
        columnName = "d%d" % downNum
        downReach = int(network.loc[network.reach == reach][columnName])
        if downReach == -1:
            continue
        cArea = network.loc[network.reach == downReach]["area"].values[0]
        area = area + cArea
        REACHES.append(downReach)
        if area > patchArea:
            break
    return REACHES, area
