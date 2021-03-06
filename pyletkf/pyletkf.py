# -*- conding: utf-8 -*

import os
import configparser
from multiprocessing import Pool
import h5py
import numpy as np
import sys
import pyletkf.pyletkf.exTool as exTool
import pyletkf.pyletkf.letkf as letkf


class LETKF_core(object):
    """
    Local Ensemble Transformed Kalman Filter implementation in Python/Cython.
    Specifically optimized for hydrological use.
    """

    required_instance_vals_grid = ['mode', 'nLon', 'east', 'assimE',
                                   'nLat', 'res', 'patch', 'assimS',
                                   'west', 'assimW', 'north', 'assimN',
                                   'ensMem', 'south', 'undef']
    required_instance_vals_vector = ['mode', 'ensMem', 'patchArea',
                                     'networkPath', 'nReach', 'undef']

    def __init__(self, configPath=None, mode="vector", use_cache=False):
        """
        initial settings.
        Args:
            configPath (str): optional. path to the DA configuration.
            mode (str): grid or vector. grid will be deplicated.
            use_cache (bool): True to use pre-cached local patch lists.
        Notes:
            Vectorized simulation is highly efficient and effective.
            only vector mode supports river network based local patch.
            Even if you are using 2d gridded models, vectorizing 2d to 1d and
            using vector mode is highly recommended.
        """

        self.mode = mode
        self.use_cache = use_cache
        self.localPatchPath = "./localPatch.obj"

        if type(configPath) == str and os.path.exists(configPath):
            # Read configuraion file.
            print("Read variables from configuration...")
            config = configparser.ConfigParser()
            config.read(configPath)

            if self.mode == "grid":
                raise NotImplementedError("mode=grid is not supported anymore. " +
                                            "Please make your data vectorized. " +
                                            "In general vectorization leads an efficiency and " +
                                            "readability of the code.")
            elif mode == "vector":
                self.ensMem = int(config.get("assimilation", "ensMem"))
                self.patchArea = float(config.get("assimilation", "patchArea"))
                self.networkPath = str(config.get("model", "networkPath"))
                self.nReach = int(config.get("model", "nReach"))
                self.localPatchPath = str(config.get("assimilation",
                                                     "localPatchPath"))
                self.networktype = str(config.get("model", "networktype"))
                self.undef = float(config.get("observation", "undef"))
                self.reaches = np.arange(0, self.nReach, 1)
                if self.networktype == "nextxy":
                    self.nLon = int(config.get("model", "nLon"))
                    self.nLat = int(config.get("model", "nLat"))
                    self.vectorinfoPath = str(config.get("model", "vectorinfopath"))
                    self.catareaPath = str(config.get("model", "catareaPath"))
                elif self.networktype == "csv" or self.networktype == "\"csv\"":
                    self.reach_start = 1

            else:
                raise IOError("mode %s is not supprted." % mode)

            print("############Check instance variables############")
            self.__showProperties()
            print("##############")

        elif not os.path.exists(configPath):
            raise IOError("{0} does not exist.".format(configPath))
        else:
            print("configuration has not been made. Please set-up by yourself")

    def initialize(self, backend="h5py"):

        if self.mode == "grid":
            # check all instance variables are set.
            raise NotImplementedError("mode=grid is not supported anymore. " +
                                      "Please make your data vectorized. " +
                                      "In general vectorization leads an efficiency and " +
                                      "readability of the code.")
        elif self.mode == "vector":
            # check all instance variables are set.
            self.__checkInstanceVals(self.required_instance_vals_vector)
            if self.use_cache:
                if backend == "pickle":
                    with open(self.localPatchPath, mode="rb") as f:
                        import pickle
                        self.patches = pickle.load(f)
                else:
                    with h5py.File(self.localPatchPath, mode="r") as f:
                        key = list(f.keys())[0]
                        patches = f[key][:].tolist()
                        self.patches = [patch.tolist() for patch in patches]
            else:
                # generate local patch
                self.patches = \
                    self.__constLocalPatch_vector()
        else:
            raise IOError("mode %s is not supoorted." % self.mode)

    def letkf_vector(self, ensembles, observation, obserr, obsvars,
                     guess="mean", nCPUs=1, smoother=False):
        """
        Data Assimilation with Local Ensemble Transformed Kalman Filter
        Args:
            ensembles (np.ndarray): [nvar,eNum,nT,nReach], state matrix
                If smoother == False, then only the assimilation at time nT
                    (last time step of the array) will happen.
                If smoother == True, whole time series up to nT
                    will be smoothed by the smoother.
            observation (np.ndarray): [nObs, nReach],
                                      undef-padded observation vector
            obserr (np.ndarray): [nObs, nReach], undef-padded observation error
            obsvars (list): either 1 or 0 in shape of [nvar]
                          with same order as observation;
                            1: included in observation
                            0: not included in observation
            guess (str): if "mean", where observation is not available replace all values
                         with ensemble mean as a single posteroir. 
                         if "prior", just use prior ensembles 
                         for posterior (no update at all).
            nCPUs (int): number of cpus used for parallelization
            smoother (bool): true to activate no cost LETKF smooter
        Returns:
            numpy.ndarray: [nvar,eNum,nT,nState] assimilated matrix
        Notes:
            Note that ensembles[:,:,nT,:] should be the same time step
            as that of observations.
        """
        outArray = ensembles.copy()
        # check shapes are correspond with instance correctly
        eNum = ensembles.shape[1]
        assert eNum == self.ensMem, "Specified emsemble member is not" + \
                                    "match with the passed array shape." \
        # main letkf
        p = Pool(nCPUs)
        argsmap = self.__get_argsmap_letkf(ensembles[:, :, -1, :], observation,
                                           obserr, self.patches, obsvars,
                                           guess, self.undef, nCPUs)
        result = p.map(submit_letkf, argsmap)
        p.close()
        xa, Ws = self.__parse_mapped_res(result)
        #xa, Ws = letkf.letkf(ensembles[:, :, -1, :], observation, obserr,
        #                     self.patches, obsvars, np.arange(0, self.nReach, 1).tolist(), self.undef)
        outArray[:, :, -1, :] = xa
        # smoother
        if smoother:
            p = Pool(nCPUs)
            argsmap = self.__get_argsmap_ncs(ensembles[:, :, 0:-1, :],
                                             self.patches, Ws, nCPUs)
            result = p.map(submit_ncs, argsmap)
            p.close()
            #xa = letkf.noCostSmoother(ensembles[:, :, 0:-1, :],
            #                          self.patches, Ws, reaches)
            xa = self.__parse_mapped_res(result)
            outArray[:, :, 0:-1, :] = xa
        return outArray, Ws

    def __get_argsmap_letkf(self, ensembles, observation, obserr, patches,
                            obsvars, guess, undef, nCPUs):
        """
        map args for letkf.letkf() along with nCPUs.
        for multiprocessing purpose.
        """
        splitted_reaches = np.array_split(self.reaches, nCPUs)
        splitted_reaches = [e.tolist() for e in splitted_reaches]
        return [[ensembles, observation, obserr, self.patches, obsvars,
                 reaches, self.undef, guess] for reaches in splitted_reaches]

    def __get_argsmap_ncs(self, ensembles, patches, Ws, nCPUs):
        """
        map args for letkf.noCostSmoother() along with nCPUs.
        for multiprocessing purpose.
        """
        splitted_reaches = np.array_split(self.reaches, nCPUs)
        splitted_reaches = [e.tolist() for e in splitted_reaches]
        return [[ensembles, self.patches, Ws, reaches]
                for reaches in splitted_reaches]

    def __parse_mapped_res(self, result):
        """
        parse results from mapped results in multiprocessing.
        multiprocessing.Pool.map() waits until all processes finish,
        and returns results in a list with the same order as input.
        this method assuming that an argument is a result of map() (sorted),
        not map_unordered() which is non-blocking but unsorted.
        """
        xa = np.zeros_like(result[0][0])
        # zero-padded for out-of-range reaches, so just sum up.
        for idx, r in enumerate(result):
            xa_i = r[0]
            xa = xa + xa_i

            # resulted list is sorted, so just extend in a same order.
            if idx == 0:
                Ws = r[1]
            else:
                Ws.extend(r[1])
        return xa, Ws

    def __checkInstanceVals(self, valList):
        keys = self.__dict__.keys()
        if len(keys) < len(valList):
            nonDefined = set(valList) - set(keys)
            raise IOError("Not all instance variables defined. %s"
                          % str(list(nonDefined)))

    def __checkLatLonShapes(self, array):
        if len(array.shape) == 2:
            lat_shape = array.shape[0]
            lon_shape = array.shape[1]
        elif len(array.shape) == 3:
            lat_shape = array.shape[1]
            lon_shape = array.shape[2]
        assert lat_shape == self.nLat, "Specified latitude number {0}".format(self.nLat) +\
                                       "is not match with the passed" +\
                                       "array shape {0}.".format(lat_shape)
        assert lon_shape == self.nLon, "Specified longitude number {0}".format(self.nLon) +\
                                       "is not match with the passed" +\
                                       "array shape {0}.".format(lon_shape)

    def __showProperties(self):
        for key in self.__dict__.keys():
            print("{0}:{1}".format(key, self.__dict__[key]))

    def __constLocalPatch_vector(self):
        if self.networktype == "nextxy":
            # 2d inputs; use nextxy format river network; binary formats
            with h5py.File(self.vectorinfoPath, "r") as f:
                map2vec = f["map2vec"][:]
                nvec = len(f["vec2lat"][:])
            nextx = np.memmap(self.networkPath, dtype=np.int32, mode="r",
                              shape=(2, self.nLat, self.nLon))[0, :, :]
            nexty = np.memmap(self.networkPath, dtype=np.int32, mode="r",
                              shape=(2, self.nLat, self.nLon))[1, :, :]
            ctmare = np.memmap(self.catareaPath, dtype=np.float32, mode="r",
                              shape=(self.nLat, self.nLon))
            PATCHES = exTool.constLocalPatch_vector_nextxy(nextx, nexty,
                                                           ctmare,
                                                           self.patchArea,
                                                           map2vec, nvec,
                                                           name=self.localPatchPath)
        else:
            # already vectorized data; csv formats
            PATCHES = exTool.constLocalPatch_vector_csv(self.networkPath,
                                                        self.patchArea,
                                                        self.nReach,
                                                        self.localPatchPath,
                                                        reach_start=self.reach_start)
        return PATCHES


def submit_letkf(args):
    result = letkf.letkf(args[0], args[1], args[2], args[3],
                         args[4], args[5], args[6], args[7])
    return result


def submit_ncs(args):
    result = letkf.noCostSmoother(args[0], args[1], args[2], args[3])
    return result


if __name__ == "__main__":

    chunk = LETKF_core("./config.ini", mode="vector", use_cache=False)
    chunk.initialize()
