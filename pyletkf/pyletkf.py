# -*- conding: utf-8 -*

import os
import configparser
import pickle
from . import exTool
from . import letkf


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
                                     'networkFile', 'nReach', 'undef']

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
        self.reach_start = 1
        self.use_cache = use_cache
        self.localPatchPath = "./localPatch.obj"

        if type(configPath) == str and os.path.exists(configPath):
            # Read configuraion file.
            print("Read variables from configuration...")
            config = configparser.ConfigParser()
            config.read(configPath)

            if mode == "grid":
                self.assimN = float(config.get("assimilation", "assimN"))
                self.assimS = float(config.get("assimilation", "assimS"))
                self.assimE = float(config.get("assimilation", "assimE"))
                self.assimW = float(config.get("assimilation", "assimW"))
                self.patch = int(config.get("assimilation", "patch"))
                self.ensMem = int(config.get("assimilation", "ensMem"))
                self.nLon = int(config.get("model", "nLon"))
                self.nLat = int(config.get("model", "nLat"))
                self.res = float(config.get("model", "res"))
                self.north = float(config.get("model", "north"))
                self.south = float(config.get("model", "south"))
                self.east = float(config.get("model", "east"))
                self.west = float(config.get("model", "west"))
                self.undef = float(config.get("observation", "undef"))

            elif mode == "vector":
                self.ensMem = int(config.get("assimilation", "ensMem"))
                self.patchArea = float(config.get("assimilation", "patchArea"))
                self.networkFile = str(config.get("model", "networkFile"))
                self.nReach = int(config.get("model", "nReach"))
                self.localPatchPath = str(config.get("assimilation",
                                                     "localPatchPath"))
                self.networktype = "nextxy"  # nextxy or csv
                self.undef = float(config.get("observation", "undef"))

            else:
                raise IOError("mode %s is not supprted." % mode)

            print("############Check instance variables############")
            self.__showProperties()
            print("##############")

    def initialize(self):

        if self.mode == "grid":
            # check all instance variables are set.
            self.__checkInstanceVals(self.required_instance_vals_grid)
            # implementaion needed
            # generate local patch
            # self.patches = self.__constLocalPatch_grid()  #  not supported

        elif self.mode == "vector":
            # check all instance variables are set.
            self.__checkInstanceVals(self.required_instance_vals_vector)
            if self.use_cache:
                with open(self.localPatchPath, "rb") as f:
                    self.patches = pickle.load(f)
            else:
                # generate local patch
                self.patches = \
                    self.__constLocalPatch_vector(reach_start=self.reach_start)
        else:
            raise IOError("mode %s is not supoorted." % self.mode)

    def letkf_grid(self, ensembles, observation, obserr, ocean, excGrids):
        """
        Data Assimilation with Local Ensemble Transformed Kalman Filter
        Args:
            ensembles (numpy.ndarray): [nLat,nLon,eNum], ensemble simulation
            observation (numpy.ndarray): [nLat,nLon], gridded observation
                                         with observed or undef values
            ocean (numpy.ndarray): [nLat,nLon], gridded ocean mask
            excGrids (numpy.ndarray): [nLat,nLon], grids to exclude
        """
        # check shapes are correspond with instance correctly
        eNum = ensembles.shape[0]
        if eNum != self.ensMem:
            raise IOError("Specified emsemble member %d is not match" +
                          "with the passed array shape %d."
                          % (self.ensMem, eNum))
        for data in [ensembles, observation, ocean, excGrids]:
            self.__checkLatLonShapes(data)
        # main letkf
        xa = letkf.letkf(ensembles, observation, obserr, ocean, excGrids,
                         self.patch, self.ensMem, self.assimE, self.assimW,
                         self.assimN, self.assimS, self.east, self.west,
                         self.north, self.south, self.res, self.undef)
        return xa

    def letkf_vector(self, ensembles, observation, obserr, smoother=False):
        """
        Data Assimilation with Local Ensemble Transformed Kalman Filter
        Args:
            ensembles (np.ndarray): [eNum,nT,nReach], ensemble state matrix
                If smoother == False, then only the assimilation at time nT
                    (last time step of the array) will happen.
                If smoother == True, whole time series up to nT
                    will be smoothed by the smoother.
            observation (np.ndarray): [nObs],
                                         undef-padded observation vector
            obserr (np.ndarray): [nReach], observation error
            smoother (bool): true to activate no cost LETKF smooter
        Returns:
            numpy.ndarray: [eNum,nT,nState] assimilated matrix
        Notes:
            Note that ensembles[:,nT,:] should be the same time step
            as that of observations.
        """
        outArray = ensembles.copy()
        # check shapes are correspond with instance correctly
        eNum = ensembles.shape[0]
        assert eNum == self.ensMem, "Specified emsemble member %d is not" + \
                                    "match with the passed array shape %d." \
                                    % (self.ensMem, eNum)
        # main letkf
        xa, Ws = letkf.letkf(ensembles[:, -1, :], observation, obserr,
                             self.patches, self.ensMem, self.undef)
        outArray[:, -1, :] = xa
        # smoother
        if smoother:
            xa = letkf.noCostSmoother(ensembles[:, 0:-1, :], self.patches, Ws)
            outArray[:, 0:-1, :] = xa
        return outArray, Ws

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
        assert lat_shape == self.nLat, "Specified latitude number %d" +\
                                       "is not match with the passed" +\
                                       "array shape %d." \
                                       % (self.nLat, lat_shape)
        assert lon_shape == self.nLon, "Specified longitude number %d" +\
                                       "is not match with the passed" +\
                                       "array shape %d." \
                                       % (self.nLon, lon_shape)

    def __showProperties(self):
        for key in self.__dict__.keys():
            print("%s:%s" % (key, self.__dict__[key]))

    def __constLocalPatch_vector(self):
        if input2d:
            # 2d inputs; use nextxy format river network; binary formats
            PATCHES = exTool.constLocalPatch_vector_nextxy(self.networkfile,
                                                           self.patchArea)
        else:
            # already vectorized data; csv formats
            PATCHES = exTool.constLocalPatch_vector_csv(self.networkFile,
                                                        self.patchArea,
                                                        self.nReach,
                                                        self.localPatchPath)
        return PATCHES


if __name__ == "__main__":

    chunk = LETKF_core("./config.ini", mode="vector", use_cache=False)
    chunk.initialize()
