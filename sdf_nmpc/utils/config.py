import os
import numpy as np
import casadi as cs
import yaml
from .math import euler2rot, get_vfov



class AttrDict(dict):
    """Makes dict accessible by attributes.
    Made partly after https://stackoverflow.com/a/1639632/6494418
    """
    def __init__(self, dictionary):
        for key in dictionary:
            self.__setitem__(key, dictionary[key])

    def __setitem__(self, key, value):
        if isinstance(value, dict):
            super(AttrDict, self).__setitem__(key, AttrDict(value))
        elif isinstance(value, list):
            super(AttrDict, self).__setitem__(key, [AttrDict(v) if isinstance(v, dict) else v for v in value])
        else:
            super(AttrDict, self).__setitem__(key, value)
        super(AttrDict, self).__setattr__(key, self[key])

    def __setattr__(self, key, value):
        self.__setitem__(key, value)


class Config(AttrDict):
    """Read all yaml config file for a given run and store into single object."""
    def __init__(self, config_file):
        ## open mission config file
        with open(config_file, 'r') as f:
            yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
        super(Config, self).__init__(yaml_dict)

        ## derived config fields
        vfov_cpt = get_vfov(self.sensor.hfov, self.sensor.aspect_ratio, self.sensor.is_spherical)
        vfov_cfg = self.sensor.vfov
        assert abs(vfov_cpt - vfov_cfg) < 0.1, 'check sensor fov in config file'

        self.sensor.B_p_C = self.robot.sensor_extrinsics.position
        self.sensor.B_R_C = euler2rot(self.robot.sensor_extrinsics.orientation)
