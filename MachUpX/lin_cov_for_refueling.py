import numpy as np
import json
import os
import copy
import scipy.optimize as opt
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import multiprocessing as mp
import subprocess as sp
import random
import time
import scipy.stats as stats
import machupX as mx

if __name__=="__main__":

    # Load scene
    scene_file = "scene.json"
    scene = mx.Scene(scene_file)
    scene.export_stl(filename="tailing_scene.stl")