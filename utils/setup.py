import os
from setuptools import setup,find_packages

#export PYTHONPATH=$PYTHONPATH:/Users/Srija/Desktop/

with open("requirements.txt") as f:
    required_packages=f.read().splitlines()

#Create necessary folders if they don't exist
dirs=[
    'src/models',
    'src/inference',
    'data',
    'saved_models/regression_model',
    'saved_models/lane_segmentation_model',
    'saved_models/object_detection_model',
    'utils',
    '',
]















