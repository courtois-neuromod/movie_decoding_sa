#!/bin/bash                                                                                          
#SBATCH --time=10:00:00
#SBATCH --account=rrg-pbellec                                                                         
#SBATCH --mem-per-cpu=30G
python fMRI_Data.py
