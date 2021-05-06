#!/bin/bash                                                                                         
#SBATCH --time=5:00:00
#SBATCH --account=rrg-pbellec                                                                         
#SBATCH --mem-per-cpu=500G
python Movie_feature_extraction.py
