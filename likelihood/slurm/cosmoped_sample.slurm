#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --time=00:30:00

#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=heatherp@princeton.edu
echo "about to load anaconda3"
module load anaconda3
echo "about to run sample cosmoped"
python ../python/sample_cosmoped_likelihood.py
