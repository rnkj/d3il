#!/bin/bash

############ GENERAL ENV SETUP ############
echo New Environment Name:
read envname

echo Creating new conda environment $envname
conda create -n $envname python=3.10.8 -y -q

eval "$(conda shell.bash hook)"
conda activate $envname

echo
echo Activating $envname
if [[ "$CONDA_DEFAULT_ENV" != "$envname" ]]
then
    echo Failed to activate conda environment.
    exit 1
fi

### Set Channel vars
conda config --add channels conda-forge
conda config --set channel_priority strict


############ PYTHON ############
echo Install mamba
conda install mamba -c conda-forge -y -q


############ REQUIRED DEPENDENCIES (PYBULLET) ############
echo Installing dependencies...

mamba install -c conda-forge mkl==2024.0.0
mamba install -c pytorch -c nvidia pytorch==1.13.1 torchvision==0.14.1 pytorch-cuda=11.7

mamba install -c conda-forge pybullet pyyaml scipy opencv pinocchio matplotlib gin-config tyro gym==0.21.0 -y -q

# Open3D for PointClouds and its dependencies. Why does it not install them directly?
mamba install -c conda-forge scikit-learn addict pandas plyfile tqdm -y -q
mamba install -c conda-forge -c open3d-admin open3d -y -q

pip install einops
pip install hydra-core==1.2.0
pip install wandb

# Robomimic
pip install termcolor

# ACT
pip install ipython

# BESO
pip install torchsde torchdiffeq

############ MUJOCO BETA SUPPORT INSTALLATION ############
mamba install -c conda-forge imageio -y -q
pip install mujoco==2.3.2

############ INSTALL D3il-Sim & FINALIZE ############
echo
echo Installing D3il-Sim Package
cd environments/d3il && pip install -e .

exit 0
