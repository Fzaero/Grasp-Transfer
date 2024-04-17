# GraspTransfer
This repository contains offical implementation of "Grasp Transfer based on Self-Aligning Implicit Representations of Local Surfaces."

## Data Creation:

First, generate watertight meshes using following codebase : https://github.com/hjwdzh/Manifold. (You can check manifold_scripts under scripts folder.)

To generate SDF values for training data, we have used modified mesh_to_sdf library (https://github.com/fzaero/mesh_to_sdf). (You can check pre_process_meshes.py under scripts folder.)

Simulation of objects require meshes to be processed with v-hacd. (You can check vhacd.py under scripts folder.)

## Installation

### Setting up the environment
Using Conda or virtual environment is recommended. Virtual environments allows easier integration with ROS. 

First setup pytorch. Then use the following command to install other packages.

```
pip install -r /path/to/requirements.txt
```

## Using the code

Easy way to use the code is to source setup_env.bash file from within the environment, then put the following code into begining of your script/notebook.

```
import sys
import os
grasp_transfer_path = os.getenv("GRASP_TRANSFER_SOURCE_DIR")
sys.path.insert(0,grasp_transfer_path)
```

Please check PCD_Grasp_Transfer_Examples_Real notebook for quick start.

