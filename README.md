# AttachmentNN

This repository contains experiments extending the coupled-oscillator model of human attachment dynamics presented in:

**"A coupled-oscillator model of human attachment dynamics evaluated in a robot dyadic interaction"**  
*Aung Htet, Alejandro Jimenez-Rodriguez, Marcantonio Gagliardi, and Tony J. Prescott*

## Overview

This repository provides a simulated MiRo robot implementation in Gazebo to study the attachment control system and collect experimental data. Additional experiments explore using neural networks to estimate physical and emotional distance as coupling functions.

## Prerequisites Installation

This project runs in a Docker container using the image `aung9htet/ubuntu-20.04:miro`.

### 1. Install Docker

If you don't have Docker installed on your system, follow these steps:

**For Ubuntu/Debian:**

Follow the official Docker installation guide: **[Install Docker on Ubuntu](https://docs.docker.com/engine/install/ubuntu/)**

Or use these quick commands:
```bash
# Update package index
sudo apt-get update

# Install required packages
sudo apt-get install -y ca-certificates curl gnupg lsb-release

# Add Docker's official GPG key
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up the repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
sudo apt-get update
sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
```

**For other operating systems:** Visit **[Docker's official installation guide](https://docs.docker.com/get-docker/)**

**Run Docker without sudo (recommended):**

Follow the post-installation steps: **[Linux post-installation steps](https://docs.docker.com/engine/install/linux-postinstall/)**

```bash
# Add your user to the docker group
sudo usermod -aG docker $USER

# Log out and log back in for group changes to take effect
# Or run: newgrp docker
```

**Install NVIDIA Docker (if you have an NVIDIA GPU):**

Official guide: **[NVIDIA Container Toolkit Installation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)**

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

**Additional Resources:**

For more information and tutorials, check out these helpful resources:
- **Docker Documentation**: [https://docs.docker.com/](https://docs.docker.com/)
- **Docker Installation Guide**: [https://docs.docker.com/engine/install/](https://docs.docker.com/engine/install/)
- **NVIDIA Container Toolkit**: [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)

### 2. Starting the Docker Container

To start the Docker container with all necessary configurations:

```bash
./run_docker.bash
```

This script will:
- Enable X11 forwarding for GUI applications
- Mount this repository's ROS package into the container's catkin workspace
- Configure GPU support
- Launch the container with the name `ros_container` (or open a new shell in it if it already exists)

**How the project is shared with the container**

The ROS package lives in this repository at `ros/Attachment-Model/` and is bind-mounted to `/root/mdk/catkin_ws/src/Attachment-Model` inside the container. The `analysis/` folder is mounted at `/root/analysis`. Code edits, recorded data and figures are therefore the same files on both sides: anything an experiment saves appears in this repository and can be committed and pushed.

The mount is set up when the container is created. If you already have a `ros_container` from before this change, remove it once (`docker rm -f ros_container`) and run `./run_docker.bash` again.

### 3. Working Inside the Container

Once inside the container, you can:
- Run experiments and simulations
- Install additional packages as needed
- Access your workspace files (if volume mounting is enabled)

To exit the container without stopping it:
```bash
# Press Ctrl+P, then Ctrl+Q
```

To exit and stop the container:
```bash
exit
```

**Note:** If you need to return to the container later, see section 6 for reattaching instructions.

### 4. Saving Your Changes

If you've made changes inside the container (installed packages, modified files, etc.) and want to save them to the Docker image:

```bash
./save_docker.bash
```

This will commit your changes to the `aung9htet/ubuntu-20.04:miro` image.

**Warning:** This will overwrite the existing image. Make sure you want to save your changes before running this command.

### 5. Removing the Container

To stop and remove the container:

```bash
# Stop the container
docker stop ros_container

# Remove the container
docker rm ros_container
```

To remove the container forcefully (if it's running):

```bash
docker rm -f ros_container
```

### 6. Reattaching to a Running Container (Optional)

If you've detached from the container using `Ctrl+P` then `Ctrl+Q` and want to reattach:

```bash
docker attach ros_container
```

Or start a new bash session in the running container:

```bash
docker exec -it ros_container bash
```

**Note:** This section is only needed if you're not using volume mounts and need to access a detached container.

### 7. Managing Docker Images

To list all Docker images:
```bash
docker images
```

To remove the Docker image (if needed):
```bash
docker rmi aung9htet/ubuntu-20.04:miro
```

### Troubleshooting Docker Issues

**Permission denied errors:**
- Make sure your user is in the docker group: `groups $USER`
- Log out and log back in after adding yourself to the docker group

**GPU not detected:**
- Verify NVIDIA drivers are installed: `nvidia-smi`
- Ensure nvidia-docker2 is properly installed

**X11 forwarding not working:**
- Run `xhost local:root` before starting the container
- Check that `DISPLAY` environment variable is set: `echo $DISPLAY`

**Container name already in use:**
- Remove the existing container: `docker rm ros_container`
- Or use a different container name in `run_docker.bash`

## Development Tools (Optional)

**Visual Studio Code Setup:**
1. Install [Visual Studio Code](https://code.visualstudio.com/)
2. Install the "Docker" extension from the VS Code marketplace
3. Install the "Dev Containers" extension for developing inside Docker containers
4. You can attach VS Code to your running container for easier development

This allows you to write and debug code directly within the Docker environment.

## Running the Experiment

Once inside the Docker container, you can run the attachment model experiment.

### Starting the Experiment

To launch the attachment model simulation:

```bash
roslaunch attachment_model relationship.launch avoidant:=0.0 ambivalent:=0.1 duration:=300
```

| Argument | Default | Meaning |
|---|---|---|
| `avoidant` | `0.0` | Avoidance level, 0.0 to 1.0 (emotional-distance coupling) |
| `ambivalent` | `0.0` | Ambivalence level, 0.0 to 1.0 (physical-distance coupling) |
| `duration` | `0` | Run length in simulation seconds; `0` runs until Ctrl+C |
| `gui` | `true` | Show the Gazebo client and the camera preview window |

**What happens when you start the experiment:**

1. **Gazebo Simulation Launches**: Opens the arena (`worlds/attachment.world`) with the parent and child MiRo robots
2. **Robot Nodes Start**: Initializes child and parent robot control nodes
3. **Attachment Model Runs**: The attachment controller executes the attachment model (coupled through the neural networks)
4. **Perception System Activates**: AprilTag detection for robot tracking and localization
5. **Data Recording Begins**:
   - Camera stream is recorded to `camera_recording.mp4`
   - Robot behavior data, robot paths and inter-robot distance are saved as NumPy dictionaries (`child_data.npy`, `parent_data.npy`)
   - Run parameters are saved to `meta.json`
   - Everything goes to a new `ros/Attachment-Model/data/experiment_data_{index}/` folder
6. **The run ends** after `duration` seconds; all nodes save their data on shutdown

### Recording a batch of experiments

`run_experiments.sh` records several conditions one after another (run it inside the container):

```bash
cd /root/mdk/catkin_ws/src/Attachment-Model
./run_experiments.sh paper     # secure, strongly avoidant, moderately ambivalent; 5 min each
./run_experiments.sh sweep     # avoidant 0.1-0.8 and ambivalent 0.1-0.8; 3 min each
./run_experiments.sh run 0.3 0.0 120   # one run: avoidant, ambivalent, seconds
```

### Note on the avoidant and ambivalent parameters

Until `experiment_data_33` the controller applied the two parameters the wrong way round: the value labelled *ambivalent* drove the avoidant (emotional-distance) coupling and the value labelled *avoidant* drove the ambivalent (physical-distance) coupling. This is fixed for all runs that have a `meta.json` (`"param_mapping": "fixed"`). The analysis scripts correct the labels of the earlier runs when loading them. The four `.pth` networks were trained on the earlier runs.

### Project Structure

The ROS package is in `ros/Attachment-Model/` (inside the container: `/root/mdk/catkin_ws/src/Attachment-Model`):

- **Source Code**: `src/`
  - `subscribers.py` - Subscribers for odometry, ground truth, actions and the emotion parameters
  - `agents.py` - Shared agent logic and data recording
  - `child_node.py` - Child robot control script
  - `parent_node.py` - Parent robot control script
  - `attachment_model_control.py` - Attachment controller
  - `perception.py` - AprilTag detection and tracking
  - `camera_recording.py` - Video recording node
  - `experiment_timer.py` - Ends a run after its duration and writes `meta.json`
- **Launch file**: `launch/relationship.launch`
- **Simulation**: `worlds/attachment.world`, `models/`
- **Experiment Data**: `data/experiment_data_{index}/`

Analysis scripts are in `analysis/` at the repository root.

## Experimental Data

Pre-collected experimental data is included in the `ros/Attachment-Model/data/` directory. Each experiment folder (`experiment_data_{index}/`) contains:
- `camera_recording.mp4` - Video recording of the robot interaction
- `child_data.npy` - Child robot behavioral data
- `parent_data.npy` - Parent robot behavioral data

### Data Structure

Each `.npy` file contains a dictionary. `child_data.npy` holds the child's series and `parent_data.npy` the parent's; the shared and position keys are in both. One sample is taken every 0.1 s.

```python
{
    'Xc': [],                           # Child [need, accumulated need]
    'Xp': [],                           # Parent [need, accumulated need]
    'ambivalence': [],                  # Ambivalence parameter values
    'avoidance': [],                    # Avoidance parameter values
    'physical_coupling_targets': [],    # Physical distance coupling targets
    'emotional_coupling_targets': [],   # Emotional distance coupling targets
    'child_states': [],                 # Child action (0 approach, 1 explore)
    'parent_states': [],                # Parent action (0 approach, 1 explore)
    'vocal_child': [],                  # Child vocalization events
    'vocal_parent': [],                 # Parent vocalization events
    'child_evidence_trace': [],         # Child evidence accumulation
    'parent_evidence_trace': [],        # Parent evidence accumulation

    # Recorded from experiment_data_34 onwards
    'time': [],                         # Simulation time (s)
    'child_odom': [],                   # Child odometry path [x, y, yaw] in the world frame (m, degrees)
    'parent_odom': [],                  # Parent odometry path [x, y, yaw] in the world frame
    'child_odom_raw': [],               # Child odometry as published (starts at 0 in the robot's own frame)
    'parent_odom_raw': [],              # Parent odometry as published
    'child_gt': [],                     # Child true position [x, y] from Gazebo (m)
    'parent_gt': [],                    # Parent true position [x, y] from Gazebo (m)
    'odom_distance': [],                # Child-parent distance from odometry (m)
    'odom_distance_norm': [],           # min(1, odom_distance / arena size)
    'gt_distance': [],                  # Child-parent distance from true positions (m)
    'gt_distance_norm': []              # min(1, gt_distance / arena size)
}
```

MiRo's wheel odometry starts at zero in each robot's own frame and drifts, most of all when the child pushes against a wall. The odometry path is saved in the world frame using the spawn poses, and the true Gazebo position is saved next to it. See `analysis/figures/improved/odometry_error.png`.

The arena is 2.61 m wide, centred on the caregiver box at (0.325, 0.325).

### Dataset Organization

**Earlier runs, no robot positions** (labels below are as recorded; the two parameters were applied the wrong way round, see the note above):
- `experiment_data_0` to `16`: training data, 3 minutes per run
  - `0`: Avoidance = 0.0, Ambivalence = 0.0
  - `1` to `8`: Avoidance = 0.1 to 0.8, Ambivalence = 0.0
  - `9` to `16`: Avoidance = 0.0, Ambivalence = 0.1 to 0.8
- `experiment_data_17` to `33`: experimental data, 1 minute per run, same order as above

**Runs with robot paths and `meta.json`** (parameters applied as labelled):
- `experiment_data_34`: secure, Avoidance = 0.0, Ambivalence = 0.1, 5 minutes
- `experiment_data_35`: strongly avoidant, Avoidance = 0.9, Ambivalence = 0.0, 5 minutes
- `experiment_data_36`: moderately ambivalent, Avoidance = 0.0, Ambivalence = 0.4, 5 minutes
- `experiment_data_37` to `44`: Avoidance = 0.1 to 0.8, Ambivalence = 0.0, 3 minutes
- `experiment_data_45` to `52`: Avoidance = 0.0, Ambivalence = 0.1 to 0.8, 3 minutes
- `experiment_data_53`: Avoidance = 0.0, Ambivalence = 0.0, 3 minutes

In the runs with avoidance 0.5 and above (`35`, `41` to `44`) the oscillator model diverges within the first seconds and the child stays in approach mode next to the caregiver. The same happens in the earlier runs `5`, `14`, `15` and `16`. `analysis/results/run_summary.csv` flags these runs in the `model_diverged` column.

## Analysis and Figures

The `analysis/` folder reproduces the robot-experiment analysis of the npj paper (in `papers/`) for this project and adds redesigned versions of the figures. Run it inside the container:

```bash
/workspace/.venv/bin/python /root/analysis/make_figures.py
```

| Output | Content |
|---|---|
| `analysis/figures/paper_style/fig11_trajectory_distance_vocalisation` | Child trajectory, distance and vocalisation per regime (paper Fig. 11) |
| `analysis/figures/paper_style/fig12_heatmaps` | 50 x 50 occupancy heat maps (paper Fig. 12) |
| `analysis/figures/paper_style/fig9_trajectory_distance_needs` | Trajectory, distance with vocalisation, needs (paper Fig. 9 layout) |
| `analysis/figures/paper_style/fig3_correlation_frequency` | Need correlation and oscillation frequency against each parameter (paper Fig. 3) |
| `analysis/figures/improved/` | Redesigned regime, heat-map, proximity, parameter-sweep and odometry-error figures |
| `analysis/results/proximity_stats.md` | Time close to the caregiver: chi-squared and Bonferroni-corrected pairwise tests (paper section 6.3) |
| `analysis/results/run_summary.csv` | One row of measures per run |

Each figure is written as PNG and PDF.

- Positions come from Gazebo's true robot position by default, because wheel odometry drifts by more than a metre within a run. Add `--source odom` to use the odometry path instead.
- "Close to the caregiver" means inside the square centred on the caregiver whose side is half the arena width.
- Periods when the exploring child stays in one heat-map cell for longer than the mean plus two standard deviations are treated as stuck and excluded, as in the paper.
- The three regimes are picked by parameter value (`REGIMES` in `make_figures.py`); the longest matching run is used.

## TCDS paper

`papers/tcds/` holds the narrative outline and the draft of the journal paper built on these
experiments; `analysis/tcds/` holds the scripts behind it. They re-implement the attachment
controllers outside ROS (`model.py`), read the trained `.pth` networks without PyTorch
(`nnweights.py`), replay the recorded Gazebo runs, compare the coupled and decoupled
controllers, evaluate the learned coupling in the notebook's agent simulator
(`sim_agents.py`), and infer the attachment parameters from behaviour with a kinematic
stand-in for the Gazebo experiment (`sim_kinematic.py`). See `papers/tcds/README.md`.

## Training the Neural Network

After collecting experimental data, you can train neural networks to learn the coupling functions between parent and child behaviors.

### Training Script Location

The neural network training script is located at:
```
/root/mdk/catkin_ws/src/Attachment-Model/src/attachment_with_nn.ipynb
```

### What the Training Script Does

The training script (`attachment_with_nn.ipynb`) performs the following:

1. **Loads Experimental Data**: Automatically searches and loads all experiment data from the `data` directory
2. **Data Preparation**: 
   - Merges child and parent data from multiple experiments
   - Splits data into training, validation, and test sets (80%/20% split)
   - Organizes data by parameter values (ambivalence and avoidance levels)
3. **Network Training**: Trains four neural networks:
   - Physical coupling for parent behavior
   - Physical coupling for child behavior
   - Emotional coupling for parent behavior
   - Emotional coupling for child behavior
4. **Model Saving**: Saves trained models as `.pth` files that can be used by the simulation

### How to Use the Training Script

**Option 1: Run in Jupyter Notebook** (Recommended)

If Jupyter is available in the container:
```bash
cd /root/mdk/catkin_ws/src/Attachment-Model/src
jupyter notebook attachment_with_nn.ipynb
```

**Option 2: Convert to Python Script and Run**

If Jupyter is not available:
```bash
cd /root/mdk/catkin_ws/src/Attachment-Model/src

# Convert notebook to Python script
jupyter nbconvert --to script attachment_with_nn.ipynb

# Run the script
python attachment_with_nn.py
```

**Option 3: Open with VS Code**

If you have VS Code attached to the container with the Jupyter extension:
1. Open the file `attachment_with_nn.ipynb` in VS Code
2. Select a Python kernel
3. Run all cells

### Training Configuration

The script can be configured to use either:
- **Recorded experimental data** (default): `use_recorded = True`
- **Simulated data**: `use_recorded = False`

To switch between modes, modify the variable in the notebook:
```python
# Set to True to use recorded data, False to use simulation
use_recorded = True
```

### Saved Model Files

After training completes, the following model files will be saved in the same directory:
- `physical_parent_coupling_model.pth` - Physical coupling network for parent
- `physical_child_coupling_model.pth` - Physical coupling network for child
- `emotional_parent_coupling_model.pth` - Emotional coupling network for parent
- `emotional_child_coupling_model.pth` - Emotional coupling network for child

These trained models are automatically used by the simulation when you run the experiment.