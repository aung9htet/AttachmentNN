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
- Mount the workspace directory for file sharing (optional)
- Configure GPU support (if available)
- Launch the container with the name `ros_container`

**Note on Volume Mounting (Optional):**

The `run_docker.bash` script includes a volume mount on line 9. This is **optional** for the current experiment since all data is saved within the Docker image itself. However, you may want to set this up if you need to:
- Share files between your host machine and the container
- Access experiment data from outside the container
- Use the container for other projects requiring local file access

To enable volume mounting, update line 9 in `run_docker.bash` to match your local directory:
```bash
--volume="/home/YOUR_USERNAME/YOUR_PATH:/root/docker-ws"
```

If you don't need this feature, you can leave it as is or comment out the line by adding `#` at the beginning.

**Alternative: Using VS Code to Transfer Files**

If you just want to share or retrieve data without setting up volume mounts, you can use Visual Studio Code's Docker extension:
1. Install the "Docker" extension in VS Code
2. Right-click on your running container in the Docker view
3. Select "Attach Visual Studio Code" to open the container filesystem
4. Use VS Code's file explorer to drag-and-drop files between your host and container
5. Or use the command palette: "Docker: Copy from Container" / "Docker: Copy to Container"

This method is convenient for occasional file transfers without permanent volume mounts.

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
roslaunch attachment_model relationship.launch
```

**What happens when you start the experiment:**

1. **Gazebo Simulation Launches**: Opens a 3D robot simulation environment with MiRo robots
2. **Robot Nodes Start**: Initializes child and parent robot control nodes
3. **Attachment Model Runs**: The central attachment controller executes the attachment model (either coupled or uncoupled through a neural network)
4. **Perception System Activates**: AprilTag detection for robot tracking and localization
5. **Data Recording Begins**: 
   - Camera streams are recorded to video files (`camera_recording.mp4`)
   - Robot behavior data is saved as NumPy arrays (`child_data.npy`, `parent_data.npy`)
   - All data is automatically saved to `/root/mdk/catkin_ws/src/Attachment-Model/data`

### Configuring Model Parameters

You can adjust the attachment model's emotional parameters to test different behaviors:

**Location**: `/root/mdk/catkin_ws/src/Attachment-Model/src/subscribers.py` (lines 88-99)

```python
class EmotionSubscriber:
    """
        Subscribe to emotion parameters.
    """
    def __init__(self):
        self.avoidant = 0.0      # Avoidance level (0.0 to 1.0)
        self.ambivalent = 0.0    # Ambivalence level (0.0 to 1.0)
        self.subscriber = rospy.Subscriber("/emotion", Emotion, self._callback)

    def _callback(self, msg):
        self.avoidant = msg.avoidant
        self.ambivalent = msg.ambivalent
```

**Parameters:**
- `self.avoidant`: Controls avoidance behavior (range: 0.0 to 1.0)
- `self.ambivalent`: Controls ambivalent behavior (range: 0.0 to 1.0)

Modify these values to explore different attachment styles and behavioral patterns.

### Project Structure

The project files are organized as follows:

- **Source Code**: `/root/mdk/catkin_ws/src/Attachment-Model/src`
  - `subscribers.py` - Contains emotion parameters and subscriber logic
  - `child_node.py` - Child robot control script
  - `parent_node.py` - Parent robot control script
  - `attachment_model_control.py` - Central attachment controller
  - `perception.py` - AprilTag detection and tracking
  - `camera_recording.py` - Video recording node

- **Experiment Data**: `/root/mdk/catkin_ws/src/Attachment-Model/data`
  - Organized by experiment runs (e.g., `experiment_data_0/`, `experiment_data_1/`, etc.)
  - Each experiment folder contains:
    - `camera_recording.mp4` - Video recording of the simulation
    - `child_data.npy` - Child robot behavioral data
    - `parent_data.npy` - Parent robot behavioral data

### Accessing Experiment Data

All experiment data is automatically saved to the `data` directory within the container. To retrieve your data:

1. **Option 1 - Using VS Code** (see Development Tools section below):
   - Attach VS Code to the running container
   - Navigate to `/root/mdk/catkin_ws/src/Attachment-Model/data`
   - Copy files to your host machine

2. **Option 2 - Using Docker commands**:
   ```bash
   # From your host machine (in a new terminal)
   docker cp ros_container:/root/mdk/catkin_ws/src/Attachment-Model/data ./experiment_results
   ```

3. **Option 3 - Save the entire container**:
   - Use `./save_docker.bash` to commit all changes to the image
   - Data will be preserved in the Docker image for future use

## Experimental Data

Pre-collected experimental data is included in the `data/` directory. Each experiment folder (`experiment_data_{index}/`) contains:
- `camera_recording.mp4` - Video recording of the robot interaction
- `child_data.npy` - Child robot behavioral data
- `parent_data.npy` - Parent robot behavioral data

### Data Structure

Each `.npy` file contains a dictionary with the following fields:
```python
{
    'Xc': [],                           # Child position coordinates
    'Xp': [],                           # Parent position coordinates
    'ambivalence': [],                  # Ambivalence parameter values
    'avoidance': [],                    # Avoidance parameter values
    'physical_coupling_targets': [],    # Physical distance coupling targets
    'emotional_coupling_targets': [],   # Emotional distance coupling targets
    'child_states': [],                 # Child internal states
    'parent_states': [],                # Parent internal states
    'vocal_child': [],                  # Child vocalization events
    'vocal_parent': [],                 # Parent vocalization events
    'child_evidence_trace': [],         # Child evidence accumulation
    'parent_evidence_trace': []         # Parent evidence accumulation
}
```

### Dataset Organization

**Training Data** (3 minutes per run):
- `experiment_data_0`: Avoidance = 0.0, Ambivalence = 0.0
- `experiment_data_1` to `8`: Avoidance = 0.1 to 0.8, Ambivalence = 0.0
- `experiment_data_9` to `16`: Avoidance = 0.0, Ambivalence = 0.1 to 0.8

**Experimental Data** (1 minute per run):
- `experiment_data_17`: Avoidance = 0.0, Ambivalence = 0.0
- `experiment_data_18` to `25`: Avoidance = 0.1 to 0.8, Ambivalence = 0.0
- `experiment_data_26` to `33`: Avoidance = 0.0, Ambivalence = 0.1 to 0.8

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