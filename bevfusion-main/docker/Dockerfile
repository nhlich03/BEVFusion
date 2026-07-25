# FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# RUN apt-get update && apt-get install wget -yq
# RUN apt-get install build-essential g++ gcc -y
# ENV DEBIAN_FRONTEND noninteractive
# RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y
# RUN apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git -y

# # Install miniconda
# ENV CONDA_DIR /opt/conda
# RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
#      /bin/bash ~/miniconda.sh -b -p /opt/conda
# # Put conda in path so we can use conda activate
# ENV PATH=$CONDA_DIR/bin:$PATH
# RUN conda install python=3.8
# RUN conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch
# RUN pip install Pillow==8.4.0
# RUN pip install tqdm
# RUN pip install torchpack
# RUN pip install mmcv==1.4.0 mmcv-full==1.4.0 mmdet==2.20.0
# RUN pip install nuscenes-devkit
# RUN pip install mpi4py==3.0.3
# RUN pip install numba==0.48.0



FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

# ===== System =====
RUN apt-get update && apt-get install -y \
    wget git curl vim \
    build-essential cmake \
    libgl1-mesa-glx libglib2.0-0 \
    libsm6 libxrender1 libxext6 \
    ffmpeg \
    openmpi-bin libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

# ===== Miniconda =====
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh

# ===== Create env =====
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main && \
    conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r && \
    conda create -y -n bevfusion python=3.8
SHELL ["conda", "run", "-n", "bevfusion", "/bin/bash", "-c"]

# ===== PyTorch =====
RUN pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# ===== Core =====
RUN pip install \
    numpy scipy matplotlib pyyaml tqdm pillow==8.4.0 \
    opencv-python

# ===== BEV deps =====
RUN pip install nuscenes-devkit
RUN pip install mmdet==2.20.0 numba

# ===== Torchpack =====
RUN pip install git+https://github.com/mit-han-lab/torchpack.git

# ===== MPI =====
RUN pip install mpi4py==3.0.3

# ===== Extra =====
RUN pip install timm einops future tensorboard

RUN pip install mmcv-full==1.7.2 \
    -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1.0/index.html

# ===== Patches for double registration and version check =====
RUN sed -i "s/force=False/force=True/g" /opt/conda/envs/bevfusion/lib/python3.8/site-packages/mmcv/utils/registry.py
RUN sed -i "s/mmcv_maximum_version = '1.5.0'/mmcv_maximum_version = '1.8.0'/g" /opt/conda/envs/bevfusion/lib/python3.8/site-packages/mmdet/__init__.py

# ===== Workdir =====
WORKDIR /workspace

CMD ["conda", "run", "--no-capture-output", "-n", "bevfusion", "/bin/bash"]