#https://hub.docker.com/_/microsoft-vscode-devcontainers

ARG PYTHON_VERSION=3.10
ARG CONDA_VERSION=22.11.1-1
ARG NUMPY_VERSION=1.23.5
ARG TF_VERSION=2.11.0
ARG REVERB_VERSION=0.10.0
ARG TF_AGENTS_VERSION=0.15.0

# Mixes between nvidia/cuda and tensorflow docker images

FROM tensorflow/tensorflow:${TF_VERSION}-gpu

# Let us install tzdata painlessly
ENV DEBIAN_FRONTEND=noninteractive

# Needed for string substitution
SHELL ["/bin/bash", "-c"]

#RUN apt-get update \
#    && apt-get install -y \
#        wget \
#    && rm -rf /var/lib/apt/lists/*

# Install miniconda
#ENV CONDA_DIR /opt/conda

#RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py${PYTHON_VERSION/./}_${CONDA_VERSION}-Linux-x86_64.sh -O ~/miniconda.sh \
#    && /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
#ENV PATH=$CONDA_DIR/bin:$PATH

#RUN python3 -m pip --no-cache-dir install --upgrade \
#    "pip<20.3" \
#    setuptools

# install tensorflow
RUN python3 -m pip install \
    dm-reverb \
    tf-agents \
    matplotlib \
    ipykernel
#    numpy==$NUMPY_VERSION \
#    tensorflow-gpu==$TF_VERSION

RUN apt-get update \
    && apt-get install -y \
        xvfb \
        ffmpeg \
        freeglut3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install \
    'imageio==2.4.0' \
    pyvirtualdisplay