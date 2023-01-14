FROM mcr.microsoft.com/devcontainers/base:bionic

RUN apt-get update \
    && export DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y git \ 
    && apt-get install -y build-essential \
    && apt-get install -y wget \
    && apt-get clean  \
    && rm -rf /var/lib/apt/lists/*

ARG PYTHON_VERSION=310
ARG CONDA_VERSION=22.11.1-1
ARG NUMPY_VERSION=

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py${PYTHON_VERSION}_${CONDA_VERSION}-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# install tensorflow
RUN conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
ENV LD_LIBRARY_PATH=$CONDA_DIR/lib/
RUN python3 -m pip install numpy==1.23.5 tensorflow==2.11.0