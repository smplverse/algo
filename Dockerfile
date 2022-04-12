FROM nvcr.io/nvidia/cuda:11.6.1-cudnn8-devel-ubi8 as base

RUN wget --quiet \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
  -O ~/miniconda.sh \
  && /bin/bash ~/miniconda.sh -b -p /opt/conda

RUN conda install -y -c conda-forge opencv

FROM base as main
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
