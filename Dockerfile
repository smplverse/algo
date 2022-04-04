FROM nvcr.io/nvidia/pytorch:22.03-py3

COPY . .
RUN pip install -r requirements.txt
RUN apt-get update \
  && apt-get install ffmpeg libsm6 libxext6 -y
RUN conda install -c conda-forge opencv
