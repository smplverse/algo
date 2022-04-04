FROM nvcr.io/nvidia/pytorch:22.03-py3 as base
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN conda install -c conda-forge opencv

FROM base as main
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
