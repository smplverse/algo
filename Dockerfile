FROM nvcr.io/nvidia/tensorflow:22.03-tf1-py3 as base
RUN apt-get update && apt-get install -y libgl1-mesa-glx
RUN conda install -c conda-forge opencv bcolz

FROM base as main
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .
