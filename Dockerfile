FROM nvcr.io/nvidia/pytorch:22.03-py3

COPY . .

RUN apt-get update && apt-get install -y libgl1-mesa-glx

RUN pip install -r requirements.txt

RUN conda install -c conda-forge opencv
