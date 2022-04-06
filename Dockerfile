FROM nvcr.io/nvidia/tensorflow:22.03-tf2-py3 as base

# install opencv
RUN apt update 
RUN apt install -y cmake g++ wget unzip libgl1-mesa-glx
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/refs/tags/4.5.5.zip
RUN unzip opencv.zip && rm opencv.zip
RUN mkdir -p build && cd build && cmake ../opencv-4.5.5 && cmake --build . -- -j 8

FROM base as main
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
