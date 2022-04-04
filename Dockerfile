FROM nvcr.io/nvidia/pytorch:22.03-py3

COPY . .
RUN pip install -r requirements.txt
