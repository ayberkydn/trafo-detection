FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
RUN apt-get update
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y git
RUN apt-get install -y build-essential

RUN pip install opencv-python  
RUN pip install numpy
RUN python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'

