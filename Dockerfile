
FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime

RUN apt-get update
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y libgl1-mesa-dev
RUN apt-get install -y git
RUN apt-get install -y build-essential

RUN pip install opencv-python  
RUN pip install numpy
RUN python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html

ARG USERNAME=ayb
RUN useradd -ms /bin/bash  $USERNAME
USER $USERNAME
