!#/bin/bash

echo -ne '이 쉘스크립트는 docker container에 root로 들어가서 환경설정을 위해 실행하는 스크립트이다.'

# Compile protobuf configs
cd /home/tensorflow/models/research/
protoc object_detection/protos/*.proto --python_out=.

cp object_detection/packages/tf2/setup.py ./
PATH=/home/tensorflow/.local/bin:$PATH
python -m pip install -U pip
python -m pip install .

pip install ipython
pip install opencv-python
pip install opencv-contrib-python
pip install imageio
