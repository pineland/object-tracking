FROM "ubuntu:bionic"

# 위 이미지를 확장하기 위해 패키지 업데이트&업그레이드
RUN apt-get update && yes | apt-get upgrade

# 작업디렉토리 생성
RUN mkdir -p /tensorflow/models

# 패키지 다운로드 및 설치 프로그램 설치
RUN apt-get install -y git python3-pip
RUN pip3 install --upgrade pip

# tensorflow 설치
RUN pip3 install tensorflow

# Tensorflow Object Detection API가 의존하는 라이브러리 설치
RUN apt-get install -y protobuf-compiler python-pil python3-lxml
RUN pip install jupyter
RUN pip install matplotlib
RUN pip install tf_slim

# 예제코드를 다운로드
RUN git clone https://github.com/tensorflow/models.git /tensorflow/models

# 작업디렉토리 이동
WORKDIR /tensorflow/models/research
 
# Tensorflow Object Detection API가 모델과 학습파라미터의 환경구성에 사용하는 protobuf의 라이브러리가 프레임워크가 사용되기 전에 컴파일되어야 한다.
RUN protoc object_detection/protos/*.proto --python_out=.

# 로컬에서 실행할 때, /tensorflow/models/research/와 slim 디렉토리가 패스설정돼야 한다.
RUN export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

# 주피터노트북 실행을 위한 환경설정. root로 실행토록 한다.
RUN jupyter notebook --generate-config --allow-root
RUN echo "c.NotebookApp.password = u'sha1:6a3f528eec40:6e896b6e4828f525a6e20e5411cd1c8075d68619'" >> /root/.jupyter/jupyter_notebook_config.py
EXPOSE 8888
CMD ["jupyter", "notebook", "--allow-root", "--notebook-dir=/tensorflow/models/research", "--ip=0.0.0.0", "--port=8888", "--no-browser"]
