

FROM pytorch/pytorch:1.8.1-cuda11.1-cudnn8-devel
ENV NVIDIA_DRIVER_CAPABILITIES all

RUN rm /etc/apt/sources.list.d/cuda.list \
    && rm /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-key del 7fa2af80 \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub


RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        software-properties-common curl vim git zip unzip unrar p7zip-full wget cmake \
        apache2 openssl libssl-dev

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
        libwebp-dev libcairo2-dev libjpeg-dev libgif-dev \
        libboost-all-dev libopencv-dev libwxgtk3.0-gtk3-dev \
        ffmpeg libgl1-mesa-glx libsm6 libxext6 libxrender-dev libx11-xcb1 \
        mesa-utils xauth xorg openbox xvfb

RUN conda install \
    'matplotlib==3.5.0' \
    'scipy==1.7.1' \
    'scikit-learn==1.0.1' \
    'scikit-image==0.18.3' \
    'imagesize==1.3.0'

RUN conda install -c conda-forge \
    'patool==1.12' \
    'addict==2.4.0' \
    'igl==2.2.1' \
    'meshplot==0.4.0'

RUN conda install \
    'jupyterlab==3.2.1'


##########################################################################


RUN conda install \
    'pillow==8.4.0'

RUN conda install -c conda-forge \
    'wandb==0.12.7'

RUN conda install -c fvcore -c iopath -c conda-forge \
    'fvcore==0.1.5.post20210915' \
    'iopath==0.1.9' \
&& conda install -c bottler \
    'nvidiacub==1.10.0' \
&& conda install -c pytorch3d \
    'pytorch3d==0.6.1'



RUN pip install \
    'pyunpack==0.2.2'

RUN pip install \
    'pygltflib==1.14.5'

RUN pip install \
    'opencv-contrib-python==4.5.4.60'

RUN pip install \
    'kornia==0.6.2'

RUN pip install \
    'cupy-cuda111==9.4.0'

RUN pip install \
    'moderngl==5.6.4' \
    'moderngl-window==2.4.1' \
    'PyQt5==5.15.6' \
    'PyQt5-Qt5==5.15.2' \
    'PyQt5-sip==12.10.1' \
    'PySDL2==0.9.11' \
    'PySide2==5.15.2.1' \
    'cffi==1.15.0' \
    'glfw==1.12.0' \
    'pymunk==6.2.1' \
    'pywavefront==1.3.3' \
    'shiboken2==5.15.2.1'

RUN pip install \
    'trimesh==3.12.3'

RUN pip install \
    'torch-fidelity==0.3.0'


# RUN pip install \
#     --extra-index-url https://developer.download.nvidia.com/compute/redist/ \
#     'nvidia-dali-cuda110==1.8.0'





