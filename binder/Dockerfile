FROM jupyter/minimal-notebook:x86_64-2022-10-09

MAINTAINER Jupyter Help <jupyter-help@brown.edu>

USER root

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get -yq dist-upgrade \
    && apt-get install -yq --no-install-recommends \
    build-essential \
    openssh-client \
    vim \ 
    curl \
    gcc \
    libopenmpi-dev \
    automake \
    libtool \
    && apt-get clean

USER $NB_UID

RUN pip install --upgrade pip

USER root

RUN apt-get install -yq --no-install-recommends \
    xvfb \
    x11-utils \
    libx11-dev \
    qtbase5-dev \
    qt5-qmake \
    libncurses-dev \
    flex \
    bison \
    && apt-get clean

ENV DISPLAY=:99

USER $NB_UID

RUN pip install vtk && \
    pip install numpy && \
    pip install scipy && \
    pip install pyqt5 && \
    pip install xvfbwrapper && \
    pip install pyvista && \
    pip install pyvistaqt && \
    pip install nibabel && \
    pip install ipyevents && \
    pip install darkdetect

USER ${NB_UID}

RUN pip install mpi4py

RUN pip install mne && pip install hnn_core

RUN git init . && \
    git remote add origin https://github.com/jonescompneurolab/hnn-core.git && \
    git fetch origin gh-pages && \
    git checkout gh-pages

ENV PYVISTA_TRAME_SERVER_PROXY_PREFIX='/proxy/'

ENV JUPYTERHUB_SERVICE_PREFIX='/proxy/'

# Add an x-server to the entrypoint. This is needed by Mayavi
ENTRYPOINT ["tini", "-g", "--", "xvfb-run"]
