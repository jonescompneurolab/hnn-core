FROM jupyter/minimal-notebook:65761486d5d3 

MAINTAINER Jupyter Help <jupyter-help@brown.edu>

USER root

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && apt-get -yq dist-upgrade \
    && apt-get install -yq --no-install-recommends \
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
RUN npm i npm@latest -g

USER $NB_UID

RUN pip install --upgrade pip
RUN npm i npm@latest -g

USER root

RUN apt-get install -yq --no-install-recommends \
    xvfb \
    x11-utils \
    libx11-dev \
    qt5-default \
    libncurses-dev \
    flex \
    bison \
    && apt-get clean

USER ${NB_UID}

RUN pip install numpy && \
    pip install scipy && \
    pip install matplotlib && \
    pip install pyqt5 && \
	pip install ipywidgets && \
    pip install pillow && \
    pip install https://api.github.com/repos/mne-tools/mne-python/zipball/master
    
RUN pip install mpi4py

RUN git clone https://github.com/neuronsimulator/nrn && \
	cd nrn && \
	./build.sh && \
	./configure --with-nrnpython=python3 --without-iv --prefix=$HOME && \
	make -j4 && \
	make install -j4 && \
	cd src/nrnpython/ && \
	python3 setup.py install

ENV PATH=${PATH}:${HOME}/x86_64/bin

RUN git clone https://github.com/jonescompneurolab/hnn-core.git && \
	cd hnn-core && \
    make && \
    python setup.py develop

ENV PATH=${PATH}:${HOME}/hnn-core/x86_64/

RUN git init . && \
    git remote add origin https://github.com/jonescompneurolab/hnn-core.git && \
	git fetch origin gh-pages && \
	git checkout gh-pages

# Add an x-server to the entrypoint. This is needed by Mayavi
ENTRYPOINT ["tini", "-g", "--", "xvfb-run"] 

# CMD ["jupyter", "notebook", "--ip", "0.0.0.0"]
