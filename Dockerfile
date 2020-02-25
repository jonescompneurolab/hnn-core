FROM ubuntu:19.04

# avoid questions from debconf
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update && \
    apt-get install -y python3-dev python3-pip python3-tk \
                       git vim iputils-ping net-tools iproute2 nano sudo \
                       telnet language-pack-en-base && \
    apt-get autoremove -y --purge && \
    apt-get clean

RUN pip3 install pip --upgrade

# create the group hnn_group and user hnn_user
# add hnn_user to the sudo group
RUN groupadd hnn_group && useradd -m -b /home/ -g hnn_group hnn_user && \
    adduser hnn_user sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R hnn_user:hnn_group /home/hnn_user && \
    chsh -s /bin/bash hnn_user

RUN echo '#!/bin/bash\n\
# write the date that base OS was updated to /base-install\n\
date -u +"%Y-%m-%dT%H:%M:%SZ" > /base-install' > /usr/local/bin/date_base_install.sh
RUN chmod +x /usr/local/bin/date_base_install.sh && \
    /usr/local/bin/date_base_install.sh

RUN echo 'export PATH=$PATH:/home/hnn_user/nrn/build/x86_64/bin\n\
export PYTHONPATH=/home/hnn_user/nrn/build/lib/python\n\
export MPLBACKEND=TkAgg\n\
export OMPI_MCA_mpi_warn_on_fork=0\n\
export OMPI_MCA_btl_openib_allow_ib=1\n\
export OMPI_MCA_btl_vader_single_copy_mechanism=none' > /home/hnn_user/hnn_envs
RUN chown hnn_user:hnn_group /home/hnn_user/hnn_envs

USER hnn_user

# allow user to specify architecture if different than x86_64
ARG CPU=x86_64
# supply the path NEURON binaries for building hnn
ENV PATH=${PATH}:/home/hnn_user/nrn/build/$CPU/bin

# use environment variables from hnn_envs
RUN echo 'source /home/hnn_user/hnn_envs' >> ~/.bashrc

# compile NEURON, only temporarily installing packages for building
RUN sudo apt-get update && \
    sudo apt-get install -y \
                       bison flex automake libtool libncurses-dev zlib1g-dev \
                       libopenmpi-dev openmpi-bin \
                       git vim iputils-ping net-tools iproute2 nano sudo \
                       telnet language-pack-en-base && \
    cd /home/hnn_user/ && \
    mkdir nrn && \
    cd nrn && \
    git clone https://github.com/neuronsimulator/nrn src && \
    cd /home/hnn_user/nrn/src && \
    git checkout 7.7 && \
    ./build.sh && \
    ./configure --with-nrnpython=python3 --with-paranrn --disable-rx3d \
      --without-iv --without-nrnoc-x11 --with-mpi \
      --prefix=/home/hnn_user/nrn/build && \
    make -j4 && \
    make install -j4 && \
    cd src/nrnpython && \
    python3 setup.py install --user && \
    cd /home/hnn_user/nrn/ && \
    rm -rf src && \
    sudo apt-get -y remove --purge bison flex zlib1g-dev && \
    sudo apt-get autoremove -y --purge && \
    sudo apt-get clean

ARG BUILD_DATE
ARG VCS_REF
ARG VCS_TAG
ARG SOURCE_BRANCH

LABEL org.label-schema.build-date=$BUILD_DATE \
      org.label-schema.vcs-url="https://github.com/jonescompneurolab/hnn-core.git" \
      org.label-schema.vcs-ref=$VCS_REF \
      org.label-schema.schema-version=$VCS_TAG

RUN sudo pip install matplotlib scipy numpy joblib psutil && \
    sudo rm -rf /home/hnn_user/.cache && \
    cd /home/hnn_user && \
    git clone https://github.com/blakecaldwell/hnn-core.git \
      --single-branch --branch $SOURCE_BRANCH && \
    cd hnn-core && \
    sudo python3 setup.py develop && \
    make

# if users open up a shell, they should go to the repo checkout
WORKDIR /home/hnn_user/hnn-core