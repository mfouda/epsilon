FROM debian:latest

# Debian packages
RUN apt-get update && apt-get install -y \
  autoconf \
  autotools-dev \
  build-essential \
  bzip2 \
  curl \
  g++ \
  git \
  gfortran \
  libc-dev \
  libgflags-dev \
  libgtest-dev \
  libopenblas-dev \
  libtool \
  make \
  python-dev \
  python-numpy \
  python-scipy \
  python-setuptools \
  unzip \
  wget \
  zip && apt-get clean

# Python and cvxpy
RUN easy_install -U pip
RUN pip install -U cvxpy

# Protobuf latest version
RUN git clone https://github.com/google/protobuf.git
RUN cd protobuf && ./autogen.sh && ./configure && make install

CMD ["bash"]
