FROM debian:latest

# Debian packages
RUN apt-get update && apt-get install -y \
  autoconf \
  autotools-dev \
  build-essential \
  bzip2 \
  curl \
  g++ \
  gfortran \
  git \
  libc-dev \
  libgflags-dev \
  libgoogle-glog-dev \
  libgtest-dev \
  libopenblas-dev \
  libquadmath0 \
  libtool \
  make \
  pkg-config \
  python-decorator \
  python-dev \
  python-setuptools \
  unzip \
  wget \
  zip && apt-get clean

# Protobuf latest version
RUN git clone https://github.com/google/protobuf.git
RUN cd protobuf && ./autogen.sh && ./configure && make install
RUN ldconfig

# Python with numpy/scipy and cvxpy
RUN easy_install -U pip
RUN pip install -U nose
RUN pip install -U numpy scipy
RUN pip install -U cvxpy

CMD ["bash"]
