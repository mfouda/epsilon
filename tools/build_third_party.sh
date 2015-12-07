#!/bin/bash -eu
#
# Build third_party dependencies (gflags, glog, protobuf)

third_party=$PWD/third_party
build=$PWD/build-cc/third_party

mkdir -p $build

cmake_flags="-DCMAKE_INSTALL_PREFIX=$build"
if [ "$(uname -s)" == "Linux" ]; then
    cmake_flags+=" -DCMAKE_CXX_FLAGS=-fPIC"
fi

# gflags
mkdir -p $third_party/gflags/build
cd $third_party/gflags/build
cmake $cmake_flags ..
make -j install

# glog
mkdir -p $third_party/glog/build
cd $third_party/glog/build
cmake $cmake_flags ..
make -j install

# protobuf
cd $third_party/protobuf
./autogen.sh
./configure --enable-static --disable-shared --prefix=$build
make -j install
