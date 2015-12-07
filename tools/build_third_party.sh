#!/bin/bash -eu
#
# Build third_party dependencies (gflags, glog, protobuf)

third_party=$PWD/third_party
build=$PWD/build-cc/third_party

mkdir -p $build

# gflags
cd $third_party/gflags
mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$build ..
make -j install

# glog
cd $third_party/glog
./configure --enable-static --disable-shared --prefix=$build
make -j install

# protobuf
cd $third_party/protobuf
./autogen.sh
./configure --enable-static --disable-shared --prefix=$build
make -j install
