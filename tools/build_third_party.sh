#!/bin/bash -eu
#
# Build third_party dependencies (gflags, glog, protobuf)

third_party=$PWD/third_party
build=$PWD/build-deps

mkdir -p $build

cmake_flags="-DCMAKE_INSTALL_PREFIX=$build"
if [ "$(uname -s)" == "Linux" ]; then
    cmake_flags+=" -DCMAKE_CXX_FLAGS=-fPIC"
    export CXXFLAGS=-fPIC
fi

# glog
mkdir -p $build/glog
cd $build/glog
cmake $cmake_flags -DWITH_GFLAGS=OFF $third_party/glog
make -j install

# protobuf
cd $third_party/protobuf
./autogen.sh
mkdir -p $build/protobuf
cd $build/protobuf
$third_party/protobuf/configure \
    --enable-static --disable-shared --without-zlib --prefix=$build
make -j install
