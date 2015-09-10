# Makefile for epsilon
#
# Tested on: Mac OS X 10.10.5, Ubuntu 15.04

# Settings
OPTFLAGS = -O3

# Internal directories
src_dir = src
proto_dir = proto
tools_dir = tools
eigen_dir = third_party/eigen
gtest_dir = third_party/googletest/googletest
build_dir = build-cc

SYSTEM = $(shell uname -s)
CC = g++
CXX = g++

CXXFLAGS += $(OPTFLAGS) -std=c++14
CXXFLAGS += -Wall -Wextra -Werror
CXXFLAGS += -Wno-sign-compare -Wno-unused-parameter -Wno-macro-redefined
CXXFLAGS += -I$(build_dir) -I$(src_dir) -I$(eigen_dir)
CXXFLAGS += -I$(gtest_dir)/include

ifeq ($(SYSTEM),Linux)
CXXFLAGS += -fPIC
endif

# NOTE(mwytock): libgflags.pc is missing on Homebrew
ifeq ($(SYSTEM),Darwin)
LIBS = protobuf libglog
LDLIBS += `pkg-config --libs $(LIBS)` -L/usr/local/lib -lgflags
CXXFLAGS += `pkg-config --cflags $(LIBS)` -I/usr/local/include
else
LIBS = protobuf libglog libgflags
LDLIBS += `pkg-config --libs $(LIBS)`
CXXFLAGS += `pkg-config --cflags $(LIBS)`
endif

common_cc = \
	epsilon/algorithms/prox_admm.cc \
	epsilon/algorithms/solver.cc \
	epsilon/expression/expression.cc \
	epsilon/expression/expression_util.cc \
	epsilon/expression/var_offset_map.cc \
	epsilon/file/file.cc \
	epsilon/operators/affine.cc \
	epsilon/operators/prox.cc \
	epsilon/parameters/local_parameter_service.cc \
	epsilon/util/dynamic_matrix.cc \
	epsilon/util/string.cc \
	epsilon/util/time.cc \
	epsilon/util/vector.cc \
	epsilon/util/vector_file.cc

common_test_cc = \
	epsilon/algorithms/algorithm_testutil.cc \
	epsilon/expression/expression_testutil.cc \
	epsilon/util/test_main.cc \
	epsilon/util/vector_testutil.cc

proto = \
	epsilon/data.proto \
	epsilon/expression.proto \
	epsilon/solver.proto \
	epsilon/solver_params.proto

tests = \
	epsilon/algorithms/prox_admm_test \
	epsilon/operators/prox_test \
	epsilon/operators/affine_test \
	epsilon/util/vector_test

libs = epsilon

# Google test
gtest_srcs = $(gtest_dir)/src/*.cc $(gtest_dir)/src/*.h

# Generated files
proto_cc  = $(proto:%.proto=$(build_dir)/%.pb.cc)
proto_obj = $(proto:%.proto=$(build_dir)/%.pb.o)
common_obj = $(common_cc:%.cc=$(build_dir)/%.o)
common_test_obj = $(common_test_cc:%.cc=$(build_dir)/%.o)
build_tests = $(tests:%=$(build_dir)/%)
build_sub_dirs = $(addprefix $(build_dir)/, $(dir $(common_cc)))
build_libs = $(libs:%=$(build_dir)/lib%.a)

# Stop make from deleting intermediate files
.SECONDARY:

all: $(build_libs)

clean:
	rm -rf $(build_dir)

# Build
$(build_dir):
	mkdir -p $(build_sub_dirs)

$(build_dir)/%.pb.cc $(build_dir)/%.pb.h: $(proto_dir)/%.proto | $(build_dir)
	protoc --proto_path=$(proto_dir) --cpp_out=$(build_dir) $<

$(build_dir)/%.pb.o: $(src_dir)/%.pb.cc | $(build_dir)
	$(COMPILE.cc) $(OUTPUT_OPTION) $<

$(build_dir)/%.o: $(src_dir)/%.cc $(proto_cc) | $(build_dir)
	$(COMPILE.cc) $(OUTPUT_OPTION) $<

$(build_dir)/libepsilon.a: $(common_obj) $(proto_obj)
	$(AR) rcs $@ $^
ifeq ($(SYSTEM),Darwin)
	ranlib $@
endif

# Test
test: $(build_tests)
	@$(tools_dir)/run_tests.sh $(build_tests)

# NOTE(mwytock): Add -Wno-missing-field-intializers to this rule to avoid error
# on OS X
$(build_dir)/gtest-all.o: $(gtest_srcs)
	$(COMPILE.cc) -I$(gtest_dir) -Wno-missing-field-initializers -c $(gtest_dir)/src/gtest-all.cc -o $@

$(build_dir)/%_test: $(build_dir)/%_test.o $(common_obj) $(proto_obj) $(common_test_obj) $(build_dir)/gtest-all.o
	$(LINK.o) $^ $(LDLIBS) -o $@
