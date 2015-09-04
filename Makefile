# TODO(mwytock): Need a better make system so that every source file doesnt
# depend on every header

# Internal directories
src_dir = src
build_dir = build
proto_dir = proto
python_dir = python
sub_dir = epsilon
third_party_dir = third_party
gtest_dir = $(third_party_dir)/googletest/googletest

CC = g++
CXX = g++
OPTFLAGS = -O3

CXXFLAGS = `pkg-config --cflags $(LIBS)`
CXXFLAGS += $(OPTFLAGS) -std=c++14
CXXFLAGS += -Wall -Wextra -Werror -Wno-sign-compare -Wno-unused-parameter
CXXFLAGS += -I$(build_dir) -I$(src_dir) -I$(third_party_dir)
CXXFLAGS += -I$(gtest_dir)/include

LDLIBS += `pkg-config --libs $(LIBS)`
LDFLAGS = -Wl,-no-as-needed

# Homebrew installed gflags doesnt show up in pkg-config in OS X
UNAME_S = $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
	LIBS = protobuf libglog libcurl libtcmalloc libprofiler
	CXXFLAGS += -I/usr/local/include
	LDLIBS += -L/usr/local/include -lgflags
else
	LIBS = protobuf libglog libcurl libtcmalloc libprofiler libgflags
endif

common_cc = \
	epsilon/algorithms/prox_admm.cc \
	epsilon/algorithms/prox_admm_test.cc \
	epsilon/algorithms/solver.cc \
	epsilon/expression/expression.cc \
	epsilon/expression/problem.cc \
	epsilon/operators/prox.cc \
	epsilon/operators/affine.cc \
	epsilon/operators/prox_test.cc \
	epsilon/parameters/local_parameter_service.cc \
	epsilon/util/dynamic_matrix.cc \
	epsilon/util/file.cc \
	epsilon/util/init.cc \
	epsilon/util/port.cc \
	epsilon/util/problems.cc \
	epsilon/util/string.cc \
	epsilon/util/synchronization.cc \
	epsilon/util/thread_pool.cc \
	epsilon/util/time.cc \
	epsilon/util/vector.cc \
	epsilon/util/vector_test.cc \

common_test_cc = \
	epsilon/algorithms/algorithm_testutil.cc \
	epsilon/expression/expression_testutil.cc \
	epsilon/util/test_main.cc \
	epsilon/util/vector_testutil.cc

proto = \
	epsilon/expression.proto \
	epsilon/prox.proto \
	epsilon/solver_params.proto

tests = \
	epsilon/algorithms/prox_admm_test \
	epsilon/operators/prox_test \
	epsilon/operators/affine_test \
	epsilon/util/vector_test

# Google test
gtest_srcs = $(gtest_dir)/src/*.cc $(gtest_dir)/src/*.h

# Generated files
proto_cc  = $(proto:%.proto=$(build_dir)/%.pb.cc)
proto_obj = $(proto:%.proto=$(build_dir)/%.pb.o)
proto_py  = $(proto:%.proto=$(python_dir)/%_pb2.py)
common_obj = $(common_cc:%.cc=$(build_dir)/%.o)
common_test_obj = $(common_test_cc:%.cc=$(build_dir)/%.o)
build_tests = $(tests:%=$(build_dir)/%)
build_sub_dirs = $(addprefix $(build_dir)/, $(dir $(common_cc)))
#build_binaries = $(binaries:%=$(build_dir)/%)

# Stop make from deleting intermediate files
.SECONDARY:

proto_py: $(proto_py)

clean:
	rm -rf $(build_dir) $(python_dir)/build
	find $(python_dir) -name '*_pb2.py*' -or -name '*.pyc' -exec rm {} \;

$(build_dir):
	mkdir -p $(build_sub_dirs)

$(build_dir)/%.pb.cc $(build_dir)/%.pb.h: $(proto_dir)/%.proto | $(build_dir)
	protoc --proto_path=$(proto_dir) --cpp_out=$(build_dir) $<

$(python_dir)/%_pb2.py: $(proto_dir)/%.proto
	protoc --proto_path=$(proto_dir) --python_out=$(python_dir) $<

$(build_dir)/%.pb.o: $(src_dir)/%.pb.cc | $(build_dir)
	$(COMPILE.cc) $(OUTPUT_OPTION) $<

$(build_dir)/%.o: $(src_dir)/%.cc $(proto_cc) | $(build_dir)
	$(COMPILE.cc) $(OUTPUT_OPTION) $<

$(build_dir)/%: $(build_dir)/%.o $(common_obj) $(proto_obj)
	$(LINK.o) $^ $(LDLIBS) -o $@

# Test-related rules
test: $(build_tests) $(proto_py)
	@$(tools_dir)/run_tests.sh $(build_tests)
	@python -m unittest discover $(python_dir)

$(build_dir)/gtest-all.o: $(gtest_srcs)
	$(COMPILE.cc) -I$(gtest_dir) -Wno-missing-field-initializers -c $(gtest_dir)/src/gtest-all.cc -o $@

$(build_dir)/gtest.a : $(build_dir)/gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

$(build_dir)/%_test: $(build_dir)/%_test.o $(common_obj) $(proto_obj) $(common_test_obj) $(build_dir)/gtest.a
	$(LINK.o) $^ $(LDLIBS) -o $@
