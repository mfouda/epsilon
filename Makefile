# Makefile for epsilon, supports OS X and Linux

# Optimization flags, use OPTFLAGS=-g when debugging
OPTFLAGS = -DNDEBUG -O3

# Internal directories
src_dir = src
proto_dir = proto
tools_dir = tools
eigen_dir = third_party/eigen
gtest_dir = third_party/googletest/googletest
build_dir = build-cc

CFLAGS += $(OPTFLAGS)
CXXFLAGS += $(OPTFLAGS) -std=c++14
CXXFLAGS += -Wall -Wextra -Werror
CXXFLAGS += -Wno-sign-compare -Wno-unused-parameter
CXXFLAGS += -I$(build_dir) -I$(src_dir) -I$(eigen_dir)
CXXFLAGS += -I$(gtest_dir)/include

# Currently, dont parallelize at Eigen level
CXXFLAGS += -DEIGEN_DONT_PARALLELIZE

# Third-party library, glmgen
glmgen_dir = third_party/glmgen/c_lib/glmgen
glmgen_CFLAGS = -I$(glmgen_dir)/include

# Third-party library, Google benchmark
benchmark_LDLIBS = -lbenchmark

# System-specific configuration
SYSTEM = $(shell uname -s)

ifeq ($(SYSTEM),Linux)
CFLAGS += -fPIC
CXXFLAGS += -fPIC
endif

# NOTE(mwytock): libgflags.pc is missing on Homebrew
ifeq ($(SYSTEM),Darwin)
LIBS = protobuf libglog
LDLIBS += `pkg-config --libs $(LIBS)` -L/usr/local/lib -lgflags
CXXFLAGS += -Wno-macro-redefined
CXXFLAGS += `pkg-config --cflags $(LIBS)` -I/usr/local/include
else
LIBS = protobuf libglog libgflags
LDLIBS += `pkg-config --libs $(LIBS)`
CXXFLAGS += `pkg-config --cflags $(LIBS)`
endif

common_cc = \
	epsilon/affine/affine.cc \
	epsilon/algorithms/prox_admm.cc \
	epsilon/algorithms/solver.cc \
	epsilon/expression/expression.cc \
	epsilon/expression/expression_util.cc \
	epsilon/expression/var_offset_map.cc \
	epsilon/file/file.cc \
	epsilon/linear/dense_matrix_impl.cc \
	epsilon/linear/diagonal_matrix_impl.cc \
	epsilon/linear/kronecker_product_impl.cc \
	epsilon/linear/linear_map.cc \
	epsilon/linear/linear_map_add.cc \
	epsilon/linear/linear_map_multiply.cc \
	epsilon/linear/scalar_matrix_impl.cc \
	epsilon/linear/sparse_matrix_impl.cc \
	epsilon/parameters/local_parameter_service.cc \
	epsilon/prox/affine.cc \
	epsilon/prox/lambda_max.cc \
	epsilon/prox/max.cc \
	epsilon/prox/neg_log_det.cc \
	epsilon/prox/newton.cc \
	epsilon/prox/non_negative.cc \
	epsilon/prox/norm_2.cc \
	epsilon/prox/norm_nuclear.cc \
	epsilon/prox/ortho_invariant.cc \
	epsilon/prox/prox.cc \
	epsilon/prox/scaled_zone.cc \
	epsilon/prox/semidefinite.cc \
	epsilon/prox/sum_exp.cc \
	epsilon/prox/sum_inv_pos.cc \
	epsilon/prox/sum_largest.cc \
	epsilon/prox/sum_logistic.cc \
	epsilon/prox/sum_neg_entr.cc \
	epsilon/prox/sum_neg_log.cc \
	epsilon/prox/sum_square.cc \
	epsilon/prox/total_variation_1d.cc \
	epsilon/prox/vector_prox.cc \
	epsilon/prox/zero.cc \
	epsilon/util/file.cc \
	epsilon/util/string.cc \
	epsilon/util/time.cc \
	epsilon/vector/block_matrix.cc \
	epsilon/vector/block_vector.cc \
	epsilon/vector/vector_file.cc \
	epsilon/vector/vector_util.cc

third_party_obj = \
	$(glmgen_dir)/src/tf/tf_dp.o

common_test_cc = \
	epsilon/algorithms/algorithm_testutil.cc \
	epsilon/expression/expression_testutil.cc \
	epsilon/util/test_main.cc \
	epsilon/vector/vector_testutil.cc

proto = \
	epsilon/expression.proto \
	epsilon/solver.proto \
	epsilon/solver_params.proto

tests = \
	epsilon/linear/kronecker_product_impl_test \
	epsilon/linear/linear_map_test \
	epsilon/vector/block_matrix_test \
	epsilon/vector/block_vector_test

binaries = \
	epsilon/benchmark

libs = epsilon

# Google test
gtest_srcs = $(gtest_dir)/src/*.cc $(gtest_dir)/src/*.h

# Generated files
proto_cc  = $(proto:%.proto=$(build_dir)/%.pb.cc)
proto_obj = $(proto:%.proto=$(build_dir)/%.pb.o)
common_obj = $(common_cc:%.cc=$(build_dir)/%.o)
common_test_obj = $(common_test_cc:%.cc=$(build_dir)/%.o)
build_tests = $(tests:%=$(build_dir)/%)
build_binaries = $(binaries:%=$(build_dir)/%)
build_sub_dirs = $(addprefix $(build_dir)/, $(dir $(common_cc)))
build_libs = $(libs:%=$(build_dir)/lib%.a)

# Third party
build_sub_dirs += $(addprefix $(build_dir)/, $(dir $(third_party_obj)))
common_obj += $(third_party_obj:%=$(build_dir)/%)

# Stop make from deleting intermediate files
.SECONDARY:

all: $(build_libs) $(build_binaries)

clean:
	rm -rf $(build_dir)

# Build rules
$(build_dir):
	mkdir -p $(build_sub_dirs)

$(build_dir)/%.pb.cc $(build_dir)/%.pb.h: $(proto_dir)/%.proto | $(build_dir)
	protoc --proto_path=$(proto_dir) --cpp_out=$(build_dir) $<

$(build_dir)/%.pb.o: $(src_dir)/%.pb.cc | $(build_dir)
	$(COMPILE.cc) $(OUTPUT_OPTION) $<

$(build_dir)/%.o: $(src_dir)/%.cc $(proto_cc) | $(build_dir)
	$(COMPILE.cc) $(OUTPUT_OPTION) $<

# Third party build rules
$(build_dir)/$(glmgen_dir)/%.o: $(glmgen_dir)/%.c | $(build_dir)
	$(COMPILE.c) $(glmgen_CFLAGS) $(OUTPUT_OPTION) $<

# Targets
$(build_dir)/libepsilon.a: $(common_obj) $(proto_obj)
	$(AR) rcs $@ $^
ifeq ($(SYSTEM),Darwin)
	ranlib $@
endif

$(build_dir)/epsilon/benchmark: $(build_dir)/epsilon/benchmark.o $(common_obj) $(proto_obj)
	$(LINK.cc) $^ $(LDLIBS) -o $@

$(build_dir)/epsilon/linear/benchmarks: $(build_dir)/epsilon/linear/benchmarks.o
	$(LINK.cc) $^ $(benchmark_LDLIBS) $(LDLIBS) -o $@

# Tests
test: $(build_tests)
	@$(tools_dir)/run_tests.sh $(build_tests)

# NOTE(mwytock): Add -Wno-missing-field-intializers to this rule to avoid error
# on OS X
$(build_dir)/gtest-all.o: $(gtest_srcs)
	$(COMPILE.cc) -I$(gtest_dir) -Wno-missing-field-initializers -c $(gtest_dir)/src/gtest-all.cc -o $@

$(build_dir)/%_test: $(build_dir)/%_test.o $(common_obj) $(proto_obj) $(common_test_obj) $(build_dir)/gtest-all.o
	$(LINK.cc) $^ $(LDLIBS) -o $@
