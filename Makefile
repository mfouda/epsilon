# TODO(mwytock): Need a better make system so that every source file doesnt
# depend on every header

CC = g++
CXX = g++
GRPC_CPP_PLUGIN = `which grpc_cpp_plugin`
GRPC_PYTHON_PLUGIN = `which grpc_python_plugin`

OPTFLAGS = -O3
LIBS = protobuf libgflags libglog libcurl libtcmalloc libprofiler
EIGEN = $(HOME)/eigen

# TODO(mwytock): Consider using pkg-config to manage these things?
CXXFLAGS = `pkg-config --cflags $(LIBS)`
CXXFLAGS += $(OPTFLAGS) -std=c++14
CXXFLAGS += -Wall -Wextra -Werror -Wno-sign-compare -Wno-unused-parameter
CXXFLAGS += -I$(build_dir) -I$(src_dir) -I$(EIGEN)

LDLIBS += `pkg-config --libs $(LIBS)`
LDLIBS += -lbenchmark                      # Google benchmark
LDLIBS += -lamd -lldl -lsuitesparseconfig  # SuiteSparse
LDLIBS += -lgrpc++_unsecure -lgrpc -lgpr   # GRPC
LDFLAGS = -Wl,-no-as-needed

# ECSNS
ECSNS = $(HOME)/go/src/github.com/mwytock/ecsns
ECSNS_PROTO = $(ECSNS)/proto/ecsns

# Google Test
GTEST_DIR = /usr/src/gtest
GTEST_SRCS_ = $(GTEST_DIR)/src/*.cc $(GTEST_DIR)/src/*.h

# Internal directories
src_dir = src
build_dir = build
proto_dir = proto
python_dir = python
sub_dir = distopt
tools_dir = tools

common_cc = \
	distopt/algorithms/consensus_epsilon.cc \
	distopt/algorithms/consensus_epsilon_sub.cc \
	distopt/algorithms/consensus_prox.cc \
	distopt/algorithms/consensus_prox_master.cc \
	distopt/algorithms/scs.cc \
	distopt/algorithms/solver.cc \
	distopt/algorithms/solver_service_impl.cc \
	distopt/expression/cone.cc \
	distopt/expression/eval.cc \
	distopt/expression/expression.cc \
	distopt/expression/graph.cc \
	distopt/expression/linear.cc \
	distopt/expression/operator.cc \
	distopt/expression/problem.cc \
	distopt/file/file.cc \
	distopt/hash/hash.cc \
	distopt/master/master_impl.cc \
	distopt/operators/cone_projection.cc \
	distopt/operators/prox.cc \
	distopt/parameters/local_parameter_service.cc \
	distopt/parameters/parameter_server_impl.cc \
	distopt/parameters/remote_parameter_service.cc \
	distopt/util/dynamic_matrix.cc \
	distopt/util/file.cc \
	distopt/util/init.cc \
	distopt/util/port.cc \
	distopt/util/problems.cc \
	distopt/util/string.cc \
	distopt/util/synchronization.cc \
	distopt/util/thread_pool.cc \
	distopt/util/time.cc \
	distopt/util/vector.cc \
	distopt/worker/data_cache.cc \
	distopt/worker/operator_cache.cc \
	distopt/worker/worker_impl.cc \
	distopt/worker/worker_pool.cc \
	ecsns/ecsns_client.cc \
	epsilon/epsilon.cc \
	epsilon/projections/negative_log_det.cc \
	epsilon/projections/non_negative.cc \
	epsilon/projections/norm1.cc \
	epsilon/projections/norm2.cc \
	epsilon/projections/norm_nuclear.cc \
	epsilon/projections/projection.cc \
	epsilon/sparse_ldl.cc

common_test_cc = \
	distopt/algorithms/algorithm_testutil.cc \
	distopt/expression/expression_testutil.cc \
	distopt/hash/hash_testutil.cc \
	distopt/util/test_main.cc \
	distopt/util/vector_testutil.cc \
	distopt/util/backends_testutil.cc \
	epsilon/projections/projection_testutil.cc

proto = \
	distopt/benchmarks.proto \
	distopt/data.proto \
	distopt/expression.proto \
	distopt/master.proto \
	distopt/operator.proto \
	distopt/parameters.proto \
	distopt/problem.proto \
	distopt/prox.proto \
	distopt/solver.proto \
	distopt/solver_params.proto \
	distopt/stats.proto \
	distopt/worker.proto

# Flaky tests:
# 	distopt/algorithms/consensus_epsilon_test
tests = \
	distopt/algorithms/consensus_epsilon_sub_test \
	distopt/algorithms/consensus_prox_master_test \
	distopt/algorithms/consensus_prox_test \
	distopt/algorithms/scs_test \
	distopt/expression/cone_test \
	distopt/expression/eval_test \
	distopt/expression/linear_test \
	distopt/expression/operator_test \
	distopt/file/file_test \
	distopt/hash/hash_test \
	distopt/operators/cone_projection_test \
	distopt/operators/prox_test \
	distopt/util/vector_test \
	epsilon/epsilon_test \
	epsilon/projections/negative_log_det_test \
	epsilon/projections/norm1_test \
	epsilon/projections/norm2_test \
	epsilon/sparse_ldl_test \
	epsilon/projections/random_test

binaries = \
	distopt/master/master \
	distopt/parameters/parameter_server \
	distopt/worker/worker \
	epsilon/benchmarks/cholesky \
	epsilon/benchmarks/quasi_definite \
	epsilon/benchmarks/solve \
	epsilon/solve


# Generated files
proto_cc  = $(proto:%.proto=$(build_dir)/%.pb.cc)
proto_cc += $(proto:%.proto=$(build_dir)/%.grpc.pb.cc)
proto_obj  = $(proto:%.proto=$(build_dir)/%.pb.o)
proto_obj += $(proto:%.proto=$(build_dir)/%.grpc.pb.o)
proto_py = $(proto:%.proto=$(python_dir)/%_pb2.py)
common_obj = $(common_cc:%.cc=$(build_dir)/%.o)
common_test_obj = $(common_test_cc:%.cc=$(build_dir)/%.o)
common_test_obj += $(build_dir)/gtest.a
build_binaries = $(binaries:%=$(build_dir)/%)
build_tests = $(tests:%=$(build_dir)/%)
build_sub_dirs = $(addprefix $(build_dir)/, $(dir $(common_cc) $(binaries)))

# Add stuff for ECSNS, in separate repository
proto_obj += build/ecsns/ecsns.pb.o
proto_obj += build/ecsns/ecsns.grpc.pb.o
build_sub_dirs += build/ecsns

# Stop make from deleting intermediate files
.SECONDARY:

all: $(build_binaries) $(proto_py)

proto_py: $(proto_py)

clean:
	rm -rf $(build_dir) $(python_dir)/build
	find $(python_dir) -name '*_pb2.py*' -or -name '*.pyc' -exec rm {} \;

$(build_dir):
	mkdir -p $(build_sub_dirs)

$(build_dir)/ecsns/%.pb.cc: $(ECSNS_PROTO)/%.proto | $(build_dir)
	protoc --proto_path=$(ECSNS_PROTO) --cpp_out=$(build_dir)/ecsns $<

$(build_dir)/ecsns/%.grpc.pb.cc: $(ECSNS_PROTO)/%.proto | $(build_dir)
	protoc --proto_path=$(ECSNS_PROTO) --grpc_out=$(build_dir)/ecsns --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN) $<

$(build_dir)/%.grpc.pb.cc $(build_dir)/%.grpc.pb.h: $(proto_dir)/%.proto | $(build_dir)
	protoc --proto_path=$(proto_dir) --grpc_out=$(build_dir) --plugin=protoc-gen-grpc=$(GRPC_CPP_PLUGIN) $<

$(build_dir)/%.pb.cc $(build_dir)/%.pb.h: $(proto_dir)/%.proto | $(build_dir)
	protoc --proto_path=$(proto_dir) --cpp_out=$(build_dir) $<

$(python_dir)/%_pb2.py: $(proto_dir)/%.proto
	protoc --proto_path=$(proto_dir) --python_out=$(python_dir) --grpc_out=$(python_dir) --plugin=protoc-gen-grpc=$(GRPC_PYTHON_PLUGIN) $<

$(build_dir)/%.pb.o: $(src_dir)/%.pb.cc | $(build_dir)
	$(COMPILE.cc) $(OUTPUT_OPTION) $<

$(build_dir)/%.o: $(src_dir)/%.cc $(proto_cc) | $(build_dir)
	$(COMPILE.cc) $(OUTPUT_OPTION) $<

$(build_dir)/%: $(build_dir)/%.o $(common_obj) $(proto_obj)
	$(LINK.o) $^ $(LDLIBS) -o $@

# Test-related rules
test: $(build_binaries) $(build_tests) $(proto_py)
	@$(tools_dir)/run_tests.sh $(build_tests)
	@python -m unittest discover $(python_dir)

$(build_dir)/gtest-all.o : $(GTEST_SRCS_)
	$(CXX) $(CPPFLAGS) -I$(GTEST_DIR) $(CXXFLAGS) -c $(GTEST_DIR)/src/gtest-all.cc -o $@

$(build_dir)/gtest.a : $(build_dir)/gtest-all.o
	$(AR) $(ARFLAGS) $@ $^

$(build_dir)/%_test: $(build_dir)/%_test.o $(common_obj) $(proto_obj) $(common_test_obj)
	$(LINK.o) $^ $(LDLIBS) -o $@
