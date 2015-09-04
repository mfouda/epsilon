
#include <grpc++/channel_arguments.h>
#include <grpc++/create_channel.h>

#include "distopt/algorithms/algorithm_testutil.h"
#include "distopt/expression/expression_testutil.h"
#include "distopt/parameters/remote_parameter_service.h"
#include "distopt/util/backends_testutil.h"
#include "distopt/util/problems.h"

Eigen::VectorXd ComputeLS(const Eigen::MatrixXd& A, const Eigen::VectorXd& b) {
  Eigen::LLT<Eigen::MatrixXd> solver;
  solver.compute(A.transpose()*A);
  return solver.solve(A.transpose()*b);
}

Problem DenseLassoProblem(int m, int n, double p) {
  srand(0);
  Eigen::MatrixXd A = Eigen::MatrixXd::Random(m, n);
  Eigen::VectorXd d = A.colwise().norm().array().inverse();
  A *= d.asDiagonal();
  Eigen::VectorXd x0 = RandomSparse(n, 1, p);
  Eigen::VectorXd b = A*x0 + sqrt(0.001)*Eigen::VectorXd::Random(m);

  const double lambda_max = (A.transpose()*b).lpNorm<Eigen::Infinity>();
  const double lambda = 0.1*lambda_max;

  Problem problem;
  *problem.mutable_objective() = BuildLasso(
      TestConstant(A), TestConstant(b), lambda, "x");
  return problem;
}

Eigen::VectorXd FetchRemoteParameter(uint64_t parameter_id) {
  std::unique_ptr<ParameterServer::Stub> stub =
      ParameterServer::NewStub(
          grpc::CreateChannel(
              GetLocalServerAddress(), grpc::InsecureCredentials(),
              grpc::ChannelArguments()));

  FetchRequest request;
  request.set_id(parameter_id);

  grpc::ClientContext context;
  FetchResponse response;
  grpc::Status status = stub->Fetch(&context, request, &response);
  CHECK(status.IsOk());

  return GetVector(response.value());
}
