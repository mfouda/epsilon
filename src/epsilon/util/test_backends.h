
void InitLocalServers();
void DestroyLocalServers();

class LocalWorkerPool : public WorkerPool {
 public:
  LocalWorkerPool();
  virtual ~LocalWorkerPool();

  virtual Worker::StubInterface* GetWorker(const std::string& key) override {
    return worker_.get();
  }

 private:
  std::unique_ptr<Worker::Stub> worker_;
  std::unique_ptr<SolverService::Stub> solver_;
};
