#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <vector>
#include <random>
#include <cmath>
#include <memory>
#include <string>
#include <stdexcept>
#include <sstream>
#include <iostream>

// --- Dominio ---
// g++ -shared -fPIC -o dqn_module.so hdqnt.cpp -I/home/hadoop/Documentos/cpp_programs/pybind/py-dqn-torch/myenv/lib/python3.12/site-packages/pybind11/include -I/home/hadoop/libtorch/include -I/home/hadoop/libtorch/include/torch/csrc/api/include -I/usr/include/python3.12 -L/home/hadoop/libtorch/lib -ltorch -ltorch_cpu -lc10 -std=c++17 -Wl,-rpath,/home/hadoop/libtorch/lib

struct State {
    std::vector<double> features;
    explicit State(std::vector<double> features) : features(std::move(features)) {}
};

struct Action {
    int id;
    explicit Action(int id) : id(id) {}
};

class DQNModel : public torch::nn::Module {
public:
    DQNModel(size_t input_size, size_t output_size, double learning_rate)
        : input_size_(input_size), output_size_(output_size), optimizer_(nullptr) {
        network_ = register_module("network", torch::nn::Sequential(
            torch::nn::Linear(input_size, 128),
            torch::nn::ReLU(),
            torch::nn::Linear(128, 64),
            torch::nn::ReLU(),
            torch::nn::Linear(64, output_size)
        ));
        for (auto& module : network_->modules()) {
            if (auto* linear = module->as<torch::nn::Linear>()) {
                torch::nn::init::xavier_uniform_(linear->weight);
                torch::nn::init::zeros_(linear->bias);
            }
        }
        optimizer_ = std::make_unique<torch::optim::Adam>(
            network_->parameters(), torch::optim::AdamOptions(learning_rate));
    }

    std::vector<double> predict(const State& state) {
        if (state.features.size() != input_size_) {
            throw std::runtime_error("Tamaño de entrada inválido");
        }
        torch::Tensor input = torch::tensor(state.features, torch::kFloat32).reshape({1, -1});
        torch::NoGradGuard no_grad;
        torch::Tensor output = network_->forward(input);
        return std::vector<double>(output.data_ptr<float>(), output.data_ptr<float>() + output.size(1));
    }

    double update(const State& state, const Action& action, double target) {
        torch::Tensor input = torch::tensor(state.features, torch::kFloat32).reshape({1, -1});
        torch::Tensor output = network_->forward(input);
        torch::Tensor q_value = output[0][action.id];
        torch::Tensor target_tensor = torch::tensor({target}, torch::kFloat32).reshape({});
        torch::Tensor loss = torch::mse_loss(q_value, target_tensor);
        optimizer_->zero_grad();
        loss.backward();
        // Gradient clipping
        for (auto& param : network_->parameters()) {
            if (param.grad().defined()) {
                param.grad().clamp_(-1.0, 1.0);
            }
        }
        optimizer_->step();
        return loss.item<double>();
    }

    size_t input_size() const { return input_size_; }
    size_t output_size() const { return output_size_; }

private:
    size_t input_size_;
    size_t output_size_;
    torch::nn::Sequential network_;
    std::unique_ptr<torch::optim::Adam> optimizer_;
};

// --- Puertos ---

class ModelRepository {
public:
    virtual void save(std::shared_ptr<DQNModel> model) = 0;
    virtual std::shared_ptr<DQNModel> load() = 0;
    virtual ~ModelRepository() = default;
};

class DQNTrainer {
public:
    virtual void train(const std::vector<State>& states, const std::vector<Action>& actions,
                      const std::vector<double>& rewards, const std::vector<bool>& dones,
                      double gamma, double learning_rate) = 0;
    virtual ~DQNTrainer() = default;
};

class DQNPredictor {
public:
    virtual Action predict(const State& state, double epsilon) = 0;
    virtual ~DQNPredictor() = default;
};

// --- Replay Buffer ---

class ReplayBuffer {
public:
    ReplayBuffer(size_t max_size = 10000) : max_size_(max_size) {}

    void add(const State& state, const Action& action, double reward, const State& next_state, bool done) {
        buffer_.emplace_back(state, action, reward, next_state, done);
        if (buffer_.size() > max_size_) buffer_.erase(buffer_.begin());
    }

    std::vector<std::tuple<State, Action, double, State, bool>> sample(size_t batch_size) {
        std::vector<std::tuple<State, Action, double, State, bool>> batch;
        if (buffer_.size() < batch_size) return batch;

        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<> dis(0, buffer_.size() - 1);
        for (size_t i = 0; i < batch_size; ++i) {
            batch.push_back(buffer_[dis(gen)]);
        }
        return batch;
    }

private:
    std::vector<std::tuple<State, Action, double, State, bool>> buffer_;
    size_t max_size_;
};

// --- Adaptadores ---

class InMemoryModelRepository : public ModelRepository {
public:
    void save(std::shared_ptr<DQNModel> model) override {
        saved_model_ = model;
    }

    std::shared_ptr<DQNModel> load() override {
        return saved_model_;
    }

private:
    std::shared_ptr<DQNModel> saved_model_;
};

// --- Capa de aplicación ---

class DQNService : public DQNTrainer, public DQNPredictor {
public:
    DQNService(std::unique_ptr<ModelRepository> repository, size_t input_size, size_t output_size, double learning_rate)
        : repository_(std::move(repository)), buffer_(10000), update_counter_(0), target_update_freq_(10) {
        auto loaded_model = repository_->load();
        if (loaded_model) {
            model_ = loaded_model;
            target_model_ = std::make_shared<DQNModel>(input_size, output_size, learning_rate);
            std::stringstream buffer;
            torch::save(model_, buffer);
            buffer.seekg(0);
            torch::load(target_model_, buffer);
        } else {
            model_ = std::make_shared<DQNModel>(input_size, output_size, learning_rate);
            target_model_ = std::make_shared<DQNModel>(input_size, output_size, learning_rate);
            std::stringstream buffer;
            torch::save(model_, buffer);
            buffer.seekg(0);
            torch::load(target_model_, buffer);
        }
    }

    DQNService(size_t input_size, size_t output_size, double learning_rate)
        : DQNService(std::make_unique<InMemoryModelRepository>(), input_size, output_size, learning_rate) {}

    void train(const std::vector<State>& states, const std::vector<Action>& actions,
               const std::vector<double>& rewards, const std::vector<bool>& dones,
               double gamma, double learning_rate) override {
        if (states.size() != actions.size() + 1 || actions.size() != rewards.size() || actions.size() != dones.size()) {
            throw std::runtime_error("Tamaños de entrada inconsistentes");
        }

        for (size_t t = 0; t < states.size() - 1; ++t) {
            buffer_.add(states[t], actions[t], rewards[t], states[t + 1], dones[t]);
        }

        for (int i = 0; i < 5; ++i) {
            auto batch = buffer_.sample(64);
            double total_loss = 0.0;
            for (const auto& [state, action, reward, next_state, done] : batch) {
                double target = reward;
                if (!done) {
                    auto next_q_values = target_model_->predict(next_state);
                    target += gamma * *std::max_element(next_q_values.begin(), next_q_values.end());
                }
                total_loss += model_->update(state, action, target);
            }
            std::cout << "Pérdida promedio (batch " << i + 1 << "): " << total_loss / batch.size() << std::endl;
            update_counter_++;
            if (update_counter_ % target_update_freq_ == 0) {
                std::stringstream buffer;
                torch::save(model_, buffer);
                buffer.seekg(0);
                torch::load(target_model_, buffer);
            }
        }
        repository_->save(model_);
    }

    Action predict(const State& state, double epsilon) override {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dis(0.0, 1.0);
        if (dis(gen) < epsilon) {
            std::uniform_int_distribution<> action_dis(0, model_->output_size() - 1);
            return Action(action_dis(gen));
        }
        auto q_values = model_->predict(state);
        int best_action = std::distance(q_values.begin(),
                                       std::max_element(q_values.begin(), q_values.end()));
        return Action(best_action);
    }

private:
    std::unique_ptr<ModelRepository> repository_;
    std::shared_ptr<DQNModel> model_;
    std::shared_ptr<DQNModel> target_model_;
    ReplayBuffer buffer_;
    size_t update_counter_;
    size_t target_update_freq_;
};

// --- Pybind11 ---

namespace py = pybind11;

PYBIND11_MODULE(dqn_module, m) {
    py::class_<State>(m, "State")
        .def(py::init<std::vector<double>>())
        .def_readwrite("features", &State::features);

    py::class_<Action>(m, "Action")
        .def(py::init<int>())
        .def_readwrite("id", &Action::id);

    py::class_<DQNService>(m, "DQNService")
        .def(py::init<size_t, size_t, double>())
        .def("train", &DQNService::train)
        .def("predict", &DQNService::predict);
}