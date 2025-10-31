#pragma once

#include "module.hpp"
#include "../ops.hpp"

namespace infinicore::nn {

class Linear : public Module {
public:
    Linear(size_t in_features, size_t out_features, bool bias = true, const Device &device = Device());

    // Forward pass: output = input @ weight.T + bias
    Tensor forward(Tensor &input) const;

    // Forward pass with residual connection (InfiniLM-style)
    // output = input @ weight.T + bias + residual
    Tensor forward(Tensor &input, Tensor &residual) const;

    // Module information
    size_t in_features() const { return in_features_; }
    size_t out_features() const { return out_features_; }
    bool has_bias() const { return has_bias_; }

    // String representation
    std::string extra_repr() const;

    // Accessors for parameters
    Tensor weight() const { return weight_; }
    Tensor bias() const { return bias_; }

protected:
    // Parameters
    Parameter weight_;
    Parameter bias_;

private:
    // Helper method for common forward computation
    Tensor compute_linear(Tensor &input) const;

    size_t in_features_;
    size_t out_features_;
    bool has_bias_;
};

} // namespace infinicore::nn
