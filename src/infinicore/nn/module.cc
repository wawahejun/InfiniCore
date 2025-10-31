#include "infinicore/nn/module.hpp"

namespace infinicore::nn {
const std::unordered_map<std::string, Parameter> &Module::state_dict() const {
    static std::unordered_map<std::string, Parameter> result;
    result.clear();

    collect_all_parameters(result, "");

    return result;
}

void Module::load_state_dict(const std::unordered_map<std::string, Tensor> &_state_dict) {
    // Collect all parameters from this module and its submodules with their full hierarchical names
    std::unordered_map<std::string, Parameter> all_params;
    collect_all_parameters(all_params, "");

    // For each parameter in this module hierarchy, load from the state dict
    for (auto &[param_full_name, param] : all_params) {
        // Look up the corresponding tensor in the input state dict using the full name
        auto it = _state_dict.find(param_full_name);
        if (it != _state_dict.end()) {
            param->copy_from(it->second);
        }
    }
}

void Module::load_parameter(const std::string &name, const Tensor &param) {
    parameters_[name]->copy_from(param);
}

void Module::load_parameter_from_blob(const std::string &name, const void *data) {
    auto param = parameters_[name];
    param.load_blob(data);
}

Tensor Module::register_parameter(const std::string &name, Parameter param) {
    parameters_[name] = param;
    return param;
}

void Module::collect_all_parameters(std::unordered_map<std::string, Parameter> &all_params, const std::string &prefix) const {
    // Add direct parameters with the given prefix
    for (const auto &[param_name, param] : parameters_) {
        std::string full_name = prefix.empty() ? param_name : prefix + "." + param_name;
        all_params[full_name] = param;
    }

    // Recursively collect parameters from submodules with extended prefix
    for (const auto &[sub_name, submodule] : submodules_) {
        std::string sub_prefix = prefix.empty() ? sub_name : prefix + "." + sub_name;
        submodule->collect_all_parameters(all_params, sub_prefix);
    }
}

} // namespace infinicore::nn
