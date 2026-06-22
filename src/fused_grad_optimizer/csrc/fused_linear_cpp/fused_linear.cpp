// C++ autograd Function for FusedLinear that eliminates the per-call Python
// overhead of torch.autograd.Function.apply().
//
// Forward: at::linear (single cuBLAS addmm).
// Backward: at::matmul for grad_input; calls back into Python ONCE per linear
// to dispatch the fused grad+optimizer kernel (which itself is in C++/CUDA via
// the cutlass_evt_b200 or kernel_fp8_state PyBind modules, so the callback is
// thin).
//
// Compared to the pure-Python FusedLinearFunction, this saves the Python
// interpreter overhead of building the autograd node + bookkeeping per linear.
// For Llama-3.1-8B (~225 linears per fwd) the estimated savings are 10-20 ms.

#include <torch/extension.h>
#include <torch/autograd.h>
#include <pybind11/pybind11.h>
#include <pybind11/functional.h>

namespace py = pybind11;
using torch::autograd::AutogradContext;
using torch::autograd::Variable;
using torch::autograd::variable_list;

// Backward callback: takes (grad_output_2d, input_2d, weight, py_state, py_config)
// and applies the fused grad+optimizer in-place to `weight` (and to optimizer
// state's m/v). Returns nothing. This is essentially `_apply_fused` from
// fused_grad_optimizer/autograd.py but called from C++ with one fewer Python
// dispatch layer.
using BwdCallback = std::function<void(
    torch::Tensor, torch::Tensor, torch::Tensor,
    py::object, py::object, bool)>;

// Global registry: Python registers the callback once at import time.
static BwdCallback g_bwd_callback;

void set_bwd_callback(BwdCallback cb) {
    g_bwd_callback = std::move(cb);
}


class FusedLinearAutograd : public torch::autograd::Function<FusedLinearAutograd> {
public:
    static torch::Tensor forward(
        AutogradContext* ctx,
        torch::Tensor input,
        torch::Tensor weight,
        c10::optional<torch::Tensor> bias,
        py::object state,
        py::object config,
        bool is_accumulating)
    {
        // Single cuBLAS addmm — same kernel nn.Linear uses.
        torch::Tensor output;
        if (bias.has_value()) {
            output = at::linear(input, weight, *bias);
        } else {
            output = at::linear(input, weight);
        }

        // Save for backward.
        if (bias.has_value()) {
            ctx->save_for_backward({input, weight, *bias});
            ctx->saved_data["has_bias"] = true;
        } else {
            ctx->save_for_backward({input, weight});
            ctx->saved_data["has_bias"] = false;
        }
        ctx->saved_data["is_accumulating"] = is_accumulating;

        // Stash Python objects on ctx (avoid GIL release while holding refs).
        {
            py::gil_scoped_acquire gil;
            ctx->saved_data["state_ptr"]  = reinterpret_cast<int64_t>(state.ptr());
            ctx->saved_data["config_ptr"] = reinterpret_cast<int64_t>(config.ptr());
            // We don't take ownership — the FusedLinear module keeps these alive.
        }

        return output;
    }

    static variable_list backward(AutogradContext* ctx, variable_list grad_outputs) {
        auto saved = ctx->get_saved_variables();
        auto input  = saved[0];
        auto weight = saved[1];
        bool has_bias = ctx->saved_data["has_bias"].toBool();
        torch::Tensor bias_saved = has_bias ? saved[2] : torch::Tensor();

        bool is_accumulating = ctx->saved_data["is_accumulating"].toBool();

        auto grad_output = grad_outputs[0];

        // Compute grad_input against the PRE-update weight (in-place update of
        // weight happens INSIDE the bwd callback below; this matters for the
        // chain rule — see comment in fused_grad_optimizer/autograd.py:48-52).
        // Reshape to 2D for matmul, then back to grad_output's full shape but
        // with weight's input-dim as the last dim.
        auto out_last_dim = grad_output.size(-1);
        auto grad_output_2d = grad_output.reshape({-1, out_last_dim});
        auto input_2d       = input.reshape({-1, input.size(-1)});

        auto grad_input_2d = at::matmul(grad_output_2d, weight);
        auto grad_input_shape = grad_output.sizes().vec();
        grad_input_shape.back() = weight.size(1);
        auto grad_input = grad_input_2d.reshape(grad_input_shape);

        // Make grad_output/input contiguous for the fused kernel.
        auto go = grad_output_2d.is_contiguous() ? grad_output_2d : grad_output_2d.contiguous();
        auto inp = input_2d.is_contiguous() ? input_2d : input_2d.contiguous();

        // Call Python callback to dispatch the fused grad+optimizer.
        {
            py::gil_scoped_acquire gil;
            py::object state  = py::reinterpret_borrow<py::object>(
                reinterpret_cast<PyObject*>(ctx->saved_data["state_ptr"].toInt()));
            py::object config = py::reinterpret_borrow<py::object>(
                reinterpret_cast<PyObject*>(ctx->saved_data["config_ptr"].toInt()));
            TORCH_CHECK(g_bwd_callback != nullptr,
                        "FusedLinearAutograd bwd callback not registered");
            g_bwd_callback(go, inp, weight, state, config, is_accumulating);
        }

        // grad_bias
        torch::Tensor grad_bias;
        if (has_bias) {
            // sum over all batch dims
            grad_bias = grad_output.reshape({-1, out_last_dim}).sum(0);
        }

        // Return one grad per forward input.
        // Inputs were: (input, weight, bias, state, config, is_accumulating)
        return {grad_input, torch::Tensor(), grad_bias,
                torch::Tensor(), torch::Tensor(), torch::Tensor()};
    }
};


torch::Tensor fused_linear_apply(
    torch::Tensor input,
    torch::Tensor weight,
    c10::optional<torch::Tensor> bias,
    py::object state,
    py::object config,
    bool is_accumulating)
{
    return FusedLinearAutograd::apply(input, weight, bias, state, config, is_accumulating);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_linear_apply", &fused_linear_apply,
          "Forward of FusedLinear in C++; backward dispatches to a registered Python callback. "
          "Eliminates the Python overhead of torch.autograd.Function.apply().");
    m.def("set_bwd_callback", &set_bwd_callback,
          "Register the Python callback that dispatches the fused grad+optimizer kernel. "
          "Signature: callback(grad_output_2d, input_2d, weight, state, config, is_accumulating).");
}
