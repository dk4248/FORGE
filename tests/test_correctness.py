"""
Correctness tests: verify that the fused kernel produces the same weight
updates as standard PyTorch (matmul + optimizer.step).

This is the ground truth check — if the fused kernel diverges from PyTorch's
AdamW/SGD by more than floating-point tolerance, something is wrong.

NOTE: On GPUs with TF32 tensor cores (Ampere+), tl.dot uses TF32 by default.
TF32 has 10-bit mantissa (vs fp32's 23-bit), so matmul results differ from
IEEE fp32 by up to ~0.5% on large reductions. Tolerances are set accordingly.
"""

import pytest
import torch

# Skip entire module if no CUDA GPU available
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required"
)

# TF32 matmul tolerance: tl.dot uses TF32 on Ampere+, which introduces
# ~1e-3 relative error per element in large reductions.
ATOL = 5e-3
RTOL = 5e-3


def _reference_sgd_step(grad_output, input, weight, lr, weight_decay):
    """Standard PyTorch: materialize grad, apply SGD."""
    grad_weight = grad_output.t() @ input
    weight_ref = weight.clone()
    weight_ref.mul_(1.0 - lr * weight_decay)
    weight_ref.add_(grad_weight.float(), alpha=-lr)
    return weight_ref


def _reference_adamw_step(grad_output, input, weight, m, v,
                          lr, beta1, beta2, eps, weight_decay, step):
    """Standard PyTorch: materialize grad, apply AdamW."""
    grad_weight = (grad_output.t() @ input).float()
    m_ref = m.clone()
    v_ref = v.clone()

    m_ref.mul_(beta1).add_(grad_weight, alpha=1 - beta1)
    v_ref.mul_(beta2).add_(grad_weight.square(), alpha=1 - beta2)

    bc1 = 1.0 - beta1 ** step
    bc2 = 1.0 - beta2 ** step
    m_hat = m_ref / bc1
    v_hat = v_ref / bc2

    weight_ref = weight.float().clone()
    weight_ref.mul_(1.0 - lr * weight_decay)
    weight_ref.add_(m_hat / (v_hat.sqrt() + eps), alpha=-lr)

    return weight_ref, m_ref, v_ref


class TestFusedGradSGD:

    @pytest.mark.parametrize("BT,V,H", [(128, 256, 128), (64, 512, 256), (256, 1024, 512)])
    def test_matches_reference(self, BT, V, H):
        from fused_grad_optimizer.kernel import fused_grad_sgd

        torch.manual_seed(42)
        device = "cuda"
        lr, wd = 0.01, 0.1

        grad_output = torch.randn(BT, V, device=device, dtype=torch.float32)
        input = torch.randn(BT, H, device=device, dtype=torch.float32)
        weight = torch.randn(V, H, device=device, dtype=torch.float32)

        weight_ref = _reference_sgd_step(grad_output, input, weight, lr, wd)

        weight_fused = weight.clone()
        fused_grad_sgd(grad_output, input, weight_fused, lr=lr, weight_decay=wd)

        torch.testing.assert_close(weight_fused, weight_ref, atol=ATOL, rtol=RTOL)


class TestFusedGradAdamW:

    @pytest.mark.parametrize("BT,V,H", [(128, 256, 128), (64, 512, 256), (256, 1024, 512)])
    def test_matches_reference(self, BT, V, H):
        from fused_grad_optimizer.kernel import fused_grad_adamw

        torch.manual_seed(42)
        device = "cuda"
        lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01
        step = 5

        grad_output = torch.randn(BT, V, device=device, dtype=torch.float32)
        input = torch.randn(BT, H, device=device, dtype=torch.float32)
        weight = torch.randn(V, H, device=device, dtype=torch.float32)
        m = torch.randn(V, H, device=device, dtype=torch.float32) * 0.01
        v = torch.randn(V, H, device=device, dtype=torch.float32).abs() * 0.01

        weight_ref, m_ref, v_ref = _reference_adamw_step(
            grad_output, input, weight, m, v, lr, beta1, beta2, eps, wd, step,
        )

        weight_fused = weight.clone()
        m_fused = m.clone()
        v_fused = v.clone()
        fused_grad_adamw(
            grad_output, input, weight_fused, m_fused, v_fused,
            lr=lr, beta1=beta1, beta2=beta2, eps=eps,
            weight_decay=wd, step=step,
        )

        torch.testing.assert_close(weight_fused, weight_ref.to(weight.dtype), atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(m_fused, m_ref, atol=ATOL, rtol=RTOL)
        torch.testing.assert_close(v_fused, v_ref, atol=ATOL, rtol=RTOL)

    def test_bf16_weight(self):
        """Verify kernel works with bf16 weights (common in LLM training)."""
        from fused_grad_optimizer.kernel import fused_grad_adamw

        torch.manual_seed(42)
        device = "cuda"
        BT, V, H = 128, 256, 128

        grad_output = torch.randn(BT, V, device=device, dtype=torch.bfloat16)
        input = torch.randn(BT, H, device=device, dtype=torch.bfloat16)
        weight = torch.randn(V, H, device=device, dtype=torch.bfloat16)
        m = torch.zeros(V, H, device=device, dtype=torch.float32)
        v = torch.zeros(V, H, device=device, dtype=torch.float32)

        # Should not crash; weight should change
        weight_before = weight.clone()
        fused_grad_adamw(grad_output, input, weight, m, v, step=1)
        assert not torch.equal(weight, weight_before), "Weight should have been updated"


# Quantized kernels have more error: int8 quantization of m/v adds ~1/127 per element
QATOL = 0.05
QRTOL = 0.05


class TestFusedGradAdamWInt8State:
    """Test the int8-quantized-state fused kernel against fp32 reference."""

    @pytest.mark.parametrize("BT,V,H", [(128, 256, 128), (64, 512, 256), (256, 1024, 512)])
    def test_matches_reference(self, BT, V, H):
        from fused_grad_optimizer.kernel import fused_grad_adamw_int8state

        torch.manual_seed(42)
        device = "cuda"
        lr, beta1, beta2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 0.01
        step = 5
        qblock = 64

        grad_output = torch.randn(BT, V, device=device, dtype=torch.float32)
        input = torch.randn(BT, H, device=device, dtype=torch.float32)
        weight = torch.randn(V, H, device=device, dtype=torch.float32)
        m = torch.randn(V, H, device=device, dtype=torch.float32) * 0.01
        v = torch.randn(V, H, device=device, dtype=torch.float32).abs() * 0.01

        # Reference: standard fp32 AdamW
        weight_ref, m_ref, v_ref = _reference_adamw_step(
            grad_output, input, weight, m, v, lr, beta1, beta2, eps, wd, step,
        )

        # Quantize m, v to int8 for the fused kernel
        scale_cols = H // qblock
        m_q = torch.zeros(V, H, dtype=torch.int8, device=device)
        v_q = torch.zeros(V, H, dtype=torch.int8, device=device)
        m_scale = torch.ones(V, scale_cols, dtype=torch.float32, device=device)
        v_scale = torch.ones(V, scale_cols, dtype=torch.float32, device=device)

        # Quantize the initial m, v into int8 format
        for b in range(scale_cols):
            h_start, h_end = b * qblock, (b + 1) * qblock
            m_block = m[:, h_start:h_end]
            v_block = v[:, h_start:h_end]
            m_s = m_block.abs().amax(dim=1) / 127.0 + 1e-12
            v_s = v_block.abs().amax(dim=1) / 127.0 + 1e-12
            m_scale[:, b] = m_s
            v_scale[:, b] = v_s
            m_q[:, h_start:h_end] = (m_block / m_s[:, None]).round().clamp(-127, 127).to(torch.int8)
            v_q[:, h_start:h_end] = (v_block / v_s[:, None]).round().clamp(-127, 127).to(torch.int8)

        weight_fused = weight.clone()
        fused_grad_adamw_int8state(
            grad_output, input, weight_fused, m_q, v_q, m_scale, v_scale,
            lr=lr, beta1=beta1, beta2=beta2, eps=eps,
            weight_decay=wd, step=step, qblock=qblock,
        )

        torch.testing.assert_close(weight_fused, weight_ref.to(weight.dtype),
                                   atol=QATOL, rtol=QRTOL)

    def test_bf16_weight(self):
        """Verify int8 state kernel works with bf16 weights."""
        from fused_grad_optimizer.kernel import fused_grad_adamw_int8state

        torch.manual_seed(42)
        device = "cuda"
        BT, V, H = 128, 256, 128
        qblock = 64

        grad_output = torch.randn(BT, V, device=device, dtype=torch.bfloat16)
        input = torch.randn(BT, H, device=device, dtype=torch.bfloat16)
        weight = torch.randn(V, H, device=device, dtype=torch.bfloat16)
        m_q = torch.zeros(V, H, dtype=torch.int8, device=device)
        v_q = torch.zeros(V, H, dtype=torch.int8, device=device)
        m_scale = torch.ones(V, H // qblock, dtype=torch.float32, device=device)
        v_scale = torch.ones(V, H // qblock, dtype=torch.float32, device=device)

        weight_before = weight.clone()
        fused_grad_adamw_int8state(
            grad_output, input, weight, m_q, v_q, m_scale, v_scale,
            step=1, qblock=qblock,
        )
        assert not torch.equal(weight, weight_before), "Weight should have been updated"
        # m_q should now have non-zero values
        assert m_q.any(), "m_q should have non-zero values after update"


class TestFusedLinearQuantized:

    def test_forward_backward_quantized(self):
        from fused_grad_optimizer.module import FusedLinear

        device = "cuda"
        torch.manual_seed(42)

        layer = FusedLinear(128, 64, optimizer_type="adamw", quantize_state=True).to(device)
        layer.train()
        layer.update_optimizer_config(lr=1e-3, step=1, weight_decay=0.01)

        x = torch.randn(8, 128, device=device, requires_grad=True)
        y = layer(x)
        loss = y.sum()

        weight_before = layer.weight.data.clone()
        loss.backward()

        assert not torch.equal(layer.weight.data, weight_before), \
            "FusedLinear(quantize_state=True) should update weight during backward"
        assert x.grad is not None, "grad_input should be computed"
        # Verify int8 state was allocated
        assert layer._state.m_q is not None, "m_q should be allocated"
        assert layer._state.m_q.dtype == torch.int8

    def test_memory_savings(self):
        """Int8 state should use less memory than bf16 state."""
        from fused_grad_optimizer.module import FusedLinear

        device = "cuda"
        V, H = 1024, 512

        layer_fp = FusedLinear(H, V, optimizer_type="adamw", quantize_state=False).to(device)
        layer_fp.train()
        layer_fp.update_optimizer_config(lr=1e-3, step=1)
        layer_fp._ensure_state()
        layer_fp._state.ensure_buffers()

        layer_q = FusedLinear(H, V, optimizer_type="adamw", quantize_state=True).to(device)
        layer_q.train()
        layer_q.update_optimizer_config(lr=1e-3, step=1)
        layer_q._ensure_state()
        layer_q._state.ensure_buffers()

        # fp state: m (V*H*2) + v (V*H*2) = 4*V*H bytes (bf16)
        fp_bytes = layer_fp._state.m.nbytes + layer_fp._state.v.nbytes
        # int8 state: m_q (V*H*1) + v_q (V*H*1) + m_scale + v_scale
        q_bytes = (layer_q._state.m_q.nbytes + layer_q._state.v_q.nbytes +
                   layer_q._state.m_scale.nbytes + layer_q._state.v_scale.nbytes)

        assert q_bytes < fp_bytes, (
            f"Quantized state ({q_bytes} bytes) should be smaller than "
            f"fp state ({fp_bytes} bytes)")


class TestFusedLinearModule:

    def test_forward_backward(self):
        from fused_grad_optimizer.module import FusedLinear

        device = "cuda"
        torch.manual_seed(42)

        layer = FusedLinear(128, 64, optimizer_type="adamw").to(device)
        layer.train()
        layer.update_optimizer_config(lr=1e-3, step=1, weight_decay=0.01)

        x = torch.randn(8, 128, device=device, requires_grad=True)
        y = layer(x)
        loss = y.sum()

        weight_before = layer.weight.data.clone()
        loss.backward()

        # Weight should have been updated in-place during backward
        assert not torch.equal(layer.weight.data, weight_before), \
            "FusedLinear should update weight during backward"

        # grad_input should flow back
        assert x.grad is not None, "grad_input should be computed"

    def test_eval_mode_no_update(self):
        from fused_grad_optimizer.module import FusedLinear

        device = "cuda"
        layer = FusedLinear(128, 64, optimizer_type="adamw").to(device)
        layer.eval()

        x = torch.randn(8, 128, device=device)
        y = layer(x)

        # Should produce output without errors in eval mode
        assert y.shape == (8, 64)

    def test_from_linear(self):
        from fused_grad_optimizer.module import FusedLinear

        device = "cuda"
        original = torch.nn.Linear(128, 64, bias=False).to(device)
        fused = FusedLinear.from_linear(original, optimizer_type="adamw")

        # Weights should be shared
        assert torch.equal(fused.weight.data, original.weight.data)


class TestFusedOptimizerManager:

    def test_excludes_fused_params(self):
        from fused_grad_optimizer.module import FusedLinear, FusedOptimizerManager

        device = "cuda"
        model = torch.nn.Sequential(
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            FusedLinear(64, 32, optimizer_type="adamw"),
        ).to(device)

        manager = FusedOptimizerManager(model)

        non_fused = manager.get_non_fused_params()
        fused_ids = manager._fused_param_ids

        # The FusedLinear weight should be excluded from non_fused
        assert id(model[2].weight) in fused_ids
        assert id(model[2].weight) not in {id(p) for p in non_fused}

        # The regular Linear params should be in non_fused
        assert id(model[0].weight) in {id(p) for p in non_fused}
