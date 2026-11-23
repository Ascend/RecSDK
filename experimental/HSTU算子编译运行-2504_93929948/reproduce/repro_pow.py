import torch
import torch_npu

torch.npu.set_device(0)
torch.npu.config.allow_internal_format = False

x = torch.full((112, 2, 64), 0.25, device="npu",
               dtype=torch.float32, requires_grad=True)

print("START: square().mean().backward()", flush=True)
x.square().mean().backward()
torch.npu.synchronize()

expected = torch.full_like(x.cpu(), 0.5 / x.numel())
torch.testing.assert_close(x.grad.cpu(), expected)
print("PURE PYTORCH BACKWARD PASSED", flush=True)
