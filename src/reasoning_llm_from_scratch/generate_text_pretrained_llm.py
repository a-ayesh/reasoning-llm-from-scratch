from .qwen3 import KVCache
import warnings
import torch

def get_device(enable_tensor_cores=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using NVIDIA CUDA GPU")
        
        if enable_tensor_cores:
            major, minor = map(int, torch.__version__.split(".")[:2])
            # PyTorch 2.9 and 2.10 still read the legacy TF32 setting in torch.compile.
            # See https://github.com/pytorch/pytorch/issues/166387
            # and https://github.com/rasbt/reasoning-from-scratch/issues/256
            if (major, minor) >= (2, 11):
                torch.backends.cuda.matmul.fp32_precision = "tf32"
                torch.backends.cudnn.conv.fp32_precision = "tf32"
            else:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True

    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using Apple Silicon GPU (MPS)")

    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device

@torch.inference_mode()
def generate_text_basic_stream(
    model,
    token_ids,
    max_new_tokens, 
    eos_token_id=None
):
    model.eval()

    for _ in range(max_new_tokens):
        out = model(token_ids)[:, -1]
        next_token = torch.argmax(out, dim=-1, keepdim=True)

        # Stop if we encounter an end-of-sequence token
        if (eos_token_id is not None
        and torch.all(next_token == eos_token_id)):
            break

        yield next_token  # Yield each token as it's generated
        
        token_ids = torch.cat([token_ids, next_token], dim=1)

@torch.inference_mode()
def generate_text_basic_stream_cache(
    model,
    token_ids,
    max_new_tokens,
    eos_token_id=None
):
    model.eval()
    cache = KVCache(n_layers=model.cfg["n_layers"])  # New
    model.reset_kv_cache()                           # New

    out = model(token_ids, cache=cache)[:, -1]
    for _ in range(max_new_tokens):
        next_token = torch.argmax(out, dim=-1, keepdim=True)

        if (eos_token_id is not None
                and torch.all(next_token == eos_token_id)):
            break

        yield next_token
        out = model(next_token, cache=cache)[:, -1]

def generate_stats(
    output_token_ids,
    start_time,
    end_time
    ):
    total_time = end_time - start_time
    print(f"\n\nTime: {total_time:.2f} sec")
    print(f"{int(output_token_ids.numel() / total_time)} tokens/sec")
    
    for name, backend in (("CUDA", getattr(torch, "cuda", None)),):
        if backend is not None and backend.is_available():

            device_type = output_token_ids.device.type
            if device_type != name.lower():
                warnings.warn(
                    f"{name} is available but tensors are on "
                    f"{device_type}. Memory stats may be 0."
                )

            # Synchronize if supported (important for async backends)
            if hasattr(backend, "synchronize"):
                backend.synchronize()

            max_mem_bytes = backend.max_memory_allocated()
            max_mem_gb = max_mem_bytes / (1024 ** 3)
            print(f"Max {name} memory allocated: {max_mem_gb:.2f} GB")

            backend.reset_peak_memory_stats()

    if torch.backends.mps.is_available():
        device_type = output_token_ids.device.type
        if device_type != "mps":
            warnings.warn(
                f"MPS is available but tensors are on {device_type}. "
                f"Memory stats may be 0."
            )

        torch.mps.synchronize()
        current_gb = torch.mps.current_allocated_memory() / (1024 ** 3)
        driver_gb = torch.mps.driver_allocated_memory() / (1024 ** 3)
        print(f"MPS current allocated (tensors): {current_gb:.2f} GB")
        print(f"MPS driver allocated (incl. cache): {driver_gb:.2f} GB")
