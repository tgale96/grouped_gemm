import torch
import grouped_gemm as gg

def benchmark(name, func, x, w, batch_sizes, iterations=50, trans_a=False, trans_b=True):
    print(f"Profiling {name}...")

    # warmup
    for _ in range(10):
        out = func(x, w, batch_sizes, trans_b=trans_b)
        out.sum().backward()
        x.grad = None
        w.grad = None
    torch.cuda.synchronize()

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        for _ in range(iterations):
            out = func(x, w, batch_sizes, trans_b=trans_b)
            out.sum().backward()
            x.grad = None
            w.grad = None

    torch.cuda.synchronize()

    total_ms = prof.key_averages().total_average().device_time_total / 1000
    avg_ms = total_ms / iterations

    # FLOPs + TFLOPs
    fwd_flops = grouped_gemm_flops(x, w, batch_sizes, trans_a=trans_a, trans_b=trans_b)
    avg_s = avg_ms * 1e-3
    fwd_tflops = fwd_flops / avg_s / 1e12

    # Your timing includes backward. If backward does two GEMMs
    # (dA and dB), total GEMM FLOPs ≈ 3x forward.
    fwd_bwd_tflops_est = 3.0 * fwd_tflops

    print(f"  -> Total GPU time: {total_ms:.2f} ms")
    print(f"  -> Time per step:  {avg_ms:.3f} ms")
    print(f"  -> Forward TFLOPs: {fwd_tflops:.2f}")
    print(f"  -> Fwd+Bwd TFLOPs (est): {fwd_bwd_tflops_est:.2f}")

    prof.export_chrome_trace(f'traces/{name}_trace.json')
    return avg_ms, fwd_tflops

def grouped_gemm_flops(x, w, batch_sizes, trans_a=False, trans_b=True):
    """
    Returns forward FLOPs for grouped GEMM.
    x: (tokens, K) when trans_a=False, else packed (sum k_i, m)
    w:
      - trans_b=True: (E, N, K)
      - trans_b=False: (E, K, N)
    batch_sizes: (E,) giving m_i (if trans_a=False) or k_i (if trans_a=True)
    """
    bs = batch_sizes.detach().to("cpu", non_blocking=True).tolist()
    E = len(bs)

    if not trans_a:
        # fixed K, variable m_i
        K = x.shape[1]
        N = w.shape[1] if trans_b else w.shape[2]
        M_total = sum(bs)
        flops = 2.0 * M_total * K * N
    else:
        # variable K_i, fixed m and n
        # x is logically A_i: (K_i, m) because A is transposed
        m = x.shape[1]
        n = w.shape[1]  # b is (tokens, n) in this mode
        flops = 0.0
        for k_i in bs:
            flops += 2.0 * m * n * k_i

    return flops

def make_batch_sizes(M, E, mode="uniform", device="cpu"):
    if mode == "uniform":
        m = M // E
        sizes = [m] * E
        sizes[0] += M - m * E
        return torch.tensor(sizes, dtype=torch.long, device=device)

    elif mode == "mild_skew":
        alpha = torch.full((E,), 2.0)
        probs = torch.distributions.Dirichlet(alpha).sample()
        sizes = torch.floor(probs * M).long()
        diff = M - int(sizes.sum().item())
        for _ in range(abs(diff)):
            idx = torch.randint(0, E, ())
            sizes[idx] += 1 if diff > 0 else -1
        return sizes.to(device)

    elif mode == "extreme_skew":
        hot_E = max(1, E // 8)
        cold_E = E - hot_E

        hot_tokens = int(0.8 * M)
        cold_tokens = M - hot_tokens

        hot_base = hot_tokens // hot_E
        cold_base = cold_tokens // max(cold_E, 1)

        sizes = []
        for i in range(E):
            if i < hot_E:
                sizes.append(hot_base)
            else:
                sizes.append(cold_base)

        sizes = torch.tensor(sizes, dtype=torch.long)
        diff = M - int(sizes.sum().item())
        for _ in range(abs(diff)):
            idx = torch.randint(0, E, ())
            sizes[idx] += 1 if diff > 0 else -1
        return sizes.to(device)

if __name__ == '__main__':

    model_config_dict = {
        "Qwen/Qwen3-30B-A3B":{
            "num_experts_per_tok": 8,
            "hidden_size": 2048,
            "num_experts": 128,
            "moe_intermediate_size": 768,
        },
    }
    
    model_name = "Qwen/Qwen3-30B-A3B"
    model_config = model_config_dict[model_name]
    seqlen = 8192
    test_case = "up_proj"

    M = seqlen * model_config["num_experts_per_tok"]
    E = model_config["num_experts"]

    if test_case == "up_proj":
        K = model_config["hidden_size"]
        N = 2 * model_config["moe_intermediate_size"]
    elif test_case == "down_proj":
        K = model_config["moe_intermediate_size"]
        N = model_config["hidden_size"]
    
    print(f"Config: {test_case} | Tokens: {M} | Experts: {E} | Shape: K={K}, N={N}")

    torch.manual_seed(42)
    
    x = torch.rand(M, K, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    w = torch.rand(E, N, K, dtype=torch.bfloat16, device='cuda', requires_grad=True)
    
    modes = ["uniform", "mild_skew", "extreme_skew"]

    for mode in modes:
        print("\n" + "=" * 30)
        print(f"Workload mode: {mode}")
        batch_sizes = make_batch_sizes(M, E, mode=mode, device='cpu')
        print(f"Statistics: min={int(batch_sizes.min())}, "
            f"max={int(batch_sizes.max())}, "
            f"mean={batch_sizes.float().mean().item():.1f}, "
            f"var={batch_sizes.float().var(unbiased=False).item():.1f}"
        )
        print("=" * 30)

        x.grad = None
        w.grad = None

        time_base, tflops_base = benchmark(
            "cuBLAS (Base)", 
            gg.ops.gmm_base, 
            x, w, batch_sizes
        )
        
        print("-" * 30)

        time_cublas, tflops_cublas = benchmark(
            "cuBLAS (Batched)", 
            gg.ops.gmm_cuBLAS, 
            x, w, batch_sizes
        )

        print("-" * 30)

        time_CUTLASS, tflops_CUTLASS = benchmark(
            "CUTLASS sm80", 
            gg.ops.gmm_CUTLASS_sm80, 
            x, w, batch_sizes
        )

        print("-" * 30)

        time_CUTLASS_sm90_cooperative, tflops_CUTLASS_sm90_cooperative = benchmark(
            "CUTLASS sm90 cooperative", 
            gg.ops.gmm_CUTLASS_sm90_cooperative, 
            x, w, batch_sizes
        )

        print("-" * 30)

        time_CUTLASS_sm90_pingpong, tflops_CUTLASS_sm90_pingpong = benchmark(
            "CUTLASS sm90 pingpong", 
            gg.ops.gmm_CUTLASS_sm90_pingpong, 
            x, w, batch_sizes
        )

        print("=" * 30)
        # print(f"Speedup: {time_base / time_cublas:.2f}x")
        # print("=" * 30)