"""
Benchmark comparing original JAX autodiff sparse jacobian/hessian
vs symbolic sympy2jax approach.

Uses perfetto traces to analyze the computation graphs after compilation.
"""

import jax
import jax.numpy as jnp
import equinox as eqx
from time import time

from jax2sympy.problems import mpc
from jax2sympy.sparsify import sparse_jacobian, sparse_hessian, get_sparsity_pattern
from jax2sympy.sparsify_sym import sparse_jacobian_sym, sparse_hessian_sym


def benchmark_quadcopter_nav(N_batch=2000):
    """Benchmark sparse jacobian/hessian computation on quadcopter_nav MPC."""

    print(f"Loading quadcopter_nav MPC problem...")
    f, h, g, x, gt, aux = mpc.quadcopter_nav(N=3)

    # Create batch of inputs
    batch_x = jnp.stack([x] * N_batch)
    print(f"Batch shape: {batch_x.shape}")

    # ========================================
    # ORIGINAL APPROACH (JAX autodiff)
    # ========================================
    print("\n" + "="*60)
    print("ORIGINAL APPROACH (JAX autodiff vmap(grad))")
    print("="*60)

    # Create sparse functions
    jac_f_coo = get_sparsity_pattern(f, x, type='jacobian')
    jac_h_coo = get_sparsity_pattern(h, x, type='jacobian')
    jac_g_coo = get_sparsity_pattern(g, x, type='jacobian')

    hes_f_coo = get_sparsity_pattern(f, x, type='hessian')
    hes_h_coo = get_sparsity_pattern(h, x, type='hessian')

    jac_f_orig = sparse_jacobian(f, jac_f_coo, (f(x).size, x.size))
    jac_h_orig = sparse_jacobian(h, jac_h_coo, (h(x).size, x.size))
    jac_g_orig = sparse_jacobian(g, jac_g_coo, (g(x).size, x.size))

    hes_f_orig = sparse_hessian(f, hes_f_coo, (f(x).size, x.size, x.size))
    hes_h_orig = sparse_hessian(h, hes_h_coo, (h(x).size, x.size, x.size))

    # Vmap and JIT
    @eqx.filter_jit
    def compute_all_orig(batch_x):
        jac_f = jax.vmap(jac_f_orig)(batch_x)
        jac_h = jax.vmap(jac_h_orig)(batch_x)
        jac_g = jax.vmap(jac_g_orig)(batch_x)
        hes_f = jax.vmap(hes_f_orig)(batch_x)
        hes_h = jax.vmap(hes_h_orig)(batch_x)
        return jac_f, jac_h, jac_g, hes_f, hes_h

    # Warmup / compile
    print("Compiling original approach...")
    out_orig = compute_all_orig(batch_x)
    jax.block_until_ready(out_orig)
    print("Compilation done.")

    # Time it
    t1 = time()
    out_orig = compute_all_orig(batch_x)
    jax.block_until_ready(out_orig)
    t2 = time()
    print(f"Original approach time: {t2-t1:.4f}s for {N_batch} batches")
    print(f"Original approach per-batch: {(t2-t1)/N_batch*1000:.4f}ms")

    # Perfetto trace
    print("Creating perfetto trace for original approach...")
    with jax.profiler.trace("tmp/sparse_original", create_perfetto_trace=True):
        out_orig = compute_all_orig(batch_x)
        jax.block_until_ready(out_orig)
    print("Trace saved to tmp/sparse_original/")

    # ========================================
    # SYMBOLIC APPROACH (sympy2jax)
    # ========================================
    print("\n" + "="*60)
    print("SYMBOLIC APPROACH (sympy2jax)")
    print("="*60)

    # Create sparse functions
    jac_f_sym = sparse_jacobian_sym(f, x)
    jac_h_sym = sparse_jacobian_sym(h, x)
    jac_g_sym = sparse_jacobian_sym(g, x)

    hes_f_sym = sparse_hessian_sym(f, x)
    hes_h_sym = sparse_hessian_sym(h, x)

    # Vmap and JIT
    @eqx.filter_jit
    def compute_all_sym(batch_x):
        jac_f = jax.vmap(jac_f_sym)(batch_x)
        jac_h = jax.vmap(jac_h_sym)(batch_x)
        jac_g = jax.vmap(jac_g_sym)(batch_x)
        hes_f = jax.vmap(hes_f_sym)(batch_x)
        hes_h = jax.vmap(hes_h_sym)(batch_x)
        return jac_f, jac_h, jac_g, hes_f, hes_h

    # Warmup / compile
    print("Compiling symbolic approach...")
    out_sym = compute_all_sym(batch_x)
    jax.block_until_ready(out_sym)
    print("Compilation done.")

    # Time it
    t1 = time()
    out_sym = compute_all_sym(batch_x)
    jax.block_until_ready(out_sym)
    t2 = time()
    print(f"Symbolic approach time: {t2-t1:.4f}s for {N_batch} batches")
    print(f"Symbolic approach per-batch: {(t2-t1)/N_batch*1000:.4f}ms")

    # Perfetto trace
    print("Creating perfetto trace for symbolic approach...")
    with jax.profiler.trace("tmp/sparse_symbolic", create_perfetto_trace=True):
        out_sym = compute_all_sym(batch_x)
        jax.block_until_ready(out_sym)
    print("Trace saved to tmp/sparse_symbolic/")

    # ========================================
    # COMPARISON
    # ========================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    # Verify outputs match (compare sparse data directly, not dense)
    print("\nVerifying outputs match (comparing sparse data)...")
    for name, o, s in [
        ("jac_f", out_orig[0], out_sym[0]),
        ("jac_h", out_orig[1], out_sym[1]),
        ("jac_g", out_orig[2], out_sym[2]),
        ("hes_f", out_orig[3], out_sym[3]),
        ("hes_h", out_orig[4], out_sym[4]),
    ]:
        # Compare sparse data values directly (same sparsity pattern)
        max_diff = jnp.max(jnp.abs(o.data - s.data))
        print(f"  {name}: max diff = {max_diff:.2e}")


if __name__ == "__main__":
    benchmark_quadcopter_nav(N_batch=2000)
