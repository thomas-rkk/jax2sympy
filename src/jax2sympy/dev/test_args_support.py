"""
Test that sparsify_sym.py handles *args correctly by comparing
against the original sparsify.py implementation.
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO

# Original sparsify.py functions
from jax2sympy.sparsify import sparse_jacobian, sparse_hessian, get_sparsity_pattern

# New symbolic approach
from jax2sympy.sparsify_sym import sparse_jacobian_sym, sparse_hessian_sym


def test_jacobian_with_args():
    """Test sparse jacobian with additional *args."""
    print("=" * 60)
    print("Testing Jacobian with *args")
    print("=" * 60)

    # Function with scaling and offset args (like in jaxipm)
    def f_with_args(x, scale, offset):
        return jnp.array([
            (x[0] - offset[0])**2 * scale[0],
            (x[1] - offset[1])**2 * scale[1],
            x[0] * x[1] * scale[0] * scale[1]
        ])

    # Sample inputs
    x = jnp.array([1.0, 2.0])
    scale = jnp.array([0.5, 2.0])
    offset = jnp.array([0.1, 0.2])

    print(f"x = {x}")
    print(f"scale = {scale}")
    print(f"offset = {offset}")

    # Test original approach - curry function for sparsity detection
    print("\n--- Original (JAX autodiff) ---")
    f_curried = lambda x: f_with_args(x, scale, offset)
    coo_pattern = get_sparsity_pattern(f_curried, x, type='jacobian')
    jac_orig_fn = sparse_jacobian(f_with_args, coo_pattern, (f_with_args(x, scale, offset).size, x.size))
    jac_orig = jac_orig_fn(x, scale, offset)
    print(f"Original sparse jacobian data: {jac_orig.data}")
    print(f"Original sparse jacobian indices: {jac_orig.indices}")

    # Test symbolic approach
    print("\n--- Symbolic (sympy2jax) ---")
    jac_sym_fn = sparse_jacobian_sym(f_with_args, x, scale, offset)
    jac_sym = jac_sym_fn(x, scale, offset)
    print(f"Symbolic sparse jacobian data: {jac_sym.data}")
    print(f"Symbolic sparse jacobian indices: {jac_sym.indices}")

    # Reference: dense jacobian
    print("\n--- Reference (dense JAX jacobian) ---")
    jac_dense = jax.jacrev(f_with_args)(x, scale, offset)
    print(f"Dense jacobian:\n{jac_dense}")

    # Compare
    print("\n--- Comparison ---")
    # Convert to dense for comparison
    orig_dense = jac_orig.todense()
    sym_dense = jac_sym.todense()

    max_diff_orig = jnp.max(jnp.abs(orig_dense - jac_dense))
    max_diff_sym = jnp.max(jnp.abs(sym_dense - jac_dense))
    max_diff_between = jnp.max(jnp.abs(orig_dense - sym_dense))

    print(f"Max diff (original vs dense): {max_diff_orig:.2e}")
    print(f"Max diff (symbolic vs dense): {max_diff_sym:.2e}")
    print(f"Max diff (original vs symbolic): {max_diff_between:.2e}")

    assert max_diff_orig < 1e-5, f"Original jacobian differs from dense by {max_diff_orig}"
    assert max_diff_sym < 1e-5, f"Symbolic jacobian differs from dense by {max_diff_sym}"
    assert max_diff_between < 1e-5, f"Original and symbolic jacobians differ by {max_diff_between}"

    print("\nJacobian with *args test PASSED!")
    return True


def test_hessian_with_args():
    """Test sparse hessian with additional *args."""
    print("\n" + "=" * 60)
    print("Testing Hessian with *args")
    print("=" * 60)

    # Scalar function with scaling and offset args
    def f_with_args(x, scale, offset):
        return jnp.sum((x - offset)**2) * scale[0] + jnp.sin(x[0] * x[1]) * scale[1]

    # Sample inputs
    x = jnp.array([1.0, 2.0])
    scale = jnp.array([0.5, 2.0])
    offset = jnp.array([0.1, 0.2])

    print(f"x = {x}")
    print(f"scale = {scale}")
    print(f"offset = {offset}")

    # Test original approach - curry function for sparsity detection
    print("\n--- Original (JAX autodiff) ---")
    f_curried = lambda x: f_with_args(x, scale, offset)
    coo_pattern = get_sparsity_pattern(f_curried, x, type='hessian')
    hess_orig_fn = sparse_hessian(f_with_args, coo_pattern, (f_with_args(x, scale, offset).size, x.size, x.size))
    hess_orig = hess_orig_fn(x, scale, offset)
    print(f"Original sparse hessian data: {hess_orig.data}")
    print(f"Original sparse hessian indices: {hess_orig.indices}")

    # Test symbolic approach
    print("\n--- Symbolic (sympy2jax) ---")
    hess_sym_fn = sparse_hessian_sym(f_with_args, x, scale, offset)
    hess_sym = hess_sym_fn(x, scale, offset)
    print(f"Symbolic sparse hessian data: {hess_sym.data}")
    print(f"Symbolic sparse hessian indices: {hess_sym.indices}")

    # Reference: dense hessian
    print("\n--- Reference (dense JAX hessian) ---")
    hess_dense = jax.hessian(f_with_args)(x, scale, offset)
    print(f"Dense hessian:\n{hess_dense}")

    # Compare
    print("\n--- Comparison ---")
    # Convert to dense for comparison
    orig_dense = hess_orig.todense()
    sym_dense = hess_sym.todense()

    max_diff_orig = jnp.max(jnp.abs(orig_dense - hess_dense))
    max_diff_sym = jnp.max(jnp.abs(sym_dense - hess_dense))
    max_diff_between = jnp.max(jnp.abs(orig_dense - sym_dense))

    print(f"Max diff (original vs dense): {max_diff_orig:.2e}")
    print(f"Max diff (symbolic vs dense): {max_diff_sym:.2e}")
    print(f"Max diff (original vs symbolic): {max_diff_between:.2e}")

    assert max_diff_orig < 1e-5, f"Original hessian differs from dense by {max_diff_orig}"
    assert max_diff_sym < 1e-5, f"Symbolic hessian differs from dense by {max_diff_sym}"
    assert max_diff_between < 1e-5, f"Original and symbolic hessians differ by {max_diff_between}"

    print("\nHessian with *args test PASSED!")
    return True


def test_varying_args():
    """Test that the jacobian/hessian change correctly when args change."""
    print("\n" + "=" * 60)
    print("Testing that derivatives vary correctly with *args")
    print("=" * 60)

    def f_with_args(x, scale):
        return jnp.array([x[0]**2 * scale[0], x[1]**2 * scale[1]])

    x = jnp.array([2.0, 3.0])
    scale1 = jnp.array([1.0, 1.0])
    scale2 = jnp.array([2.0, 0.5])

    # Create symbolic jacobian function
    jac_sym_fn = sparse_jacobian_sym(f_with_args, x, scale1)

    # Evaluate at different scales
    jac_scale1 = jac_sym_fn(x, scale1)
    jac_scale2 = jac_sym_fn(x, scale2)

    print(f"Jacobian at scale={scale1}: {jac_scale1.todense()}")
    print(f"Jacobian at scale={scale2}: {jac_scale2.todense()}")

    # Expected: d/dx0 (x0^2 * s0) = 2*x0*s0
    # At x=[2,3], scale=[1,1]: jac = [[4, 0], [0, 6]]
    # At x=[2,3], scale=[2,0.5]: jac = [[8, 0], [0, 3]]

    expected1 = jnp.array([[4.0, 0.0], [0.0, 6.0]])
    expected2 = jnp.array([[8.0, 0.0], [0.0, 3.0]])

    assert jnp.allclose(jac_scale1.todense(), expected1), "Scale1 jacobian incorrect"
    assert jnp.allclose(jac_scale2.todense(), expected2), "Scale2 jacobian incorrect"

    print("\nVarying *args test PASSED!")
    return True


def test_vmap_with_args():
    """Test that vmap works correctly with *args."""
    print("\n" + "=" * 60)
    print("Testing vmap with *args")
    print("=" * 60)

    def f_with_args(x, scale):
        return jnp.array([x[0]**2 * scale, x[1]**2 * scale])

    x = jnp.array([2.0, 3.0])
    scale = jnp.array(1.5)

    # Create batch
    batch_x = jnp.stack([x, x * 2, x * 3])  # 3 different x values
    batch_scale = jnp.array([1.0, 2.0, 0.5])  # 3 different scales

    # Create symbolic jacobian
    jac_sym_fn = sparse_jacobian_sym(f_with_args, x, scale)

    # Vmap over both x and scale
    vmapped_jac = jax.vmap(jac_sym_fn)(batch_x, batch_scale)

    print(f"Batch x shape: {batch_x.shape}")
    print(f"Batch scale shape: {batch_scale.shape}")
    print(f"Vmapped jacobian data shape: {vmapped_jac.data.shape}")
    print(f"Vmapped jacobian indices shape: {vmapped_jac.indices.shape}")

    # Verify each batch element
    for i in range(3):
        single_jac = jac_sym_fn(batch_x[i], batch_scale[i])
        assert jnp.allclose(vmapped_jac.data[i], single_jac.data), f"Batch {i} data mismatch"

    print("\nvmap with *args test PASSED!")
    return True


def test_coo_pattern_parameter():
    """Test that providing coo_pattern filters to specified indices."""
    print("\n" + "=" * 60)
    print("Testing coo_pattern parameter")
    print("=" * 60)

    # Function with more outputs than we want to compute
    def f_large(x):
        return jnp.array([
            x[0]**2,      # output 0: depends on x[0]
            x[1]**2,      # output 1: depends on x[1]
            x[0] * x[1],  # output 2: depends on x[0], x[1]
            x[0] + x[1],  # output 3: depends on x[0], x[1]
        ])

    x = jnp.array([2.0, 3.0])

    # Full jacobian would be 4x2
    # Let's only compute a subset of entries
    subset_coo = np.array([
        [0, 0],  # df[0]/dx[0] = 2*x[0] = 4
        [2, 1],  # df[2]/dx[1] = x[0] = 2
    ])

    # Create symbolic jacobian with filtered COO
    jac_sym_fn = sparse_jacobian_sym(f_large, x, coo_pattern=subset_coo, out_shape=(4, 2))
    jac_result = jac_sym_fn(x)

    print(f"Requested COO pattern:\n{subset_coo}")
    print(f"Result data: {jac_result.data}")
    print(f"Result indices:\n{jac_result.indices}")

    # Verify values
    expected_data = jnp.array([4.0, 2.0])  # 2*x[0]=4, x[0]=2
    assert jnp.allclose(jac_result.data, expected_data), f"Expected {expected_data}, got {jac_result.data}"
    assert jnp.array_equal(jac_result.indices, subset_coo), "COO indices mismatch"

    print("\ncoo_pattern parameter test PASSED!")
    return True


def test_out_shape_parameter():
    """Test that out_shape parameter correctly sets the BCOO shape."""
    print("\n" + "=" * 60)
    print("Testing out_shape parameter")
    print("=" * 60)

    def f_simple(x):
        return jnp.array([x[0]**2, x[1]**2])

    x = jnp.array([2.0, 3.0])

    # Request larger output shape than natural
    large_shape = (5, 4)
    jac_fn = sparse_jacobian_sym(f_simple, x, out_shape=large_shape)
    result = jac_fn(x)

    print(f"Requested out_shape: {large_shape}")
    print(f"Result shape: {result.shape}")

    assert result.shape == large_shape, f"Expected shape {large_shape}, got {result.shape}"

    # Verify the dense result is zero-padded correctly
    dense = result.todense()
    print(f"Dense result:\n{dense}")

    # Original 2x2 jacobian is [[4, 0], [0, 6]]
    # Should be embedded in top-left of 5x4
    assert dense[0, 0] == 4.0
    assert dense[1, 1] == 6.0
    assert jnp.sum(dense) == 10.0  # Only non-zero entries

    print("\nout_shape parameter test PASSED!")
    return True


if __name__ == "__main__":
    test_jacobian_with_args()
    test_hessian_with_args()
    test_varying_args()
    test_vmap_with_args()
    test_coo_pattern_parameter()
    test_out_shape_parameter()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
