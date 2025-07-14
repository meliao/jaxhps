from jaxhps.quadrature import (
    chebyshev_points,
    first_kind_chebyshev_points,
    chebyshev_weights,
    affine_transform,
)

import jax.numpy as jnp
import logging


class Test_chebyshev_weights:
    def test_0(self) -> None:
        """Checks to make sure things run correctly."""
        p = 6
        bounds = jnp.array([-1, 1])
        w = chebyshev_weights(p, bounds)
        assert w.shape == (p,)
        assert not jnp.any(jnp.isnan(w))
        assert not jnp.any(jnp.isinf(w))

        p = 7
        bounds = jnp.array([-1, 1])
        w = chebyshev_weights(p, bounds)
        assert w.shape == (p,)
        assert not jnp.any(jnp.isnan(w))
        assert not jnp.any(jnp.isinf(w))

    def test_1(self) -> None:
        """Checks that a constant function has the expected integral."""

        p = 4
        bounds = jnp.array([-1, 1])
        w = chebyshev_weights(p, bounds)
        assert w.shape == (p,)
        print("test_1: w = ", w)
        f = jnp.ones(p)
        integral = w @ f
        print("test_1: integral = ", integral)
        expected = 2
        assert jnp.allclose(integral, expected)

        p = 5
        bounds = jnp.array([-1, 1])
        w = chebyshev_weights(p, bounds)
        assert w.shape == (p,)
        print("test_1: w = ", w)
        f = jnp.ones(p)
        integral = w @ f
        print("test_1: integral = ", integral)
        expected = 2
        assert jnp.allclose(integral, expected)

    def test_2(self) -> None:
        """Checks a nonzero polynomial over [0,1]"""
        p = 6
        bounds = jnp.array([0, 1])
        w = chebyshev_weights(p, bounds)
        print("test_2: w = ", w)
        assert w.shape == (p,)

        def f(x):
            return x**3

        pts = affine_transform(chebyshev_points(p), bounds)
        f_evals = f(pts)
        integral = w @ f_evals
        expected = 1 / 4

        print("test_2: integral = ", integral)
        print("test_2: expected = ", expected)

        assert jnp.allclose(integral, expected)

        p = 7
        bounds = jnp.array([0, 1])
        w = chebyshev_weights(p, bounds)
        print("test_2: w = ", w)
        assert w.shape == (p,)

        def f(x):
            return x**3

        pts = affine_transform(chebyshev_points(p), bounds)
        f_evals = f(pts)
        integral = w @ f_evals
        expected = 1 / 4

        print("test_2: integral = ", integral)
        print("test_2: expected = ", expected)

        assert jnp.allclose(integral, expected)

    def test_3(self) -> None:
        """Observed this test case in the wild."""
        p = 16
        bounds = jnp.array([0.5, 0.625])
        w = chebyshev_weights(p, bounds)
        # Make sure not nan or inf
        print("test_3: w = ", w)
        assert not jnp.any(jnp.isnan(w))
        assert not jnp.any(jnp.isinf(w))


class Test_chebyshev_points:
    def test_0(self) -> None:
        n = 2
        p = chebyshev_points(n)
        expected_p = jnp.array([-1, 1.0])
        assert jnp.allclose(p, expected_p)

    def test_1(self) -> None:
        n = 3
        p = chebyshev_points(n)
        expected_p = jnp.array([-1, 0.0, 1.0])
        print(p)
        assert jnp.allclose(p, expected_p)


class Test_affine_transform:
    def test_0(self) -> None:
        """Check that the affine transform works as expected."""
        n = 10
        x = chebyshev_points(n)[0]
        ab = jnp.array([0, 1])
        y = affine_transform(x, ab)
        expected_y = x * 0.5 + 0.5
        assert jnp.allclose(y, expected_y)


class Test_first_kind_chebyshev_points:
    def test_0(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        n = 2
        p = first_kind_chebyshev_points(n)
        logging.info("p = %s", p)
        expected_p = jnp.array([-1 / jnp.sqrt(2), 1.0 / jnp.sqrt(2)])
        logging.info("expected_p = %s", expected_p)
        assert jnp.allclose(p, expected_p)
