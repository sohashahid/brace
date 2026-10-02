import numpy as np
import pytest

from brace_f1.baselines import (
    HAZARD_CLASS_ORDER,
    RegularizedSideHazard,
    TunedDeterministicResidualDynamics,
    forecast_constant_turn_rate,
    forecast_constant_velocity,
)
from brace_f1.forecast import forecast_ctra_particles
from brace_f1.geometry import CircuitCorridor
from brace_f1.io import DataValidationError
from brace_f1.residual import RESIDUAL_TARGET_NAMES


def _square_ring() -> CircuitCorridor:
    return CircuitCorridor.from_arrays(
        centerline=np.asarray([[-7.5, -7.5], [7.5, -7.5], [7.5, 7.5], [-7.5, 7.5]]),
        boundary_a=np.asarray([[-10.0, -10.0], [10.0, -10.0], [10.0, 10.0], [-10.0, 10.0]]),
        boundary_b=np.asarray([[-5.0, -5.0], [5.0, -5.0], [5.0, 5.0], [-5.0, 5.0]]),
    )


def test_constant_velocity_and_constant_turn_rate_are_distinct_registered_baselines() -> None:
    # Straight travel exits the square outer loop; positive turn rate bends along it.
    state = np.asarray([[7.5, 0.0, 10.0, 0.0, np.pi / 2.0, 1.0, 99.0, -99.0]])
    cv = forecast_constant_velocity(state, _square_ring(), n_particles=8, seed=1, dt_s=0.05)
    ctrv = forecast_constant_turn_rate(state, _square_ring(), n_particles=8, seed=1, dt_s=0.05)
    assert cv.no_exit_probability[0, -1] == pytest.approx(0.0)
    assert ctrv.no_exit_probability[0, -1] == pytest.approx(1.0)
    np.testing.assert_allclose(
        cv.no_exit_probability + cv.outcome_probability.sum(axis=(2, 3)), 1.0
    )
    np.testing.assert_allclose(
        ctrv.no_exit_probability + ctrv.outcome_probability.sum(axis=(2, 3)), 1.0
    )


def test_tuned_deterministic_residual_selects_alpha_on_validation_only() -> None:
    X_fit = np.asarray([[0.0], [1.0]])
    Y_fit = np.zeros((2, 5))
    Y_fit[:, 0] = [0.0, 1.0]
    X_validation = np.asarray([[2.0], [3.0]])
    Y_validation = np.zeros((2, 5))
    Y_validation[:, 0] = [2.0, 3.0]
    model = TunedDeterministicResidualDynamics.fit(
        X_fit,
        Y_fit,
        X_validation=X_validation,
        Y_validation=Y_validation,
        feature_names=("x",),
        target_names=RESIDUAL_TARGET_NAMES,
        alpha_grid=(0.0, 1.0, 100.0),
    )
    assert model.selected_alpha == pytest.approx(0.0)
    assert tuple(model.validation_mse_by_alpha) == (0.0, 1.0, 100.0)
    np.testing.assert_allclose(model.predict(X_validation), Y_validation, atol=1e-12)
    assert len(model.content_hash) == 64
    with pytest.raises(TypeError):
        model.validation_mse_by_alpha[0.0] = 999.0


def test_tuned_deterministic_residual_is_forecast_compatible() -> None:
    X = np.asarray([[0.0], [1.0], [2.0], [3.0]])
    Y = np.zeros((4, len(RESIDUAL_TARGET_NAMES)))
    model = TunedDeterministicResidualDynamics.fit(
        X,
        Y,
        X_validation=X,
        Y_validation=Y,
        feature_names=("context",),
        target_names=RESIDUAL_TARGET_NAMES,
        alpha_grid=(0.0, 1.0),
    )

    result = forecast_ctra_particles(
        np.asarray([[7.5, 0.0, 8.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
        _square_ring(),
        residual_model=model,
        residual_features=np.asarray([[1.0]]),
        horizons_s=(0.5,),
        n_particles=2,
        seed=1,
    )

    assert result.no_exit_probability.shape == (1, 1)


def test_regularized_side_hazard_returns_competing_probability_mass() -> None:
    X = np.asarray([[-3.0], [-2.0], [-1.0], [0.0], [1.0], [2.0], [3.0]])
    y = np.asarray(["left", "left", "left", "no_exit", "right", "right", "right"])
    model = RegularizedSideHazard.fit(X, y, feature_names=("signed_state",), regularization=0.1)
    assert HAZARD_CLASS_ORDER == ("no_exit", "left", "right")
    step = model.predict_step_probabilities(np.asarray([[-3.0], [3.0]]))
    np.testing.assert_allclose(step.sum(axis=1), 1.0, atol=1e-12)
    assert step[0, 1] > step[0, 2]
    assert step[1, 2] > step[1, 1]
    cumulative = model.predict_horizon_probabilities(
        np.asarray([[-3.0], [3.0]]), horizons_s=(0.25, 0.5, 1.0, 1.5), dt_s=0.05
    )
    assert cumulative.shape == (2, 4, 3)
    np.testing.assert_allclose(cumulative.sum(axis=2), 1.0, atol=1e-12)
    assert np.all(np.diff(cumulative[:, :, 0], axis=1) <= 1e-12)
    assert (
        model.content_hash
        == RegularizedSideHazard.fit(
            X, y, feature_names=("signed_state",), regularization=0.1
        ).content_hash
    )
    assert not model.coefficients.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        model.coefficients[0, 0] = 999.0


def test_hazard_and_tuning_reject_unregistered_inputs() -> None:
    with pytest.raises(DataValidationError, match="labels"):
        RegularizedSideHazard.fit(
            np.asarray([[0.0], [1.0]]),
            np.asarray(["left", "crash"]),
            feature_names=("x",),
        )
    with pytest.raises(DataValidationError, match="alpha_grid"):
        TunedDeterministicResidualDynamics.fit(
            np.asarray([[0.0], [1.0]]),
            np.zeros((2, 5)),
            X_validation=np.asarray([[2.0]]),
            Y_validation=np.zeros((1, 5)),
            feature_names=("x",),
            target_names=RESIDUAL_TARGET_NAMES,
            alpha_grid=(),
        )
