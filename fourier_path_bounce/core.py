"""Reusable JAX-Fourier path optimization.

The public path convention is always false vacuum at ``t=0`` and true vacuum
at ``t=1``.  The reduced action and adaptive mode scan match the arithmetic in
the repository's published D=3 and D=4 benchmark scripts.
"""

from __future__ import annotations

import hashlib
import math
import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Mapping, Sequence

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from scipy.optimize import minimize


Array = np.ndarray
Potential = Callable[[Any], Any]


class FourierPathError(RuntimeError):
    """Base error for public Fourier-path operations."""


class InputValidationError(FourierPathError, ValueError):
    """Raised when endpoint, setting, or path data are inconsistent."""


class PotentialEvaluationError(FourierPathError):
    """Raised when a potential does not return finite scalar values."""


@dataclass(frozen=True)
class OptimizerSettings:
    method: str = "L-BFGS-B"
    maxiter: int = 800
    ftol: float = 1.0e-10
    gtol: float = 1.0e-7
    maxls: int = 50
    coefficient_bound_factor: float = 0.5
    coefficient_bound: float | None = None

    def validate(self) -> None:
        if self.method != "L-BFGS-B":
            raise InputValidationError("Only the validated L-BFGS-B optimizer is supported")
        if self.maxiter < 1 or self.maxls < 1:
            raise InputValidationError("maxiter and maxls must be positive")
        if self.ftol <= 0.0 or self.gtol <= 0.0:
            raise InputValidationError("ftol and gtol must be positive")
        if self.coefficient_bound is not None and self.coefficient_bound <= 0.0:
            raise InputValidationError("coefficient_bound must be positive")
        if self.coefficient_bound_factor <= 0.0:
            raise InputValidationError("coefficient_bound_factor must be positive")


@dataclass(frozen=True)
class ModeSelectionSettings:
    modes: tuple[int, ...] = (1, 2, 3, 4, 5)
    relative_tolerance: float = 1.0e-3
    patience: int = 3
    zero_start: bool = True
    warm_previous: bool = True
    warm_best: bool = False
    random_starts: int = 0
    random_scale: float = 0.03
    seed: int = 20260508

    def validate(self, *, external_start_available: bool = False) -> None:
        """Validate the mode-selection settings.

        ``external_start_available`` records that the caller will supply an
        additional start point (``optimize_fourier_path(initial_coefficients=...)``).
        User-supplied coefficients are a complete start strategy on their own, so
        when one is present the built-in start flags may all be disabled.
        """
        if not self.modes or any(int(m) != m or m < 1 for m in self.modes):
            raise InputValidationError("modes must be a nonempty sequence of positive integers")
        if tuple(sorted(set(self.modes))) != self.modes:
            raise InputValidationError("modes must be strictly increasing and unique")
        if self.relative_tolerance < 0.0 or self.patience < 1:
            raise InputValidationError("relative_tolerance must be nonnegative and patience positive")
        if self.random_starts < 0 or self.random_scale < 0.0:
            raise InputValidationError("random_starts and random_scale must be nonnegative")
        if not (
            self.zero_start
            or self.random_starts
            or self.warm_previous
            or self.warm_best
            or external_start_available
        ):
            raise InputValidationError(
                "at least one optimizer start strategy is required: enable zero_start, "
                "warm_previous, warm_best, or random_starts, or pass explicit "
                "initial_coefficients to optimize_fourier_path"
            )


@dataclass(frozen=True)
class FourierPathSettings:
    dimension: int = 4
    n_grid: int = 260
    profile: str = "cubic_smoothstep_raw_t"
    optimizer: OptimizerSettings = field(default_factory=OptimizerSettings)
    mode_selection: ModeSelectionSettings = field(default_factory=ModeSelectionSettings)

    def validate(self, *, external_start_available: bool = False) -> None:
        if self.dimension not in (3, 4):
            raise InputValidationError("dimension must be 3 or 4 for the validated action formulas")
        if self.n_grid < 3:
            raise InputValidationError("n_grid must be at least 3")
        if self.profile != "cubic_smoothstep_raw_t":
            raise InputValidationError(
                "the reusable production interface currently supports only the published "
                "cubic_smoothstep_raw_t profile"
            )
        self.optimizer.validate()
        self.mode_selection.validate(external_start_available=external_start_available)


@dataclass(frozen=True)
class ModeResult:
    n_modes: int
    action: float
    start_label: str
    optimizer_success: bool
    message: str
    nit: int
    nfev: int
    njev: int
    gradient_norm: float
    relative_change_from_previous: float | None
    global_best_action: float
    global_best_n_modes: int
    no_improve_count: int
    compile_time_s: float
    solve_time_s: float
    start_actions: Mapping[str, float]


@dataclass
class FourierPathResult:
    """Structured result of the adaptive Fourier optimization.

    ``path_points`` has shape ``(n_grid, n_fields)`` and is oriented from the
    false vacuum to the true vacuum. ``coefficients`` has shape
    ``(n_fields, selected_modes)``.
    """

    false_vacuum: Array
    true_vacuum: Array
    coefficients: Array
    selected_modes: int
    parameter: Array
    path_points: Array
    action_proxy: float
    optimizer_success: bool
    optimizer_message: str
    adaptive_converged: bool
    stop_reason: str
    history: tuple[ModeResult, ...]
    settings: FourierPathSettings
    potential_false: float
    potential_true: float
    compile_time_s: float
    solve_time_s: float
    total_time_s: float
    gradient_norm: float
    coefficient_norm: float
    coefficient_bound: float
    coefficient_bound_saturation_count: int
    invalid_interval_count: int
    min_v_minus_vt: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_fields(self) -> int:
        return int(self.false_vacuum.size)

    @property
    def orientation(self) -> str:
        return "false_to_true"

    @property
    def path_sha256(self) -> str:
        return hashlib.sha256(np.asarray(self.path_points, dtype="<f8").tobytes()).hexdigest()

    def sample_at(self, parameter_values: Sequence[float]) -> Array:
        """Evaluate the optimized curve at explicit raw-t locations."""
        values = np.asarray(parameter_values, dtype=float)
        if values.ndim != 1 or values.size < 1 or not np.all(np.isfinite(values)):
            raise InputValidationError("parameter_values must be a finite one-dimensional array")
        if np.any(values < 0.0) or np.any(values > 1.0) or np.any(np.diff(values) <= 0.0):
            raise InputValidationError("parameter_values must be strictly increasing in [0, 1]")
        return reconstruct_path(
            self.false_vacuum, self.true_vacuum, self.coefficients, values
        )

    def sample(
        self,
        n_points: int,
        *,
        sampling: str = "parameter",
        include_endpoints: bool = True,
    ) -> Array:
        """Sample the optimized curve without rerunning the optimizer."""
        if not isinstance(n_points, int) or n_points < (2 if include_endpoints else 1):
            raise InputValidationError("n_points is too small for the requested endpoint policy")
        total = n_points if include_endpoints else n_points + 2
        query = np.linspace(0.0, 1.0, total)
        if sampling == "parameter":
            path = reconstruct_path(
                self.false_vacuum, self.true_vacuum, self.coefficients, query
            )
        elif sampling == "arclength":
            dense_t = np.linspace(0.0, 1.0, max(8193, 64 * total + 1))
            dense_path = reconstruct_path(
                self.false_vacuum, self.true_vacuum, self.coefficients, dense_t
            )
            segment = np.linalg.norm(np.diff(dense_path, axis=0), axis=1)
            cumulative = np.concatenate(([0.0], np.cumsum(segment)))
            if not np.isfinite(cumulative[-1]) or cumulative[-1] <= 0.0:
                raise InputValidationError("cannot arc-length sample a degenerate path")
            normalized = cumulative / cumulative[-1]
            parameter_query = np.interp(query, normalized, dense_t)
            path = reconstruct_path(
                self.false_vacuum, self.true_vacuum, self.coefficients, parameter_query
            )
        else:
            raise InputValidationError("sampling must be 'parameter' or 'arclength'")
        path[0] = self.false_vacuum
        path[-1] = self.true_vacuum
        return path if include_endpoints else path[1:-1]

    def interior_points(self, k: int, *, sampling: str = "parameter") -> Array:
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            raise InputValidationError("K must be an integer >= 1")
        return self.sample(k + 2, sampling=sampling, include_endpoints=True)[1:-1]

    def summary(self) -> dict[str, Any]:
        return {
            "n_fields": self.n_fields,
            "orientation": self.orientation,
            "selected_modes": self.selected_modes,
            "path_shape": list(self.path_points.shape),
            "coefficient_shape": list(self.coefficients.shape),
            "action_proxy": self.action_proxy,
            "optimizer_success": self.optimizer_success,
            "adaptive_converged": self.adaptive_converged,
            "stop_reason": self.stop_reason,
            "compile_time_s": self.compile_time_s,
            "solve_time_s": self.solve_time_s,
            "total_time_s": self.total_time_s,
            "gradient_norm": self.gradient_norm,
            "coefficient_norm": self.coefficient_norm,
            "invalid_interval_count": self.invalid_interval_count,
            "min_v_minus_vt": self.min_v_minus_vt,
            "path_sha256": self.path_sha256,
            "settings": asdict(self.settings),
            "metadata": self.metadata,
        }


def _validate_endpoints(false_vacuum: Sequence[float], true_vacuum: Sequence[float]) -> tuple[Array, Array]:
    false = np.asarray(false_vacuum, dtype=float)
    true = np.asarray(true_vacuum, dtype=float)
    if false.ndim != 1 or true.ndim != 1 or false.size == 0:
        raise InputValidationError("vacua must be nonempty one-dimensional coordinate arrays")
    if false.shape != true.shape:
        raise InputValidationError(
            f"vacuum dimensions differ: false {false.shape}, true {true.shape}"
        )
    if not np.all(np.isfinite(false)) or not np.all(np.isfinite(true)):
        raise InputValidationError("vacuum coordinates must be finite")
    if np.array_equal(false, true):
        raise InputValidationError("false and true vacua must be distinct")
    return false.copy(), true.copy()


def _potential_scalar(potential: Potential, point: Array, label: str) -> float:
    try:
        value = np.asarray(potential(jnp.asarray(point)), dtype=float)
    except Exception as exc:
        raise PotentialEvaluationError(f"potential failed at {label}: {exc}") from exc
    if value.shape != ():
        raise PotentialEvaluationError(
            f"potential must return one scalar for one point; got shape {value.shape} at {label}"
        )
    result = float(value)
    if not np.isfinite(result):
        raise PotentialEvaluationError(f"potential returned a non-finite value at {label}")
    return result


def _validate_gradient(gradient: Potential | None, endpoints: tuple[Array, Array]) -> None:
    if gradient is None:
        return
    for label, point in zip(("false vacuum", "true vacuum"), endpoints):
        try:
            value = np.asarray(gradient(np.asarray(point)), dtype=float)
        except Exception as exc:
            raise PotentialEvaluationError(f"potential gradient failed at {label}: {exc}") from exc
        if value.shape != point.shape or not np.all(np.isfinite(value)):
            raise PotentialEvaluationError(
                f"potential gradient at {label} must have finite shape {point.shape}; got {value.shape}"
            )


def fourier_basis(parameter: Sequence[float], n_modes: int) -> Array:
    t = np.asarray(parameter, dtype=float)
    if t.ndim != 1 or n_modes < 1:
        raise InputValidationError("parameter must be one-dimensional and n_modes positive")
    return np.asarray([np.sin((k + 1) * np.pi * t) for k in range(n_modes)])


def reconstruct_path(
    false_vacuum: Sequence[float],
    true_vacuum: Sequence[float],
    coefficients: Any,
    parameter: Sequence[float],
) -> Array:
    false, true = _validate_endpoints(false_vacuum, true_vacuum)
    coeffs = np.asarray(coefficients, dtype=float)
    t = np.asarray(parameter, dtype=float)
    if coeffs.ndim != 2 or coeffs.shape[0] != false.size or coeffs.shape[1] < 1:
        raise InputValidationError(
            f"coefficients must have shape (n_fields, n_modes); got {coeffs.shape}"
        )
    if t.ndim != 1 or t.size < 1 or not np.all(np.isfinite(t)):
        raise InputValidationError("parameter must be a finite one-dimensional array")
    if np.any(t < 0.0) or np.any(t > 1.0):
        raise InputValidationError("path parameter values must lie in [0, 1]")
    basis = fourier_basis(t, coeffs.shape[1])
    path = false[None, :] + t[:, None] * (true - false)[None, :] + basis.T @ coeffs.T
    endpoint_zero = np.isclose(t, 0.0, rtol=0.0, atol=0.0)
    endpoint_one = np.isclose(t, 1.0, rtol=0.0, atol=0.0)
    path[endpoint_zero] = false
    path[endpoint_one] = true
    return path


def action_prefactor(dimension: int) -> float:
    if dimension not in (3, 4):
        raise InputValidationError("dimension must be 3 or 4")
    omega = 2.0 * np.pi ** (0.5 * dimension) / math.gamma(0.5 * dimension)
    return ((dimension - 1.0) ** (dimension - 1.0)) * omega / dimension


def _expand_coefficients(coefficients: Array, n_fields: int, new_modes: int) -> Array:
    old = np.asarray(coefficients, dtype=float).reshape(n_fields, -1)
    expanded = np.zeros((n_fields, new_modes), dtype=float)
    copied = min(old.shape[1], new_modes)
    expanded[:, :copied] = old[:, :copied]
    return expanded.ravel()


def _build_action(
    potential: Potential,
    false: Array,
    true: Array,
    potential_false: float,
    potential_true: float,
    settings: FourierPathSettings,
    n_modes: int,
) -> tuple[Callable[[Array], tuple[float, Array]], float]:
    t_np = np.linspace(0.0, 1.0, settings.n_grid)
    t = jnp.asarray(t_np)
    basis = jnp.asarray(fourier_basis(t_np, n_modes))
    false_jax = jnp.asarray(false)
    true_jax = jnp.asarray(true)
    vf = jnp.asarray(potential_false)
    vt_end = jnp.asarray(potential_true - potential_false)
    prefactor = jnp.asarray(action_prefactor(settings.dimension))
    dimension = float(settings.dimension)

    def point_potential(point: Any) -> Any:
        value = jnp.asarray(potential(point))
        return jnp.reshape(value, ())

    values = jax.vmap(point_potential)

    def action(flat_coefficients: Any) -> Any:
        coefficients = flat_coefficients.reshape((false.size, n_modes))
        straight = false_jax[None, :] + t[:, None] * (true_jax - false_jax)[None, :]
        path = straight + basis.T @ coefficients.T
        # Work in the benchmark's false-vacuum-zero energy convention.
        v_path = values(path) - vf
        vt = t * t * (3.0 - 2.0 * t) * vt_end
        delta_vt = vt[1:] - vt[:-1]
        minus_delta_vt = -delta_vt
        ds = jnp.linalg.norm(path[1:] - path[:-1], axis=1)
        vdiff = v_path[:-1] + v_path[1:] - vt[:-1] - vt[1:]

        if settings.dimension == 3:
            # This regulator and invalid-interval penalty are inherited exactly
            # from the published D=3 benchmark implementation.
            safe_vdiff = jnp.maximum(vdiff, 1.0e-300)
            safe_minus_delta_vt = jnp.maximum(minus_delta_vt, 1.0e-300)
            terms = (
                safe_vdiff ** (0.5 * dimension)
                * ds**dimension
                / safe_minus_delta_vt ** (dimension - 1.0)
            )
            bad = jnp.any(
                (vdiff <= 0.0) | (minus_delta_vt <= 0.0) | (~jnp.isfinite(terms))
            )
            total = prefactor * jnp.sum(terms)
            return jnp.where(bad, total + 1.0e50, total)

        denominator = -(delta_vt**3)
        terms = (vdiff**2) * (ds**4) / denominator
        return 0.5 * 27.0 * (jnp.pi**2) * jnp.sum(terms)

    value_gradient = jax.jit(jax.value_and_grad(action))
    compile_start = time.perf_counter()
    value0, gradient0 = value_gradient(jnp.zeros(false.size * n_modes))
    value0.block_until_ready()
    gradient0.block_until_ready()
    compile_time = time.perf_counter() - compile_start

    def scipy_function(x: Array) -> tuple[float, Array]:
        value, gradient = value_gradient(jnp.asarray(x))
        value.block_until_ready()
        gradient.block_until_ready()
        return float(value), np.asarray(gradient, dtype=float)

    return scipy_function, compile_time


def _path_diagnostics(
    potential: Potential,
    path: Array,
    potential_false: float,
    potential_true: float,
    dimension: int,
) -> tuple[int, float]:
    try:
        values = np.asarray(
            [_potential_scalar(potential, point, f"path point {i}") for i, point in enumerate(path)]
        ) - potential_false
    except PotentialEvaluationError:
        return path.shape[0] - 1, float("nan")
    t = np.linspace(0.0, 1.0, path.shape[0])
    vt = (potential_true - potential_false) * t * t * (3.0 - 2.0 * t)
    vdiff = values[:-1] + values[1:] - vt[:-1] - vt[1:]
    minus_delta_vt = -(vt[1:] - vt[:-1])
    invalid = (vdiff <= 0.0) | (minus_delta_vt <= 0.0)
    if dimension == 4:
        ds = np.linalg.norm(np.diff(path, axis=0), axis=1)
        terms = (vdiff**2) * ds**4 / (-(vt[1:] - vt[:-1]) ** 3)
        invalid |= ~np.isfinite(terms)
    return int(np.count_nonzero(invalid)), float(np.min(values - vt))


def optimize_fourier_path(
    potential: Potential,
    false_vacuum: Sequence[float],
    true_vacuum: Sequence[float],
    *,
    settings: FourierPathSettings | None = None,
    potential_gradient: Potential | None = None,
    initial_coefficients: Any | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> FourierPathResult:
    """Optimize an endpoint-preserving Fourier path for a user potential.

    The potential must accept one JAX array of shape ``(n_fields,)`` and return
    a scalar. It must be differentiable by JAX. The optional gradient is
    validated for downstream solver use but is not used by the Fourier action.
    """
    cfg = settings or FourierPathSettings()
    # Explicit initial coefficients count as a start strategy in their own right,
    # so the built-in start flags may all be disabled when one is supplied. The
    # shape and finiteness of the array are checked below, once n_fields is known.
    cfg.validate(external_start_available=initial_coefficients is not None)
    false, true = _validate_endpoints(false_vacuum, true_vacuum)
    _validate_gradient(potential_gradient, (false, true))
    potential_false = _potential_scalar(potential, false, "false vacuum")
    potential_true = _potential_scalar(potential, true, "true vacuum")
    if not potential_true < potential_false:
        raise InputValidationError(
            "the true-vacuum potential must be lower than the false-vacuum potential"
        )

    selection = cfg.mode_selection
    optimizer = cfg.optimizer
    endpoint_distance = float(np.linalg.norm(true - false))
    bound = (
        float(optimizer.coefficient_bound)
        if optimizer.coefficient_bound is not None
        else optimizer.coefficient_bound_factor * endpoint_distance
    )
    user_start: Array | None = None
    if initial_coefficients is not None:
        user_start = np.asarray(initial_coefficients, dtype=float)
        if user_start.ndim != 2 or user_start.shape[0] != false.size:
            raise InputValidationError(
                "initial_coefficients must have shape (n_fields, n_initial_modes)"
            )
        if not np.all(np.isfinite(user_start)):
            raise InputValidationError("initial_coefficients must be finite")

    best_action = float("inf")
    best_coefficients: Array | None = None
    best_modes: int | None = None
    best_run: dict[str, Any] | None = None
    previous_coefficients: Array | None = None
    previous_modes: int | None = None
    previous_mode_action: float | None = None
    no_improve = 0
    adaptive_converged = False
    stop_reason = "max_modes_reached"
    history: list[ModeResult] = []
    total_compile = 0.0
    total_solve = 0.0
    started = time.perf_counter()

    for mode_index, n_modes in enumerate(selection.modes, start=1):
        try:
            objective, compile_time = _build_action(
                potential, false, true, potential_false, potential_true, cfg, n_modes
            )
        except Exception as exc:
            raise PotentialEvaluationError(
                "potential must return a scalar and be differentiable by JAX"
            ) from exc
        total_compile += compile_time
        starts: list[tuple[str, Array]] = []
        if selection.zero_start:
            starts.append(("zero", np.zeros(false.size * n_modes)))
        if user_start is not None:
            starts.append(
                ("user", _expand_coefficients(user_start, false.size, n_modes))
            )
        if selection.warm_previous and previous_coefficients is not None:
            starts.append(
                (
                    "warm_previous",
                    _expand_coefficients(previous_coefficients, false.size, n_modes),
                )
            )
        if selection.warm_best and best_coefficients is not None:
            starts.append(
                ("warm_best", _expand_coefficients(best_coefficients, false.size, n_modes))
            )
        for random_index in range(selection.random_starts):
            seed = (
                selection.seed
                + 1000 * false.size
                + 37 * n_modes
                + 17 * random_index
                + 991 * mode_index
            )
            generator = np.random.default_rng(seed)
            random_start = generator.normal(
                scale=selection.random_scale * bound, size=false.size * n_modes
            )
            starts.append(
                (f"random_{random_index + 1}", np.clip(random_start, -bound, bound))
            )
        if not starts:
            starts.append(("zero_fallback", np.zeros(false.size * n_modes)))

        run_results: list[dict[str, Any]] = []
        for label, x0 in starts:
            solve_started = time.perf_counter()
            result = minimize(
                objective,
                x0,
                method=optimizer.method,
                jac=True,
                bounds=[(-bound, bound)] * (false.size * n_modes),
                options={
                    "maxiter": optimizer.maxiter,
                    "ftol": optimizer.ftol,
                    "gtol": optimizer.gtol,
                    "maxls": optimizer.maxls,
                },
            )
            solve_time = time.perf_counter() - solve_started
            total_solve += solve_time
            _, final_gradient = objective(np.asarray(result.x, dtype=float))
            run_results.append(
                {
                    "label": label,
                    "action": float(result.fun),
                    "coefficients": np.asarray(result.x, dtype=float),
                    "success": bool(result.success),
                    "message": str(result.message),
                    "nit": int(getattr(result, "nit", -1)),
                    "nfev": int(getattr(result, "nfev", -1)),
                    "njev": int(getattr(result, "njev", -1)),
                    "gradient_norm": float(np.linalg.norm(final_gradient)),
                    "solve_time": solve_time,
                }
            )
        finite_runs = [run for run in run_results if np.isfinite(run["action"])]
        if not finite_runs:
            raise FourierPathError(f"all optimization starts returned non-finite actions at N_m={n_modes}")
        mode_best = min(finite_runs, key=lambda run: run["action"])
        mode_action = float(mode_best["action"])
        relative_change = (
            None
            if previous_mode_action is None or mode_action == 0.0
            else abs(mode_action - previous_mode_action) / abs(mode_action)
        )
        if mode_action < best_action:
            improvement = (
                float("inf")
                if not np.isfinite(best_action)
                else (best_action - mode_action) / max(abs(mode_action), 1.0e-300)
            )
            no_improve = 0 if improvement > selection.relative_tolerance else no_improve + 1
            best_action = mode_action
            best_coefficients = np.asarray(mode_best["coefficients"], dtype=float)
            best_modes = n_modes
            best_run = mode_best
        else:
            no_improve += 1
        assert best_modes is not None
        history.append(
            ModeResult(
                n_modes=n_modes,
                action=mode_action,
                start_label=str(mode_best["label"]),
                optimizer_success=bool(mode_best["success"]),
                message=str(mode_best["message"]),
                nit=int(mode_best["nit"]),
                nfev=int(mode_best["nfev"]),
                njev=int(mode_best["njev"]),
                gradient_norm=float(mode_best["gradient_norm"]),
                relative_change_from_previous=relative_change,
                global_best_action=best_action,
                global_best_n_modes=best_modes,
                no_improve_count=no_improve,
                compile_time_s=compile_time,
                solve_time_s=sum(float(run["solve_time"]) for run in run_results),
                start_actions={str(run["label"]): float(run["action"]) for run in run_results},
            )
        )
        previous_coefficients = np.asarray(mode_best["coefficients"], dtype=float)
        previous_modes = n_modes
        previous_mode_action = mode_action
        if no_improve >= selection.patience:
            adaptive_converged = True
            stop_reason = (
                f"global best did not improve by more than {selection.relative_tolerance:g} "
                f"for {selection.patience} mode steps"
            )
            break

    if best_coefficients is None or best_modes is None or best_run is None:
        raise FourierPathError("optimization produced no finite result")
    coefficients = best_coefficients.reshape(false.size, best_modes)
    parameter = np.linspace(0.0, 1.0, cfg.n_grid)
    path = reconstruct_path(false, true, coefficients, parameter)
    invalid_count, min_v_minus_vt = _path_diagnostics(
        potential, path, potential_false, potential_true, cfg.dimension
    )
    tolerance = max(1.0, bound) * 1.0e-10
    saturated = int(np.count_nonzero(np.abs(np.abs(coefficients) - bound) <= tolerance))
    total_time = time.perf_counter() - started
    return FourierPathResult(
        false_vacuum=false,
        true_vacuum=true,
        coefficients=coefficients,
        selected_modes=best_modes,
        parameter=parameter,
        path_points=path,
        action_proxy=best_action,
        optimizer_success=bool(best_run["success"]),
        optimizer_message=str(best_run["message"]),
        adaptive_converged=adaptive_converged,
        stop_reason=stop_reason,
        history=tuple(history),
        settings=cfg,
        potential_false=potential_false,
        potential_true=potential_true,
        compile_time_s=total_compile,
        solve_time_s=total_solve,
        total_time_s=total_time,
        gradient_norm=float(best_run["gradient_norm"]),
        coefficient_norm=float(np.linalg.norm(coefficients)),
        coefficient_bound=bound,
        coefficient_bound_saturation_count=saturated,
        invalid_interval_count=invalid_count,
        min_v_minus_vt=min_v_minus_vt,
        metadata={
            "energy_convention": "false_vacuum_shifted_to_zero",
            "profile": cfg.profile,
            "potential_gradient_supplied": potential_gradient is not None,
            **dict(metadata or {}),
        },
    )


__all__ = [
    "FourierPathError",
    "FourierPathResult",
    "FourierPathSettings",
    "InputValidationError",
    "ModeResult",
    "ModeSelectionSettings",
    "OptimizerSettings",
    "PotentialEvaluationError",
    "action_prefactor",
    "fourier_basis",
    "optimize_fourier_path",
    "reconstruct_path",
]
