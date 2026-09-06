# Changelog

## v0.1.1

A diagnostics and validation release. **No objective, tolerance, or numerical
setting changed.** Optimizer outputs are bit-for-bit identical to v0.1.0 at
identical settings, verified across the API example, the FindBounce example and
its `K=1`/`K=4` exports, the CosmoTransitions driver, and the xSM benchmark at
T = 85 GeV.

### FindBounce iteration-limit and unavailable-diagnostic reporting

v0.1.0 set `Status="ok"` whenever FindBounce returned a finite action. FindBounce
breaks its deformation loop on `switchPath || iter == maxItePath`, where
`switchPath` is set by `PathTolerance` or `ActionTolerance` and is not exposed,
so a run stopping at its iteration cap was indistinguishable from one that met a
solver tolerance.

`RunFindBounceWithFourier` now reports `PathIterations`,
`ConfiguredMaxPathIterations` (including FindBounce's own default when the option
is omitted), `PathIterationLimitReached`, `PathTolerance`, `ActionTolerance`, a
`PathConvergence` classification, and a `ConvergenceEvidence` statement. A run
that reaches the cap returns `returned_at_path_iteration_limit`; a finite action
whose termination cannot be classified returns `termination_unverified`. Single
field runs, for which FindBounce performs no path deformation, are classified
separately. `FindBounceTerminationReport` is exported so a direct `FindBounce`
call can be classified the same way.

`Status="ok"` is retained with its documented meaning: FindBounce stopped below
the configured cap, so its own `PathTolerance`/`ActionTolerance` rule fired. That
is the solver's stopping rule, **not** a proof that the action is numerically
converged and not a claim of path uniqueness.

The documented Wolfram example now runs `K=4` at both an insufficient budget of
10 and a sufficient budget of 20, showing the two statuses and the ~1% action
difference between them. It makes no `K`-independence or universal-convergence
claim.

### Reliable JSON output for unsuccessful or limited runs

The `"ResultFile"` option previously wrote a **zero-byte** file whenever the
summary contained `Missing[...]`, that is for exactly the failed, timed-out, and
limit-reached runs whose record matters most: `Export[..., "RawJSON"]` truncated
the file and the enclosing `Quiet@Check` hid the error. Summaries are now
JSON-encodable in every status. Actions, imported initial paths, and the
`BounceFunction` are returned in every status, so a truncated or failed run
remains diagnosable. Timeouts now emit their own message instead of reusing the
"returned `$Failed`" text.

### User-supplied coefficients as the sole optimization start

`ModeSelectionSettings.validate` rejected a configuration whose only start point
was the caller's `initial_coefficients`, even though the optimizer already
handled that case correctly. `validate` now accepts a keyword-only
`external_start_available` flag, which `optimize_fourier_path` supplies, so
`zero_start`, `warm_previous`, `warm_best` and `random_starts` may all be
disabled when explicit coefficients are given. The no-argument behaviour is
unchanged, and the error message now names `initial_coefficients` as one of the
ways to satisfy the requirement.

### Validation corrections

`initial_coefficients` of shape `(n_fields, 0)` is now rejected with
`n_initial_modes >= 1`, matching the contract `reconstruct_path` already
enforced; it previously passed validation and silently behaved as a zero start.

### Detection of optimization that makes no progress

Bounded L-BFGS-B can terminate on its own `ftol` test at its start point,
reporting `CONVERGENCE: RELATIVE REDUCTION OF F <= FACTR*EPSMCH` after a single
iteration without having moved. On the `d=3` objective this happens routinely
when the starts are small, including the default straight-line start: the
returned coefficients are all zero even though the straight path is not
stationary there. v0.1.0 reported such runs as `optimizer_success=True` and
`adaptive_converged=True`, the latter because every mode step returned the same
stalled action and the adaptive scan read that as convergence.

`optimize_fourier_path` now records, per run, whether the optimizer left its
start point and the bound-projected gradient norm. When no start at any mode
count moved while the projected gradient stayed above `gtol`, the result reports
`optimizer_made_no_progress=True`, `optimizer_success=False`,
`adaptive_converged=False`, and a `stop_reason` naming `random_starts` and
`initial_coefficients` as the remedies. The flag is serialized and reloads as
`False` for v0.1.0 archives.

**This is a diagnostic, not a new optimizer and not a guarantee of convergence.**
The underlying line-search behaviour belongs to SciPy's bounded L-BFGS-B on this
objective and is deliberately not worked around, so that numerical results are
unchanged. It reports only the specific case in which nothing was optimized at
all; it does not certify that a run which *did* make progress reached a
stationary point or a global minimum. For `d=3`, set
`ModeSelectionSettings.random_starts >= 1` or supply `initial_coefficients`;
outcomes remain seed-dependent. Always check `optimizer_made_no_progress`,
`gradient_norm`, and `adaptive_converged` before using a path.

### Compatibility

Additive on both sides, with two intended behaviour changes.

* Python: new `FourierPathResult.optimizer_made_no_progress` field (keyword
  default `False`); `validate()` gains a keyword-only argument whose default
  reproduces v0.1.0 behaviour. NPZ files round-trip in both directions. A
  stalled `d=3` run that previously reported `optimizer_success=True` now
  reports `False`.
* Wolfram: seven new result keys and the `FindBounceTerminationReport` export.
  Truncated runs return `returned_at_path_iteration_limit` and unclassifiable
  ones `termination_unverified` instead of `ok`, so code branching on
  `Status === "ok"` will treat them as non-OK. Both bundled drivers were updated.

### Tests

The public suite grows from 34 to 48 tests, all passing, including genuine
FindBounce 1.1.0 runs of the documented `K=4` case at both budgets.

## v0.1.0

Initial release.
