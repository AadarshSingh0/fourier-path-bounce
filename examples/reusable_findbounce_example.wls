(* Run after examples/reusable_findbounce_example.py from the repository root.

   This example deliberately runs the K=4 initializer twice: once with the
   original MaxPathIterations -> 10 budget, which is NOT sufficient for this
   case, and once with MaxPathIterations -> 20, which is. Compare the two
   printed summaries: the first stops exactly at its limit and is reported as
   "returned_at_path_iteration_limit", the second stops on FindBounce's own
   tolerance and is reported as "ok".

   The lesson is the reason this example exists: a finite Action does not mean
   the calculation converged. Always inspect Status, PathIterations,
   ConfiguredMaxPathIterations, PathIterationLimitReached and
   ConvergenceEvidence before using an action. Even a run below the limit only
   tells you that FindBounce's PathTolerance/ActionTolerance rule fired; it is
   not a proof of numerical convergence, and actions obtained with different K
   are not expected to agree. *)

Get[FileNameJoin[{Directory[], "fourier_path_bounce", "wolfram", "FourierPathBounce.wl"}]];
Needs["FourierPathBounce`"];

fields = {x, y};
potential = 1/4 (x^2 - 1)^2 + 1/2 (y - 0.35 (x^2 - 1))^2 - 0.1 x;
gradient = {D[potential, x], D[potential, y]};
falseVacuum = {-0.9456492739235919, -0.03701160775472421};
trueVacuum = {1.0466805318046022, 0.033439047480567696};
metadataFile = FileNameJoin[{Directory[], "example_output", "findbounce_K4", "initializer_metadata.json"}];

reportedKeys = {"Status", "Action", "PathIterations", "ConfiguredMaxPathIterations",
  "PathIterationLimitReached", "PathConvergence", "PathTolerance", "ActionTolerance",
  "FieldPointCount", "IntermediatePointCount", "ConvergenceEvidence"};

runWithBudget[maxPathIterations_Integer] := FourierPathBounce`RunFindBounceWithFourier[
  potential, fields, falseVacuum, trueVacuum, 4, metadataFile,
  "Gradient" -> gradient,
  "FindBounceOptions" -> {
    "MaxPathIterations" -> maxPathIterations,
    "MaxRadiusIterations" -> 200,
    "PathTolerance" -> 0.01
  },
  "TimeLimit" -> 120
];

Print["=== K=4 with the insufficient budget MaxPathIterations -> 10 ==="];
truncated = runWithBudget[10];
Print[KeyTake[truncated, reportedKeys]];

Print[""];
Print["=== K=4 with a sufficient budget MaxPathIterations -> 20 ==="];
converged = runWithBudget[20];
Print[KeyTake[converged, reportedKeys]];

Print[""];
Print["Action at the limit    : ", InputForm[truncated["Action"]]];
Print["Action below the limit : ", InputForm[converged["Action"]]];
Print["Relative difference    : ",
  InputForm[Abs[converged["Action"] - truncated["Action"]]/Abs[converged["Action"]]]];
Print["The first run was truncated, not converged; its action must not be quoted."];
