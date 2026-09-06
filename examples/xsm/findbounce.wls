(* Genuine FindBounce handoff for examples/xsm/benchmark.py. *)

outputDirectory = Environment["XSM_EXAMPLE_OUTPUT"];
wrapperFile = Environment["FOURIER_PATH_BOUNCE_WOLFRAM"];
If[! StringQ[outputDirectory] || outputDirectory === "",
  Print["XSM_EXAMPLE_OUTPUT is not set."]; Exit[2]
];
If[! StringQ[wrapperFile] || ! FileExistsQ[wrapperFile],
  Print["FOURIER_PATH_BOUNCE_WOLFRAM does not name FourierPathBounce.wl."]; Exit[2]
];
If[! Quiet@Check[Needs["FindBounce`"] ; True, False],
  Print["FindBounce is unavailable in this Wolfram installation."]; Exit[3]
];
(* Loaded before the straight-initialized run so both runs are classified by
   the same termination logic. *)
Get[wrapperFile];
Needs["FourierPathBounce`"];

parameterFile = FileNameJoin[{outputDirectory, "xsm_wolfram_parameters.json"}];
metadataFile = FileNameJoin[{outputDirectory, "findbounce_input", "xsm_initializer_metadata.json"}];
resultFile = FileNameJoin[{outputDirectory, "findbounce_results.json"}];
parameters = Quiet@Check[Import[parameterFile, "RawJSON"], $Failed];
If[! AssociationQ[parameters], Print["Cannot read xSM parameter JSON."]; Exit[2]];

lambdaH = N[parameters["lambda_h"]];
lambdaS = N[parameters["lambda_s"]];
lambdaM = N[parameters["lambda_m"]];
muH2 = N[parameters["mu_h_squared"]];
muS2 = N[parameters["mu_s_squared"]];
temperature = N[parameters["temperature"]];
falseVacuum = N[parameters["false_vacuum"]];
trueVacuum = N[parameters["true_vacuum"]];

potential = -muH2 h^2/2 + muS2 s^2/2 + lambdaH h^4/4 +
  lambdaS s^4/4 + lambdaM h^2 s^2/4;
gradient = {D[potential, h], D[potential, s]};
hessian = {{D[potential, h, h], D[potential, h, s]},
  {D[potential, s, h], D[potential, s, s]}};
commonOptions = {
  "Gradient" -> gradient,
  "Hessian" -> hessian,
  "MaxPathIterations" -> 10,
  "MaxRadiusIterations" -> 200,
  "PathTolerance" -> 0.01,
  "ActionTolerance" -> 10.^(-6),
  Dimension -> 3
};

straightTiming = AbsoluteTiming[
  straight = TimeConstrained[
    FindBounce`FindBounce[
      potential, {h, s}, {falseVacuum, trueVacuum},
      "FieldPoints" -> 31, Sequence @@ commonOptions
    ],
    300,
    $Failed
  ];
];
If[straight === $Failed || ! NumericQ[straight["Action"]],
  Print["The straight-initialized FindBounce calculation failed."]; Exit[4]
];

fourierTiming = AbsoluteTiming[
  fourierResult = FourierPathBounce`RunFindBounceWithFourier[
    potential, {h, s}, falseVacuum, trueVacuum, 3, metadataFile,
    "Gradient" -> gradient,
    "FindBounceOptions" -> {
      "Hessian" -> hessian,
      "MaxPathIterations" -> 10,
      "MaxRadiusIterations" -> 200,
      "PathTolerance" -> 0.01,
      "ActionTolerance" -> 10.^(-6)
    },
    "TimeLimit" -> 300
  ];
];
If[! AssociationQ[fourierResult] ||
   ! MemberQ[{"ok", "returned_at_path_iteration_limit"}, fourierResult["Status"]],
  Print["The Fourier-initialized FindBounce calculation failed."]; Exit[4]
];
fourier = fourierResult["Bounce"];
(* Classify the straight-initialized run with the same rules. *)
straightTermination = FourierPathBounce`FindBounceTerminationReport[straight, commonOptions, 2];

Export[
  FileNameJoin[{outputDirectory, "findbounce_straight_path.csv"}],
  N[straight["Path"]], "CSV"
];
Export[
  FileNameJoin[{outputDirectory, "findbounce_fourier_path.csv"}],
  N[fourier["Path"]], "CSV"
];

result = <|
  "solver" -> "FindBounce",
  "version" -> "1.1.0",
  "temperature_GeV" -> temperature,
  "straight" -> Join[<|
    "action_S3" -> N[straight["Action"]],
    "S3_over_T" -> N[straight["Action"]]/temperature,
    "path_iterations" -> straight["PathIterations"],
    "solver_time_s" -> straightTiming[[1]],
    "field_points" -> 31
  |>, straightTermination],
  "fourier" -> Join[<|
    "status" -> fourierResult["Status"],
    "action_S3" -> N[fourier["Action"]],
    "S3_over_T" -> N[fourier["Action"]]/temperature,
    "path_iterations" -> fourier["PathIterations"],
    "solver_time_s" -> fourierTiming[[1]],
    "field_points" -> 31,
    "interior_points_K" -> 29
  |>, KeyDrop[fourierResult, Complement[Keys[fourierResult],
      {"ConfiguredMaxPathIterations", "PathIterationLimitReached", "PathTolerance",
       "ActionTolerance", "PathConvergence", "ConvergenceEvidence"}]]],
  "relative_action_difference" ->
    N[(fourier["Action"] - straight["Action"])/straight["Action"]]
|>;
(* Missing[...] is not JSON-encodable; keep a readable record either way. *)
Export[resultFile, Replace[result, m_Missing :> ToString[m, InputForm], {3}], "RawJSON"];
Print[ExportString[result, "RawJSON"]];
