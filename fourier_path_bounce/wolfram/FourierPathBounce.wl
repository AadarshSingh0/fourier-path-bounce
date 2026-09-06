(* Generic FindBounce handoff for fourier_path_bounce Python exports. *)
BeginPackage["FourierPathBounce`"];

ImportFourierPathPoints::usage =
  "ImportFourierPathPoints[source, falseVacuum, trueVacuum] validates and returns an open false-to-true path. source may be a metadata JSON file, a full-path CSV file, or an explicit numeric point matrix.";
RunFindBounceWithFourier::usage =
  "RunFindBounceWithFourier[potential, fields, falseVacuum, trueVacuum, dimension, source, opts] invokes the standard FindBounce solver with the Fourier path supplied as FieldPoints.";
FindBounceTerminationReport::usage =
  "FindBounceTerminationReport[bounce, findBounceOptions, fieldCount] reports how a FindBounce calculation terminated: the observed path-iteration count, the effective configured limit, whether that limit was reached, the tolerances that define the solver's own stopping rule, and what those facts do and do not establish about convergence.";

ImportFourierPathPoints::source = "Cannot read Fourier path source `1`.";
ImportFourierPathPoints::matrix = "The Fourier path must be a finite numeric matrix.";
ImportFourierPathPoints::dims = "Path point dimension `1` does not equal vacuum dimension `2`.";
ImportFourierPathPoints::orient = "The path must start at the supplied false vacuum and end at the supplied true vacuum.";
ImportFourierPathPoints::meta = "Metadata is missing required false-to-true open-path information.";
RunFindBounceWithFourier::input = "Fields and vacua must be consistent finite vectors.";
RunFindBounceWithFourier::potential = "The potential is not finite and numeric at both supplied vacua.";
RunFindBounceWithFourier::gradient = "The supplied gradient does not evaluate to a finite vector of the field dimension.";
RunFindBounceWithFourier::options = "FindBounceOptions must be a list of rules and may not override FieldPoints, Gradient, or Dimension.";
RunFindBounceWithFourier::package = "FindBounce could not be loaded in this Wolfram environment.";
RunFindBounceWithFourier::failed = "FindBounce returned $Failed or did not expose a finite numeric Action.";
RunFindBounceWithFourier::pathlimit = "FindBounce stopped at the configured MaxPathIterations limit (`1`). The returned action is finite but path deformation may have been truncated; rerun with a larger limit to distinguish truncation from convergence at the limit.";

Begin["`Private`"];

finiteNumericQ[value_] := NumericQ[N[value]] && FreeQ[N[value], _DirectedInfinity | Indeterminate | ComplexInfinity];

(* --- FindBounce termination semantics --------------------------------------
   FindBounce 1.1.0 runs

       While[iter <= maxItePath || switchPath,
         ... ; If[switchPath || iter == maxItePath || bottomless, Break[]]; ... ; iter++]
       results["PathIterations"] = iter

   `switchPath` is set to True when the relative path displacement falls below
   "PathTolerance" (MultiFieldBounce) or the relative action change falls below
   "ActionTolerance" (ParameterInFieldSpace). The loop therefore exits either
   because one of those solver tolerances was met or because the iteration cap
   was hit, and FindBounce does not expose which. Consequently:

     PathIterations <  cap  ->  a solver tolerance was met (not a proof that the
                                action is numerically converged, only that
                                FindBounce's own stopping rule fired);
     PathIterations == cap  ->  ambiguous; truncation cannot be excluded.

   For a single field FindBounce forces maxItePath = 0 and performs no path
   deformation at all, so the limit concept does not apply. --------------------*)

normalizedOptionKeys[rules_List] := Association @@ (
  Function[rule, ToString[First[rule]] -> Last[rule]] /@ rules);

findBounceDefaultOption[name_String] := Quiet@Check[
  Lookup[normalizedOptionKeys[Options[FindBounce`FindBounce]], name, Missing["Unavailable"]],
  Missing["Unavailable"]];

effectiveFindBounceOption[userOptions_List, name_String] := Module[{supplied},
  supplied = Lookup[normalizedOptionKeys[userOptions], name, Missing["NotSet"]];
  If[MissingQ[supplied], findBounceDefaultOption[name], supplied]
];

FindBounceTerminationReport[bounce_, userOptions_List, fieldCount_Integer] := Module[
  {pathIterations, cap, pathTolerance, actionTolerance, singleField, limitReached,
   convergence, evidence},
  pathIterations = If[bounce === $Failed || bounce === Null,
    Missing["Unavailable"],
    Quiet@Check[bounce["PathIterations"], Missing["Unavailable"]]];
  cap = effectiveFindBounceOption[userOptions, "MaxPathIterations"];
  pathTolerance = effectiveFindBounceOption[userOptions, "PathTolerance"];
  actionTolerance = effectiveFindBounceOption[userOptions, "ActionTolerance"];
  singleField = fieldCount === 1;
  limitReached = TrueQ[
    Not[singleField] && IntegerQ[pathIterations] && IntegerQ[cap] && pathIterations >= cap];
  convergence = Which[
    singleField, "not_applicable_single_field",
    Not[IntegerQ[pathIterations]] || Not[IntegerQ[cap]], "unknown",
    limitReached, "not_established_limit_reached",
    True, "solver_tolerance_satisfied"];
  evidence = Switch[convergence,
    "not_applicable_single_field",
      "FindBounce performs no path deformation for a single field; MaxPathIterations does not apply.",
    "unknown",
      "The path-iteration count or the configured limit is unavailable, so termination cannot be classified.",
    "not_established_limit_reached",
      "Path deformation stopped at the configured MaxPathIterations limit. FindBounce does not report whether its PathTolerance/ActionTolerance criterion was also met, so truncation cannot be excluded. Rerun with a larger limit and compare actions.",
    "solver_tolerance_satisfied",
      "Path deformation stopped before the configured limit, so FindBounce's own PathTolerance/ActionTolerance criterion fired. That is the solver's stopping rule, not a proof that the action is numerically converged; confirm by varying the tolerances, the limit, and the number of field points."];
  <|
    "PathIterations" -> pathIterations,
    "ConfiguredMaxPathIterations" -> cap,
    "PathIterationLimitReached" -> limitReached,
    "PathTolerance" -> pathTolerance,
    "ActionTolerance" -> actionTolerance,
    "PathConvergence" -> convergence,
    "ConvergenceEvidence" -> evidence
  |>
];
finiteVectorQ[value_] := VectorQ[value, finiteNumericQ];
finiteMatrixQ[value_] := MatrixQ[value, finiteNumericQ];

loadSource[source_String] := Module[{extension, metadata, fullPathFile},
  If[! FileExistsQ[source], Message[ImportFourierPathPoints::source, source]; Return[$Failed]];
  extension = ToLowerCase[FileExtension[source]];
  Switch[extension,
    "json",
      metadata = Quiet@Check[Import[source, "RawJSON"], $Failed];
      If[! AssociationQ[metadata], Message[ImportFourierPathPoints::source, source]; Return[$Failed]];
      If[
        Lookup[metadata, "format", Missing[]] =!= "fourier_path_bounce.findbounce_points" ||
        Lookup[metadata, "orientation", Missing[]] =!= "false_to_true" ||
        Lookup[metadata, "polygon", Missing[]] =!= "open",
        Message[ImportFourierPathPoints::meta]; Return[$Failed]
      ];
      fullPathFile = FileNameJoin[{DirectoryName[ExpandFileName[source]], Lookup[metadata, "full_path_csv", ""]}];
      If[! FileExistsQ[fullPathFile], Message[ImportFourierPathPoints::source, fullPathFile]; Return[$Failed]];
      Quiet@Check[N@Import[fullPathFile, "CSV"], $Failed],
    "csv", Quiet@Check[N@Import[source, "CSV"], $Failed],
    _, Message[ImportFourierPathPoints::source, source]; $Failed
  ]
];
loadSource[source_List] := N[source];
loadSource[source_] := (Message[ImportFourierPathPoints::source, source]; $Failed);

ImportFourierPathPoints[source_, falseVacuum_List, trueVacuum_List] := Module[
  {path, dimension, tolerance = 10^-10},
  If[! finiteVectorQ[falseVacuum] || ! finiteVectorQ[trueVacuum] || Length[falseVacuum] =!= Length[trueVacuum],
    Message[RunFindBounceWithFourier::input]; Return[$Failed]
  ];
  dimension = Length[falseVacuum];
  path = loadSource[source];
  If[path === $Failed, Return[$Failed]];
  If[! finiteMatrixQ[path], Message[ImportFourierPathPoints::matrix]; Return[$Failed]];
  If[Length[path] < 3, Message[ImportFourierPathPoints::matrix]; Return[$Failed]];
  If[Dimensions[path][[2]] =!= dimension,
    Message[ImportFourierPathPoints::dims, Dimensions[path][[2]], dimension]; Return[$Failed]
  ];
  If[Norm[path[[1]] - N[falseVacuum]] > tolerance || Norm[path[[-1]] - N[trueVacuum]] > tolerance,
    Message[ImportFourierPathPoints::orient]; Return[$Failed]
  ];
  (* Replace round-tripped values only after orientation has been validated. *)
  path[[1]] = N[falseVacuum];
  path[[-1]] = N[trueVacuum];
  If[AnyTrue[path[[2 ;; -2]], Norm[# - path[[1]]] <= tolerance || Norm[# - path[[-1]]] <= tolerance &],
    Message[ImportFourierPathPoints::orient]; Return[$Failed]
  ];
  path
];

Options[RunFindBounceWithFourier] = {
  "Gradient" -> Automatic,
  "FindBounceOptions" -> {},
  "TimeLimit" -> Infinity,
  "ResultFile" -> None
};

RunFindBounceWithFourier[
  potential_, fields_List, falseVacuum_List, trueVacuum_List,
  dimension_Integer, source_, OptionsPattern[]
] := Module[
  {path, gradient, userOptions, forbidden, timeLimit, resultFile, variables,
   potentialValues, gradientValues, callOptions, timed, runtime, bounce,
   action, status, message = "", summary, result, termination, exportable},

  If[
    dimension < 2 || Length[fields] < 1 || Length[fields] =!= Length[falseVacuum] ||
    Length[fields] =!= Length[trueVacuum] || ! finiteVectorQ[falseVacuum] || ! finiteVectorQ[trueVacuum],
    Message[RunFindBounceWithFourier::input]; Return[$Failed]
  ];
  If[! Quiet@Check[Needs["FindBounce`"]; True, False],
    Message[RunFindBounceWithFourier::package]; Return[$Failed]
  ];
  path = ImportFourierPathPoints[source, falseVacuum, trueVacuum];
  If[path === $Failed, Return[$Failed]];
  potentialValues = Quiet@Check[N[potential /. Thread[fields -> #]] & /@ {falseVacuum, trueVacuum}, $Failed];
  If[potentialValues === $Failed || ! finiteVectorQ[potentialValues],
    Message[RunFindBounceWithFourier::potential]; Return[$Failed]
  ];

  gradient = OptionValue["Gradient"];
  If[gradient =!= Automatic,
    gradientValues = Quiet@Check[N[gradient /. Thread[fields -> #]] & /@ {falseVacuum, trueVacuum}, $Failed];
    If[gradientValues === $Failed || Dimensions[gradientValues] =!= {2, Length[fields]} || ! finiteMatrixQ[gradientValues],
      Message[RunFindBounceWithFourier::gradient]; Return[$Failed]
    ]
  ];
  userOptions = OptionValue["FindBounceOptions"];
  forbidden = {"FieldPoints", "Gradient", Dimension};
  If[! ListQ[userOptions] || ! And @@ (MatchQ[#, _Rule | _RuleDelayed] & /@ userOptions) ||
     AnyTrue[userOptions, MemberQ[forbidden, First[#]] &],
    Message[RunFindBounceWithFourier::options]; Return[$Failed]
  ];
  timeLimit = OptionValue["TimeLimit"];
  resultFile = OptionValue["ResultFile"];
  variables = If[Length[fields] == 1, First[fields], fields];
  callOptions = Join[
    {"FieldPoints" -> path},
    If[gradient === Automatic, {}, {"Gradient" -> gradient}],
    {Dimension -> dimension},
    userOptions
  ];

  Block[{$MessageList = {}},
    timed = TimeConstrained[
      AbsoluteTiming[
        Quiet@Check[
          FindBounce`FindBounce[
            potential, variables, {falseVacuum, trueVacuum}, Sequence @@ callOptions
          ],
          $Failed
        ]
      ],
      timeLimit,
      "TIMEOUT"
    ];
    message = ToString[$MessageList, InputForm];
  ];
  If[timed === "TIMEOUT",
    runtime = N[timeLimit]; bounce = $Failed; action = Missing["Timeout"]; status = "timeout",
    runtime = timed[[1]]; bounce = timed[[2]];
    action = If[bounce === $Failed, Missing["Failed"], Quiet@Check[N[bounce["Action"]], Missing["ActionFailed"]]];
    status = If[finiteNumericQ[action], "ok", "failed"]
  ];
  (* A finite action is not by itself evidence of convergence: classify how
     FindBounce actually stopped before assigning a successful status. *)
  termination = FindBounceTerminationReport[bounce, userOptions, Length[fields]];
  If[status === "ok" && TrueQ[termination["PathIterationLimitReached"]],
    status = "returned_at_path_iteration_limit"
  ];
  summary = Join[<|
    "Status" -> status,
    "Action" -> action,
    "RuntimeSeconds" -> runtime,
    "Messages" -> message,
    "Dimension" -> dimension,
    "FieldDimension" -> Length[fields],
    "IntermediatePointCount" -> Length[path] - 2,
    "FieldPointCount" -> Length[path],
    "Orientation" -> "false_to_true",
    "Polygon" -> "open"
  |>, termination];
  (* Missing[...] cannot be encoded as JSON. Without this substitution Export
     truncates the file and the enclosing Check hides the error, so precisely
     the failed, timed-out, and limit-reached runs left a zero-byte record. *)
  exportable = Replace[summary, m_Missing :> ToString[m, InputForm], {1}];
  If[StringQ[resultFile], Quiet@Check[Export[resultFile, exportable, "RawJSON"], Null]];
  Which[
    status === "failed" || status === "timeout",
      Message[RunFindBounceWithFourier::failed],
    status === "returned_at_path_iteration_limit",
      Message[RunFindBounceWithFourier::pathlimit, termination["ConfiguredMaxPathIterations"]]
  ];
  (* The action, the imported initial path, and the BounceFunction (whose
     "Path" and "Radii" carry the deformed geometry) are always returned, so a
     limit-reached or failed run remains fully diagnosable. *)
  result = Join[summary, <|"InitialPath" -> path, "Bounce" -> bounce|>];
  result
];

End[];
EndPackage[];
