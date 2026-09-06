(* Genuine FindBounce integration for the reusable two-field example. *)
Get[FileNameJoin[{Directory[], "fourier_path_bounce", "wolfram", "FourierPathBounce.wl"}]];

metadataFile = Environment["FOURIER_PATH_METADATA"];
resultFile = Environment["FOURIER_PATH_RESULT"];
If[! StringQ[metadataFile] || ! FileExistsQ[metadataFile], Print["Missing FOURIER_PATH_METADATA"]; Exit[2]];
If[! StringQ[resultFile], Print["Missing FOURIER_PATH_RESULT"]; Exit[2]];

fields = {x, y};
potential = 1/4 (x^2 - 1)^2 + 1/2 (y - 0.35 (x^2 - 1))^2 - 0.1 x;
gradient = {D[potential, x], D[potential, y]};
falseVacuum = {-0.9456492739235919, -0.03701160775472421};
trueVacuum = {1.0466805318046022, 0.033439047480567696};

result = FourierPathBounce`RunFindBounceWithFourier[
  potential, fields, falseVacuum, trueVacuum, 4, metadataFile,
  "Gradient" -> gradient,
  (* 20 is sufficient for the K=4 case of this example; 10 is not. See
     examples/reusable_findbounce_example.wls. *)
  "FindBounceOptions" -> {
    "MaxPathIterations" -> 20,
    "MaxRadiusIterations" -> 200,
    "PathTolerance" -> 0.01
  },
  "TimeLimit" -> 120,
  "ResultFile" -> resultFile
];
If[result === $Failed, Exit[3]];
Print[ExportString[KeyDrop[result, {"Bounce", "InitialPath"}], "RawJSON"]];
(* Exit codes distinguish the outcomes rather than collapsing a truncated or
   unclassifiable run into a generic failure:
     0 solver tolerance satisfied below the configured limit
     4 no finite action (solver failure or timeout)
     5 finite action returned exactly at the path-iteration limit
     6 finite action whose termination could not be classified *)
Which[
  result["Status"] === "ok", Exit[0],
  result["Status"] === "returned_at_path_iteration_limit",
    Print["Path deformation stopped at the configured limit; the action is not established as converged."];
    Exit[5],
  result["Status"] === "termination_unverified",
    Print["FindBounce returned a finite action but no usable termination diagnostics."];
    Exit[6],
  True, Exit[4]
];
