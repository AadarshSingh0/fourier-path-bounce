(* Run after examples/reusable_findbounce_example.py from the repository root. *)
Get[FileNameJoin[{Directory[], "fourier_path_bounce", "wolfram", "FourierPathBounce.wl"}]];
Needs["FourierPathBounce`"];

fields = {x, y};
potential = 1/4 (x^2 - 1)^2 + 1/2 (y - 0.35 (x^2 - 1))^2 - 0.1 x;
gradient = {D[potential, x], D[potential, y]};
falseVacuum = {-0.9456492739235919, -0.03701160775472421};
trueVacuum = {1.0466805318046022, 0.033439047480567696};
metadataFile = FileNameJoin[{Directory[], "example_output", "findbounce_K4", "initializer_metadata.json"}];

result = FourierPathBounce`RunFindBounceWithFourier[
  potential, fields, falseVacuum, trueVacuum, 4, metadataFile,
  "Gradient" -> gradient,
  "FindBounceOptions" -> {
    "MaxPathIterations" -> 10,
    "MaxRadiusIterations" -> 200,
    "PathTolerance" -> 0.01
  },
  "TimeLimit" -> 120
];
Print[KeyDrop[result, {"Bounce", "InitialPath"}]];
