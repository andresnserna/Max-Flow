# Max Flow

Max Flow models a road network as a directed graph and answers evacuation, all-pairs travel-time, critical-road-closure, and evacuation-capacity queries. The CLI is `Max Flow/project_cli.py`; the shipped default dataset is `tiny`.

## Run it

Run commands from the repository root. Python 3.10+ is recommended; the project uses only the Python standard library.

```sh
python3 "Max Flow/project_cli.py" --help
python3 "Max Flow/project_cli.py" --query evac
```

Choose another shipped dataset by passing its node and edge files:

```sh
python3 "Max Flow/project_cli.py" \
  --nodes "Max Flow/data/instances/small/nodes.tsv" \
  --edges "Max Flow/data/instances/small/edges.tsv" \
  --query evac
```

The input files are tab-separated. `nodes.tsv` has `id`, `name`, `kind`, and `pass_time` columns. `kind` is `zone`, `shelter`, or `junction`. `edges.tsv` has `u`, `v`, `minutes`, `capacity`, and `oneway`; `oneway=1` creates only `u → v`, while `oneway=0` creates arcs in both directions. Travel-time queries use `minutes`; flow uses `capacity`.

## Queries

All queries use the `tiny` dataset by default. To use a different dataset, pass its files with `--nodes` and `--edges` (see [Run it](#run-it)).

### Evacuation assignments

```sh
python3 "Max Flow/project_cli.py" --query evac
```

Prints one tab-separated row per zone: zone name, nearest reachable shelter name (or `NONE`), and travel time in minutes (or `INF`). The assignment is found with one reverse Dijkstra search. If shelters tie, the lower shelter ID wins.

Close one directed road arc and recompute assignments with `--close U V`:

```sh
python3 "Max Flow/project_cli.py" --query evac --close 6 7
```

On a two-way road, only the requested direction is removed.

### All-pairs travel times

```sh
python3 "Max Flow/project_cli.py" --query apsp
python3 "Max Flow/project_cli.py" --query apsp --algo floyd-warshall
```

Prints `source_id`, `target_id`, and distance for each ordered pair of distinct nodes. Unreachable pairs print `INF`. `--algo` accepts `dijkstra` (default), `bellman-ford`, `johnson`, or `floyd-warshall`.

### Critical road closure

```sh
python3 "Max Flow/project_cli.py" --query critical
```

Prints the directed edge `u`, `v` whose closure gives the worst post-closure evacuation travel time, followed by that time. It prints `NONE` when there is no candidate. A newly unreachable zone contributes `INF`; ties use the lowest `(u, v)` pair.

### Maximum evacuation flow

```sh
python3 "Max Flow/project_cli.py" --query flow
python3 "Max Flow/project_cli.py" --query flow --shelter-cap 50
python3 "Max Flow/project_cli.py" --query flow --method fattest --verbose
```

The flow query connects a synthetic source (ID `V`) to every zone and every shelter to a synthetic sink (ID `V+1`), where `V` is the number of real nodes. Without `--shelter-cap`, each shelter is effectively uncapped. `--shelter-cap K` limits each shelter's intake to `K`.

The first output line is the maximum-flow value. Following lines describe the minimum cut as tab-separated `u`, `v`, `capacity` road rows and, when capped, `shelter_id`, `SHELTER`, `K` rows for binding shelter limits. Road endpoints are printed source-side first, and road and shelter rows are sorted as required. The listed capacities sum to the maximum-flow value.

`--method` accepts `bfs` (default), `dfs`, or `fattest`. `--source` and `--sink` can optionally select an explicit source/sink flow instead of the all-zone/all-shelter evacuation model; supply both together.

## Counters and diagnostics

For flow, `--counters` writes `counters.txt` in the current working directory with `augmenting_paths` and `residual_examinations`. The latter counts each residual arc examined during path searches and the final reachability pass for the cut. Other query types retain the `Relaxations` and `PQ_Ops` counters.

```sh
python3 "Max Flow/project_cli.py" --query flow --counters --verbose
```

`--verbose` writes the graph size and query counters to standard error; result rows remain on standard output. `--close` applies only to `evac`, `--algo` only to `apsp`, and flow options only to `flow`.

## Experiments

Run the 2.1 shelter-capacity search on the shipped tiny, small, and medium instances:

```sh
python3 "Max Flow/experiments.py" --experiment 2_1 --verbose
```

The runner prints an estimated first-drop capacity before the search, then reports tested `(K, max flow)` values and the breakpoint. Set the lowest tested capacity per dataset with repeatable `--k DATASET=VALUE` options, for example:

```sh
python3 "Max Flow/experiments.py" --experiment 2_1 --k tiny=10 --k medium=20
```

Run the report's 2.1 search with the selected lower-bound capacities, including the existing generated 30-node cases:

```sh
python3 "Max Flow/experiments.py" --experiment 2_1 --verbose \
  --k tiny=145 \
  --k small=240 \
  --k medium=185 \
  --k generated-30-s2-seed9=105 \
  --k generated-30-s4-seed9=150
```

Generate deterministic same-size instances with different shelter counts using a fixed seed:

```sh
python3 "Max Flow/experiments.py" --experiment 2_1 \
  --generated-nodes 200 --shelters 2 4 6 --seed 9 --verbose
```

Generated node and edge files are saved under `Max Flow/results/generated/`. Experiment attempts are recorded in `Max Flow/results/experiment_2_1_generated.csv`. The other study runners are `--experiment 2_2` (widen minimum-cut roads) and `--experiment 2_3` (close each road and compare with the minimum cut); each writes CSV files under `Max Flow/results/`.

## Tests

Run the standard-library test suite from the project directory:

```sh
cd "Max Flow"
python3 -m unittest test_project.py
```

The tests cover the shipped datasets, query formatting, flow algorithms, multi-zone/shelter flow with capped and uncapped intake, cut output, and flow counters.
