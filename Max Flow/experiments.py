"""Experiment runners for the graph query performance study."""

from __future__ import annotations

import csv
import argparse
import subprocess
import sys
import math
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from time import perf_counter
from typing import Callable, TypeVar
from data.netgen import build, write
from road_network import RoadNetwork
from graph_flow import Graph as FlowGraph


T = TypeVar("T")
BASE = Path(__file__).parent
DATASETS = ("tiny", "small", "medium")
APSP_ALGORITHMS = ("dijkstra", "floyd-warshall", "johnson")
MAX_FLOYD_CELLS = 4_000_000


@dataclass(frozen=True)
class TimedResult:
    value: object | None
    seconds: float
    timed_out: bool = False


class ThreadedRunner:
    """Run a callable in a worker while a separate thread measures its runtime."""

    @staticmethod
    def run(function: Callable[[Event], T], *, stop_after: float | None = 30.0) -> TimedResult:
        started = Event()
        finished = Event()
        stop_event = Event()
        measured: list[float] = []
        values: list[T] = []
        errors: list[BaseException] = []
        timed_out: list[bool] = []

        def clock() -> None:
            start_time = perf_counter()
            started.set()
            if stop_after is None:
                finished.wait()
                timed_out.append(False)
            elif finished.wait(stop_after):
                timed_out.append(False)
            else:
                stop_event.set()
                finished.wait()
                timed_out.append(True)
            measured.append(perf_counter() - start_time)

        def work() -> None:
            started.wait()
            try:
                values.append(function(stop_event))
            except TimeoutError:
                timed_out.append(True)
            except BaseException as error:
                errors.append(error)
            finally:
                finished.set()

        timer_thread = Thread(target=clock, name="experiment-clock")
        worker_thread = Thread(target=work, name="experiment-worker")
        timer_thread.start()
        worker_thread.start()
        try:
            worker_thread.join()
        except KeyboardInterrupt:
            stop_event.set()
            worker_thread.join()
            timer_thread.join()
            raise
        timer_thread.join()

        if errors:
            raise errors[0]
        return TimedResult(values[0] if values else None, measured[0], any(timed_out))


def with_stop(counters: Counters, stop_event: Event) -> Counters:
    counters.stop_event = stop_event
    return counters

def experiment_2_1(shelter_capacity: int | None = None, datasets=DATASETS) -> tuple[int, ...]:
    """When do shelters become the problem? \n
For each network, find the maximum flow with shelters uncapped, and then progressively lower 
the shelter cap and watch the Max Flow. At some value of K our binding constraint will become 
shelter capacity rather than the capacity of our roads. """

    if shelter_capacity is not None and shelter_capacity < 0:
        raise ValueError("shelter_capacity cannot be negative")

    results: list[int] = []

    for dataset in datasets:
        directory = BASE / "data" / "instances" / dataset
        network = RoadNetwork.load(directory / "nodes.tsv", directory / "edges.tsv")
        if not network.zones or not network.shelters:
            raise ValueError(f"{dataset}: max flow requires at least one zone and shelter")

        results.append(_max_flow_with_shelter_capacity(network, shelter_capacity))

    return tuple(results)

def _max_flow_with_shelter_capacity(
    network: RoadNetwork, shelter_capacity: int | None
) -> int:
    """Compute the zone-to-shelter max flow for one loaded network."""
    if not network.zones or not network.shelters:
        raise ValueError("max flow requires at least one zone and shelter")

    source, sink = "__experiment_source__", "__experiment_sink__"
    graph = FlowGraph()
    for node_id in sorted(network.nodes):
        graph.add_node(node_id)
    for road in network.roads:
        if road.oneway:
            graph.add_edge(road.u, road.v, road.capacity)
        else:
            graph.add_undirected_edge(road.u, road.v, road.capacity)

    uncapped_capacity = max(1, sum(road.capacity for road in network.roads))
    cap = uncapped_capacity if shelter_capacity is None else shelter_capacity
    for zone_id in network.zones:
        graph.add_edge(source, zone_id, uncapped_capacity)
    for shelter_id in network.shelters:
        graph.add_edge(shelter_id, sink, cap)
    return graph.max_flow(source, sink).value

def experiment_2_2(datasets=DATASETS, *, increase: int = 50) -> tuple[Path, ...]:
    """Impact of road widening \n
Test the result of adding 50 to every edge in your minimum cut, and report how the max 
flow changes in each instance.  """
    if increase < 0:
        raise ValueError("increase cannot be negative")

    output_files: list[Path] = []
    output_dir = BASE / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        directory = BASE / "data" / "instances" / dataset
        network = RoadNetwork.load(directory / "nodes.tsv", directory / "edges.tsv")
        if not network.zones or not network.shelters:
            raise ValueError(f"{dataset}: max flow requires at least one zone and shelter")

        source, sink = "__experiment_source__", "__experiment_sink__"
        uncapped_capacity = max(1, sum(road.capacity for road in network.roads))

        def build_flow_graph(widened_road_id: int | None = None) -> FlowGraph:
            graph = FlowGraph()

            for node_id in sorted(network.nodes):
                graph.add_node(node_id)

            for road in network.roads:
                capacity = road.capacity + (increase if road.id == widened_road_id else 0)
                if road.oneway:
                    graph.add_edge(road.u, road.v, capacity)
                else:
                    graph.add_undirected_edge(road.u, road.v, capacity)
            
            for zone_id in network.zones:
                graph.add_edge(source, zone_id, uncapped_capacity)
            
            for shelter_id in network.shelters:
                graph.add_edge(shelter_id, sink, uncapped_capacity)
            
            return graph

        baseline = build_flow_graph().max_flow(source, sink)
        
        # Limit the trials to the road edges in the original minimum cut.
        cut_pairs = {(u, v) for u, v, _ in baseline.cut_edges}
        
        cut_roads = [
            road for road in network.roads
            if (road.u, road.v) in cut_pairs
            or (not road.oneway and (road.v, road.u) in cut_pairs)
        ]

        output_path = output_dir / f"experiment_2_2_{dataset}.csv"
       
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("attempt#", "u", "v", "weight before", "weight after",
                             "max flow after", "max flow delta"))
            for attempt, road in enumerate(cut_roads, start=1):
                after = build_flow_graph(road.id).max_flow(source, sink).value
                writer.writerow((attempt, road.u, road.v, road.capacity,
                                 road.capacity + increase, after, after - baseline.value))
        output_files.append(output_path)

    return tuple(output_files)

def experiment_2_3(datasets=DATASETS) -> tuple[Path, ...]:
    """Is the most “dangerous” road also the most narrow?  \n
Our min-cut finds the roads that limit how many people can be 
pushed through the network at the same time. Run your closure experiment and the max-flow 
simultaneously (i.e. find the max flow with each edge removed), and check whether the critical 
edge appears in the minimum cut set.  """
    output_files: list[Path] = []
    output_dir = BASE / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset in datasets:
        directory = BASE / "data" / "instances" / dataset
        network = RoadNetwork.load(directory / "nodes.tsv", directory / "edges.tsv")
        if not network.zones or not network.shelters:
            raise ValueError(f"{dataset}: max flow requires at least one zone and shelter")

        source, sink = "__experiment_source__", "__experiment_sink__"
        uncapped_capacity = max(1, sum(road.capacity for road in network.roads))

        def build_flow_graph(closed_road_id: int | None = None) -> FlowGraph:
            graph = FlowGraph()
            for node_id in sorted(network.nodes):
                graph.add_node(node_id)
            for road in network.roads:
                if road.id == closed_road_id:
                    continue
                if road.oneway:
                    graph.add_edge(road.u, road.v, road.capacity)
                else:
                    graph.add_undirected_edge(road.u, road.v, road.capacity)
            for zone_id in network.zones:
                graph.add_edge(source, zone_id, uncapped_capacity)
            for shelter_id in network.shelters:
                graph.add_edge(shelter_id, sink, uncapped_capacity)
            return graph

        baseline = build_flow_graph().max_flow(source, sink)
        cut_pairs = {(u, v) for u, v, _ in baseline.cut_edges}
        output_path = output_dir / f"experiment_2_3_{dataset}.csv"
        with output_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(("attempt#", "u", "v", "oneway", "weight",
                             "max flow after", "max flow delta", "in min cut"))
            for attempt, road in enumerate(network.roads, start=1):
                after = build_flow_graph(road.id).max_flow(source, sink).value
                is_in_cut = ((road.u, road.v) in cut_pairs or
                             (not road.oneway and (road.v, road.u) in cut_pairs))
                writer.writerow((attempt, road.u, road.v, road.oneway, road.capacity,
                                 after, after - baseline.value, is_in_cut))
        output_files.append(output_path)

    return tuple(output_files)

def exp_2_1_runner(
    datasets=DATASETS,
    *,
    k_by_dataset: dict[str, int] | None = None,
    verbose: bool = False,
    instances: dict[str, Path] | None = None,
):
    """Find each dataset's shelter-capacity breakpoint with a halving/bisection search.

    ``k_by_dataset`` sets the lowest capacity to consider for each dataset
    (default 0). The returned dictionary stores each dataset's baseline, every
    (K, max-flow) attempt, and the highest K at which flow falls below baseline.
    """
    selected = tuple(datasets)
    k_by_dataset = k_by_dataset or {}
    unknown = set(k_by_dataset) - set(selected)
    if unknown:
        raise ValueError(f"K values supplied for unselected datasets: {', '.join(sorted(unknown))}")
    if any(value < 0 for value in k_by_dataset.values()):
        raise ValueError("K values cannot be negative")

    results: dict[str, dict[str, object]] = {}
    for dataset in selected:
        directory = (instances or {}).get(dataset, BASE / "data" / "instances" / dataset)
        network = RoadNetwork.load(directory / "nodes.tsv", directory / "edges.tsv")
        stop_k = k_by_dataset.get(dataset, 0)
        if stop_k < 0:
            raise ValueError("K values cannot be negative")

        baseline = _max_flow_with_shelter_capacity(network, None)
        shelter_threshold = math.ceil(baseline / len(network.shelters)) if network.shelters else 0
        if verbose:
            print(f"{dataset}: first drop in max flow at K={shelter_threshold}", flush=True)
        attempts: list[tuple[int, int]] = []

        def attempt(capacity: int) -> int:
            flow = _max_flow_with_shelter_capacity(network, capacity)
            attempts.append((capacity, flow))
            if verbose:
                print(f"{dataset}: K={capacity}, max flow={flow}", flush=True)
            return flow

        if baseline == 0:
            dropped_k = None
            unchanged_k = None
        else:
            # Start near two-thirds of the baseline, then halve until the
            # shelter cap lowers flow. Integer arithmetic rounds upward.
            capacity = max(stop_k, (2 * baseline + 2) // 3)
            same_k = baseline
            drop_k: int | None = None
            while True:
                flow = attempt(capacity)
                if flow < baseline:
                    drop_k = capacity
                    break
                same_k = capacity
                if capacity == stop_k:
                    break
                capacity = max(stop_k, capacity // 2)

            if drop_k is None:
                dropped_k = unchanged_k = None
            else:
                # Maintain a known dropped K and a higher K with baseline flow;
                # integer bisection converges to adjacent capacities.
                while same_k - drop_k > 1:
                    midpoint = (drop_k + same_k) // 2
                    flow = attempt(midpoint)
                    if flow < baseline:
                        drop_k = midpoint
                    else:
                        same_k = midpoint
                dropped_k, unchanged_k = drop_k, same_k

        results[dataset] = {
            "baseline": baseline,
            "predicted_first_drop_k": shelter_threshold,
            "attempts": tuple(attempts),
            "first_drop_k": dropped_k,
            "last_unchanged_k": unchanged_k,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run graph network experiments.")
    parser.add_argument("--experiment", choices=("2_1", "2_2", "2_3"), default="2_1")
    parser.add_argument(
        "--k", action="append", default=[], metavar="DATASET=VALUE",
        help="Experiment 2.1 lowest K to search; may be repeated (default: 0 per dataset).",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print each capacity and max-flow result during experiment 2.1.",
    )
    parser.add_argument("--generated-nodes", type=int, help="generate same-size experiment 2.1 instances")
    parser.add_argument("--shelters", type=int, nargs="+", default=[], help="shelter counts for generated cases")
    parser.add_argument("--seed", type=int, default=1, help="repeatable generator seed")
    args = parser.parse_args()
    
    if args.experiment == "2_1":
        k_by_dataset: dict[str, int] = {}
        for specification in args.k:
            try:
                dataset, raw_value = specification.split("=", 1)
                value = int(raw_value)
            except ValueError as error:
                raise SystemExit(f"Invalid --k value {specification!r}; expected DATASET=VALUE") from error
            if dataset not in DATASETS and not dataset.startswith("generated-"):
                raise SystemExit(f"Unknown dataset in --k: {dataset!r}")
            if dataset in k_by_dataset:
                raise SystemExit(f"--k was specified more than once for {dataset!r}")
            if value < 0:
                raise SystemExit("--k values must be non-negative")
            k_by_dataset[dataset] = value

        instances: dict[str, Path] = {}

        if args.generated_nodes is not None:
            if args.generated_nodes < 3 or not args.shelters or any(count < 1 for count in args.shelters):
                raise SystemExit("--generated-nodes requires size >= 3 and positive --shelters counts")
            for count in sorted(set(args.shelters)):
                name = f"generated-{args.generated_nodes}-s{count}-seed{args.seed}"
                directory = BASE / "results" / "generated" / name
                write(build(args.generated_nodes, args.seed, n_shelter=count), directory)
                instances[name] = directory
        elif args.shelters:
            raise SystemExit("--shelters requires --generated-nodes")

        # A generated instance can be selected by naming it in --k, which
        # allows running an already generated case without regenerating it.
        for dataset in k_by_dataset:
            if dataset.startswith("generated-") and dataset not in instances:
                directory = BASE / "results" / "generated" / dataset
                if (directory / "nodes.tsv").is_file() and (directory / "edges.tsv").is_file():
                    instances[dataset] = directory
        
        results = exp_2_1_runner(datasets=(*DATASETS, *instances), k_by_dataset=k_by_dataset,
                                 verbose=args.verbose, instances=instances)
        
        for dataset, result in results.items():
            print(f"{dataset}: baseline max flow={result['baseline']}")
            print(f"  First drop in max flow at: K={result['predicted_first_drop_k']}")
           
            if result["first_drop_k"] is None:
                print(f"  No shelter-capacity drop found at or above K={k_by_dataset.get(dataset, 0)}")
            else:
                print(
                    f"  First drop: K={result['first_drop_k']} "
                    f"(flow below baseline); K={result['last_unchanged_k']} still preserves baseline"
                )
           
            print(f"  Attempts: {result['attempts']}")
       
        if instances:
            output = BASE / "results" / "experiment_2_1_generated.csv"
            with output.open("w", newline="", encoding="utf-8") as file:
                writer = csv.writer(file)
                writer.writerow(("instance", "shelters", "seed", "baseline", "predicted_first_drop_k", "K", "max_flow"))
                for name, result in results.items():
                    if name not in instances:
                        continue
                    network = RoadNetwork.load(instances[name] / "nodes.tsv", instances[name] / "edges.tsv")
                    for capacity, flow in result["attempts"]:
                        writer.writerow((name, len(network.shelters), args.seed, result["baseline"],
                                         result["predicted_first_drop_k"], capacity, flow))
            print(output)
    elif args.experiment == "2_2":
        for output_file in experiment_2_2():
            print(output_file)
    else:
        for output_file in experiment_2_3():
            print(output_file)



if __name__ == "__main__":
    main()
