"""Command-line entry point for COSC 4367 Project 2."""

from __future__ import annotations

import argparse
import sys
from math import inf
from pathlib import Path

from graph_services import Counters, all_pairs, critical_closure, evacuations, max_flow
from road_network import RoadNetwork


DEFAULT_INSTANCE = Path(__file__).parent / "data" / "instances" / "tiny"
APSP_ALGORITHMS = {"dijkstra", "bellman-ford", "johnson", "floyd-warshall"}


def _format_distance(value: float) -> str:
    return "INF" if value == inf else str(int(value))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Solve Graphs as Networks queries.")
    parser.add_argument("--nodes", type=Path, default=DEFAULT_INSTANCE / "nodes.tsv",
                        help="path to nodes TSV file")
    parser.add_argument("--edges", type=Path, default=DEFAULT_INSTANCE / "edges.tsv",
                        help="path to edges TSV file")
    parser.add_argument("--query", required=True, choices=("evac", "apsp", "critical", "flow"),
                        help="query to run")
    parser.add_argument("--close", nargs=2, type=int, metavar=("U", "V"),
                        help="show evacuation results with directed edge U->V closed")
    parser.add_argument(
        "--algo",
        choices=sorted(APSP_ALGORITHMS),
        help="APSP algorithm (dijkstra, bellman-ford, johnson, or floyd-warshall)",
    )
    parser.add_argument("--source", type=int, help="optional explicit flow source node ID")
    parser.add_argument("--sink", type=int, help="optional explicit flow sink node ID")
    parser.add_argument("--shelter-cap", type=int, help="maximum intake for each shelter")
    parser.add_argument(
        "--method", choices=("dfs", "bfs", "fattest"), default="bfs",
        help="maximum-flow path selection method (default: bfs)",
    )
    parser.add_argument("--counters", action="store_true", help="write operation counts to counters.txt")
    parser.add_argument("--verbose", action="store_true", help="print search summary to stderr")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.algo is not None and args.query != "apsp":
        parser.error("--algo is only applicable to --query apsp")
    if args.close is not None and args.query != "evac":
        parser.error("--close is only applicable to --query evac")
    if args.query == "flow":
        if (args.source is None) != (args.sink is None):
            parser.error("--source and --sink must be supplied together")
        if args.shelter_cap is not None and args.shelter_cap < 0:
            parser.error("--shelter-cap must be non-negative")
    elif args.source is not None or args.sink is not None or args.method != "bfs" or args.shelter_cap is not None:
        parser.error("--source, --sink, --method, and --shelter-cap are only applicable to --query flow")

    try:
        network = RoadNetwork.load(args.nodes, args.edges)
        counter = Counters()

        if args.query == "evac":
            excluded_arcs = frozenset()
            if args.close is not None:
                road = network.find_road(*args.close)
                excluded_arcs = frozenset({(road.id, *args.close)})
            for row in evacuations(network, excluded_arcs=excluded_arcs, counters=counter):
                shelter = "NONE" if row.shelter_id is None else network.nodes[row.shelter_id].name
                print(
                    f"{network.nodes[row.zone_id].name}\t{shelter}\t"
                    f"{_format_distance(row.minutes)}"
                )
        elif args.query == "apsp":
            distances = all_pairs(network, algorithm=args.algo or "dijkstra", counter=counter)
            for source in sorted(distances):
                for target in sorted(distances[source]):
                    if source != target:
                        print(
                            f"{source}\t{target}\t"
                            f"{_format_distance(distances[source][target])}"
                        )
        elif args.query == "critical":
            result = critical_closure(network, counter=counter)
            if result is None:
                print("NONE")
            else:
                print(f"{result.u}\t{result.v}\t{_format_distance(result.worst_minutes)}")
        else:
            result = max_flow(
                network, args.source, args.sink, method=args.method, counters=counter,
                shelter_cap=args.shelter_cap,
            )
            print(result.value)
            side = set(result.source_side)
            road_lines = set()
            shelter_lines = set()
            for u, v, capacity in result.cut_edges:
                if u in network.nodes and v in network.nodes:
                    road_lines.add((u, v, capacity))
                elif args.shelter_cap is not None and u in network.shelters:
                    shelter_lines.add((u, "SHELTER", capacity))
            for u, v, capacity in sorted(road_lines):
                print(f"{u}\t{v}\t{capacity}")
            for shelter_id, _, capacity in sorted(shelter_lines):
                print(f"{shelter_id}\tSHELTER\t{args.shelter_cap}")

        if args.counters:
            counters_path = Path.cwd() / "counters.txt"
            if args.query == "flow":
                contents = (f"augmenting_paths\t{counter.augmenting_paths}\n"
                            f"residual_examinations\t{counter.residual_examinations}\n")
            else:
                contents = f"Relaxations\t{counter.relaxations}\nPQ_Ops\t{counter.pq_ops}\n"
            counters_path.write_text(contents, encoding="utf-8")
        if args.verbose:
            metrics = (f"augmenting_paths={counter.augmenting_paths} "
                       f"residual_examinations={counter.residual_examinations}" if args.query == "flow"
                       else f"relaxations={counter.relaxations} PQ_Ops={counter.pq_ops}")
            print(f"V={len(network.nodes)} roads={len(network.roads)} "
                  f"arcs={sum(map(len, network.adjacency().values()))} {metrics}", file=sys.stderr)
    except (OSError, ValueError, RuntimeError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
