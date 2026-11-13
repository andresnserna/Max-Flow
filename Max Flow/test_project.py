"""Focused tests against the shipped tiny instance and expected outputs."""

from __future__ import annotations

import csv
import io
import os
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from math import inf
from pathlib import Path

from project_cli import main as cli_main
from graph_services import all_pairs, critical_closure, evacuations
from road_network import Node, Road, RoadNetwork
from graph_flow import Graph as FlowGraph, make_graph
from graph_services import Counters


INSTANCE = Path(__file__).parent / "data" / "instances" / "tiny"
NODES = INSTANCE / "nodes.tsv"
EDGES = INSTANCE / "edges.tsv"


class MaxFlowTests(unittest.TestCase):
    def test_all_zones_to_shelters_and_shelter_cap_cut_output(self) -> None:
        network = RoadNetwork(
            nodes={
                0: Node(0, "Z0", "zone", 0), 1: Node(1, "Z1", "zone", 0),
                2: Node(2, "S", "shelter", 0), 3: Node(3, "J", "junction", 0),
            },
            roads=(Road(0, 0, 2, 1, 3, True), Road(1, 1, 2, 1, 4, True)),
        )
        with tempfile.TemporaryDirectory() as directory:
            nodes_path, edges_path = Path(directory) / "nodes.tsv", Path(directory) / "edges.tsv"
            nodes_path.write_text("id\tname\tkind\tpass_time\n0\tZ0\tzone\t0\n1\tZ1\tzone\t0\n2\tS\tshelter\t0\n3\tJ\tjunction\t0\n")
            edges_path.write_text("u\tv\tminutes\tcapacity\toneway\n0\t2\t1\t3\t1\n1\t2\t1\t4\t1\n")
            uncapped = io.StringIO()
            with redirect_stdout(uncapped):
                cli_main(["--nodes", str(nodes_path), "--edges", str(edges_path), "--query", "flow"])
            self.assertEqual(uncapped.getvalue().splitlines(), ["7", "0\t2\t3", "1\t2\t4"])
            capped = io.StringIO()
            with redirect_stdout(capped):
                cli_main(["--nodes", str(nodes_path), "--edges", str(edges_path), "--query", "flow", "--shelter-cap", "5"])
            self.assertEqual(capped.getvalue().splitlines(), ["5", "2\tSHELTER\t5"])
            tight_roads = io.StringIO()
            with redirect_stdout(tight_roads):
                cli_main(["--nodes", str(nodes_path), "--edges", str(edges_path), "--query", "flow", "--shelter-cap", "9"])
            self.assertEqual(tight_roads.getvalue().splitlines(), ["7", "0\t2\t3", "1\t2\t4"])

    def test_flow_counter_definitions_include_final_reachability(self) -> None:
        graph = FlowGraph()
        graph.add_edge(0, 1, 2)
        graph.add_edge(1, 2, 2)
        counters = Counters()
        result = graph.max_flow(0, 2, counters=counters)
        self.assertEqual(counters.augmenting_paths, 1)
        self.assertEqual(result.augmenting_paths, 1)
        # Two BFS searches plus the final source-reachability scan examine arcs.
        self.assertGreater(counters.residual_examinations, 0)
        self.assertEqual(result.residual_examinations, counters.residual_examinations)

    def test_class_example_all_methods_and_counters(self) -> None:
        for method in ("dfs", "bfs", "fattest"):
            graph = FlowGraph()
            for u, v, capacity in (
                ("s", "a", 4), ("s", "b", 2), ("a", "d", 4),
                ("a", "c", 2), ("a", "b", 1), ("b", "d", 2),
                ("c", "t", 3), ("d", "t", 3),
            ):
                graph.add_edge(u, v, capacity)
            counters = Counters()
            result = graph.max_flow("s", "t", method=method, counters=counters)
            self.assertEqual(result.value, 5)
            self.assertEqual(sum(capacity for _, _, capacity in result.cut_edges), 5)
            self.assertGreater(counters.relaxations, 0)
            self.assertEqual(result.relaxations, counters.relaxations)
            self.assertEqual(result.pq_ops, counters.pq_ops)
            if method == "fattest":
                self.assertGreater(counters.pq_ops, 0)
            if method == "bfs":
                self.assertGreater(counters.pq_ops, 0)

    def test_edge_cases(self) -> None:
        graph = FlowGraph()
        graph.add_node("s")
        graph.add_node("t")
        for method in ("dfs", "bfs", "fattest"):
            result = graph.max_flow("s", "t", method=method)
            self.assertEqual((result.value, result.augmentations), (0, 0))

        graph.add_edge("s", "t", 7)
        for method in ("dfs", "bfs", "fattest"):
            result = graph.max_flow("s", "t", method=method)
            self.assertEqual((result.value, result.augmentations), (7, 1))

    def test_random_graph_methods_agree(self) -> None:
        import random

        random.seed(4367)
        for _ in range(8):
            graph = make_graph(7, 16, 1, 8)
            values = [graph.max_flow(0, 6, method=method).value
                      for method in ("dfs", "bfs", "fattest")]
            self.assertEqual(values[0], values[1])
            self.assertEqual(values[1], values[2])


class ProjectQueryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.network = RoadNetwork.load(NODES, EDGES)

    def test_load(self) -> None:
        self.assertEqual(len(self.network.nodes), 20)
        self.assertEqual(len(self.network.roads), 26)
        self.assertEqual(sum(not road.oneway for road in self.network.roads), 21)
        self.assertEqual(sum(road.oneway for road in self.network.roads), 5)
        self.assertEqual(len(self.network.zones), 5)
        self.assertEqual(len(self.network.shelters), 3)
        self.assertEqual(sum(node.pass_time > 0 for node in self.network.nodes.values()), 2)

    def test_evac(self) -> None:
        expected = {}
        with (INSTANCE / "expected.evac.tsv").open(encoding="utf-8") as file:
            for zone, shelter, minutes in csv.reader(file, delimiter="\t"):
                expected[zone] = (shelter, int(minutes))

        actual = {
            self.network.nodes[row.zone_id].name: (
                self.network.nodes[row.shelter_id].name if row.shelter_id is not None else "NONE",
                row.minutes,
            )
            for row in evacuations(self.network)
        }
        self.assertEqual(actual, expected)

    def test_apsp_algorithms(self) -> None:
        expected = {}

        with (INSTANCE / "expected.apsp.tsv").open(encoding="utf-8") as file:
            for source, target, minutes in csv.reader(file, delimiter="\t"):
                expected[(int(source), int(target))] = int(minutes)
                
        for algorithm in ("dijkstra", "bellman-ford", "johnson"):
            result = all_pairs(self.network, algorithm=algorithm)
            actual = {
                (source, target): result[source][target]
                for source in sorted(result)
                for target in sorted(result[source])
                if source != target
            }
            self.assertEqual(actual, expected)

    def test_floyd_warshall(self) -> None:
        dijkstra_result = all_pairs(self.network, algorithm="dijkstra")
        floyd_result = all_pairs(self.network, algorithm="floyd-warshall")
        self.assertEqual(floyd_result, dijkstra_result)

    def test_cli_outputs(self) -> None:
        # Run the actual CLI on both shipped TSV datasets and compare its rows
        # with the expected shelter assignments and all-pairs travel times.
        for instance_name in ("tiny", "small"):
            with self.subTest(instance=instance_name):
                instance = Path(__file__).parent / "data" / "instances" / instance_name
                nodes_path = instance / "nodes.tsv"
                edges_path = instance / "edges.tsv"

                for query, expected_name in (
                    ("evac", "expected.evac.tsv"),
                    ("apsp", "expected.apsp.tsv"),
                ):
                    with self.subTest(query=query):
                        expected = (instance / expected_name).read_text(
                            encoding="utf-8"
                        ).splitlines()
                        output = io.StringIO()
                        with redirect_stdout(output):
                            exit_code = cli_main([
                                "--nodes", str(nodes_path),
                                "--edges", str(edges_path),
                                "--query", query,
                            ])

                        self.assertEqual(exit_code, 0)
                        self.assertEqual(output.getvalue().splitlines(), expected)

    def test_cli_critical_output(self) -> None:
        expected = (INSTANCE / "expected.critical.tsv").read_text(
            encoding="utf-8"
        ).splitlines()
        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = cli_main([
                "--nodes", str(NODES), "--edges", str(EDGES), "--query", "critical",
            ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(output.getvalue().splitlines(), expected)

    def test_cli_apsp_non_default_algorithms(self) -> None:
        expected = (INSTANCE / "expected.apsp.tsv").read_text(
            encoding="utf-8"
        ).splitlines()
        for algorithm in ("bellman-ford", "johnson", "floyd-warshall"):
            with self.subTest(algorithm=algorithm):
                output = io.StringIO()
                with redirect_stdout(output):
                    exit_code = cli_main([
                        "--nodes", str(NODES), "--edges", str(EDGES),
                        "--query", "apsp", "--algo", algorithm,
                    ])
                self.assertEqual(exit_code, 0)
                self.assertEqual(output.getvalue().splitlines(), expected)

    def test_cli_unreachable_evacuation_format(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            nodes_path = Path(directory) / "nodes.tsv"
            edges_path = Path(directory) / "edges.tsv"
            nodes_path.write_text(
                "id\tname\tkind\tpass_time\n"
                "0\tIsolated_Zone\tzone\t0\n"
                "1\tIsolated_Shelter\tshelter\t0\n",
                encoding="utf-8",
            )
            edges_path.write_text(
                "u\tv\tminutes\tcapacity\toneway\n", encoding="utf-8"
            )
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = cli_main([
                    "--nodes", str(nodes_path), "--edges", str(edges_path),
                    "--query", "evac",
                ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(output.getvalue().splitlines(), ["Isolated_Zone\tNONE\tINF"])

    def test_loader_rejects_malformed_files(self) -> None:
        valid_nodes = (
            "id\tname\tkind\tpass_time\n"
            "0\tZone\tzone\t0\n"
            "1\tShelter\tshelter\t0\n"
        )
        valid_edges_header = "u\tv\tminutes\tcapacity\toneway\n"
        cases = (
            ("missing node column", "id\tname\tkind\n0\tZone\tzone\n", valid_edges_header),
            ("invalid node integer", valid_nodes.replace("0\tZone", "x\tZone"), valid_edges_header),
            ("negative pass time", valid_nodes.replace("Zone\tzone\t0", "Zone\tzone\t-1"), valid_edges_header),
            ("unknown endpoint", valid_nodes, valid_edges_header + "0\t9\t2\t10\t1\n"),
            ("zero travel time", valid_nodes, valid_edges_header + "0\t1\t0\t10\t1\n"),
            (
                "duplicate road",
                valid_nodes,
                valid_edges_header + "0\t1\t2\t10\t0\n1\t0\t2\t10\t0\n",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            nodes_path = Path(directory) / "nodes.tsv"
            edges_path = Path(directory) / "edges.tsv"
            for name, nodes_text, edges_text in cases:
                with self.subTest(case=name):
                    nodes_path.write_text(nodes_text, encoding="utf-8")
                    edges_path.write_text(edges_text, encoding="utf-8")
                    with self.assertRaises(ValueError):
                        RoadNetwork.load(nodes_path, edges_path)

    def test_critical(self) -> None:
        with (INSTANCE / "expected.critical.tsv").open(encoding="utf-8") as file:
            expected = next(csv.reader(file, delimiter="\t"))
        result = critical_closure(self.network)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual([result.u, result.v, result.worst_minutes], [int(x) for x in expected])

    def test_unreachable(self) -> None:
        network = RoadNetwork(nodes=self.network.nodes, roads=())
        result = evacuations(network)
        self.assertTrue(any(row.minutes == inf for row in result))

    def test_route_tie(self) -> None:
        network = RoadNetwork(
            nodes={
                0: Node(0, "Zone", "zone", 0),
                1: Node(1, "First", "shelter", 0),
                2: Node(2, "Second", "shelter", 0),
            },
            roads=(
                Road(0, 0, 1, 4, 10, True),
                Road(1, 0, 2, 4, 10, True),
            ),
        )
        result = evacuations(network)
        self.assertEqual((result[0].shelter_id, result[0].minutes), (1, 4))

    def test_close(self) -> None:
        """
        Verify --close removes only the requested arc of a two-way road.
        """
        network = RoadNetwork.load(NODES, EDGES)
        road = network.find_road(6, 7)
        self.assertFalse(road.oneway)

        baseline = network.adjacency()
        self.assertIn(7, {arc.target for arc in baseline[6]})
        self.assertIn(6, {arc.target for arc in baseline[7]})

        closed = network.adjacency(excluded_arcs=frozenset({(road.id, 6, 7)}))
        self.assertNotIn(7, {arc.target for arc in closed[6]})
        self.assertIn(6, {arc.target for arc in closed[7]})

        output = io.StringIO()
        with redirect_stdout(output):
            exit_code = cli_main([
                "--nodes", str(NODES), "--edges", str(EDGES),
                "--query", "evac", "--close", "6", "7",
            ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(
            output.getvalue().splitlines(),
            [
                "Riverside\tJefferson_Elem\t53",
                "Millbrook\tJefferson_Elem\t41",
                "Eastgate\tRiverside_HS\t19",
                "Harbor_Flats\tJefferson_Elem\t28",
                "Old_Town\tCivic_Center\t7",
            ],
        )

    def test_critical_tie_with_inf_prefers_lower_edge_ids(self) -> None:
        """If multiple shortest-path edges produce INF, choose the lower (u, v)."""
        network = RoadNetwork(
            nodes={
                0: Node(0, "Zone", "zone", 0),
                1: Node(1, "Junction", "junction", 0),
                3: Node(3, "Shelter", "shelter", 0),
            },
            roads=(
                Road(0, 0, 1, 2, 10, True),
                Road(1, 1, 3, 3, 10, True),
            ),
        )

        result = critical_closure(network)

        self.assertIsNotNone(result)
        assert result is not None
        self.assertEqual((result.u, result.v), (0, 1))
        self.assertEqual(result.worst_minutes, inf)

        with tempfile.TemporaryDirectory() as directory:
            nodes_path = Path(directory) / "nodes.tsv"
            edges_path = Path(directory) / "edges.tsv"
            nodes_path.write_text(
                "id\tname\tkind\tpass_time\n"
                "0\tZone\tzone\t0\n"
                "1\tJunction\tjunction\t0\n"
                "3\tShelter\tshelter\t0\n",
                encoding="utf-8",
            )
            edges_path.write_text(
                "u\tv\tminutes\tcapacity\toneway\n"
                "0\t1\t2\t10\t1\n"
                "1\t3\t3\t10\t1\n",
                encoding="utf-8",
            )
            output = io.StringIO()
            with redirect_stdout(output):
                exit_code = cli_main([
                    "--nodes", str(nodes_path), "--edges", str(edges_path),
                    "--query", "critical",
                ])
            self.assertEqual(exit_code, 0)
            self.assertEqual(output.getvalue().strip(), "0\t1\tINF")

    def test_counters_file_and_non_pq_algorithms_report_zero(self) -> None:
        """Bellman-Ford and Floyd-Warshall write zero priority-queue operations."""
        original_cwd = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            os.chdir(directory)
            try:
                for algorithm in ("bellman-ford", "floyd-warshall"):
                    with self.subTest(algorithm=algorithm):
                        with redirect_stdout(io.StringIO()):
                            exit_code = cli_main([
                                "--nodes", str(NODES), "--edges", str(EDGES),
                                "--query", "apsp", "--algo", algorithm, "--counters",
                            ])
                        self.assertEqual(exit_code, 0)
                        counters = Path("counters.txt").read_text(encoding="utf-8")
                        self.assertRegex(counters, r"(?m)^Relaxations\t\d+$")
                        self.assertRegex(counters, r"(?m)^PQ_Ops\t0$")
            finally:
                os.chdir(original_cwd)

    def test_verbose_summary_goes_to_stderr(self) -> None:
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            exit_code = cli_main([
                "--nodes", str(NODES), "--edges", str(EDGES),
                "--query", "evac", "--verbose",
            ])

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(stdout.getvalue().splitlines()), len(self.network.zones))
        self.assertIn("V=20 roads=26 arcs=47", stderr.getvalue())
        self.assertRegex(stderr.getvalue(), r"relaxations=\d+ PQ_Ops=\d+")
        self.assertNotIn("V=20", stdout.getvalue())

    def test_cli_rejects_incompatible_flags_invalid_ids_and_bad_input(self) -> None:
        invalid_args = (
            ["--nodes", str(NODES), "--edges", str(EDGES), "--query", "evac", "--algo", "dijkstra"],
            ["--nodes", str(NODES), "--edges", str(EDGES), "--query", "critical", "--close", "6", "7"],
            ["--nodes", str(NODES), "--edges", str(EDGES), "--query", "evac", "--close", "6", "999"],
        )
        for argv in invalid_args:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    cli_main(argv)
                self.assertEqual(error.exception.code, 2)

        with tempfile.TemporaryDirectory() as directory:
            nodes_path = Path(directory) / "nodes.tsv"
            edges_path = Path(directory) / "edges.tsv"
            nodes_path.write_text("id\tname\tkind\n0\tZone\tzone\n", encoding="utf-8")
            edges_path.write_text("u\tv\tminutes\tcapacity\toneway\n", encoding="utf-8")
            with redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    cli_main([
                        "--nodes", str(nodes_path), "--edges", str(edges_path),
                        "--query", "evac",
                    ])
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
