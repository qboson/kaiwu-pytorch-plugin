"""DPT metrics use the root-reachable observations in synthetic neighbor graphs."""
import importlib
import importlib.util
import math
import unittest


def pairwise_tau_b(left, right):
    """Independent pair-count reference; infinities compare as ordered ties."""
    concordant = discordant = left_ties = right_ties = 0
    for first in range(len(left)):
        for second in range(first):
            x = (left[first] > left[second]) - (left[first] < left[second])
            y = (right[first] > right[second]) - (right[first] < right[second])
            if x == 0 and y == 0:
                continue
            if x == 0:
                left_ties += 1
            elif y == 0:
                right_ties += 1
            elif x == y:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt(
        (concordant + discordant + left_ties)
        * (concordant + discordant + right_ties)
    )
    return (concordant - discordant) / denominator


class TestFiniteDptMetrics(unittest.TestCase):
    observations = []

    @classmethod
    def setUpClass(cls):
        try:
            cls.benchmark = importlib.import_module(
                "example.qvae_cell.evaluate_benchmark"
            )
        except ModuleNotFoundError as error:
            if error.name in {"anndata", "scanpy"} and importlib.util.find_spec(
                error.name
            ) is None:
                raise unittest.SkipTest(
                    f"Optional benchmark dependency {error.name!r} is not installed"
                ) from error
            raise
        cls.np = importlib.import_module("numpy")
        cls.pd = importlib.import_module("pandas")
        cls.anndata = importlib.import_module("anndata")

    def make_data(self, connected=False, with_pca=True):
        np = self.np
        data = self.anndata.AnnData(np.zeros((120, 3), dtype=np.float64))
        data.obs_names = [f"cell_{index:03d}" for index in range(120)]
        data.obs["label"] = ["root"] + ["other"] * 119

        def representation(groups):
            result = np.empty((120, 3), dtype=np.float64)
            for component, members in enumerate(groups):
                t = np.arange(len(members), dtype=np.float64) / (len(members) - 1)
                result[members] = np.column_stack(
                    (
                        t + 50 * component,
                        0.08 * np.sin(2 * np.pi * t) + 0.013 * t * t,
                        0.04 * np.cos(3 * np.pi * t) + 0.007 * t * t * t,
                    )
                )
            return result

        if connected:
            data.obsm["X_qvae"] = representation([list(range(120))])
            if with_pca:
                data.obsm["X_pca"] = data.obsm["X_qvae"][:, [1, 0, 2]].copy()
        else:
            data.obsm["X_qvae"] = representation(
                [list(range(60)), list(range(60, 120))]
            )
            if with_pca:
                data.obsm["X_pca"] = representation(
                    [
                        [0] + list(range(39, 0, -1)) + list(range(60, 80)),
                        list(range(40, 60)) + list(range(80, 120)),
                    ]
                )
        return data

    def reference(self, data, key, expected_component_sizes):
        owned = data.copy()
        pseudotime = self.benchmark.run_dpt(
            owned, key, "root", "label", 8, "observed_pseudotime"
        )
        graph = owned.obsp["connectivities"].tocsr()
        adjacency = []
        for row in range(len(data)):
            start, end = graph.indptr[row:row + 2]
            adjacency.append(
                {int(column) for column, weight in zip(
                    graph.indices[start:end], graph.data[start:end]
                ) if weight > 0}
            )

        def reachable(seed):
            visited, pending = {seed}, [seed]
            while pending:
                row = pending.pop()
                for column in adjacency[row] - visited:
                    visited.add(column)
                    pending.append(column)
            return visited

        root_component = reachable(0)
        unseen, components = set(range(len(data))), []
        while unseen:
            component = reachable(min(unseen))
            components.append(component)
            unseen -= component
        self.assertEqual(sorted(map(len, components)), expected_component_sizes)
        values = [float(value) for value in pseudotime]
        self.assertTrue(all(not math.isnan(value) for value in values))
        finite = {index for index, value in enumerate(values) if math.isfinite(value)}
        self.assertEqual(finite, root_component)
        self.assertTrue(all(values[index] == math.inf for index in set(range(120)) - finite))
        record = {
            "representation": key,
            "component_sizes": sorted(map(len, components)),
            "root_reachable": sorted(root_component),
            "finite_indices": sorted(finite),
            "pseudotime": [value if math.isfinite(value) else "inf" for value in values],
            "positive_graph_neighbors": [sorted(neighbors) for neighbors in adjacency],
        }
        return values, finite, record

    def evaluate_without_input_mutation(self, data):
        before = data.copy()
        result = self.benchmark.evaluate_dpt(data, "X_qvae", "label", "root", 8)
        self.np.testing.assert_array_equal(data.X, before.X)
        self.pd.testing.assert_frame_equal(data.obs, before.obs)
        self.assertEqual(dict(data.uns), dict(before.uns))
        self.assertEqual(set(data.obsm), set(before.obsm))
        self.assertEqual(set(data.obsp), set(before.obsp))
        for key in data.obsm:
            self.np.testing.assert_array_equal(data.obsm[key], before.obsm[key])
        self.assertEqual(len(result), 1)
        return result.iloc[0]

    def check_comparison(self, connected):
        data = self.make_data(connected=connected)
        sizes = [120] if connected else [60, 60]
        left, left_finite, left_record = self.reference(data, "X_qvae", sizes)
        right, right_finite, right_record = self.reference(data, "X_pca", sizes)
        common = sorted(left_finite & right_finite)
        self.assertEqual(len(common), 120 if connected else 40)
        expected_tau = pairwise_tau_b(
            [left[index] for index in common], [right[index] for index in common]
        )
        all_tau = pairwise_tau_b(left, right)
        if not connected:
            self.assertGreaterEqual(abs(all_tau - expected_tau), 0.05)
        actual = self.evaluate_without_input_mutation(data)
        self.observations.append({
            "case": "connected" if connected else "disconnected",
            "left": left_record, "right": right_record,
            "finite_intersection": common, "reference_tau": expected_tau,
            "all_observations_tau": all_tau,
            "actual_count": int(actual["n_cells_with_dpt"]),
            "actual_tau": float(actual["kendall_tau_vs_X_pca"]),
            "actual_trajectory": float(actual["trajectory_conservation_vs_X_pca"]),
        })
        with self.subTest(invariant="finite count"):
            self.assertEqual(int(actual["n_cells_with_dpt"]), len(left_finite))
        with self.subTest(invariant="finite tau"):
            self.assertAlmostEqual(float(actual["kendall_tau_vs_X_pca"]), expected_tau, delta=1e-12)
        with self.subTest(invariant="finite trajectory"):
            self.assertAlmostEqual(float(actual["trajectory_conservation_vs_X_pca"]), (expected_tau + 1) / 2, delta=1e-12)

    def test_disconnected_representations_use_common_finite_cells(self):
        self.check_comparison(connected=False)

    def test_connected_finite_metrics_are_preserved(self):
        self.check_comparison(connected=True)

    def test_without_pca_reports_only_finite_count(self):
        data = self.make_data(with_pca=False)
        _, finite, record = self.reference(data, "X_qvae", [60, 60])
        actual = self.evaluate_without_input_mutation(data)
        self.observations.append({
            "case": "without_pca", "reference": record,
            "actual_count": int(actual["n_cells_with_dpt"]),
            "actual_columns": list(actual.index),
        })
        self.assertNotIn("kendall_tau_vs_X_pca", actual.index)
        self.assertNotIn("trajectory_conservation_vs_X_pca", actual.index)
        self.assertEqual(int(actual["n_cells_with_dpt"]), len(finite))


if __name__ == "__main__":
    unittest.main()
