import unittest
from collections import Counter

from scripts.synthesize_paired_data import (
    balanced_cross_sequence,
    role_models,
)


class SynthesisScheduleTests(unittest.TestCase):
    def assert_balanced(self, left_count, right_count, total):
        left = [f"s{i}" for i in range(left_count)]
        right = [f"m{i}" for i in range(right_count)]
        pairs = balanced_cross_sequence(
            left,
            right,
            total,
            seed=8236,
            label=f"test:{left_count}:{right_count}:{total}",
        )
        self.assertEqual(len(pairs), total)
        left_margin = Counter(item[0] for item in pairs)
        right_margin = Counter(item[1] for item in pairs)
        cells = Counter(pairs)
        self.assertLessEqual(
            max(left_margin.values()) - min(left_margin.values()),
            1,
        )
        self.assertLessEqual(
            max(right_margin.values()) - min(right_margin.values()),
            1,
        )
        all_cells = [
            cells[(left_item, right_item)]
            for left_item in left
            for right_item in right
        ]
        self.assertLessEqual(max(all_cells) - min(all_cells), 1)

    def test_small_schedules(self):
        self.assert_balanced(4, 4, 50)
        self.assert_balanced(3, 4, 50)

    def test_batch_schedules(self):
        self.assert_balanced(4, 4, 500)
        self.assert_balanced(3, 4, 500)

    def test_all_three_api_roles_are_distinct(self):
        models = [
            "qwen3.6-flash",
            "deepseek-v4-flash",
            "glm-5.2",
            "MiniMax-M2.5",
        ]
        for generator in models:
            planner, judge = role_models(generator, models)
            assigned = [generator, planner, judge]
            self.assertEqual(len(set(assigned)), 3)
            self.assertNotIn("deepseek", planner.casefold())


if __name__ == "__main__":
    unittest.main()
