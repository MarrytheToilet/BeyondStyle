import unittest

from src.data.proposition_alignment import (
    continuation_quality,
    extract_json_object,
    longest_shared_run,
    validate_content_plan,
    validate_relaxed_judgment,
)


class PropositionAlignmentTests(unittest.TestCase):
    def test_extract_json_object_with_fence(self):
        value = extract_json_object('```json\n{"language": "en"}\n```')
        self.assertEqual(value, {"language": "en"})

    def test_longest_shared_run_uses_language_units(self):
        self.assertEqual(
            longest_shared_run(
                "A small red fox crossed the quiet road.",
                "Yesterday the red fox crossed the quiet field.",
                "en",
            ),
            5,
        )
        self.assertEqual(
            longest_shared_run("研究团队公布结果", "该研究团队公布数据", "zh"),
            6,
        )

    def test_valid_plan_is_canonicalized(self):
        plan, issues = validate_content_plan(
            {
                "language": "en",
                "propositions": [
                    "The committee met on Tuesday.",
                    "It approved the proposal.",
                    "Implementation will begin in June.",
                ],
                "required_entities": ["the committee", "June"],
                "discourse_endpoint": "The decision and its timing are stated.",
            },
            language="en",
            human_continuation=(
                "On Tuesday a vote was held. Members backed the proposal. "
                "The change is due to take effect during June."
            ),
        )
        self.assertEqual(issues, [])
        self.assertEqual(plan["language"], "en")

    def test_plan_rejects_long_verbatim_copy(self):
        _, issues = validate_content_plan(
            {
                "language": "en",
                "propositions": [
                    (
                        "one two three four five six seven eight nine ten "
                        "eleven twelve thirteen fourteen fifteen sixteen "
                        "seventeen eighteen nineteen"
                    ),
                    "Another fact is stated.",
                    "A final event occurs.",
                ],
                "required_entities": [],
                "discourse_endpoint": "The account ends.",
            },
            language="en",
            human_continuation=(
                "one two three four five six seven eight nine ten eleven "
                "twelve thirteen fourteen fifteen sixteen seventeen eighteen "
                "nineteen twenty"
            ),
        )
        self.assertTrue(any(
            issue.startswith("proposition_copies_human_wording:")
            for issue in issues
        ))

    def test_continuation_is_rejected_not_rewritten(self):
        generated = "Here is the continuation: A result occurred."
        quality = continuation_quality(
            generated,
            language="en",
            human_continuation="A different account of the result.",
            prefix="Earlier context.",
            target_units=8,
        )
        self.assertFalse(quality["accepted"])
        self.assertIn("meta_preamble", quality["issues"])
        self.assertEqual(generated, "Here is the continuation: A result occurred.")

    def test_moderate_plan_overlap_is_diagnostic_not_a_hard_rejection(self):
        quality = continuation_quality(
            (
                "The committee approved the proposal after a public meeting "
                "and implementation will begin during the month of June while "
                "the secretariat prepares detailed guidance for local offices. "
                "Separately, regional teams will test the new process and "
                "publish their observations later in the year."
            ),
            language="en",
            human_continuation=(
                "Members backed the measure following public discussion. "
                "The change starts in June."
            ),
            prefix="Earlier context describes the committee.",
            content_plan_text=(
                "The committee approved the proposal after a public meeting "
                "and implementation will begin during the month of June while "
                "the secretariat prepares detailed guidance for local offices."
            ),
            target_units=40,
        )
        self.assertTrue(quality["accepted"])
        self.assertTrue(quality["plan_serialization_diagnostic"])

    def test_extreme_plan_serialization_is_rejected(self):
        plan = (
            "The committee approved the proposal after a public meeting "
            "and implementation will begin during the month of June while "
            "the secretariat prepares detailed guidance for local offices."
        )
        quality = continuation_quality(
            plan,
            language="en",
            human_continuation=(
                "Members backed the measure following public discussion. "
                "The change starts in June."
            ),
            prefix="Earlier context describes the committee.",
            content_plan_text=plan,
            target_units=26,
        )
        self.assertFalse(quality["accepted"])
        self.assertIn("extreme_plan_serialization", quality["issues"])

    def test_relaxed_judgment_allows_minor_omissions(self):
        judgment, issues = validate_relaxed_judgment(
            {
                "core_content_coverage": "mostly_complete",
                "missing_core_facts": ["A minor date was omitted."],
                "major_unsupported_claims": [],
                "major_contradictions": [],
                "verdict": "pass",
                "brief_reason": "The central account is preserved.",
            },
        )
        self.assertEqual(issues, [])
        self.assertEqual(
            judgment["core_content_coverage"],
            "mostly_complete",
        )


if __name__ == "__main__":
    unittest.main()
