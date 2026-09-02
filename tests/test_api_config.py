import os
import unittest
from unittest.mock import patch

from src.config import get_api_base_url, get_api_key, get_api_models


class ApiConfigTests(unittest.TestCase):
    def test_canonical_api_names(self):
        with patch.dict(
            os.environ,
            {
                "API_KEY": "canonical-key",
                "API_BASE_URL": "https://example.invalid/v1",
                "API_MODELS": "model-a, model-b",
            },
            clear=True,
        ):
            self.assertEqual(get_api_key(), "canonical-key")
            self.assertEqual(
                get_api_base_url(),
                "https://example.invalid/v1",
            )
            self.assertEqual(get_api_models(), ["model-a", "model-b"])

    def test_noncanonical_aliases_are_ignored(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "unused-key",
                "OPENAI_BASE_URL": "https://unused.invalid/v1",
                "API_MODEL": "only-model",
            },
            clear=True,
        ):
            self.assertIsNone(get_api_key(required=False))
            self.assertEqual(
                get_api_base_url(),
                "https://api.openai.com/v1",
            )
            self.assertEqual(get_api_models(), ["only-model"])


if __name__ == "__main__":
    unittest.main()
