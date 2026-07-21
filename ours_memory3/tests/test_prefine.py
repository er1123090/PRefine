from __future__ import annotations

import json
import unittest

from ours_memory3.prefine import baseline_memory_block, build_prefine_latent


class PrefineTests(unittest.TestCase):
    def test_generation_verification_state_machine_is_shared_baseline_only(self) -> None:
        calls: list[tuple[int, int, float, bool]] = []

        def complete(messages, seed, max_tokens, temperature, json_object):
            calls.append((seed, max_tokens, temperature, json_object))
            if messages[0]["role"] == "system" and "Verification" in messages[0]["content"]:
                return json.dumps({"valid": True, "feedback": ""})
            return json.dumps({"reasoning": "stable", "implicit_pref": "prefer low cost"})

        history = {
            "sessions": [
                {
                    "session_index": 0,
                    "dialogue": [{"role": "user", "message": "hello"}],
                    "api_calls": ['Restaurant(price="cheap")'],
                },
                {
                    "session_index": 1,
                    "dialogue": [{"role": "user", "message": "again"}],
                    "api_calls": ['Restaurant(price="cheap")'],
                },
            ]
        }
        latent = build_prefine_latent(
            history=history,
            complete=complete,
            seed=7,
            maximum_attempts=2,
        )
        block = baseline_memory_block(latent=latent, history=history)
        self.assertEqual(latent["implicit_pref"], "prefer low cost")
        self.assertEqual(len(calls), 4)
        self.assertIn("[Implicit Preferences]", block)
        self.assertIn("[Past API History]", block)
        self.assertIn('Restaurant(price="cheap")', block)


if __name__ == "__main__":
    unittest.main()

