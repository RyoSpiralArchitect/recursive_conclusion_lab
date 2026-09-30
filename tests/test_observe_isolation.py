import unittest

from recursive_conclusion_lab import (
    ConclusionMode,
    DelayedMentionLeakPolicy,
    DelayedMentionMode,
    ExperimentConfig,
    RecursiveConclusionSession,
    build_adapter,
)


class ObserveIsolationTests(unittest.TestCase):
    def run_one_turn(self, delayed_mention_mode: DelayedMentionMode):
        config = ExperimentConfig(
            memory_every=0,
            conclusion_every=1,
            conclusion_mode=ConclusionMode.OBSERVE,
            delayed_mention_every=0,
            delayed_mention_mode=delayed_mention_mode,
            delayed_mention_leak_policy=DelayedMentionLeakPolicy.ON,
            latent_convergence_every=0,
            deferred_intent_every=0,
        )
        session = RecursiveConclusionSession(
            adapter=build_adapter("dummy", "dummy-v1"),
            config=config,
        )
        return session.user_turn("Help me choose a conclusion for this report.")

    def test_observe_only_keeps_conclusion_probe_out_of_reply_prompt(self):
        result = self.run_one_turn(DelayedMentionMode.OBSERVE)

        self.assertTrue(result["conclusion_probe"])
        self.assertEqual(result["suppressed_delayed_mentions"], [])
        self.assertEqual(result["injected_delayed_mentions"], [])
        self.assertNotIn("Delayed mention targets currently remain latent", result["system_prompt"])
        self.assertNotIn(result["latest_conclusion_line"].removeprefix("CONCLUSION: "), result["system_prompt"])

    def test_soft_fire_retains_leak_suppression(self):
        result = self.run_one_turn(DelayedMentionMode.SOFT_FIRE)

        self.assertTrue(result["conclusion_probe"])
        self.assertEqual(len(result["suppressed_delayed_mentions"]), 1)
        self.assertIn("Delayed mention targets currently remain latent", result["system_prompt"])
        self.assertIn(result["suppressed_delayed_mentions"][0]["text"], result["system_prompt"])


if __name__ == "__main__":
    unittest.main()
