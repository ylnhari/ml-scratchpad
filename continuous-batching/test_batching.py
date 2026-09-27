"""Synthetic checks for scheduler correctness and invalid workload handling."""

import unittest

from batching import (
    GPT,
    Request,
    occupancy_continuous,
    occupancy_static,
    run_continuous,
    run_static,
    standalone_greedy,
    verify_equivalence,
)


class ContinuousBatchingTests(unittest.TestCase):
    def test_zero_and_negative_generation_lengths_fail_before_scheduling(self):
        for scheduler in (run_static, run_continuous):
            for length in (0, -1):
                with self.subTest(scheduler=scheduler.__name__, length=length):
                    request = Request(0, [1], length)
                    with self.assertRaisesRegex(ValueError, "gen_len must be a positive integer"):
                        scheduler(None, [request], B=1)

        for occupancy in (occupancy_static, occupancy_continuous):
            for length in (0, -1):
                with self.subTest(occupancy=occupancy.__name__, length=length):
                    with self.assertRaisesRegex(ValueError, "must be a positive integer"):
                        occupancy([length], B=1)

    def test_empty_workloads_keep_the_batch_axis(self):
        self.assertEqual(occupancy_static([], B=3).shape, (0, 3))
        self.assertEqual(occupancy_continuous([], B=3).shape, (0, 3))

    def test_continuous_scheduler_accepts_one_shot_iterables(self):
        model = GPT(vocab_size=8, block_size=12, n_embd=8, n_head=2, n_layer=1, seed=4)
        requests = [
            Request(0, [1, 2], 2),
            Request(1, [3, 4], 3),
            Request(2, [5, 6], 1),
        ]
        expected = {request.rid: standalone_greedy(model, request) for request in requests}
        actual, _, useful_steps = run_continuous(model, (r for r in requests), B=2)

        self.assertEqual(actual, expected)
        self.assertEqual(useful_steps, sum(request.gen_len for request in requests))

    def test_both_schedulers_match_untrained_tiny_model_greedy_outputs(self):
        model = GPT(vocab_size=8, block_size=12, n_embd=8, n_head=2, n_layer=1, seed=9)
        requests = [
            Request(0, [1, 2], 2),
            Request(1, [3, 4], 3),
            Request(2, [5, 6], 1),
        ]

        self.assertTrue(verify_equivalence(model, requests, B=2))


if __name__ == "__main__":
    unittest.main()
