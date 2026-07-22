from __future__ import annotations

import copy
import random
import unittest

from fastwam_ood_eval.diagnostics.rng_isolation import RngIsolation, RngSnapshot


class _State:
    def __init__(self, value):
        self.value = copy.deepcopy(value)

    def clone(self):
        return _State(self.value)

    def __eq__(self, other):
        return isinstance(other, _State) and self.value == other.value


class _FakeNumpyRandom:
    def __init__(self):
        self.state = ("MT19937", [7, 11, 13], 3)
        self.seed_calls: list[int] = []

    def get_state(self):
        return copy.deepcopy(self.state)

    def set_state(self, state):
        self.state = copy.deepcopy(state)

    def seed(self, seed):
        self.seed_calls.append(seed)
        self.state = ("seeded", [seed], 0)


class _FakeNumpy:
    def __init__(self):
        self.random = _FakeNumpyRandom()


class _FakeCuda:
    def __init__(self):
        self.states = [_State("cuda:0"), _State("cuda:1")]
        self.seed_calls: list[int] = []

    def is_available(self):
        return True

    def get_rng_state_all(self):
        return [state.clone() for state in self.states]

    def set_rng_state_all(self, states):
        self.states = [state.clone() for state in states]

    def manual_seed_all(self, seed):
        self.seed_calls.append(seed)
        self.states = [_State(("seeded", seed, index)) for index in range(len(self.states))]


class _FakeTorch:
    def __init__(self):
        self.cpu_state = _State("cpu")
        self.cuda = _FakeCuda()
        self.seed_calls: list[int] = []

    def get_rng_state(self):
        return self.cpu_state.clone()

    def set_rng_state(self, state):
        self.cpu_state = state.clone()

    def manual_seed(self, seed):
        self.seed_calls.append(seed)
        self.cpu_state = _State(("seeded", seed))


class RngIsolationTests(unittest.TestCase):
    def setUp(self):
        self._python_state = random.getstate()

    def tearDown(self):
        random.setstate(self._python_state)

    def test_snapshot_captures_and_restores_cpu_and_every_cuda_device(self):
        fake_numpy = _FakeNumpy()
        fake_torch = _FakeTorch()
        expected_numpy = copy.deepcopy(fake_numpy.random.state)
        expected_cpu = fake_torch.cpu_state.clone()
        expected_cuda = [state.clone() for state in fake_torch.cuda.states]

        snapshot = RngSnapshot.capture(
            numpy_module=fake_numpy,
            torch_module=fake_torch,
        )
        fake_numpy.random.state = ("changed", [], 0)
        fake_torch.cpu_state = _State("changed")
        fake_torch.cuda.states = [_State("changed:0"), _State("changed:1")]

        snapshot.restore()

        self.assertEqual(fake_numpy.random.state, expected_numpy)
        self.assertEqual(fake_torch.cpu_state, expected_cpu)
        self.assertEqual(fake_torch.cuda.states, expected_cuda)

    def test_context_seeds_diagnostics_then_restores_all_streams_on_exception(self):
        fake_numpy = _FakeNumpy()
        fake_torch = _FakeTorch()
        original_numpy = copy.deepcopy(fake_numpy.random.state)
        original_cpu = fake_torch.cpu_state.clone()
        original_cuda = [state.clone() for state in fake_torch.cuda.states]

        random.seed(871)
        state_before_probe = random.getstate()
        oracle = random.Random()
        oracle.setstate(state_before_probe)
        expected_next_python_value = oracle.random()
        diagnostic_seed = 2**32 + 19

        with self.assertRaisesRegex(ValueError, "probe failed"):
            with RngIsolation(
                diagnostic_seed,
                numpy_module=fake_numpy,
                torch_module=fake_torch,
            ):
                self.assertEqual(fake_numpy.random.seed_calls, [19])
                self.assertEqual(fake_torch.seed_calls, [diagnostic_seed])
                self.assertEqual(fake_torch.cuda.seed_calls, [diagnostic_seed])
                random.random()
                fake_numpy.random.state = ("advanced", [1], 0)
                fake_torch.cpu_state = _State("advanced")
                fake_torch.cuda.states = [_State("advanced:0"), _State("advanced:1")]
                raise ValueError("probe failed")

        self.assertEqual(random.random(), expected_next_python_value)
        self.assertEqual(fake_numpy.random.state, original_numpy)
        self.assertEqual(fake_torch.cpu_state, original_cpu)
        self.assertEqual(fake_torch.cuda.states, original_cuda)

    def test_context_works_when_numpy_and_torch_are_unavailable(self):
        random.seed(99)
        expected_state = random.getstate()
        oracle = random.Random()
        oracle.setstate(expected_state)
        expected_next = oracle.random()

        with RngIsolation(123, numpy_module=None, torch_module=None):
            random.random()

        self.assertEqual(random.random(), expected_next)

    def test_boolean_seed_is_rejected(self):
        with self.assertRaisesRegex(TypeError, "not bool"):
            RngIsolation(True, numpy_module=None, torch_module=None)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
