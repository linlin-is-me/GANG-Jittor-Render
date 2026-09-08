"""Focused CPU tests; no claim of CUDA or framework integration coverage."""
import unittest
from unittest.mock import patch
import numpy as np
import tempfile
from pathlib import Path
from types import SimpleNamespace
from collections import namedtuple
from inference_raster_benchmark import RasterCapture, load_packet
import benchmark_inference_versions as bench
from compare_inference_benchmarks import image_metrics


class BenchmarkTests(unittest.TestCase):
    def test_shared_raster_roundtrip(self):
        Settings = namedtuple('Settings', 'image_height bg')
        class Raster:
            def __init__(self, raster_settings):
                self.settings = raster_settings
            def __call__(self, **kwargs):
                return kwargs
        renderer = SimpleNamespace(GaussianRasterizer=Raster, GaussianRasterizationSettings=Settings)
        capture = RasterCapture(renderer, lambda x: x)
        renderer.GaussianRasterizer(Settings(10, np.zeros(3, np.float32)))(
            means3D=np.ones((7, 3), np.float32), shs=None)
        with tempfile.TemporaryDirectory() as path:
            capture.save(Path(path), 'view')
            renderer.GaussianRasterizer = Raster
            raster, kwargs, count = load_packet(Path(path), 'view', renderer, np.copy, 'jittor')
            self.assertEqual(count, 7)
            self.assertTrue(kwargs['inference_only'])
            self.assertTrue(kwargs['return_aux'])
            self.assertIsNone(kwargs['shs'])
            np.testing.assert_array_equal(kwargs['means3D'], np.ones((7, 3), np.float32))
            self.assertEqual(raster.settings.image_height, 10)

    def test_ssim_identity(self):
        image = np.random.default_rng(42).random((3, 20, 25))
        self.assertAlmostEqual(image_metrics(image, image)['ssim_clamped'], 1.)

    def test_sync_and_copy_order(self):
        events = []
        def render():
            events.append('render')
            return {'render': np.ones((3, 2, 2))}
        def host(x):
            events.append('copy')
            return x
        with patch.object(bench.time, 'perf_counter', side_effect=[1., 1.01, 1.015]):
            _, gpu, total = bench.measure(render, lambda: events.append('sync'), host)
        self.assertEqual(events, ['sync', 'render', 'sync', 'copy', 'sync'])
        self.assertAlmostEqual(gpu, 10)
        self.assertAlmostEqual(total, 15)

    def test_fps_uses_total_time(self):
        self.assertEqual(bench.summary([1, 3])['fps'], 500)

    def test_invalid_timings(self):
        for values in ([], [0], [-1], [np.nan]):
            with self.assertRaises(ValueError):
                bench.summary(values)

    def test_bad_output(self):
        with self.assertRaises(FloatingPointError):
            bench.measure(lambda: {'render': np.array([np.inf])}, lambda: None, lambda x: x)

    def test_dtype_mismatch_not_hidden(self):
        with self.assertRaises(ValueError):
            bench.assert_equal(np.zeros(2, np.float64), np.zeros(2, np.float32), 'x')

    def test_camera_origin(self):
        camera = {'rotation': np.eye(3), 'position': [1, 2, 3],
                  'width': 100, 'height': 80, 'fx': 50, 'fy': 40}
        arrays, _, _ = bench.camera_arrays(camera)
        origin = np.array([1, 2, 3, 1], np.float32) @ arrays['world_view_transform']
        np.testing.assert_array_equal(origin, [0, 0, 0, 1])
        self.assertEqual(arrays['full_proj_transform'].dtype, np.float32)

    def test_pt_pbr_mapping(self):
        payload = [np.ones(1, np.float32) for _ in range(22)]
        payload[12] = 2.
        state = payload[:16] + [None, None] + payload[19:22]
        for j, p in ((13, 13), (14, 14), (15, 15), (19, 18), (20, 20), (21, 19)):
            payload[j] = {'0.weight': np.array([j], np.float32)}
            state[p] = payload[j].copy()
        bench.verify_pt_state(state, payload, lambda x: x)
        state[19], state[20] = state[20], state[19]
        with self.assertRaises(ValueError):
            bench.verify_pt_state(state, payload, lambda x: x)


if __name__ == '__main__':
    unittest.main()
