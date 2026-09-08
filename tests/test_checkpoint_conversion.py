"""CPU conversion/loader contract tests; no Jittor or GPU required."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('convert', ROOT/'tools/convert_pytorch_checkpoint.py')
converter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(converter)


def fixture():
    n, k, f = 2, 2, 4
    def mlp(outputs, marker):
        return {key: np.full(shape, marker, np.float32) for key, shape in
                [('0.weight',(f,f+3)), ('0.bias',(f,)),
                 ('2.weight',(outputs,f)), ('2.bias',(outputs,))]}
    items = [None]*21
    for idx, shape in [(0,(n,3)),(1,(n,1)),(2,(n,k,3)),(3,(n,f)),
                       (5,(n,6)),(6,(n,4)),(7,(n,1))]:
        items[idx] = np.zeros(shape, np.float32)
    items[12] = 1.0
    for idx, outputs in [(13,k),(14,7*k),(15,3*k),(18,3*k),(19,k),(20,k)]:
        items[idx] = mlp(outputs, idx)
    light = {'base':np.zeros((6,16,16,3),np.float32),
             'lgtSGs':np.ones((16,10),np.float32),
             'specular_reflectance':np.ones((1,3),np.float32),
             'sg_roughness':np.ones((1,1),np.float32)}
    meta = dict(standard_dist=10., voxel_size=.01, levels=10, init_level=5,
                fork=2, base_layer=10, dist2level='round', progressive=True, extend=1.1)
    return (items,40000), light, meta


class ConversionTests(unittest.TestCase):
    def test_roundtrip_through_public_loader(self):
        args = fixture()
        arrays = converter.convert_payload(*args)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.npz'
            converter.write_npz(path, arrays)
            with np.load(path, allow_pickle=False) as loaded:
                for key, value in arrays.items():
                    np.testing.assert_array_equal(value, loaded[key])
            tree = ast.parse((ROOT/'tools/relight_envmap.py').read_text(encoding='utf-8'))
            fn = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='load_named_npz')
            ns = {'np':np}
            exec(compile(ast.Module(body=[fn],type_ignores=[]),'public_loader','exec'),ns)
            model, light, meta = ns['load_named_npz'](path)
            np.testing.assert_array_equal(model[19]['0.weight'], args[0][0][18]['0.weight'])
            np.testing.assert_array_equal(model[20]['0.weight'], args[0][0][19]['0.weight'])
            np.testing.assert_array_equal(model[21]['0.weight'], args[0][0][20]['0.weight'])
            self.assertEqual(float(meta['standard_dist']),10.)
            self.assertEqual(int(meta['levels']),10)
            self.assertEqual(light['base'].shape,(6,16,16,3))
            self.assertIsNone(model[11])

    def test_explicit_alternative_pbr_order(self):
        arrays = converter.convert_payload(*fixture(),pbr_order='albedo-roughness-metallic')
        self.assertTrue((arrays['mlp_roughness_w1']==19).all())
        self.assertTrue((arrays['mlp_matallic_w1']==20).all())

    def test_rejects_unsupported_layouts(self):
        for mutate in (lambda a: a[0][0].pop(),
                       lambda a: a[0][0].__setitem__(16,{}),
                       lambda a: a[0][0].__setitem__(2,np.zeros((4,3),np.float32)),
                       lambda a: a[0][0][14].__setitem__('0.weight',np.zeros((4,8),np.float32))):
            args = fixture(); mutate(args)
            with self.assertRaises(ValueError): converter.convert_payload(*args)

    def test_rejects_missing_metadata_nonfinite_and_precision_loss(self):
        args = fixture(); del args[2]['standard_dist']
        with self.assertRaisesRegex(ValueError,'standard_dist'): converter.convert_payload(*args)
        args = fixture(); args[0][0][0][0,0] = np.nan
        with self.assertRaisesRegex(ValueError,'finite'): converter.convert_payload(*args)
        args = fixture(); args[0][0][0] = args[0][0][0].astype(np.float64)
        with self.assertRaisesRegex(ValueError,'FP32'): converter.convert_payload(*args)

    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'model.npz'
            converter.write_npz(path, converter.convert_payload(*fixture()))
            before = path.read_bytes()
            with self.assertRaises(FileExistsError): converter.write_npz(path,{})
            self.assertEqual(before,path.read_bytes())

    def test_light_pickle_requires_explicit_trust(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'light.npy'
            np.save(path,fixture()[1])
            with self.assertRaisesRegex(ValueError,'trust-pickle'): converter.load_light(path)
            self.assertEqual(set(converter.load_light(path,True)),set(fixture()[1]))

    def test_entries_forward_metadata(self):
        for file in ('relight_envmap.py','render_learned_light.py'):
            tree = ast.parse((ROOT/'tools'/file).read_text(encoding='utf-8'))
            calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call)
                     and isinstance(n.func,ast.Attribute) and n.func.attr=='restore_numpy']
            self.assertTrue(calls)
            self.assertTrue(all(any(k.arg=='metadata' for k in c.keywords) for c in calls))


if __name__ == '__main__': unittest.main()
