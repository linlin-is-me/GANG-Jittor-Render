"""CPU checks of the released benchmark configuration and archived measurements."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'docs/benchmarks/20260908'


class MeasuredReleaseTests(unittest.TestCase):
    def test_wrapper_uses_measured_parameters(self):
        spec=importlib.util.spec_from_file_location('measured',ROOT/'tools/render_measured.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        cmd=module.command(SimpleNamespace(weights=Path('weights'),output=Path('out'),
                                           mode='full',rounds=5,smoke=False))
        for flag,value in [('--sg-reduce-backend','vector3_cuda'),('--offset-layout','flattened'),
                           ('--warmup','2'),('--rounds','5'),('--framework','jittor'),('--mode','full')]:
            self.assertEqual(cmd[cmd.index(flag)+1],value)
        self.assertEqual(Path(cmd[cmd.index('--source-root')+1]),ROOT)

    def test_raw_timings_reproduce_summary(self):
        summary=json.loads((DATA/'three_run_summary.json').read_text())
        for family in ('local','pytorch'):
            rows=[]
            for run in ('A','B','C'):
                part=[json.loads(line) for line in (DATA/f'full-{family}-{run}.jsonl').read_text().splitlines()]
                self.assertEqual(len(part),120)
                self.assertEqual(len({r['view'] for r in part}),24)
                rows.extend(part)
            values=np.array([r['forward_ms'] for r in rows])
            self.assertTrue(np.isfinite(values).all())
            expected=next(r for r in summary['timings'] if r['family']==family and r['mode']=='full')
            self.assertAlmostEqual(float(values.mean()),expected['mean_ms'],places=8)
            self.assertEqual(len(values),expected['count'])

    def test_contract_and_snapshot_files_exist(self):
        contract=json.loads((DATA/'garden_res4.json').read_text())
        self.assertEqual(contract['resolution'],4)
        self.assertEqual(contract['anchor_count'],597027)
        self.assertEqual(len(contract['views']),24)
        inventory=json.loads((DATA/'source_inventory.json').read_text())
        for entry in inventory['files']:
            self.assertTrue((ROOT/entry['path']).is_file(),entry['path'])
        for path in ('train.py','tools/train_first.py','tools/train_second.py'):
            self.assertFalse((ROOT/path).exists())

    def test_stable_build_dependencies_exist(self):
        path=ROOT/'submodules/light_gaussian'
        spec=importlib.util.spec_from_file_location('provenance',path/'rasterizer_provenance.py')
        module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        for relative in module.CONTROLLED_SOURCES:
            self.assertTrue((path/relative).is_file(),relative)
        self.assertTrue((ROOT/'tools/write_rasterizer_manifest.py').is_file())


if __name__=='__main__': unittest.main()
