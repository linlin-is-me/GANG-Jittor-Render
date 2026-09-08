"""Dependency-free checks for the local SG opt-in boundary."""
import ast
import argparse
from pathlib import Path
from types import SimpleNamespace
import unittest


class SGContractTests(unittest.TestCase):
    def setUp(self):
        self.tree=ast.parse((Path(__file__).resolve().parents[1]/'scene/NVDIFFREC/light.py').read_text(encoding='utf-8'))
        self.cls=next(n for n in self.tree.body if isinstance(n,ast.ClassDef) and n.name=='Hybridlight')

    def test_native_constructor_default(self):
        init=next(n for n in self.cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
        assignments=[n for n in ast.walk(init) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) and t.attr=='sg_reduce_backend' for t in n.targets)]
        self.assertEqual(len(assignments),1)
        self.assertEqual(ast.literal_eval(assignments[0].value),'native')

    def test_sg_rejects_training_before_tensor_work(self):
        fn=next(n for n in self.cls.body if isinstance(n,ast.FunctionDef) and n.name=='sg_render')
        body=[]
        for n in fn.body:
            if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Tuple): break
            body.append(n)
        guard=ast.FunctionDef(name='guard',args=fn.args,body=body,decorator_list=[])
        ns={'jt':SimpleNamespace(flags=SimpleNamespace(no_grad=False))}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[])),'sg_guard','exec'),ns)
        with self.assertRaisesRegex(RuntimeError,'inference-only'):
            ns['guard'](SimpleNamespace(sg_reduce_backend='vector3_cuda'),*[None]*6)
        ns['guard'](SimpleNamespace(sg_reduce_backend='native'),*[None]*6)
        with self.assertRaises(ValueError):
            ns['guard'](SimpleNamespace(sg_reduce_backend='invalid'),*[None]*6)

    def test_cli_choices_and_defaults(self):
        root = Path(__file__).resolve().parents[1]
        for filename in ('render_views.py', 'render_learned_light.py'):
            tree = ast.parse((root/'tools'/filename).read_text(encoding='utf-8'))
            calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                     and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
                     and n.args and isinstance(n.args[0], ast.Constant)
                     and n.args[0].value == '--sg-reduce-backend']
            self.assertEqual(len(calls), 1)
            parser = argparse.ArgumentParser()
            parser.add_argument('--sg-reduce-backend', **{
                k.arg: ast.literal_eval(k.value) for k in calls[0].keywords})
            self.assertEqual(parser.parse_args([]).sg_reduce_backend, 'native')
            self.assertEqual(parser.parse_args(['--sg-reduce-backend', 'vector3_cuda']).sg_reduce_backend,
                             'vector3_cuda')
            render_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
                            and isinstance(n.func, ast.Name) and n.func.id == 'render']
            no_grad = [n for n in ast.walk(tree) if isinstance(n, ast.With)
                       and any(isinstance(i.context_expr, ast.Call)
                               and isinstance(i.context_expr.func, ast.Attribute)
                               and i.context_expr.func.attr == 'no_grad' for i in n.items)]
            self.assertTrue(render_calls)
            for call in render_calls:
                self.assertTrue(any(call in list(ast.walk(n)) for n in no_grad))

    def test_native_helpers_and_norm_clamp(self):
        dot = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == '_sg_dot3')
        native = dot.body[0].body[0].value
        expected = ast.parse('jt.sum(a*b, dim=-1, keepdim=True)', mode='eval').body
        self.assertEqual(ast.dump(native), ast.dump(expected))
        norm = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == '_sg_norm3')
        self.assertEqual(ast.dump(norm.body[0].body[0].value),
                         ast.dump(ast.parse('jt.norm(a, dim=-1, keepdim=True)', mode='eval').body))
        self.assertEqual(ast.dump(norm.body[-1].value),
                         ast.dump(ast.parse('norm3(a, eps=1e-30)', mode='eval').body))

    def test_fp64_casts_precede_dot(self):
        fn = next(n for n in self.tree.body if isinstance(n, ast.FunctionDef) and n.name == 'lambda_trick')
        assignment = next(n for n in ast.walk(fn) if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'd_f64' for t in n.targets))
        expected = ast.parse('_sg_dot3(lobe1.float64(), lobe2.float64(), sg_reduce_backend)', mode='eval').body
        self.assertEqual(ast.dump(assignment.value), ast.dump(expected))

    def test_operator_rejects_training_and_cpu_before_tensor_work(self):
        path = Path(__file__).resolve().parents[1]/'scene/NVDIFFREC/sg_vector3.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'dot3')
        guard = ast.FunctionDef(name='guard', args=fn.args, body=fn.body[:2], decorator_list=[])
        ns = {'jt': SimpleNamespace(flags=SimpleNamespace(no_grad=False, use_cuda=False))}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[])), 'op_guard', 'exec'), ns)
        with self.assertRaisesRegex(RuntimeError, 'no_grad'):
            ns['guard'](None, None)
        ns['jt'].flags.no_grad = True
        with self.assertRaisesRegex(RuntimeError, 'CUDA'):
            ns['guard'](None, None)

if __name__=='__main__': unittest.main()
