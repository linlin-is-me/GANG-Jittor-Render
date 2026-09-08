"""Run the archived Garden res4 benchmark with its measured Jittor settings."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def command(args):
    result = [sys.executable, str(ROOT/'tools/benchmark_inference_versions.py'),
              '--framework', 'jittor', '--label', 'measured-local',
              '--source-root', str(ROOT), '--weights', str(args.weights.resolve()),
              '--contract', str(ROOT/'docs/benchmarks/20260908/garden_res4.json'),
              '--output', str(args.output.resolve()), '--offset-layout', 'flattened',
              '--sg-reduce-backend', 'vector3_cuda', '--mode', args.mode,
              '--warmup', '2', '--rounds', str(args.rounds)]
    if args.smoke:
        result.append('--smoke')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True,
                        help='Directory containing the measured model.npz and outputs.log')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('full','rgb'), default='full')
    parser.add_argument('--rounds', type=int, default=5)
    parser.add_argument('--smoke', action='store_true', help='First three fixed views only')
    args = parser.parse_args()
    if args.rounds < 1:
        parser.error('--rounds must be positive')
    if args.output.exists():
        parser.error('output exists; choose a new directory')
    subprocess.run(command(args), check=True)


if __name__ == '__main__':
    main()
