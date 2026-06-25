"""Print the reported HVAC-MFR + STDC2 comparison-table metrics.

This helper is intended for reproducing the final row used in the paper-style
comparison table. It keeps the displayed decimal places consistent with the
table format.
"""

import argparse


METRICS = {
    'method': 'HVAC-MFR',
    'year': '-',
    'backbone': 'STDC2',
    'params_m': 13.1,
    'gflops': 117.8,
    'miou': 79.2,
}


def print_markdown() -> None:
    print('| Method | Year | Backbone | Params (M) | GFLOPs | mIoU (%) |')
    print('|---|---:|---|---:|---:|---:|')
    print(
        f"| {METRICS['method']} | {METRICS['year']} | "
        f"{METRICS['backbone']} | {METRICS['params_m']:.1f} | "
        f"{METRICS['gflops']:.1f} | {METRICS['miou']:.1f} |")


def print_plain() -> None:
    print('Method      Year    Backbone    Params (M)    GFLOPs    mIoU (%)')
    print(
        f"{METRICS['method']:<11} {METRICS['year']:<7} "
        f"{METRICS['backbone']:<11} {METRICS['params_m']:<13.1f} "
        f"{METRICS['gflops']:<9.1f} {METRICS['miou']:.1f}")


def print_tsv() -> None:
    print('Method\tYear\tBackbone\tParams (M)\tGFLOPs\tmIoU (%)')
    print(
        f"{METRICS['method']}\t{METRICS['year']}\t"
        f"{METRICS['backbone']}\t{METRICS['params_m']:.1f}\t"
        f"{METRICS['gflops']:.1f}\t{METRICS['miou']:.1f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Print HVAC-MFR + STDC2 reported table metrics.')
    parser.add_argument(
        '--format',
        choices=('markdown', 'plain', 'tsv'),
        default='markdown',
        help='Output format. Defaults to markdown.')
    args = parser.parse_args()

    if args.format == 'markdown':
        print_markdown()
    elif args.format == 'plain':
        print_plain()
    else:
        print_tsv()


if __name__ == '__main__':
    main()
