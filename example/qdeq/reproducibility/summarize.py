"""Recompute the audited 96-step groups; uses only the Python standard library."""
import csv
import json
from pathlib import Path
import statistics


def main():
    root = Path(__file__).resolve().parent
    with (root / 'runs_96.csv').open(encoding='utf8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    logs = {item['file']: item for item in json.loads(
        (root / 'log_evidence.json').read_text(encoding='utf8'))['logs']}
    seen = set()
    for row in rows:
        key = (row['dataset'], row['variant'], int(row['seed']))
        if key in seen:
            raise ValueError('Duplicate run: {}'.format(key))
        seen.add(key)
        expected = {metric: float(row[metric]) for metric in ('mse', 'mae')}
        if logs[row['log_file']]['final_metrics'] != [expected]:
            raise ValueError('Log/result mismatch: {}'.format(key))
    expected_keys = {(d, v, s) for d in ('weather', 'ECL', 'traffic')
                     for v in ('anderson', 'ising_fp32', 'ising_int8')
                     for s in (2023, 42, 2024)}
    if seen != expected_keys:
        raise ValueError('Missing or unexpected runs')
    print('| Dataset | Variant | n | MSE mean +/- sample SD | MAE mean +/- sample SD |')
    print('| --- | --- | --- | --- | --- |')
    for dataset in ('weather', 'ECL', 'traffic'):
        for variant in ('anderson', 'ising_fp32', 'ising_int8'):
            group = [r for r in rows if r['dataset'] == dataset and r['variant'] == variant]
            values = []
            for metric in ('mse', 'mae'):
                series = [float(r[metric]) for r in group]
                values.append('{:.4f} +/- {:.4f}'.format(statistics.mean(series), statistics.stdev(series)))
            print('| {} | {} | {} | {} | {} |'.format(dataset, variant, len(group), *values))


if __name__ == '__main__':
    main()
