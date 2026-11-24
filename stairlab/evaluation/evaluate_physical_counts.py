"""Compare saved visible-step counts to reviewed counts of physical levels."""

import argparse
import json
from pathlib import Path
from statistics import mean


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def summarize(rows, name):
    errors = [row['predictions'][name] - row['reference'] for row in rows
              if row['reference'] is not None]
    if not errors:
        return {'scored': 0}
    return {'scored': len(errors), 'mae': mean(map(abs, errors)), 'signed_error': mean(errors),
            'exact': sum(e == 0 for e in errors), 'under': sum(e < 0 for e in errors),
            'over': sum(e > 0 for e in errors), 'within_one': sum(abs(e) <= 1 for e in errors)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--labels', nargs=2, metavar=('COHORT', 'FILE'), action='append', required=True)
    parser.add_argument('--prediction', nargs=3, metavar=('NAME', 'FILE', 'COUNT_KEY'), action='append', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    predictions = {}
    for name, path, field in args.prediction:
        if name in predictions:
            parser.error(f'Duplicate method name: {name}')
        entries = read_json(path)['per_image']
        counts = {row['image']: row[field] for row in entries}
        if len(counts) != len(entries) or any(type(value) is not int or value < 0 for value in counts.values()):
            parser.error(f'Invalid or duplicate counts in {path}')
        predictions[name] = counts
    rows, seen = [], set()
    for cohort, path in args.labels:
        for label in read_json(path)['images']:
            image = label['image']
            if image in seen:
                parser.error(f'Image present more than once in label manifest: {image}')
            seen.add(image)
            count = label['count']
            if count is not None and (type(count) is not int or count < 0):
                parser.error(f'Invalid physical count for {image}')
            missing = [name for name, counts in predictions.items() if image not in counts]
            if missing:
                parser.error(f'Missing predictions for {image}: {missing}')
            rows.append({'image': image, 'cohort': cohort, 'scene': label.get('scene', 'unspecified'),
                         'reference': count, 'predictions': {name: counts[image] for name, counts in predictions.items()}})
    cohort_names = sorted({row['cohort'] for row in rows})
    grouped = {'all': rows, **{c: [row for row in rows if row['cohort'] == c] for c in cohort_names}}
    for subset in ('night', 'other'):
        grouped[f'new_{subset}'] = [row for row in rows if row['cohort'] == 'new'
                                   and ('night' in row['image'].lower()) == (subset == 'night')]
    report = {'metric_reference': 'visible physical levels', 'unknown_excluded': True,
              'summary': {key: {'images': len(v), 'pending': sum(r['reference'] is None for r in v),
                                'methods': {name: summarize(v, name) for name in predictions}}
                          for key, v in grouped.items()}, 'per_image': rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=2))


if __name__ == '__main__':
    main()
