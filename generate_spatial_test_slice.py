#!/usr/bin/env python3
"""Generate a high-difficulty spatial stress-test slice of the MSQA test set.

This script loads the master MSQA/Beacon3D test annotation file used by
``scenecot/scripts/test/full_training_msqa_beacon3d_test_moe.sh`` (i.e.
``${SCENECOT_MSR3D_ANNO_DIR}/situated_qa_test_pure_txt.json``, which is what
``QACOTScanNetMSR3D.load_anno`` in ``scenecot/data/datasets.py`` ingests) and
keeps only entries that are spatially demanding:

  1. The question contains an explicit directional/relative token:
     'behind', 'underneath', 'directly above', 'closest to', 'furthest'.
  2. The question requires relative tracking between multiple foreground
     objects (e.g. "Is the backpack to the left or right of the chair?"),
     detected via relational phrasing combined with >= 2 annotated
     foreground object ids.

Entries matching either criterion are kept. Items are copied verbatim from
the source JSON (same keys, same order), so the output file has exactly the
same structure as the original test JSON and can be ingested seamlessly by
the evaluator. To run the evaluator on the slice, point
``SCENECOT_MSR3D_ANNO_DIR`` at a directory where this file is renamed (or
symlinked) to ``situated_qa_test_pure_txt.json``.

Usage:
    python generate_spatial_test_slice.py
    python generate_spatial_test_slice.py --input /path/to/situated_qa_test_pure_txt.json
    python generate_spatial_test_slice.py --output data/msqa_spatial_stress_test.json
"""

import argparse
import json
import os
import re
import sys

# Explicit directional/relative tokens (criterion 1).
DIRECTIONAL_TOKENS = [
    'behind',
    'underneath',
    'directly above',
    'closest to',
    'furthest',
]

# Relational phrasings indicating relative tracking between objects
# (criterion 2, combined with multiple foreground objects).
RELATIVE_TRACKING_PATTERNS = [
    r'\bto the (?:left|right) of\b',
    r'\bon the (?:left|right) (?:side )?of\b',
    r'\b(?:left|right) or (?:left|right)\b',          # "left or right (of) ..."
    r'\bin front of\b',
    r'\bbetween the\b',
    r'\bnext to the\b',
    r'\bcloser to\b',
    r'\bfarther (?:from|away)\b',
    r'\bopposite (?:to|of|side)\b',
    r'\bfacing the\b',
]

_DIRECTIONAL_RE = re.compile(
    r'(?:' + '|'.join(r'\b' + re.escape(tok) + r'\b' for tok in DIRECTIONAL_TOKENS) + r')',
    flags=re.IGNORECASE,
)
_RELATIVE_RE = re.compile(
    r'(?:' + '|'.join(RELATIVE_TRACKING_PATTERNS) + r')',
    flags=re.IGNORECASE,
)


def default_input_path():
    """Resolve the master test JSON the same way the test shell script does."""
    data_root = os.environ.get('SCENECOT_DATA_ROOT', './data_assets')
    cot_data_root = os.environ.get(
        'SCENECOT_COT_DATA_ROOT', os.path.join(data_root, 'scenecot_cot_data')
    )
    anno_dir = os.environ.get(
        'SCENECOT_MSR3D_ANNO_DIR', os.path.join(cot_data_root, 'MSQA')
    )
    return os.path.join(anno_dir, 'situated_qa_test_pure_txt.json')


def has_directional_token(question):
    return _DIRECTIONAL_RE.search(question) is not None


def requires_relative_tracking(item):
    question = item.get('question', '')
    if _RELATIVE_RE.search(question) is None:
        return False
    # Relative tracking is only "hard" when multiple distinct foreground
    # objects are involved.
    obj_ids = item.get('obj_ids', [])
    return len(set(obj_ids)) >= 2


def filter_spatial_items(items):
    kept = []
    n_directional = 0
    n_relative = 0
    for item in items:
        question = item.get('question', '')
        directional = has_directional_token(question)
        relative = requires_relative_tracking(item)
        if directional:
            n_directional += 1
        if relative:
            n_relative += 1
        if directional or relative:
            kept.append(item)
    return kept, n_directional, n_relative


def main():
    parser = argparse.ArgumentParser(
        description='Create a high-difficulty spatial slice of the MSQA test set.'
    )
    parser.add_argument(
        '--input', default=None,
        help='Path to the master MSQA test JSON '
             '(default: resolved from SCENECOT_* env vars, i.e. '
             '${SCENECOT_MSR3D_ANNO_DIR}/situated_qa_test_pure_txt.json)',
    )
    parser.add_argument(
        '--output', default='data/msqa_spatial_stress_test.json',
        help='Where to save the filtered subset '
             '(default: data/msqa_spatial_stress_test.json)',
    )
    args = parser.parse_args()

    input_path = args.input or default_input_path()
    if not os.path.exists(input_path):
        sys.exit(
            f'Error: master test JSON not found at: {input_path}\n'
            'Set SCENECOT_MSR3D_ANNO_DIR (or SCENECOT_DATA_ROOT / '
            'SCENECOT_COT_DATA_ROOT) or pass --input explicitly.'
        )

    print(f'Loading master test annotations from: {input_path}')
    with open(input_path, 'r', encoding='utf-8') as f:
        items = json.load(f)
    if not isinstance(items, list):
        sys.exit(f'Error: expected a JSON list of QA entries, got {type(items).__name__}')
    print(f'Loaded {len(items)} entries')

    kept, n_directional, n_relative = filter_spatial_items(items)

    print('Filter summary:')
    print(f'  matched directional tokens {DIRECTIONAL_TOKENS}: {n_directional}')
    print(f'  matched relative tracking (multi-object): {n_relative}')
    print(f'  total kept (union): {len(kept)} / {len(items)} '
          f'({100.0 * len(kept) / max(len(items), 1):.1f}%)')

    type_counts = {}
    for item in kept:
        qa_type = item.get('type', 'unknown')
        type_counts[qa_type] = type_counts.get(qa_type, 0) + 1
    for qa_type in sorted(type_counts):
        print(f'    type "{qa_type}": {type_counts[qa_type]}')

    out_dir = os.path.dirname(args.output)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(kept, f, indent=2, ensure_ascii=False)
    print(f'Saved spatial stress-test slice to: {args.output}')
    print('To evaluate on this slice, rename/symlink it to '
          '"situated_qa_test_pure_txt.json" inside a directory and point '
          'SCENECOT_MSR3D_ANNO_DIR (data.cotqa.msr3d.anno_dir) at that directory.')


if __name__ == '__main__':
    main()
