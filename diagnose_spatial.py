#!/usr/bin/env python3
"""
diagnose_spatial.py

Parse a SceneCOT evaluation log (e.g. eval_107560.log / eval_output_107560.txt)
and print the failed (em_flag = 0) 'spatial relationship' questions in a
readable format so failure patterns can be spotted manually.

What the log actually contains per sample (from model/scenecot_agent.py and
evaluator/msqa_eval_cot_grounding.py):

    output with visual cues: <raw CoT, special tokens as rare unicode chars>
    output after visual cues: <continuation after injected visual cues>
    ...
    ************* Evaluation *************
    answer_gts: ['<gt answer>']
    answer_pred: <pred answer>
    em_flag: 0

Scene ID / question / QA type are NOT printed to stdout; the evaluator saves
them (in the same sample order) to results.json. If you pass
--results-json, those fields are joined by index. Otherwise the script
classifies samples as 'spatial relationship' from the <think_type> content
of the generated CoT.

Usage:
    python diagnose_spatial.py eval_107560.log
    python diagnose_spatial.py eval_107560.log --results-json experiments/.../results.json --top 20
"""

import argparse
import ast
import json
import os
import re
import sys
import textwrap

# ---------------------------------------------------------------------------
# CoT tag definitions (mirrors scenecot/data/cot_utils.py)
# ---------------------------------------------------------------------------

COT_INDICATORS = {
    "think_type": ["<think_type>", "</think_type>"],
    "think_grd": ["<think_grd>", "</think_grd>"],
    "think_rgn": ["<think_rgn>", "</think_rgn>"],
    "OBJ": ["[OBJ]"],
    "think_task": ["<think_task>", "</think_task>"],
    "list_obj_prob": ["<list_obj_prob>"],
    "list_obj_loc_prob": ["<list_obj_loc_prob>"],
    "list_rgn_obj": ["<list_rgn_obj>"],
    "highlight_obj": ["<highlight_obj>"],
    "img_token_indicator": ["<img_start>", "<img_end>"],
    "obj_prob": ["<obj_prob>", "</obj_prob>"],
    "obj_cap": ["<obj_cap>", "</obj_cap>"],
    "obj_loc_prob": ["<obj_loc_prob>", "</obj_loc_prob>"],
    "obj_loc_plr_prob": ["<obj_loc_plr_prob>", "</obj_loc_plr_prob>"],
    "list_obj_loc_plr_prob": ["<list_obj_loc_plr_prob>"],
    "think_sum": ["<think_sum>", "</think_sum>"],
    "answer": ["<answer>", "</answer>"],
}

COT_INDICATORS_LIST = []
for _k, _v in COT_INDICATORS.items():
    COT_INDICATORS_LIST.extend(_v)

# Fallback token->indicator mapping, derived from the last
# len(COT_INDICATORS_LIST) entries of VICUNA_ACTION_TOKENS in
# scenecot/data/data_utils.py. Used if the repo source can't be located.
_FALLBACK_TOKEN_CHARS = [
    'ტ', '开', '列', '获', '教', '少', '息', '始', 'ṃ', '松',
    'ﬁ', '间', 'ா', '政', '자', 'ब', 'Ա', 'ป', 'श', 'ļ',
    '『', 'ম', '』', '宮', 'ボ', '┌', 'Υ', '동',
]


def _load_token_chars_from_repo():
    """Rebuild the CoT token list by reading VICUNA_ACTION_TOKENS straight
    out of scenecot/data/data_utils.py (no heavy imports needed)."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, 'scenecot', 'data', 'data_utils.py'),
        os.path.join(here, 'data', 'data_utils.py'),
    ]
    for path in candidates:
        if not os.path.exists(path):
            continue
        try:
            src = open(path, encoding='utf-8').read()
            m = re.search(r'VICUNA_ACTION_TOKENS\s*=\s*(\{.*?\})\n', src, re.DOTALL)
            if not m:
                continue
            tokens = ast.literal_eval(m.group(1))
            keys = list(tokens.keys())
            return keys[-len(COT_INDICATORS_LIST):]
        except Exception:
            continue
    return None


def build_detokenize_map():
    """{rare unicode char -> human readable indicator like <think_type>}"""
    chars = _load_token_chars_from_repo() or _FALLBACK_TOKEN_CHARS
    if len(chars) != len(COT_INDICATORS_LIST):
        chars = _FALLBACK_TOKEN_CHARS
    return {tok: ind for ind, tok in zip(COT_INDICATORS_LIST, chars)}


DETOKENIZE = build_detokenize_map()


def detokenize(text):
    for tok, ind in DETOKENIZE.items():
        text = text.replace(tok, ind)
    return text


def parse_cot_answer(text):
    """Same regex extraction as data/cot_utils.py:parse_cot_answer."""
    parsed = {}
    for key, markers in COT_INDICATORS.items():
        start, end = markers[0], markers[-1]
        m = re.search(re.escape(start) + r"(.*?)" + re.escape(end), text, re.DOTALL)
        if m:
            parsed[key] = m.group(1).strip()
        elif len(markers) == 1 and start in text:
            parsed[key] = True
        else:
            parsed[key] = None
    return parsed


# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

# Lines that start a new "field" in the log. Anything else is treated as a
# continuation of the field currently being captured (CoT text can be
# multi-line because the injected <obj_loc_prob> content starts with '\n').
MARKERS = [
    ('cot_with', re.compile(r'^output with visual cues:\s?(.*)$')),
    ('cot_after', re.compile(r'^output after visual cues:\s?(.*)$')),
    ('eval_start', re.compile(r'^\*+ Evaluation \*+\s*$')),
    ('answer_gts', re.compile(r'^answer_gts:\s?(.*)$')),
    ('answer_pred', re.compile(r'^answer_pred:\s?(.*)$')),
    ('em_flag', re.compile(r'^em_flag:\s?(.*)$')),
    # noise we recognize only so it terminates a capture
    ('noise', re.compile(r'^(LIST_OBJ_PROB_TOKEN_ID:|sequence:|record:|\$\$\$+|Successfully load|obj_prob_oracle_content:|cot_(mask|no)_|cfg\.grounding)')),
]


def classify_line(line):
    for name, rx in MARKERS:
        m = rx.match(line)
        if m:
            return name, (m.group(1) if m.groups() else '')
    return None, None


def parse_log(path):
    """Returns (cot_records, eval_records).

    cot_records: list of {'with': str, 'after': str}
    eval_records: list of {'answer_gts': [...], 'answer_pred': str, 'em_flag': int}
    """
    cot_records = []
    eval_records = []

    cur_field = None          # name of the field being captured
    cur_lines = []            # accumulated lines for that field
    cur_cot = None            # cot record under construction
    cur_eval = None           # eval record under construction

    def flush():
        nonlocal cur_field, cur_lines, cur_cot, cur_eval
        if cur_field is None:
            return
        value = '\n'.join(cur_lines).strip()
        if cur_field == 'cot_with':
            cur_cot = {'with': value, 'after': ''}
            cot_records.append(cur_cot)
        elif cur_field == 'cot_after':
            if cur_cot is None:
                cur_cot = {'with': '', 'after': ''}
                cot_records.append(cur_cot)
            cur_cot['after'] = value
            cur_cot = None
        elif cur_field in ('answer_gts', 'answer_pred', 'em_flag'):
            if cur_eval is not None:
                cur_eval[cur_field] = value
        cur_field = None
        cur_lines = []

    with open(path, encoding='utf-8', errors='replace') as f:
        for raw in f:
            line = raw.rstrip('\n')
            name, first = classify_line(line)
            if name is None:
                if cur_field is not None:
                    cur_lines.append(line)
                continue
            flush()
            if name == 'eval_start':
                cur_eval = {'answer_gts': '', 'answer_pred': '', 'em_flag': ''}
                eval_records.append(cur_eval)
            elif name == 'noise':
                pass
            else:
                cur_field = name
                cur_lines = [first]
    flush()

    # post-process eval records
    cleaned = []
    for rec in eval_records:
        gts_raw = rec.get('answer_gts', '')
        try:
            gts = ast.literal_eval(gts_raw)
            if not isinstance(gts, list):
                gts = [str(gts)]
        except (ValueError, SyntaxError):
            gts = [gts_raw]
        flag_raw = rec.get('em_flag', '').strip().lower()
        if flag_raw in ('1', 'true'):
            flag = 1
        elif flag_raw in ('0', 'false'):
            flag = 0
        else:
            flag = None  # malformed block
        cleaned.append({
            'answer_gts': [str(g) for g in gts],
            'answer_pred': rec.get('answer_pred', ''),
            'em_flag': flag,
        })
    return cot_records, cleaned


# ---------------------------------------------------------------------------
# results.json join (scene_id / question / type)
# ---------------------------------------------------------------------------

def load_results_json(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def extract_question(instruction):
    """Prompt format for llava/vicuna is 'USER: {situation} {question} ASSISTANT:'."""
    if not instruction:
        return ''
    m = re.search(r'USER:\s*(.*?)\s*ASSISTANT:', instruction, re.DOTALL)
    if m:
        return m.group(1).strip()
    return instruction.strip()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

WIDTH = 100


def wrap(text, indent='    '):
    if not text:
        return indent + '(empty)'
    out = []
    for para in str(text).split('\n'):
        wrapped = textwrap.wrap(para, width=WIDTH - len(indent)) or ['']
        out.extend(indent + w for w in wrapped)
    return '\n'.join(out)


def hr(char='-'):
    return char * WIDTH


def main():
    ap = argparse.ArgumentParser(description='Diagnose failed spatial-relationship QA from a SceneCOT eval log.')
    ap.add_argument('log', help='Path to the eval log file (e.g. eval_107560.log)')
    ap.add_argument('--results-json', default=None,
                    help='Path to the results.json written by the evaluator (provides scene_id, question, type). '
                         'Entries are joined to log samples by order.')
    ap.add_argument('--top', type=int, default=20, help='Number of failures to display (default: 20)')
    ap.add_argument('--type', dest='qa_type', default='spatial relationship',
                    help="QA type to filter on (default: 'spatial relationship')")
    args = ap.parse_args()

    cot_records, eval_records = parse_log(args.log)
    if not eval_records:
        sys.exit(f"No '************* Evaluation *************' blocks found in {args.log}. "
                 "Is this the right log file?")

    results = None
    if args.results_json:
        results = load_results_json(args.results_json)
        if len(results) != len(eval_records):
            print(f"[warn] results.json has {len(results)} entries but the log has "
                  f"{len(eval_records)} evaluation blocks; joining by index up to the shorter one.\n")

    if len(cot_records) != len(eval_records):
        print(f"[warn] found {len(cot_records)} CoT blocks vs {len(eval_records)} evaluation blocks; "
              f"pairing by order may be off for some samples.\n")

    n_type = 0
    failures = []
    for i, ev in enumerate(eval_records):
        cot = cot_records[i] if i < len(cot_records) else {'with': '', 'after': ''}
        raw_cot = detokenize((cot['with'] + cot['after']).strip())
        parsed = parse_cot_answer(raw_cot)

        meta = results[i] if results and i < len(results) else None
        if meta is not None:
            qa_type = meta.get('type', '')
            is_spatial = (qa_type == args.qa_type)
        else:
            # no results.json: infer from the model's own <think_type> content
            think_type = (parsed.get('think_type') or '').lower()
            is_spatial = args.qa_type.lower() in think_type

        if not is_spatial:
            continue
        n_type += 1
        if ev['em_flag'] == 0:
            failures.append({
                'index': i,
                'scene_id': meta.get('scene_id', '(unknown — pass --results-json)') if meta else '(unknown — pass --results-json)',
                'question': extract_question(meta.get('instruction', '')) if meta else '(unknown — pass --results-json)',
                'gt': ev['answer_gts'],
                'pred': ev['answer_pred'],
                'raw_cot': raw_cot,
                'parsed': parsed,
            })

    print(hr('='))
    print(f"SPATIAL RELATIONSHIP FAILURE DIAGNOSIS  ({os.path.basename(args.log)})")
    print(hr('='))
    print(f"Total evaluation blocks parsed : {len(eval_records)}")
    print(f"'{args.qa_type}' samples       : {n_type}"
          + ('' if results else '   (inferred from <think_type>; pass --results-json for exact types)'))
    print(f"Failures (em_flag = 0)         : {len(failures)}")
    print(f"Showing top {min(args.top, len(failures))} failures")
    print(hr('='))

    for rank, f in enumerate(failures[:args.top], start=1):
        p = f['parsed']
        print()
        print(hr('='))
        print(f"FAILURE #{rank}   (sample index {f['index']})")
        print(hr('='))
        print(f"Scene ID : {f['scene_id']}")
        print("Question :")
        print(wrap(f['question']))
        print()
        print(f"Ground Truth : {' | '.join(f['gt']) if f['gt'] else '(empty)'}")
        print(f"Predicted    : {f['pred'] or '(empty)'}")
        print()
        print("--- CoT: <think_type> ---")
        print(wrap(p.get('think_type')))
        print("--- CoT: <think_task> ---")
        print(wrap(p.get('think_task')))
        print("--- CoT: <obj_loc_prob> (agent-relative x,y,z,w,h,d + prob) ---")
        print(wrap(p.get('obj_loc_prob')))
        if p.get('obj_loc_plr_prob'):
            print("--- CoT: <obj_loc_plr_prob> (polar: angle°, distance) ---")
            print(wrap(p.get('obj_loc_plr_prob')))
        if p.get('think_rgn'):
            print("--- CoT: <think_rgn> ---")
            print(wrap(p.get('think_rgn')))
        print()
        print("--- Full raw CoT (de-tokenized) ---")
        print(wrap(f['raw_cot']))

    print()
    print(hr('='))
    if not failures:
        print("No failed spatial relationship samples found.")
    elif len(failures) > args.top:
        print(f"... {len(failures) - args.top} more failures not shown (use --top {len(failures)} to see all).")
    print(hr('='))


if __name__ == '__main__':
    main()
