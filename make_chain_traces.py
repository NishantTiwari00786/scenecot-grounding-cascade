#!/usr/bin/env python3
"""
make_chain_traces.py

Builds annotated reasoning chain traces for cascade examples. Instead of showing
low quality object images, this shows SceneCOT's actual step by step output (the
task type, the grounding list, and the final answer) and points out exactly where
the grounding error happened and how it led to the wrong answer.

This is the most faithful qualitative result, because it shows the real chain our
experiment reads, not a reconstructed picture.

Output: a markdown file with one annotated trace per example.
"""

import json, os, re, argparse

# An object only counts as grounded if the model was at least this confident.
GROUND_THRESHOLD = 0.5

# Words that appear in the reasoning text but are not real object names. We skip
# them so they do not get mistaken for grounded objects.
STOPWORDS = {
    'prob','probability','the','is','a','an','of','at','my','i','now','so',
    'need','to','object','objects','answer','question','based','on','and','all',
    'find','should','list','potential','ground','this','corresponding','it','you',
}


def parse_obj_prob_raw(text):
    # Return the raw text inside the <obj_prob> tags, which is the actual grounding
    # list the model produced.
    m = re.search(r'<obj_prob>(.*?)</obj_prob>', text, re.DOTALL)
    return m.group(1).strip() if m else ''


def parse_obj_prob(text):
    # Turn the grounding text into a list of (object, confidence) pairs.
    body = parse_obj_prob_raw(text) or text
    pairs = re.findall(r'([a-zA-Z][a-zA-Z ]*?)[:\s]\s*([01]\.\d+)', body)
    out = []
    for label, prob in pairs:
        label = label.strip().lower()
        if label in STOPWORDS:        # skip junk words like "the" or "probability"
            continue
        out.append((label, float(prob)))
    return out


def grounded_labels(text, th=GROUND_THRESHOLD):
    # Keep only the objects the model was confident about, return their names.
    return set(l for l, p in parse_obj_prob(text) if p >= th)


def extract_answer(text):
    # Pull the final answer and clean off the unicode noise so it compares cleanly.
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if m:
        ans = m.group(1)
    else:
        parts = re.split(r'answer the question[^\n]*', text, flags=re.IGNORECASE)
        ans = parts[-1] if len(parts) > 1 else text
    return re.sub(r'[^a-zA-Z0-9 ]', '', ans).strip().lower()


def extract_think_type(text):
    # Pull the task recognition text, which is what kind of question the model
    # decided this is (step 1 of the chain).
    m = re.search(r'<think_type>(.*?)</think_type>', text, re.DOTALL)
    return m.group(1).strip() if m else ''


def top_grounded(text, k=6):
    # Return the top k grounded objects by confidence, so the trace shows the
    # strongest few instead of the whole long list.
    pairs = parse_obj_prob(text)
    pairs.sort(key=lambda x: -x[1])
    return pairs[:k]


def clean_q(instruction):
    # Get a readable question string by dropping the "ASSISTANT:" tail.
    q = instruction.split('ASSISTANT')[0].strip()
    return q[-200:].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)   # the results.json
    ap.add_argument('--out', default='chain_traces.md')  # output markdown file
    ap.add_argument('--n', type=int, default=6)   # how many traces to write
    args = ap.parse_args()

    data = json.load(open(args.results))

    # Collect cascade cases: wrong grounding AND wrong answer, where the grounded
    # object is clearly different from the correct one, so the trace reads cleanly.
    cascades = []
    for ex in data:
        if ex['type'] not in ('counting', 'existence'):
            continue
        gt_text = ex['response_gt'][0] if isinstance(ex['response_gt'], list) else ex['response_gt']
        pred_grd = grounded_labels(ex['response_pred'])   # what model grounded
        gt_grd = grounded_labels(gt_text)                 # what it should ground
        if not gt_grd or not pred_grd:
            continue
        grd_ok = len(pred_grd & gt_grd) > 0               # did grounding overlap?
        pa, ga = extract_answer(ex['response_pred']), extract_answer(gt_text)
        ans_ok = (pa == ga) and pa != ''
        if grd_ok or ans_ok:           # we only want cascades (both wrong)
            continue

        # skip near duplicate labels (plural/singular of the same word)
        mg = sorted(pred_grd)[0]
        sg = sorted(gt_grd)[0]
        if mg == sg or mg.rstrip('s') == sg.rstrip('s'):
            continue

        # for the annotation, pick the single most confident correct object
        gt_pairs = [(l, p) for l, p in parse_obj_prob(gt_text) if p >= GROUND_THRESHOLD]
        gt_pairs.sort(key=lambda x: -x[1])
        best_correct = gt_pairs[0][0] if gt_pairs else sg
        # and the single most confident object the model wrongly grounded
        pred_pairs = [(l, p) for l, p in parse_obj_prob(ex['response_pred']) if p >= GROUND_THRESHOLD]
        pred_pairs.sort(key=lambda x: -x[1])
        best_wrong = pred_pairs[0][0] if pred_pairs else mg

        cascades.append({
            'type': ex['type'], 'question': clean_q(ex['instruction']),
            'think_type': extract_think_type(ex['response_pred']),
            'top_grounded': top_grounded(ex['response_pred']),
            'best_wrong': best_wrong, 'best_correct': best_correct,
            'answer': pa, 'correct_answer': ga,
        })

    # Write the annotated traces as markdown.
    lines = []
    lines.append("# Qualitative Results: Annotated Reasoning Chains\n")
    lines.append("Each example shows SceneCOT's actual step by step output on a "
                 "cascade case, where a grounding error led directly to a wrong answer. "
                 "We annotate where in the chain the error occurred.\n")

    for i, c in enumerate(cascades[:args.n], 1):
        lines.append(f"\n## Example {i} ({c['type']})\n")
        lines.append(f"**Question:** {c['question']}\n")
        lines.append("**SceneCOT's reasoning chain:**\n")
        lines.append("```")
        lines.append(f"Step 1  recognize task : {c['think_type']}")
        # show the grounding list the model actually produced
        grd_str = "  ".join(f"{lab} {p:.2f}" for lab, p in c['top_grounded'])
        lines.append(f"Step 3  grounding      : {grd_str}")
        lines.append(f"Step 4  answer         : {c['answer']}")
        lines.append("```")
        # the plain English explanation of the cascade
        lines.append(f"\n**What went wrong:** the grounding step committed to "
                     f"`{c['best_wrong']}`, but the correct object was "
                     f"`{c['best_correct']}`. Because the answer is built only "
                     f"from the grounded objects, the model answered "
                     f"`{c['answer']}` when the correct answer was "
                     f"`{c['correct_answer']}`. The error began in grounding and the "
                     f"sequential pipeline had no way to recover.\n")

    with open(args.out, 'w') as f:
        f.write("\n".join(lines))
    print(f"Wrote {len(cascades[:args.n])} annotated traces to {args.out}")
    print(f"(found {len(cascades)} total cascade candidates)")


if __name__ == '__main__':
    main()
