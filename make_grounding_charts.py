#!/usr/bin/env python3
"""
make_grounding_charts.py

Builds bar chart cards for cascade examples. Each card shows the grounding step's
actual confidence scores as horizontal bars: the object the model wrongly grounded
is red, the correct object is green, and the rest are gray. A dashed line marks the
0.5 threshold. This makes the cascade visible: the model was confident about the
wrong object while the correct one scored low or was not grounded at all.
"""

import json, os, re, argparse
import matplotlib
matplotlib.use("Agg")              # render to files, no screen needed
import matplotlib.pyplot as plt

GROUND_THRESHOLD = 0.5
STOPWORDS = {
    'prob','probability','the','is','a','an','of','at','my','i','now','so',
    'need','to','object','objects','answer','question','based','on','and','all',
    'find','should','list','potential','ground','this','corresponding','it','you',
}


def parse_obj_prob(text):
    # Pull the (object, confidence) pairs out of the grounding section.
    m = re.search(r'<obj_prob>(.*?)</obj_prob>', text, re.DOTALL)
    body = m.group(1) if m else text
    pairs = re.findall(r'([a-zA-Z][a-zA-Z ]*?)[:\s]\s*([01]\.\d+)', body)
    out = []
    for label, prob in pairs:
        label = label.strip().lower()
        if label in STOPWORDS:
            continue
        out.append((label, float(prob)))
    return out


def grounded_labels(text, th=GROUND_THRESHOLD):
    return set(l for l, p in parse_obj_prob(text) if p >= th)


def extract_answer(text):
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if m:
        ans = m.group(1)
    else:
        parts = re.split(r'answer the question[^\n]*', text, flags=re.IGNORECASE)
        ans = parts[-1] if len(parts) > 1 else text
    return re.sub(r'[^a-zA-Z0-9 ]', '', ans).strip().lower()


def clean_q(instruction):
    # Get the readable question, dropping the ASSISTANT tail and keeping the end.
    q = instruction.split('ASSISTANT')[0].strip()
    return q[-120:].strip()


def top_pairs(text, k=5):
    # Top k grounded objects by confidence, deduped by label.
    pairs = parse_obj_prob(text)
    seen, out = set(), []
    for lab, p in sorted(pairs, key=lambda x: -x[1]):
        if lab in seen:
            continue
        seen.add(lab)
        out.append((lab, p))
        if len(out) >= k:
            break
    return out


def make_chart(info, out_path):
    # Build one bar chart card and save it.
    labels = [lab for lab, _ in info['bars']]
    probs = [p for _, p in info['bars']]

    # add the correct object as a marked bar only if it is not already shown
    # (case insensitive, ignoring singular/plural) so we never duplicate it
    already = any(l == info['correct_obj'] or l.rstrip('s') == info['correct_obj'].rstrip('s')
                  for l in labels)
    if not already:
        labels.append(f"{info['correct_obj']} (correct)")
        probs.append(info['correct_prob'])

    # color: top wrong object red, correct object green, others gray
    colors = []
    for i, lab in enumerate(labels):
        if 'correct' in lab:
            colors.append('#2a8844')
        elif i == 0:
            colors.append('#cc3333')
        else:
            colors.append('#bbbbbb')

    fig, ax = plt.subplots(figsize=(7.5, 4))
    y = range(len(labels))
    bars = ax.barh(list(y), probs, color=colors)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.axvline(0.5, color='#888', linestyle='--', linewidth=1)
    ax.set_xlabel('grounding confidence', fontsize=9)

    # write the probability at the end of each bar
    for b, p in zip(bars, probs):
        if p > 0.01:
            ax.text(p + 0.02, b.get_y() + b.get_height()/2, f"{p:.2f}",
                    va='center', fontsize=8)

    ax.set_title(f"Q: {info['question']}", fontsize=11, loc='left', pad=14)

    note = (f"The model grounded '{labels[0]}' at {probs[0]:.2f} and answered "
            f"'{info['answer']}'. The correct object '{info['correct_obj']}' "
            f"{'scored only %.2f' % info['correct_prob'] if info['correct_prob'] > 0 else 'was not grounded'}. "
            f"Correct answer: '{info['correct_answer']}'.  [CASCADE]")
    fig.text(0.02, 0.01, note, fontsize=8.5, wrap=True, va='bottom')

    plt.tight_layout(rect=[0, 0.10, 1, 1])
    plt.savefig(out_path, dpi=140, bbox_inches='tight')
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--out', default='grounding_charts')
    ap.add_argument('--n', type=int, default=6)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    data = json.load(open(args.results))

    cascades = []
    for ex in data:
        if ex['type'] not in ('counting', 'existence'):
            continue
        gt_text = ex['response_gt'][0] if isinstance(ex['response_gt'], list) else ex['response_gt']
        pred_grd = grounded_labels(ex['response_pred'])
        gt_grd = grounded_labels(gt_text)
        if not gt_grd or not pred_grd:
            continue
        grd_ok = len(pred_grd & gt_grd) > 0
        pa, ga = extract_answer(ex['response_pred']), extract_answer(gt_text)
        ans_ok = (pa == ga) and pa != ''
        if grd_ok or ans_ok:           # want cascades only
            continue

        # the correct object = most confident ground truth grounded label
        gt_pairs = sorted([(l, p) for l, p in parse_obj_prob(gt_text) if p >= GROUND_THRESHOLD],
                          key=lambda x: -x[1])
        if not gt_pairs:
            continue
        correct_obj = gt_pairs[0][0]

        bars = top_pairs(ex['response_pred'], k=5)
        if not bars:
            continue
        # skip near duplicate (model top object basically equals correct object)
        if bars[0][0] == correct_obj or bars[0][0].rstrip('s') == correct_obj.rstrip('s'):
            continue

        # did the correct object appear in the model's grounding at all? (its prob)
        pred_all = dict(parse_obj_prob(ex['response_pred']))
        correct_prob = pred_all.get(correct_obj, 0.0)

        cascades.append({
            'question': clean_q(ex['instruction']),
            'bars': bars,
            'correct_obj': correct_obj,
            'correct_prob': correct_prob,
            'answer': pa,
            'correct_answer': ga,
        })

    built = 0
    for info in cascades:
        if built >= args.n:
            break
        out_path = os.path.join(args.out, f"grounding_{built+1:02d}.png")
        make_chart(info, out_path)
        print(f"Built {out_path}: grounded '{info['bars'][0][0]}' "
              f"({info['bars'][0][1]:.2f}) vs correct '{info['correct_obj']}'")
        built += 1
    print(f"\nDone. Built {built} charts in {args.out}/ (found {len(cascades)} cascades)")


if __name__ == '__main__':
    main()
