#!/usr/bin/env python3
"""
make_scene_renders.py

Renders the actual 3D ScanNet scene for cascade examples, highlighting the object
the model wrongly grounded (red) and the correct object (green) in the real room.
This is the true scene visual: it shows where in the room each object is.
"""

import json, os, re, argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

GROUND_THRESHOLD = 0.5
STOPWORDS = {
    'prob','probability','the','is','a','an','of','at','my','i','now','so',
    'need','to','object','objects','answer','question','based','on','and','all',
    'find','should','list','potential','ground','this','corresponding','it','you',
}

SCAN_BASE = None  # set from args


def parse_obj_prob(text):
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
    q = instruction.split('ASSISTANT')[0].strip()
    return q[-110:].strip()


def load_scene(scene_id):
    # returns xyz, rgb, instance_ids, {instance_id: label}
    pcd_path = f"{SCAN_BASE}/pcd_with_global_alignment/{scene_id}.pth"
    lbl_path = f"{SCAN_BASE}/instance_id_to_label/{scene_id}.pth"
    if not (os.path.exists(pcd_path) and os.path.exists(lbl_path)):
        return None
    xyz, rgb, _sem, inst = torch.load(pcd_path, map_location='cpu', weights_only=False)
    inst_labels = torch.load(lbl_path, map_location='cpu', weights_only=False)
    rgb = np.asarray(rgb, dtype=float)
    if rgb.max() > 1.5:
        rgb = rgb / 255.0
    rgb = np.clip(rgb, 0, 1)
    return np.asarray(xyz), rgb, np.asarray(inst), inst_labels


def find_instance_ids(inst_labels, target_label):
    # return all instance ids whose label matches the target (with simple variants)
    target = target_label.lower().strip()
    ids = []
    for iid, lab in inst_labels.items():
        lab = str(lab).lower().strip()
        if lab == target or lab.rstrip('s') == target.rstrip('s') or target in lab or lab in target:
            ids.append(iid)
    return ids


def render(scene_id, xyz, rgb, inst, wrong_ids, correct_ids, info, out_path):
    fig, ax = plt.subplots(figsize=(8, 7))
    # base room (everything not highlighted) in its real colors
    highlight = np.isin(inst, wrong_ids + correct_ids)
    base = ~highlight
    ax.scatter(xyz[base, 0], xyz[base, 1], c=rgb[base], s=2, alpha=0.45)
    # wrong object red
    if wrong_ids:
        w = np.isin(inst, wrong_ids)
        ax.scatter(xyz[w, 0], xyz[w, 1], c='#d32f2f', s=10,
                   label=f"model grounded: {info['wrong_label']}")
    # correct object green
    if correct_ids:
        c = np.isin(inst, correct_ids)
        ax.scatter(xyz[c, 0], xyz[c, 1], c='#2a8844', s=10,
                   label=f"correct object: {info['correct_label']}")
    ax.set_aspect('equal')
    ax.legend(loc='upper right', fontsize=9)
    ax.axis('off')
    ax.set_title(f"Q: {info['question']}", fontsize=10, loc='left')
    note = (f"Model answered '{info['answer']}' (correct: '{info['correct_answer']}'). "
            f"Top-down view of {scene_id}.  [CASCADE]")
    fig.text(0.02, 0.02, note, fontsize=8.5, va='bottom')
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    plt.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close()


def main():
    global SCAN_BASE
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--scan_base', required=True,
                    help='path to .../SceneVerse/ScanNet/scan_data')
    ap.add_argument('--out', default='scene_renders')
    ap.add_argument('--n', type=int, default=6)
    args = ap.parse_args()
    SCAN_BASE = args.scan_base

    os.makedirs(args.out, exist_ok=True)
    data = json.load(open(args.results))

    built = 0
    for ex in data:
        if built >= args.n:
            break
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
        if grd_ok or ans_ok:
            continue

        wrong_label = sorted(pred_grd)[0]
        correct_label = sorted(gt_grd)[0]
        if wrong_label == correct_label or wrong_label.rstrip('s') == correct_label.rstrip('s'):
            continue

        scene = load_scene(ex['scene_id'])
        if scene is None:
            continue
        xyz, rgb, inst, inst_labels = scene

        wrong_ids = find_instance_ids(inst_labels, wrong_label)
        correct_ids = find_instance_ids(inst_labels, correct_label)
        # need at least the correct object present in the scene to be meaningful
        if not correct_ids:
            continue

        info = {
            'question': clean_q(ex['instruction']),
            'wrong_label': wrong_label,
            'correct_label': correct_label,
            'answer': pa,
            'correct_answer': ga,
        }
        out_path = os.path.join(args.out, f"scene_{built+1:02d}.png")
        render(ex['scene_id'], xyz, rgb, inst, wrong_ids, correct_ids, info, out_path)
        print(f"Built {out_path}: {ex['scene_id']} | grounded '{wrong_label}' vs correct '{correct_label}'")
        built += 1

    print(f"\nDone. Built {built} scene renders in {args.out}/")


if __name__ == '__main__':
    main()
