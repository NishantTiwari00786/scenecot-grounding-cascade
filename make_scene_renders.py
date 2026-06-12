#!/usr/bin/env python3
"""
make_scene_renders.py

Renders the actual 3D ScanNet scene for cascade examples in a clean angled 3D
view, highlighting the object the model wrongly grounded (bold red) and the
correct object (bold green) in the real room. Only renders examples where BOTH
objects exist in the scene, so every image shows the contrast.
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
    # Get the actual question: prefer the last sentence ending in '?', so the
    # title does not start mid-sentence like "trol and privacy...".
    q = instruction.split('ASSISTANT')[0].strip()
    sentences = re.split(r'(?<=[.?!])\s+', q)
    questions = [s.strip() for s in sentences if '?' in s]
    if questions:
        return questions[-1]
    return q[-110:].strip()


def load_scene(scene_id):
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
    target = target_label.lower().strip()
    ids = []
    for iid, lab in inst_labels.items():
        lab = str(lab).lower().strip()
        if lab == target or lab.rstrip('s') == target.rstrip('s') or target in lab or lab in target:
            ids.append(iid)
    return ids


def _draw_ring(ax, pts, color):
    # draw a ring around a compact cluster to help the eye find it.
    if len(pts) < 20:
        return
    cx, cy, cz = pts[:, 0].mean(), pts[:, 1].mean(), pts[:, 2].mean()
    spread = np.percentile(np.linalg.norm(pts[:, :2] - [cx, cy], axis=1), 90)
    if spread > 1.2:          # cluster too spread out, a ring would look wrong
        return
    r = max(0.3, spread * 1.4)
    theta = np.linspace(0, 2 * np.pi, 60)
    ax.plot(cx + r * np.cos(theta), cy + r * np.sin(theta), cz,
            color=color, lw=2.0, alpha=0.85, zorder=6)


def render(scene_id, xyz, rgb, inst, wrong_ids, correct_ids, info, out_path):
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')

    highlight = np.isin(inst, wrong_ids + correct_ids)
    base = ~highlight
    base_idx = np.where(base)[0]
    if len(base_idx) > 45000:
        base_idx = np.random.choice(base_idx, size=45000, replace=False)
    ax.scatter(xyz[base_idx, 0], xyz[base_idx, 1], xyz[base_idx, 2],
               c=rgb[base_idx], s=2.5, alpha=0.30, linewidths=0, zorder=1)

    w = np.isin(inst, wrong_ids)
    ax.scatter(xyz[w, 0], xyz[w, 1], xyz[w, 2], c='#e02424', s=42, alpha=1.0,
               edgecolors='#5a0000', linewidths=0.4, zorder=5,
               label=f"model grounded: {info['wrong_label']} (WRONG)")
    _draw_ring(ax, xyz[w], '#e02424')

    c = np.isin(inst, correct_ids)
    ax.scatter(xyz[c, 0], xyz[c, 1], xyz[c, 2], c='#1f9d4d', s=42, alpha=1.0,
               edgecolors='#003313', linewidths=0.4, zorder=5,
               label=f"correct object: {info['correct_label']}")
    _draw_ring(ax, xyz[c], '#1f9d4d')

    ax.view_init(elev=38, azim=-72)
    ax.set_box_aspect((1, 1, 0.32))
    ax.legend(loc='upper right', fontsize=11, framealpha=0.92)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('none')
    ax.grid(False)

    plt.title(info['question'], fontsize=13, weight='bold', pad=10)
    note = (f"Model grounded '{info['wrong_label']}' (red) and answered "
            f"'{info['answer']}'. Correct object '{info['correct_label']}' (green) "
            f"was present. Correct answer: '{info['correct_answer']}'.  Scene {scene_id}.")
    fig.text(0.5, 0.035, note, ha='center', fontsize=10)
    plt.savefig(out_path, dpi=160, bbox_inches='tight')
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
        if not correct_ids or not wrong_ids:
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
