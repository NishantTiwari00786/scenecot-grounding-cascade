#!/usr/bin/env python3
"""
make_scene_renders.py

Renders cascade examples as smooth reconstructed 3D meshes (paper style), with a
tight 3D bounding box around EACH instance of the object the model wrongly grounded
(red) and each instance of the correct object (green). Only renders examples where
both objects exist in the scene.
"""

import json, os, re, argparse
import numpy as np
import torch
import open3d as o3d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.lines import Line2D

GROUND_THRESHOLD = 0.5
STOPWORDS = {
    'prob','probability','the','is','a','an','of','at','my','i','now','so',
    'need','to','object','objects','answer','question','based','on','and','all',
    'find','should','list','potential','ground','this','corresponding','it','you',
}

SCAN_BASE = None


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
    return np.asarray(xyz, dtype=float), rgb, np.asarray(inst), inst_labels


def find_instance_ids(inst_labels, target_label):
    target = target_label.lower().strip()
    ids = []
    for iid, lab in inst_labels.items():
        lab = str(lab).lower().strip()
        if lab == target or lab.rstrip('s') == target.rstrip('s') or target in lab or lab in target:
            ids.append(iid)
    return ids


def reconstruct_mesh(xyz, rgb01):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(rgb01.astype(np.float64))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.15, max_nn=30))
    cam = np.array([xyz[:, 0].mean(), xyz[:, 1].mean(), xyz[:, 2].max() + 2.0])
    pcd.orient_normals_towards_camera_location(cam)
    mesh, _dens = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
    mesh = mesh.crop(pcd.get_axis_aligned_bounding_box())
    return mesh


def draw_bbox_for_instance(ax, pts, color):
    if len(pts) < 8:
        return
    mn = pts.min(axis=0); mx = pts.max(axis=0)
    pad = (mx - mn) * 0.12 + 0.04
    mn = mn - pad; mx = mx + pad
    x0, y0, z0 = mn; x1, y1, z1 = mx
    c = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],
                  [x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
    e = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
    for a, b in e:
        ax.plot([c[a,0],c[b,0]],[c[a,1],c[b,1]],[c[a,2],c[b,2]],
                color=color, lw=2.3, zorder=20)


def render(scene_id, xyz, rgb, inst, wrong_ids, correct_ids, info, out_path):
    mesh = reconstruct_mesh(xyz, rgb)
    verts = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    vcol = np.asarray(mesh.vertex_colors)
    if len(verts) == 0 or len(faces) == 0:
        return False

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection='3d')

    fc = vcol[faces].mean(axis=1) if len(vcol) else None
    ax.add_collection3d(Poly3DCollection(verts[faces], facecolors=fc,
                                         edgecolors='none', alpha=1.0, zorder=1))

    for iid in wrong_ids:
        draw_bbox_for_instance(ax, xyz[inst == iid], '#e02424')
    for iid in correct_ids:
        draw_bbox_for_instance(ax, xyz[inst == iid], '#1f9d4d')

    ax.set_xlim(verts[:, 0].min(), verts[:, 0].max())
    ax.set_ylim(verts[:, 1].min(), verts[:, 1].max())
    ax.set_zlim(verts[:, 2].min(), verts[:, 2].max())
    ax.view_init(elev=45, azim=-70)
    ax.set_box_aspect((1, 1, 0.35))
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.fill = False
        pane.set_edgecolor('none')
    ax.grid(False)

    handles = [
        Line2D([0], [0], color='#e02424', lw=3,
               label=f"model grounded: {info['wrong_label']} (WRONG)"),
        Line2D([0], [0], color='#1f9d4d', lw=3,
               label=f"correct object: {info['correct_label']}"),
    ]
    ax.legend(handles=handles, loc='upper right', fontsize=11, framealpha=0.92)

    plt.title(info['question'], fontsize=13, weight='bold', pad=10)
    note = (f"Model grounded '{info['wrong_label']}' (red) and answered "
            f"'{info['answer']}'. Correct object '{info['correct_label']}' (green) "
            f"was present. Correct answer: '{info['correct_answer']}'.  Scene {scene_id}.")
    fig.text(0.5, 0.035, note, ha='center', fontsize=10)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    return True


def main():
    global SCAN_BASE
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)
    ap.add_argument('--scan_base', required=True)
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
        try:
            ok = render(ex['scene_id'], xyz, rgb, inst, wrong_ids, correct_ids, info, out_path)
        except Exception as e:
            print(f"  skip {ex['scene_id']}: {e}")
            continue
        if ok:
            print(f"Built {out_path}: {ex['scene_id']} | grounded '{wrong_label}' vs correct '{correct_label}'")
            built += 1

    print(f"\nDone. Built {built} scene renders in {args.out}/")


if __name__ == '__main__':
    main()
