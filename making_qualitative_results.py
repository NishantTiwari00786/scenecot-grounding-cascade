"""

qualitative results file

For qualitative results: it turns our text finding into visual evidience, for each question where SceneCOT grounded the wrong object, this script
builds a side by side card using real ScanNet object images 

LEFT - the object the model actually grounded (its mistake, shown in red)
RIGHT - the object it should have grounded (the correct one, shown in green) 



with each image having question written underneath it. 


The images are from data_assets/scenecot_imgs/imgs/scannet, every every file is named like: scene066_001_inst42_books_0,jpeg 


"""

import json, os, re, argparse 
from collections import defaultdict
import matplotlib
matplotlib.use("Agg") # rendering to files
import matplotlib.pyplot as plt
import matplotlib.image as mpimg 


# sam grounding and stopwords threshold we used in grounding cascade 

GROUND_THRESHOLD = 0.5
STOPWORDS = {
    'prob','probability','the','is','a','an','of','at','my','i','now','so',
    'need','to','object','objects','answer','question','based','on','and','all',
    'find','should','list','potential','ground','this','corresponding','it','you',
}



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



def build_scene_index(img_dir):
    index = defaultdict(dict)
    for f in os.listdir(img_dir):
        m = re.match(r'(scene\d+_\d+)_inst(\d+)_(.+)_\d+\.jpg', f)
        if m:
            scene, _, label = m.group(1), m.group(2), m.group(3)
            # setdefault keeps the first image seen for each label in a scene
            index[scene].setdefault(label.lower(), f)
    return index


def find_image(scene_index, scene_id, label):

# find the image file for one object label in one scene, this method is approx matching, it only shows the object CLASS involved, not neccessaarily the exact 3D instance
    imgs = scene_index.get(scene_id, {})
    if not label:
        return None
    if label in imgs:                                       # exact
        return imgs[label]
    for v in {label.rstrip('s'), label + 's', label.rstrip('es')}:  # plural/singular
        if v in imgs:
            return imgs[v]
    for k in imgs:                                          # loose: one contains the other
        if label in k or k in label:
            return imgs[k]
    return None

def is_strong_example(model_grounded, should_ground):
    if not model_grounded:
        return False
    mg, sg = model_grounded.lower().strip(), should_ground.lower().strip()
    if mg == sg:
        return False
    if mg.rstrip('s') == sg.rstrip('s'):
        return False
    mg_words, sg_words = mg.split(), sg.split()
    if mg_words and sg_words and mg_words[-1] == sg_words[-1] and (mg in sg or sg in mg):
        return False
    return True


def make_card(info, scene_index, img_dir, out_path):
    
    # bulds one side of card by saving it as PNG, returns true if it builts return false it cannot


    grd_img = find_image(scene_index, info['scene_id'], info['model_grounded'])
    cor_img = find_image(scene_index, info['scene_id'], info['should_ground'])
    if not cor_img:
        return False     # if we cannot even show the correct object, skip this example

    fig, axes = plt.subplots(1, 2, figsize=(8, 4.2))

    # left: what the model grounded (red title)
    if grd_img:
        axes[0].imshow(mpimg.imread(os.path.join(img_dir, grd_img)))
        axes[0].set_title(f"Model grounded:\n{info['model_grounded']}", fontsize=11, color="#b00")
    else:
        # model grounded nothing, or an object with no image: show text instead
        axes[0].text(0.5, 0.5, f"Model grounded:\n{info['model_grounded'] or 'nothing'}\n(no image)",
                     ha="center", va="center", fontsize=11, color="#b00")
    axes[0].axis("off")

    # right: the correct object (green title)
    axes[1].imshow(mpimg.imread(os.path.join(img_dir, cor_img)))
    axes[1].set_title(f"Should have grounded:\n{info['should_ground']}", fontsize=11, color="#070")
    axes[1].axis("off")

    # caption: the question, the model's answer, the correct answer, and the bucket
    caption = (f"Q: {info['question']}\nAnswered '{info['answer']}' "
               f"(correct: '{info['correct_answer']}')  [{info['bucket']}]")
    fig.suptitle(caption, fontsize=9, y=0.04, wrap=True)
    plt.tight_layout(rect=[0, 0.06, 1, 1])   # leave room at the bottom for the caption
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    return True







def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True)    # the results.json
    ap.add_argument('--img_dir', required=True)    # folder of ScanNet object images
    ap.add_argument('--out', default='qualitative_cards')  # output folder
    ap.add_argument('--n', type=int, default=10)   # how many cards to build
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    data = json.load(open(args.results))
    scene_index = build_scene_index(args.img_dir)
    # ---- find the interesting examples: cases where grounding was WRONG ----
    candidates = []
    for ex in data:
        if ex['type'] not in ('counting', 'existence'):
            continue
        gt_text = ex['response_gt'][0] if isinstance(ex['response_gt'], list) else ex['response_gt']
        pred_grd = grounded_labels(ex['response_pred'])
        gt_grd = grounded_labels(gt_text)
        if not gt_grd:                          # skip if no ground truth grounding
            continue
        grd_ok = len(pred_grd & gt_grd) > 0     # did grounding share any correct object?
        pa, ga = extract_answer(ex['response_pred']), extract_answer(gt_text)
        ans_ok = (pa == ga) and pa != ''
        if grd_ok:                              # we only want WRONG grounding cases
            continue
        # wrong grounding + wrong answer = CASCADE; wrong grounding + right answer = LUCKY
        bucket = 'CASCADE' if not ans_ok else 'LUCKY'
        model_grounded = sorted(pred_grd)[0] if pred_grd else ''
        should_ground = sorted(gt_grd)[0]
        # only keep cards that show a clearly different wrong object
        if not is_strong_example(model_grounded, should_ground):
            continue
        candidates.append({
            'scene_id': ex['scene_id'],
            # last bit of the question text, dropping the "ASSISTANT:" tail
            'question': ex['instruction'].split('ASSISTANT')[0][-110:].strip(),
            'model_grounded': model_grounded,    # one wrong object
            'should_ground': should_ground,      # one correct object
            'answer': pa, 'correct_answer': ga, 'bucket': bucket,
        })
    # put cascades first, since the cascade is our main story
    candidates.sort(key=lambda c: 0 if c['bucket'] == 'CASCADE' else 1)
    built = 0
    for info in candidates:
        if built >= args.n:
            break
        out_path = os.path.join(args.out, f"card_{built+1:02d}_{info['bucket']}.png")
        if make_card(info, scene_index, args.img_dir, out_path):
            print(f"Built {out_path}: '{info['model_grounded']}' vs '{info['should_ground']}'")
            built += 1
    print(f"\nDone. Built {built} cards in {args.out}/")





if __name__ == '__main__':
    main()



