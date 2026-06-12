#!/usr/bin/env python3
"""

The script measures where SceneCOT asnwers from on grounding dependent questions. Since ScenCOT
answers in fixed 4 step order, and the final answer is built only from the objects chosen in grounding step.
If grounding picked up the wrong object, then answer is reasoning about the wrong thing. This is called grounding cascade

Scripts reads the model saved reasonshing chain, pulls out what model grounded and what it answered, compared both to the ground 
truth, and sorts every question in 4 groups

right grounding + right answer = the pipeline worked
wrong grounding + wrong answer = CASCADE (the failure we study)
wrong grounding + right answer = LUCKY guess (right answer, wrong reasn) 
right grounding + wrong answer = reasoning failure

The scripts prints counts, the %, and few traced examples 

"""

import json
import re
import argparse

GROUND_THRESHOLD = 0.5 # grounding threshold

# words that are in the reasoning text, but are not object names, skipping them so they are not mistaken for grounded objects
STOPWORDS = {
    'prob', 'probability', 'the', 'is', 'a', 'an', 'of', 'at', 'my', 'i',
    'now', 'so', 'need', 'to', 'object', 'objects', 'answer', 'question',
    'based', 'on', 'and', 'all', 'find', 'should', 'list', 'potential',
    'ground', 'this', 'corresponding', 'it', 'you',
}


# pulls out grounding list out of one reasning chain 

# the grounding step: <obj_prob> 0.75 boxes 0.64 desk 0.63<obj_prob?

# it returns a list of (object_name, confidence pairs) = (book, 0.75)


def parse_obj_prob(text):
    m = re.search(r'<obj_prob>(.*?)</obj_prob>', text, re.DOTALL)
    body = m.group(1) if m else text
    pairs = re.findall(r'([a-zA-Z][a-zA-Z ]*?)[:\s]\s*([01]\.\d+)', body)
    out = []
    for label, prob in pairs:
        label = label.strip().lower()
        if label in STOPWORDS: # dropping junk words
            continue
        out.append((label, float(prob))) # convert text to real number = 0.75
    return out


# it keeps only the confident objects (> Threshold) and return their names as a set.

def grounded_labels(obj_probs, threshold): 
    return set(label for label, prob in obj_probs if prob >= threshold)

# pulls the final answer out of one reasoning chain and clean, the model answer is inside <answer>...</answer> tags
# the ground truth answer has no tags, it comes after the phrase

def extract_answer(text):
    m = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL)
    if m:
        ans = m.group(1)
    else:
        parts = re.split(r'answer the question[^\n]*', text, flags=re.IGNORECASE)
        ans = parts[-1] if len(parts) > 1 else text
    ans = re.sub(r'[^a-zA-Z0-9 ]', '', ans).strip().lower()
    return ans



# this func checks if grounding is correct, by checking if grounding counts as correct if the model grounded at least one of the correct object 

def grounding_is_correct(pred_labels, gt_labels):
    if not gt_labels:
        return None
    return len(pred_labels & gt_labels) > 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', required=True) # path to results.json
    ap.add_argument('--types', nargs='+', default=['counting', 'existence']) # question types we are using
    ap.add_argument('--threshold', type=float, default=GROUND_THRESHOLD) # grounding cutoff
    ap.add_argument('--examples', type=int, default=5) # traced examples per bucket 
    args = ap.parse_args()
    

    # loaded all predictions, then keep only questions types we asked for
    data = json.load(open(args.results))
    target = [x for x in data if x['type'] in args.types]
    print(f"Analyzing {len(target)} questions of types {args.types}")
    print(f"Grounding threshold: {args.threshold}\n")
    
    # one list per quantitatve we analyzed
    buckets = {
        'right_grd_right_ans': [],
        'wrong_grd_wrong_ans': [],
        'wrong_grd_right_ans': [],
        'right_grd_wrong_ans': [],
        'unevaluable': [],
    }
    
    # the loop for classifying every question
    for ex in target:
        pred_text = ex['response_pred']
        gt_raw = ex['response_gt']
        gt_text = gt_raw[0] if isinstance(gt_raw, list) else gt_raw
           
        # what model grounded vs what it should gave grounded
        pred_grd = grounded_labels(parse_obj_prob(pred_text), args.threshold)
        gt_grd = grounded_labels(parse_obj_prob(gt_text), args.threshold)
        grd_ok = grounding_is_correct(pred_grd, gt_grd)
        
        # model answer vs correct answer
        pred_ans = extract_answer(pred_text)
        gt_ans = extract_answer(gt_text)
        ans_ok = (pred_ans == gt_ans) and pred_ans != ''

        rec = {
            'type': ex['type'],
            'question': ex['instruction'][-140:],
            'pred_grounded': sorted(pred_grd),
            'gt_grounded': sorted(gt_grd),
            'pred_answer': pred_ans,
            'gt_answer': gt_ans,
        }

        if grd_ok is None:
            buckets['unevaluable'].append(rec)
        elif grd_ok and ans_ok:
            buckets['right_grd_right_ans'].append(rec)
        elif not grd_ok and not ans_ok:
            buckets['wrong_grd_wrong_ans'].append(rec)
        elif not grd_ok and ans_ok:
            buckets['wrong_grd_right_ans'].append(rec)
        else:
            buckets['right_grd_wrong_ans'].append(rec)

    total = sum(len(v) for v in buckets.values())
    evaluable = total - len(buckets['unevaluable'])
    wrong_ans = len(buckets['wrong_grd_wrong_ans']) + len(buckets['right_grd_wrong_ans'])
    

    #calc and organizing them in buckets
    print("=" * 62)
    print("GROUNDING ERROR CASCADE - CROSS-TABULATION")
    print("=" * 62)
    print(f"{'Category':<34}{'Count':>8}{'% eval':>12}")
    print("-" * 62)
    labels = {
        'right_grd_right_ans': 'Right grounding + Right answer',
        'wrong_grd_wrong_ans': 'Wrong grounding + Wrong answer (CASCADE)',
        'wrong_grd_right_ans': 'Wrong grounding + Right answer (LUCKY)',
        'right_grd_wrong_ans': 'Right grounding + Wrong answer (REASONING)',
    }
    for k, lab in labels.items():
        n = len(buckets[k])
        pct = 100 * n / evaluable if evaluable else 0
        print(f"{lab:<34}{n:>8}{pct:>11.1f}%")
    print(f"{'Unevaluable (no GT grounding)':<34}{len(buckets['unevaluable']):>8}")
    print("-" * 62)
    print(f"{'TOTAL evaluable':<34}{evaluable:>8}")

    print("\n" + "=" * 62)
    print("KEY FINDINGS")
    print("=" * 62)
    if wrong_ans:
        casc = len(buckets['wrong_grd_wrong_ans'])
        print(f"\nOf {wrong_ans} WRONG answers:")
        print(f"  {casc} ({100*casc/wrong_ans:.1f}%) trace to WRONG GROUNDING (cascade)")
        print(f"  {len(buckets['right_grd_wrong_ans'])} "
              f"({100*len(buckets['right_grd_wrong_ans'])/wrong_ans:.1f}%) "
              f"had correct grounding but wrong reasoning")
    lucky = len(buckets['wrong_grd_right_ans'])
    if evaluable:
        print(f"\nLucky guesses (wrong grounding, right answer): "
              f"{lucky} ({100*lucky/evaluable:.1f}% of evaluable)")
        print("  -> inflate accuracy without true grounding")
        print("     = the grounding-QA coherence problem")

    print("\n" + "=" * 62)
    print("TRACED EXAMPLES: CASCADE (wrong grounding -> wrong answer)")
    print("=" * 62)
    for r in buckets['wrong_grd_wrong_ans'][:args.examples]:
        print(f"\n[{r['type']}] ...{r['question']}")
        print(f"  Model grounded: {r['pred_grounded']}")
        print(f"  Should ground:  {r['gt_grounded']}")
        print(f"  Answered '{r['pred_answer']}' (correct: '{r['gt_answer']}')")

    print("\n" + "=" * 62)
    print("TRACED EXAMPLES: LUCKY GUESSES (wrong grounding -> right answer)")
    print("=" * 62)
    for r in buckets['wrong_grd_right_ans'][:args.examples]:
        print(f"\n[{r['type']}] ...{r['question']}")
        print(f"  Model grounded: {r['pred_grounded']}")
        print(f"  Should ground:  {r['gt_grounded']}")
        print(f"  Answered '{r['pred_answer']}' (matched '{r['gt_answer']}')")


if __name__ == '__main__':
    main()
