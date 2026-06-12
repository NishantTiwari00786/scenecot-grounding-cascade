# EE243 Final Project: A Structural Limitation in SceneCOT's Grounded Reasoning

**Course:** EE 243 Advanced Computer Vision, Spring 2026, UC Riverside
**Team:** Nishant Tiwari, Simarpal Singh

**Paper studied:** SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes (ICLR 2026)

Project webpage: UPDATE LINK HERE

Video walkthrough: UPDATE LINK HERE

---

## Summary

SceneCOT answers questions about 3D rooms by reasoning in four fixed steps, and a
middle step called grounding is supposed to point to the specific object the
question is about. We first reproduce SceneCOT's evaluation, then we expose a
**structural limitation**: the four steps run in a strict sequence with no way to
check or correct the grounding step, so any grounding error is final and flows
straight into the answer. We show that on the two most grounding dependent
question types, the large majority of wrong answers come from this exact failure,
and separately that a large share of the model's correct answers are produced
without correct grounding, which means its accuracy overstates how much it
actually understands the scene.

This is not the claim "the model sometimes makes mistakes." Every model makes
mistakes. The point is about the **design**: SceneCOT commits to a single
grounding result at inference with no recovery mechanism, and we demonstrate
empirically that this design choice is where its failures concentrate.

## SIMAR ADD SPATIAL STUFF HERE ## 

---

## Table of contents

1. [How SceneCOT works, and how grounding works inside it](#how-scenecot-works-and-how-grounding-works-inside-it)
2. [Research lineage](#research-lineage)
3. [The limitation we target, stated precisely](#the-limitation-we-target-stated-precisely)
4. [Datasets: what SceneCOT trains on and what we test on](#datasets-what-scenecot-trains-on-and-what-we-test-on)
5. [Part 1: Reproducing the baseline](#part-1-reproducing-the-baseline)
6. [Part 2: How we got from the baseline to our experiment](#part-2-how-we-got-from-the-baseline-to-our-experiment)
7. [Method: exactly what we measured](#method-exactly-what-we-measured)
8. [Results](#results)
9. [Qualitative results: seeing the grounding error](#qualitative-results-seeing-the-grounding-error)
10. [Success cases](#success-cases)
11. [Limitations of our analysis](#limitations-of-our-analysis)
12. [Part 3: Spatial reasoning experiment](#part-3-spatial-reasoning-experiment-option-a)
13. [How to reproduce](#how-to-reproduce)
14. [Repository structure](#repository-structure)
15. [Acknowledgements](#acknowledgements)

---

## How SceneCOT works, and how grounding works inside it

To explain the limitation, we first have to be precise about how SceneCOT
produces an answer, and especially how its grounding step actually works.

SceneCOT is built on the LLaVA-1.5 vision language model (Vicuna-7B backbone, LoRA
fine tuned). For every question it runs four steps in a fixed order, and each step
consumes the output of the previous one:

1. **Task recognition.** The model emits a `<think_type>` token deciding the
   question category (counting, existence, spatial, navigation, and so on). This
   choice controls which coordinate representation the later steps use.

2. **Region localization.** A rule based symbolic engine uses the agent's position
   and orientation to restrict the scene to the relevant sub region, expressed as
   cardinal directions (left, right) or clock directions (3 o'clock). This narrows
   the set of candidate objects before any neural grounding happens.

3. **Entity grounding.** This is the step our project is about, so here is the
   mechanism in detail:
   - First, an off the shelf 3D instance segmenter, **Mask3D**, has already broken
     the room into a fixed set of candidate object proposals (one mask per detected
     object).
   - The grounding module, which is based on **PQ3D**, then scores each of those
     proposals for how well it matches the question, and emits an object
     probability list. In the saved reasoning chain this appears as a section
     called `<obj_prob>`, for example: `book 0.75 boxes 0.64 desk 0.63`.
   - The objects with high probability are taken as "the objects the model
     grounded."

4. **Grounded reasoning.** The symbolic engine turns the grounded objects into a
   visual clue (for counting and existence, the set of grounded objects; for
   spatial, their coordinates), and the language model produces the final answer
   **only from that clue**.

A deliberate design choice is the hybrid neural plus symbolic split: the geometry
math (clock directions, distances) is computed by fixed rules, not learned, so the
spatial math is exact. The strength the paper emphasizes is **grounding QA
coherence**, a metric that rewards the model for getting the grounding and the
answer correct together, where SceneCOT scores well above prior models.

**The single fact that drives our whole project:** the four steps are sequential,
and step 4 can only ever see the objects step 3 grounded. There is no step that
re-examines or corrects grounding. This is what makes a grounding error
unrecoverable, which we develop in the section below.

---

## Research lineage

SceneCOT builds on four earlier lines of work. Each solved part of the problem and
left something open.

| Paper | What it contributed | What it left open |
|-------|--------------------|-------------------|
| Chain-of-Thought Prompting (Wei et al. 2022) | Reasoning step by step before answering greatly improves accuracy on hard problems | Works only on text, with no understanding of 3D space or images |
| PQ3D (2024) | A single architecture that handles different 3D inputs and answers prompts from a shared 3D representation; SceneCOT reuses it as the grounding module | Answers in one pass, so it cannot break a question into multiple reasoning steps |
| Chat-Scene (NeurIPS 2024) | Turns a scene into discrete object tokens an LLM can read, giving strong object referencing; introduced the detector-proposal style grounding SceneCOT also relies on | Depends on the object detector being correct, with no way to recover if it is wrong |
| LEO (ICML 2024) | An embodied agent combining perception, reasoning, and action in one model | Built for general action, so it is weak on deep step by step spatial reasoning |

The common gap is that none combine step by step reasoning with explicit object
grounding. SceneCOT fills it. But note the inherited weakness from PQ3D and
Chat-Scene: SceneCOT's grounding is only as good as the Mask3D proposals and the
PQ3D scoring, and like its predecessors it has no mechanism to recover when that
grounding is wrong. Our experiment measures the consequence of that inherited
weakness inside SceneCOT's sequential design.

---

## The limitation we target, stated precisely

SceneCOT's reasoning chain is sequential and committed. At inference time:

- The grounding module produces one object probability list and the pipeline
  proceeds with it.
- There is **no ground truth available at inference**. The model cannot check its
  own grounding against a correct answer, because at runtime there is no correct
  answer to check against. It has only its own scores.
- There is **no later step that revisits grounding**. Step 4 reasons over whatever
  step 3 produced and emits an answer.

Put together, this means a grounding error in step 3 is **structurally
unrecoverable**: the model has no signal that it grounded the wrong object and no
stage in which it could correct course. The wrong object propagates into the
answer. We call this propagation a **grounding cascade**.

This is the precise sense in which it is a limitation of the architecture, not
just model imperfection. A different design (for example, one that grounds several
candidates and reasons over alternatives, or that produces a confidence flag and
re-grounds when confidence is low) could in principle recover. SceneCOT's
straight-through sequential design cannot. Our experiment quantifies how much of
SceneCOT's behavior this actually explains.

We also examine the reverse direction. SceneCOT's headline strength is grounding QA
coherence, the agreement between grounding and answer. We do **not** claim to
measure that metric. Instead we measure a related but distinct thing: the fraction
of **correct answers that were produced without correct grounding**. When that
fraction is large, it means raw answer accuracy is an inflated picture of how much
the model truly grounds its answers in the scene. This is a statement about the
gap between accuracy and grounding, which is different from the coherence metric
the paper reports.

---

## Datasets: what SceneCOT trains on and what we test on

Because the architecture and the data matter together, here is the full dataset
picture.

**What SceneCOT was trained on: SceneCOT-185K.** The authors assembled a training
set of about 185,000 examples, each carrying a full step by step reasoning chain
(this reasoning annotation is the genuinely new part). It has two sources:

| Portion | Size | Origin | What it constitutes |
|---------|------|--------|---------------------|
| MSQA | 145.6K | An existing situated reasoning benchmark, built on ScanNet indoor scans | Questions that place an agent at a position and orientation in a real scanned room and ask about nearby objects (counting, existence, spatial, navigation, and more) |
| GQA3D | 40K | Created by the authors using GPT-4o | Object centric question answer pairs auto generated from object images, used to strengthen object level grounding |

The underlying 3D scenes come from **ScanNet**, a large public collection of real
indoor room scans (offices, bathrooms, kitchens). The authors did not collect new
scenes; they reused ScanNet and added reasoning chains and the GQA3D questions.

**What we tested on.** Our experiment uses the **MSQA test split** as produced by
our baseline run, stored in `QACOTScanNetMSR3D/results.json` (826 questions across
nine types). From these we focus on the two most grounding dependent types:

| Type | Count | Why we chose it |
|------|-------|-----------------|
| Counting | 133 | Requires finding and enumerating the right objects, so grounding directly determines the count |
| Existence | 96 | Requires finding the object to say whether it is present |

Together that is 229 questions. Of these, 190 were **evaluable** (132 counting, 58
existence); the rest were skipped because their ground truth reasoning chain had no
grounded object to compare against. We deliberately left out types like spatial and
navigation here, because those mix in coordinate to language conversion that would
blur a clean test of grounding. (Spatial is studied separately in Part 3.)

**Important note on ground truth grounding.** The "objects that should have been
grounded" come from the ground truth reasoning chains in the released data. We use
these only as an **after the fact measuring stick** to judge the model's grounding.
They are not available to the model at inference; the model never sees them. This
is exactly why the limitation is structural, as explained above.

---

## Part 1: Reproducing the baseline

Before any experiment we reproduced the authors' evaluation to get a trusted
starting point.

**What the run does.** We ran SceneCOT's full grounded question answering
evaluation on an NVIDIA RTX A6000 GPU. The run loaded the LLaVA backbone, the PQ3D
grounding module, and the trained SceneCOT weights, then answered every test
question and scored itself. It took about three hours and forty seven minutes. The
full console log is in `eval_output_107559.txt`; a short summary is in
`BASELINE_RESULTS.md`.

**What it produced.** For every test question the model saved the question, the
correct answer, and its own complete reasoning chain, including the `<obj_prob>`
grounding section and the final answer. These predictions are in
`experiments/SceneCOT_msqa_beacon3d_test_moe/eval_results/`, with the MSQA file at
`QACOTScanNetMSR3D/results.json` (826 questions) and the GQA3D file at
`QACOTScanNetGQA3D/results.json`.

**Baseline scores we obtained**, confirming the setup works:

| Metric | Score |
|--------|------:|
| Overall (em_refined) | 52.1% |
| Existence | 65.4% |
| Spatial | 51.3% |
| Appearance | 47.1% |

Term notes: **GQA3D / Grounded QA** is the Beacon3D style benchmark where the model
must ground and answer correctly together. **em_refined** is refined exact match,
which compares the predicted and correct answer after light text cleanup so minor
formatting differences are not penalized. The rows below Overall are the same
metric restricted to one question type.

---

## Part 2: How we got from the baseline to our experiment

The baseline gives accuracy numbers but does not explain **why** the model is
wrong. The key realization is that the saved reasoning chains already contain the
grounding step's `<obj_prob>` list, so we can see exactly which objects the model
chose before answering, and compare them to the objects it should have chosen.

That means we can run the entire analysis on the saved predictions, with no GPU and
no model rerun. The baseline tells us how often the model is right; our analysis
opens up each chain and tells us where the failures originate. This is the
diagnostic layer the standard evaluation does not provide.

Concretely, our two questions are:

1. When the model gives a **wrong** answer, did the failure originate in grounding
   (wrong object chosen) or in reasoning (right object chosen, but the final step
   still wrong)?
2. When the model gives a **right** answer, was it actually grounded in the correct
   object, or did the model reach the right answer without correct grounding?

The first question tests whether the cascade is the dominant failure mode. The
second tests whether the model's accuracy is genuinely grounded.

---

## Method: exactly what we measured

We analyze the counting and existence questions. For each question we extract four
things from the saved chains:

- **Objects the model grounded:** every entry in its `<obj_prob>` list with
  confidence at least 0.5.
- **Objects it should have grounded:** the same, from the ground truth chain.
- **Grounding correct (yes/no):** yes if the model grounded at least one of the
  correct object types. We use this lenient rule on purpose; a stricter rule would
  only make grounding look worse, so our cascade numbers are conservative.
- **Answer correct (yes/no):** yes if the model's final answer matches the correct
  answer after text normalization.

Crossing the two yes/no facts gives four groups:

| | Answer right | Answer wrong |
|---|---|---|
| **Grounding right** | Correct (pipeline worked) | Reasoning failure (grounding fine, reasoning slipped) |
| **Grounding wrong** | Correct answer without correct grounding | Cascade (wrong object caused wrong answer) |

The script that does this is `grounding_cascade_analysis.py`. It is fully commented
and reproducible.

---

## Results

Quantitative Results

| Category | Counting (n=132) | Existence (n=58) |
|----------|-----------------:|-----------------:|
| Right grounding, right answer | 8 (6.1%) | 19 (32.8%) |
| Wrong grounding, wrong answer (cascade) | 61 (46.2%) | 15 (25.9%) |
| Wrong grounding, right answer | 48 (36.4%) | 24 (41.4%) |
| Right grounding, wrong answer (reasoning) | 15 (11.4%) | 0 (0.0%) |

**Finding 1: failures concentrate in grounding, and the sequential design makes
them final.** Of all wrong answers, 83.5 percent across both types trace to the
model grounding the wrong object (80.3 percent for counting, 100 percent for
existence). Reasoning failures, where grounding was correct but the answer still
wrong, are rare. Because the pipeline has no step that revisits grounding, these
grounding errors cannot be corrected and propagate straight to the answer. This is
the cascade, and it is the dominant failure mode, which is the structural
limitation we set out to demonstrate.

**Finding 2: a large share of correct answers are not grounded, so accuracy
overstates real understanding.** On counting, 36.4 percent of answers were correct
even though the model grounded the wrong object. Counting has many possible answers,
so this is not coincidence the way a yes/no guess could be. The model frequently
arrives at the right number while attending to the wrong objects. We are careful
here: this is not a claim about the paper's coherence metric. It is a measurement of
the gap between answer accuracy and correct grounding, and it shows that raw
accuracy is an inflated picture of how grounded the model's answers really are.

To distinguish the two wrong-grounding groups clearly:

- A **cascade** is a chain reaction. The wrong grounded object is fed into the
  answering step, so the answer comes out wrong. The grounding error caused the
  answer error.
- A **correct answer without correct grounding** is the opposite surprise. The model
  also grounded the wrong object, but the answer happened to be right anyway, so the
  correctness is not explained by grounding.

---

## Qualitative results: seeing the grounding error

Text findings are stronger when you can see the actual objects involved. For each
wrong-grounding case we build a side by side card from real ScanNet object images:
on the left, the object the model grounded (its mistake); on the right, the object
it should have grounded. The script `make_qualitative_cards.py` produces these.

The clearest example is a counting cascade: for "How many chairs are at your 11
o'clock?" the grounding step returned a single merged proposal labeled "stack of
chairs" rather than separate chairs, so the model counted zero individual chairs
when the answer was one. The card shows the merged object next to a correct single
chair, making the cause of the error visible.

(Generated cards are in the `qualitative_cards/` folder. See the webpage for the
selected set.)

---

## Success cases

When grounding succeeds, the pipeline behaves as designed and the answer is tied to
the correct object.

| Question | What the model grounded | Answer |
|----------|------------------------|--------|
| Is there a copier in the room? | copier | yes (correct) |
| Is there a whiteboard at your 5 o'clock? | whiteboard | yes (correct) |
| Is there a backpack on the table? | backpack | yes (correct) |

These show the intended behavior: the model finds the exact object the question
asks about and answers from it. They also confirm our analysis is not simply
labeling everything a failure; when grounding is right, the pipeline works.

---

## Limitations of our analysis

We are explicit about where our method could be improved.

First, our grounding-correct rule is a label-overlap heuristic. We match object
labels as text, so a singular/plural or synonym mismatch (for example "pictures"
versus "picture") can be counted as a grounding miss. This means our cascade
percentage may be slightly overestimated, although the effect is far too large to
be explained by label noise.

Second, existence questions are yes/no, so a wrong guess still matches the correct
answer half the time by chance. We avoid leaning on this by reporting counting
separately, where the larger answer space removes the coincidence, and we treat the
counting numbers as our strongest evidence.

Third, our result is a strong correlation backed by the known sequential design, not
a controlled intervention. The natural stronger test is to substitute oracle
grounding (force the correct object) and measure how much the answers improve. The
paper's own oracle style experiments point the same direction, and we note this as
the clear next step.

---

## Part 3: Spatial reasoning experiment (Option A)

This part studies how SceneCOT handles spatial relationship questions and is led by
Simarpal Singh.

(Simar to add: the hypothesis, the method, the results, and the success and failure
cases. Related files already in the repository are `diagnose_spatial.py`,
`generate_spatial_test_slice.py`, `test_rotation_resolution.py`,
`run_eval_spatial.sh`, and the `results/` folder.)

---

## How to reproduce

The analysis runs on the saved predictions, so no GPU or model rerun is needed.

Combined analysis on counting and existence:

```bash
python grounding_cascade_analysis.py \
  --results experiments/SceneCOT_msqa_beacon3d_test_moe/eval_results/QACOTScanNetMSR3D/results.json
```

Each type separately, which is what the results table reports:

```bash
python grounding_cascade_analysis.py --results PATH_TO_RESULTS.json --types counting --examples 3
python grounding_cascade_analysis.py --results PATH_TO_RESULTS.json --types existence --examples 3
```

Build the results chart:

```bash
python make_cascade_chart.py
```

Build the qualitative image cards (uses the ScanNet object images):

```bash
python make_qualitative_cards.py \
  --results experiments/SceneCOT_msqa_beacon3d_test_moe/eval_results/QACOTScanNetMSR3D/results.json \
  --img_dir data_assets/scenecot_imgs/imgs/scannet \
  --out qualitative_cards --n 10
```

To reproduce the baseline predictions from scratch, `run_eval.sh` runs the full
SceneCOT evaluation on one GPU and writes the prediction files used above.

---

## Repository structure

```
EE243_Scenecot_Project/
  scenecot/                          original SceneCOT codebase (git submodule)
  experiments/
    SceneCOT_msqa_beacon3d_test_moe/
      eval_results/
        QACOTScanNetMSR3D/results.json    MSQA predictions, 826 questions
        QACOTScanNetGQA3D/results.json    GQA3D and Beacon3D predictions
  results/                           spatial experiment outputs (Option A)
  grounding_cascade_analysis.py      our Option B analysis script
  make_cascade_chart.py              makes the results chart
  make_qualitative_cards.py          builds the side by side image cards
  qualitative_cards/                 generated qualitative result images
  diagnose_spatial.py                Option A, spatial failure parsing
  generate_spatial_test_slice.py     Option A, spatial stress test slice
  test_rotation_resolution.py        Option A, rotation resolution test
  run_eval.sh                        batch script for the baseline evaluation
  run_eval_spatial.sh                batch script for the spatial evaluation
  eval_output_107559.txt             full baseline evaluation log
  BASELINE_RESULTS.md                baseline numbers and file locations
  README.md
```

---

## Acknowledgements

We thank the authors of SceneCOT for releasing their code, weights, and data, and
the EE 243 course staff and the UCR CSE HPC cluster for the compute used in this
project. This is a course project for analysis and educational purposes, and all
rights to the original SceneCOT work belong to its authors.
