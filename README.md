# EE243 Final Project: Analyzing the Grounding Bottleneck in SceneCOT

**Course:** EE 243 Advanced Computer Vision, Spring 2026, UC Riverside
**Team:** Nishant Tiwari, Simarpal Singh

**Paper studied:** SceneCOT: Eliciting Grounded Chain-of-Thought Reasoning in 3D Scenes (ICLR 2026)

Project webpage: UPDATE LINK HERE

Video walkthrough: UPDATE LINK HERE

---

## What this project is about

3D Vision-Language Models can look at a scanned room and answer questions about
it, but they often give an answer that sounds right without actually looking at
the correct object. SceneCOT tries to fix this by making the model reason in
clear steps and point to a specific object before it answers.

In this project we did two things. First, we reproduced SceneCOT's evaluation
using the authors' public code and weights to confirm we could run it and get
sensible numbers. Second, we took those baseline outputs and ran our own
analysis to find a specific weakness in how the model works. Our finding is
that SceneCOT's step by step design has a built in problem: when the grounding
step picks the wrong object, the final answer is almost always wrong too. We
also found that the model often gets the right answer while looking at the wrong
object, which means a lot of its accuracy is not actually earned through correct
grounding.

The course asked for depth over breadth, so we focused on two clear bottlenecks
and studied it carefully with both success and failure cases instead of touching
many topics shallowly.

---

## Table of contents

1. [Research lineage](#research-lineage)
2. [The frontier paper: SceneCOT](#the-frontier-paper-scenecot)
3. [Part 1: Reproducing the baseline](#part-1-reproducing-the-baseline)
4. [Part 2: Our experiment (Option B, grounding cascade)](#part-2-grounding-cascade)
5. [Results](#results)
6. [Success cases](#success-cases)
7. [Failure cases](#failure-cases)
8. [Limitations of our analysis](#limitations-of-our-analysis)
9. [Part 3: Spatial reasoning experiment (Option A)](#part-3-spatial-reasoning-experiment)
10. [How to reproduce](#how-to-reproduce)
11. [Repository structure](#repository-structure)
12. [Acknowledgements](#acknowledgements)

---

## Research lineage

SceneCOT builds on four earlier lines of work. Each one solved part of the
problem and left something open for the next.

| Paper | What it contributed | What it left open |
|-------|--------------------|-------------------|
| Chain-of-Thought Prompting (Wei et al. 2022) | Showed that making a model reason step by step before answering greatly improves accuracy on hard problems | Works only on text, with no understanding of 3D space or images |
| PQ3D (2024) | A single architecture that handles different 3D inputs and answers prompts from a shared 3D representation | Answers in one pass, so it cannot break a question into multiple reasoning steps |
| Chat-Scene (NeurIPS 2024) | Turns a scene into a list of object tokens that a language model can read, which gives strong object referencing | Depends on the object detector being correct, and has no real multi step reasoning |
| LEO (ICML 2024) | An embodied agent that combines seeing, reasoning, and acting in one model | Built for general action, so it is weak on deep step by step spatial reasoning |

The common gap across all four is that none of them combine step by step
reasoning with explicit object grounding that is checked at every step. That is
exactly what SceneCOT adds. The full write up of each paper is on the project
webpage.

---

## The frontier paper: SceneCOT

SceneCOT is built on top of the LLaVA-1.5 language model and answers every 3D
question using four steps:

1. Task recognition. The model first decides what kind of question it is, such
   as counting, existence, spatial, or navigation. This decision controls how
   the later steps work.
2. Region localization. A rule based engine narrows the scene down to the part
   the question is about, using directions like left, right, or clock positions
   such as 3 o'clock.
3. Entity grounding. A grounding module finds the target object and outputs a
   list of candidate objects with confidence scores. In the saved output this
   shows up as a section called obj_prob.
4. Grounded reasoning. The model builds a clue from the grounded objects, for
   example a list of objects for counting, and then produces the final answer
   from that clue.

An important design choice is that the geometry math, like turning coordinates
into clock directions, is done by fixed rules instead of being learned. This
keeps the spatial math exact. The strength the paper highlights is grounding
coherence, meaning the answer is actually tied to the correct object rather than
being a lucky guess.

---

## Part 1: Reproducing the baseline

Before doing any experiment we reproduced the authors' evaluation so we had a
trusted starting point.

**What the baseline run does.** We ran SceneCOT's full grounded question
answering evaluation on an NVIDIA RTX A6000 GPU. The model loaded the LLaVA
backbone, the PQ3D grounding module, and the trained SceneCOT weights, then
answered every test question and scored itself. The full run took about three
hours and forty seven minutes and produced two prediction files, one for each
benchmark. The complete console log is saved in eval_output_107559.txt, and a
short summary of the numbers is in BASELINE_RESULTS.md.

**What the baseline produced.** For every test question the model saved the
question, the correct answer, and its own full reasoning chain including the
grounding step and the final answer. These predictions are stored in
experiments/SceneCOT_msqa_beacon3d_test_moe/eval_results/. The MSQA file
(QACOTScanNetMSR3D/results.json) contains 826 situated reasoning questions,
and the GQA3D file (QACOTScanNetGQA3D/results.json) contains the Beacon3D
style grounded question answering set.

**Where the data comes from.** The questions are from the MSQA benchmark, which
is built on the ScanNet dataset of real indoor room scans. Each MSQA question
places an agent at a position and orientation in the room and asks something
about the surrounding objects. The ground truth reasoning chains, including
which objects should be grounded, come from the SceneCOT-185K training data
released with the paper.

**Baseline scores we obtained.** These confirmed our setup was working, and they
are the reference point for the rest of the project.

| Metric | Score |
|--------|------:|
| Overall (em_refined) | 52.1% |
| Existence | 65.4% |
| Spatial | 51.3% |
| Appearance | 47.1% |

A quick note on the terms so the table is clear:

- GQA3D, also called Grounded QA, is the Beacon3D style benchmark where the
  model must both ground the correct object and answer correctly.
- em_refined stands for refined exact match. It checks whether the model's
  predicted answer matches the correct answer after light cleanup of the text,
  so small formatting differences are not penalized.
- The rows below Overall are the same metric measured on a single question type
  only, so Existence is the exact match score on existence questions, Spatial on
  spatial relationship questions, and Appearance on appearance questions.

---

## Part 2: Our experiment (Option B, grounding cascade)

**From baseline to experiment.** The baseline tells us how often the model is
right, but it does not tell us why it is wrong. We wanted to look inside the
reasoning chains the baseline already saved and find out where the failures
actually come from. The useful detail is that the saved predictions include the
grounding step's object list, so we can see exactly which objects the model
chose before answering. This let us run our whole analysis on the existing
prediction files without rerunning the model on the GPU. In other words, the
baseline gives accuracy numbers, and our analysis explains the reason behind
those numbers, which the baseline evaluation does not do on its own.

**The idea we are testing.** SceneCOT answers in a fixed order, and the final
answering step can only use the objects that the grounding step handed it. It
works like an assembly line. If the grounding step picks the wrong object, the
answering step has no way to recover, because it is reasoning about the wrong
thing. We call this a grounding cascade. Our two questions are:

1. When the model gives a wrong answer, is it mostly because grounding failed,
   or because the reasoning failed even though grounding was correct?
2. When the model gives a right answer, did it actually ground the correct
   object, or did it get lucky while looking at the wrong object?

**How we measured it.** We focused on counting and existence questions, because
these depend most directly on grounding the right objects. For each question we
read two things from the saved output. From the model's prediction we took every
object in its grounding list with a confidence of at least 0.5, which gives the
set of objects the model actually grounded. From the ground truth chain we took
the same thing, which gives the objects the model should have grounded. We then
checked two simple yes or no facts: did the model ground at least one of the
correct object types, and was the final answer correct. Putting these together
gives four groups:

- Right grounding and right answer. The pipeline worked as intended.
- Wrong grounding and wrong answer. This is the cascade, where a grounding
  mistake led directly to a wrong answer.
- Wrong grounding and right answer. This is a lucky guess, where the model was
  correct even though it grounded the wrong object.
- Right grounding and wrong answer. This is a genuine reasoning failure, where
  grounding was fine but the final step still went wrong.

The script that does this is grounding_cascade_analysis.py.

---

## Results

The chart below shows the four groups for counting and existence questions. The
exact numbers follow underneath.


| Category | Counting (n=132) | Existence (n=58) |
|----------|-----------------:|-----------------:|
| Right grounding, right answer | 8 (6.1%) | 19 (32.8%) |
| Wrong grounding, wrong answer (cascade) | 61 (46.2%) | 15 (25.9%) |
| Wrong grounding, right answer (lucky) | 48 (36.4%) | 24 (41.4%) |
| Right grounding, wrong answer (reasoning) | 15 (11.4%) | 0 (0.0%) |

**Finding 1: most failures come from grounding, not reasoning.** Of all the
wrong answers, 83.5 percent across both types trace back to the model grounding
the wrong object. For counting it is 80.3 percent and for existence it is 100
percent. Very few wrong answers happened when grounding was actually correct.
This supports the cascade idea: the grounding step is the real bottleneck, and
because the steps run in order, its mistakes carry straight through to the
answer.

**Finding 2: many correct answers are not actually grounded.** On counting
questions, 36.4 percent of answers were correct even though the model grounded
the wrong object. Counting has many possible answers, so this cannot be
explained by guessing. The model is often landing on the right number while
looking at the wrong objects. This is direct evidence of the grounding
coherence problem that the paper set out to solve, showing it is still present.

To make the difference between the two failure groups clear:

- A cascade is a real chain reaction. The model grounded the wrong object, so
  the wrong object went into the answering step, so the answer came out wrong.
  The grounding mistake caused the answer mistake.
- A lucky guess is the opposite surprise. The model also grounded the wrong
  object, but the final answer still happened to match the correct one. The
  answer is right, but not for the right reason, so it does not show real
  understanding of the scene.

---

## Success cases

When the grounding step works, the whole pipeline behaves the way it should and
the answer is tied to the correct object.

| Question | What the model grounded | Answer |
|----------|------------------------|--------|
| Is there a copier in the room? | copier | yes (correct) |
| Is there a whiteboard at your 5 o'clock? | whiteboard | yes (correct) |
| Is there a backpack on the table? | backpack | yes (correct) |

These show SceneCOT's intended strength. The model found the exact object the
question asked about and answered correctly based on it.

---

## Failure cases

**Cascade failures, where wrong grounding led to a wrong answer.**

| Question | Model grounded | Should have grounded | Answer (correct) |
|----------|---------------|---------------------|------------------|
| How many chairs are at your 11 o'clock? | stack of chairs (merged together) | chair | zero (one) |
| How many soap dishes are present? | container | soap dish | two (three) |
| How many toilet paper holders are behind you? | toilet paper rolls | toilet paper holder | two (one) |

The chairs example is the clearest. The grounding step returned one merged blob
labeled stack of chairs instead of separate chairs, so the model counted zero
individual chairs. The mistake started in the grounding step and the answering
step could not fix it.

**Lucky guesses, where wrong grounding still gave a right answer.**

| Question | Model grounded | Should have grounded | Answer |
|----------|---------------|---------------------|--------|
| How many soap dishes are in the shower area behind you? | shelf | soap dish | three (correct by luck) |
| Is there any object made of glass at your 2 o'clock? | nothing above 0.5 | mirror | yes (correct by luck) |

Here the model grounded the wrong thing, or nothing at all, but still produced
the right answer. The answer looks correct on the scoreboard but was not reached
by actually finding the right object.

---

## Limitations of our analysis

We want to be clear about where our method could be improved.

First, the way we decide whether grounding is correct is approximate. We count
grounding as correct if the model grounded at least one object whose label
matches a correct object label. This is a simple text match, so a small mismatch
like pictures versus picture can be counted as wrong even when the meaning is the
same. Because of this, our cascade percentage may be slightly higher than the
true value. The effect we see is large enough that this does not change the main
conclusion, but it is worth stating.

Second, existence questions are yes or no, so a wrong guess still has a fifty
percent chance of matching the correct answer by coincidence. To avoid leaning
on this, we report counting separately, where there are many possible answers
and coincidence is far less likely. Our strongest evidence comes from the
counting numbers for this reason.

Third, our analysis shows a strong link between wrong grounding and wrong
answers, but it does not perform a controlled test that forces the grounding to
be correct. A stronger version of this experiment would replace the model's
grounding with perfect grounding and then measure how much the answers improve.
We note this as a natural next step.

---

## Part 3: Spatial reasoning experiment (Option A)

This section covers the second experiment in the project, which studies how
SceneCOT handles spatial relationship questions. This part of the work was led
by Simarpal Singh.

(Simar to add: the hypothesis, the method, the results, and the success and
failure cases. Related files already in the repository are diagnose_spatial.py,
generate_spatial_test_slice.py, test_rotation_resolution.py,
run_eval_spatial.sh, and the results/ folder.)

---

## How to reproduce

Our analysis runs on the saved prediction files, so you do not need a GPU or a
model rerun to reproduce the numbers.

Run the combined analysis on counting and existence questions:

```bash
python grounding_cascade_analysis.py \
  --results experiments/SceneCOT_msqa_beacon3d_test_moe/eval_results/QACOTScanNetMSR3D/results.json
```

Run each question type on its own, which is what we report in the results table:

```bash
python grounding_cascade_analysis.py --results PATH_TO_RESULTS.json --types counting --examples 3
python grounding_cascade_analysis.py --results PATH_TO_RESULTS.json --types existence --examples 3
```

Make the results chart:

```bash
python make_cascade_chart.py
```

To reproduce the baseline predictions from scratch, the batch script
run_eval.sh runs the full SceneCOT evaluation on one GPU and writes the same
prediction files used above.

---

## Repository structure

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
diagnose_spatial.py                Option A, spatial failure parsing
generate_spatial_test_slice.py     Option A, spatial stress test slice
test_rotation_resolution.py        Option A, rotation resolution test
run_eval.sh                        batch script for the baseline evaluation
run_eval_spatial.sh                batch script for the spatial evaluation
eval_output_107559.txt             full baseline evaluation log
BASELINE_RESULTS.md                baseline numbers and file location

## Acknowledgements

We thank the authors of SceneCOT for releasing their code, weights, and data,
and the EE 243 course staff and the UCR CSE HPC cluster for the compute used in
this project. This is a course project for analysis and educational purposes,
and all rights to the original SceneCOT work belong to its authors.
