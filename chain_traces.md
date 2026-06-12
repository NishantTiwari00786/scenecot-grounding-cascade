# Qualitative Results: Annotated Reasoning Chains

Each example shows SceneCOT's actual step by step output on a cascade case, where a grounding error led directly to a wrong answer. We annotate where in the chain the error occurred.


## Example 1 (existence)

**Question:** USER: In front of you, at a middle distance, there's a white louvered window for light control and privacy. To your left, near you, there's an empty brown paper bag. Is there a book at your 5 o'clock?

**SceneCOT's reasoning chain:**

```
Step 1  recognize task : This is a existence question, so I need to ground the corresponding objects to answer it.
Step 3  grounding      : boxes 0.73  kitchen counter 0.08  table 0.06  sink 0.06  open kitchen cabinet 0.01  open kitchen cabinet 0.01
Step 4  answer         : no
```

**What went wrong:** the grounding step committed to `boxes`, but the correct object was `book`. Because the answer is built only from the grounded objects, the model answered `no` when the correct answer was `yes`. The error began in grounding and the sequential pipeline had no way to recover.


## Example 2 (existence)

**Question:** n the trash can. There is a cluttered white desk at your 12 o'clock, a black office chair in use at your 3 o'clock, and a wall at your 4 o'clock. Is there a cup to the left of the shelf on your right?

**SceneCOT's reasoning chain:**

```
Step 1  recognize task : This is a existence question, so I need to ground the corresponding objects to answer it.
Step 3  grounding      : bookshelves 0.84  stacks of cups 0.55  pillows 0.04  plant 0.03  file cabinets 0.01  cabinets 0.01
Step 4  answer         : no
```

**What went wrong:** the grounding step committed to `bookshelves`, but the correct object was `cup`. Because the answer is built only from the grounded objects, the model answered `no` when the correct answer was `yes`. The error began in grounding and the sequential pipeline had no way to recover.


## Example 3 (existence)

**Question:** USER: You are painting the door. There is a beige sink with an oval bowl shape and smooth ceramic texture at your 5 o'clock. There is a bottle at your 5 o'clock. Is there a mirror at your 5 o'clock?

**SceneCOT's reasoning chain:**

```
Step 1  recognize task : This is a existence question, so I need to ground the corresponding objects to answer it.
Step 3  grounding      : stick 0.78  mirror 0.34  sink 0.09  bathroom vanity 0.01  bathroom vanity 0.01  doors 0.00
Step 4  answer         : no
```

**What went wrong:** the grounding step committed to `stick`, but the correct object was `mirror`. Because the answer is built only from the grounded objects, the model answered `no` when the correct answer was `yes`. The error began in grounding and the sequential pipeline had no way to recover.


## Example 4 (existence)

**Question:** ER: You are knocking on the door. In the back, at a middle distance, there's a white plastic object for dispensing paper towels and a metallic silver rail for support. Is there a backpack in the room?

**SceneCOT's reasoning chain:**

```
Step 1  recognize task : This is a existence question, so I need to ground the corresponding objects to answer it.
Step 3  grounding      : paper towel dispenser 0.86  soap dispenser 0.24  soap dispenser 0.24  toilet paper rolls 0.08  sink 0.05  handicap bar 0.04
Step 4  answer         : no
```

**What went wrong:** the grounding step committed to `paper towel dispenser`, but the correct object was `backpack`. Because the answer is built only from the grounded objects, the model answered `no` when the correct answer was `yes`. The error began in grounding and the sequential pipeline had no way to recover.


## Example 5 (existence)

**Question:** d rectangular prism shape at your 8 o'clock direction. To your right, at a middle distance, there's a white lamp. To your left, near you, there's a white mirror. Is there a cabinet at your 11 o'clock?

**SceneCOT's reasoning chain:**

```
Step 1  recognize task : This is a existence question, so I need to ground the corresponding objects to answer it.
Step 3  grounding      : closet walls 0.60  ceiling light 0.12  shower wall 0.04  shower wall 0.04  shower wall 0.04  doorframe 0.02
Step 4  answer         : no
```

**What went wrong:** the grounding step committed to `closet walls`, but the correct object was `cabinet`. Because the answer is built only from the grounded objects, the model answered `no` when the correct answer was `yes`. The error began in grounding and the sequential pipeline had no way to recover.


## Example 6 (existence)

**Question:** ite front-loading washing machine. There is a white rectangular prism clothes dryer for clothes drying and a white irregular clothing in your 5 o'clock direction. Is there a picture at your 9 o'clock?

**SceneCOT's reasoning chain:**

```
Step 1  recognize task : This is a existence question, so I need to ground the corresponding objects to answer it.
Step 3  grounding      : trash can 0.54  bulletin board 0.51  bulletin board 0.01  bulletin board 0.01
Step 4  answer         : no
```

**What went wrong:** the grounding step committed to `trash can`, but the correct object was `picture`. Because the answer is built only from the grounded objects, the model answered `no` when the correct answer was `yes`. The error began in grounding and the sequential pipeline had no way to recover.
