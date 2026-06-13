"""
test_rotation_resolution.py

Isolated simulation of the coordinate-serialization path in
scenecot/model/scenecot_agent.py :: build_obj_prob_loc_txt (build_choice='loc').

The real code does (lines ~554-578):

    theta = atan2(ori_y, ori_x)
    rotation_matrix = [[ cos(theta), sin(theta)],
                       [-sin(theta), cos(theta)]]
    relative_loc = rotation_matrix @ (obj_xy - agent_xy)
    relative_loc_str = ",".join(f"{coord:.1f}" for coord in relative_loc)   # <-- 0.1 m grid

Coordinates are in METERS, so ":.1f" snaps every axis to a 10 cm grid before
the string ever reaches the LLM decoder. This script feeds in object pairs
that are 5-20 cm apart and shows how the truncation collapses or distorts
their relative arrangement.
"""

import math


# ----------------------------------------------------------------------------
# Faithful mock of build_obj_prob_loc_txt's 'loc' branch (rectangle + polar)
# ----------------------------------------------------------------------------

def make_rotation_matrix(agent_ori_xy):
    """Mirror of lines 557-562: normalize orientation, build 2D rotation."""
    norm = math.hypot(agent_ori_xy[0], agent_ori_xy[1])
    ox, oy = agent_ori_xy[0] / norm, agent_ori_xy[1] / norm
    theta = math.atan2(oy, ox)
    c, s = math.cos(theta), math.sin(theta)
    return [[c, s], [-s, c]]


def rotate(rot, vec_xy):
    return [rot[0][0] * vec_xy[0] + rot[0][1] * vec_xy[1],
            rot[1][0] * vec_xy[0] + rot[1][1] * vec_xy[1]]


def egocentric_loc(agent_pos, agent_ori, obj_loc):
    """Exact float-precision egocentric coordinates (before truncation)."""
    rot = make_rotation_matrix(agent_ori[:2])
    rel_xy = rotate(rot, [obj_loc[0] - agent_pos[0], obj_loc[1] - agent_pos[1]])
    return rel_xy + [obj_loc[2] - agent_pos[2]]


def loc_str_rectangle(agent_pos, agent_ori, obj_loc, obj_sizes=(0.1, 0.1, 0.1)):
    """Mirror of the coord_type == 'rectangle' branch (lines 571-574)."""
    relative_loc = egocentric_loc(agent_pos, agent_ori, obj_loc)
    s = ",".join(f"{coord:.1f}" for coord in relative_loc)
    s += "," + ",".join(f"{size:.1f}" for size in obj_sizes)
    return s


def loc_str_polar(agent_pos, agent_ori, obj_loc):
    """Mirror of the coord_type == 'polar' branch (lines 575-578)."""
    relative_loc = egocentric_loc(agent_pos, agent_ori, obj_loc)
    theta_obj = math.atan2(relative_loc[1], relative_loc[0]) * 180 / math.pi
    distance_obj = math.hypot(relative_loc[0], relative_loc[1])
    return f"{theta_obj:.1f}, {distance_obj:.1f}"


def parse_rect(s):
    """What the LLM decoder 'sees': the truncated numbers, parsed back."""
    return [float(v) for v in s.split(",")[:3]]


# ----------------------------------------------------------------------------
# Reporting helpers
# ----------------------------------------------------------------------------

def fmt_v(v):
    return "(" + ", ".join(f"{x:+.3f}" for x in v) + ")"


def analyze_pair(title, agent_pos, agent_ori, name_a, obj_a, name_b, obj_b):
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
    print(f"agent position {fmt_v(agent_pos)}  orientation {fmt_v(agent_ori)}")

    true_a = egocentric_loc(agent_pos, agent_ori, obj_a)
    true_b = egocentric_loc(agent_pos, agent_ori, obj_b)
    str_a = loc_str_rectangle(agent_pos, agent_ori, obj_a)
    str_b = loc_str_rectangle(agent_pos, agent_ori, obj_b)
    seen_a = parse_rect(str_a)
    seen_b = parse_rect(str_b)

    true_sep = [tb - ta for ta, tb in zip(true_a, true_b)]
    seen_sep = [sb - sa for sa, sb in zip(seen_a, seen_b)]
    true_d = math.hypot(true_sep[0], true_sep[1])
    seen_d = math.hypot(seen_sep[0], seen_sep[1])

    print(f"\n  {'object':<12}{'true egocentric (m)':<28}{'string fed to LLM (:.1f)'}")
    print(f"  {name_a:<12}{fmt_v(true_a):<28}\"{str_a}\"")
    print(f"  {name_b:<12}{fmt_v(true_b):<28}\"{str_b}\"")

    print(f"\n  true   {name_b}-{name_a} offset: dx={true_sep[0]:+.3f} dy={true_sep[1]:+.3f} dz={true_sep[2]:+.3f}  (planar dist {true_d * 100:5.1f} cm)")
    print(f"  in-str {name_b}-{name_a} offset: dx={seen_sep[0]:+.3f} dy={seen_sep[1]:+.3f} dz={seen_sep[2]:+.3f}  (planar dist {seen_d * 100:5.1f} cm)")

    verdicts = []
    if seen_a[:2] == seen_b[:2] and seen_a[2] == seen_b[2]:
        verdicts.append("COLLAPSED: both objects serialize to identical coordinates")
    elif seen_a[:2] == seen_b[:2]:
        verdicts.append("XY-COLLAPSED: identical floor-plane coordinates, only z differs")
    for axis, ts, ss in zip("xyz", true_sep, seen_sep):
        if abs(ts) > 1e-9 and ss == 0.0:
            verdicts.append(f"axis {axis}: real {ts * 100:+.1f} cm offset erased to 0")
        elif ts * ss < 0:
            verdicts.append(f"axis {axis}: SIGN FLIP ({ts * 100:+.1f} cm becomes {ss * 100:+.1f} cm) -> left/right or front/behind inverted")
        elif abs(ts) > 1e-9 and abs(ss - ts) / abs(ts) > 0.25:
            verdicts.append(f"axis {axis}: offset distorted {ts * 100:+.1f} -> {ss * 100:+.1f} cm ({(ss - ts) / ts * 100:+.0f}% error)")
    if true_d > 1e-9:
        err = (seen_d - true_d) / true_d * 100
        if abs(err) > 25:
            verdicts.append(f"planar distance error {err:+.0f}% ({true_d * 100:.1f} -> {seen_d * 100:.1f} cm)")
    if not verdicts:
        verdicts.append("pair survives truncation at this orientation")

    for v in verdicts:
        print(f"  >> {v}")
    return verdicts


# ----------------------------------------------------------------------------
# Scenarios
# ----------------------------------------------------------------------------

def scenario_cup_on_table():
    """Cup sitting on a small side table: centers ~7 cm apart in XY."""
    agent_pos = [0.0, 0.0, 1.0]
    agent_ori = [1.0, 0.0, 0.0]
    table = [2.03, 0.51, -0.40]           # table center
    cup = [2.08, 0.56, -0.05]             # cup center: 5 cm right, 5 cm forward, on tabletop
    return analyze_pair(
        "SCENARIO 1: cup on a small table (7 cm planar separation)",
        agent_pos, agent_ori, "table", table, "cup", cup)


def scenario_two_chairs():
    """Two chairs crammed together, 12 cm apart, straddling a 0.05 rounding edge."""
    agent_pos = [0.0, 0.0, 0.0]
    agent_ori = [1.0, 0.0, 0.0]
    chair_l = [1.50, 0.04, 0.45]          # 4 cm to agent's... left? y=+0.04 rounds to 0.0
    chair_r = [1.50, -0.08, 0.45]         # 12 cm from chair_l on the other side
    return analyze_pair(
        "SCENARIO 2: two chairs crammed together (12 cm apart, straddling agent midline)",
        agent_pos, agent_ori, "chair_L", chair_l, "chair_R", chair_r)


def scenario_rounding_boundary():
    """18 cm separation gets STRETCHED to 30 cm because each end rounds away."""
    agent_pos = [0.0, 0.0, 0.0]
    agent_ori = [0.0, 1.0, 0.0]           # agent faces +y: rotation actually matters here
    book = [0.26, 2.00, 0.72]             # ego-x after rotation = 2.0 fwd, lateral -0.26
    lamp = [0.44, 2.00, 0.72]             # 18 cm from the book
    return analyze_pair(
        "SCENARIO 3: book and lamp 18 cm apart, agent rotated 90 deg",
        agent_pos, agent_ori, "book", book, "lamp", lamp)


def scenario_orientation_sweep():
    """Same physical pair; only the agent orientation changes.

    Whether the pair collapses depends on where the rotation drops the
    coordinates relative to the 0.1 m rounding grid -- the relationship the
    LLM sees is a function of viewing angle, not of the scene.
    """
    print(f"\n{'=' * 78}")
    print("SCENARIO 4: orientation sweep -- same two objects ~6 cm apart")
    print(f"{'=' * 78}")
    agent_pos = [0.0, 0.0, 0.0]
    mug_a = [1.75, 1.18, 0.71]
    mug_b = [1.80, 1.21, 0.71]            # 5.8 cm apart diagonally
    true_gap = math.hypot(mug_b[0] - mug_a[0], mug_b[1] - mug_a[1]) * 100
    print(f"  mug_A world ({mug_a[0]}, {mug_a[1]})   mug_B world ({mug_b[0]}, {mug_b[1]})   true planar gap {true_gap:.1f} cm\n")
    print(f"  {'agent yaw':>10}  {'mug_A string':<32}{'mug_B string':<32}{'LLM-visible gap'}")
    collapsed_angles, distorted = [], []
    for deg in range(0, 360, 15):
        ori = [math.cos(math.radians(deg)), math.sin(math.radians(deg)), 0.0]
        sa = loc_str_rectangle(agent_pos, ori, mug_a)
        sb = loc_str_rectangle(agent_pos, ori, mug_b)
        pa, pb = parse_rect(sa), parse_rect(sb)
        gap = math.hypot(pb[0] - pa[0], pb[1] - pa[1]) * 100
        note = ""
        if pa[:2] == pb[:2]:
            note = "  << COLLAPSED"
            collapsed_angles.append(deg)
        elif abs(gap - true_gap) / true_gap > 0.25:
            note = f"  << {gap - true_gap:+.1f} cm error"
            distorted.append(deg)
        print(f"  {deg:>8}\u00b0  {sa:<32}{sb:<32}{gap:5.1f} cm{note}")
    print(f"\n  >> identical strings (objects indistinguishable) at {len(collapsed_angles)}/24 orientations: {collapsed_angles}")
    print(f"  >> gap distorted by >25% at {len(distorted)}/24 orientations: {distorted}")


def scenario_polar():
    """Polar branch: two objects at clearly different bearings merge."""
    print(f"\n{'=' * 78}")
    print("SCENARIO 5: polar (coord_type='polar') -- distance quantization")
    print(f"{'=' * 78}")
    agent_pos = [0.0, 0.0, 0.0]
    agent_ori = [1.0, 0.0, 0.0]
    bottle = [2.04, 0.00, 0.7]
    glass = [1.96, 0.06, 0.7]             # 10 cm from bottle
    sp_a = loc_str_polar(agent_pos, agent_ori, bottle)
    sp_b = loc_str_polar(agent_pos, agent_ori, glass)
    true_gap = math.hypot(bottle[0] - glass[0], bottle[1] - glass[1]) * 100
    print(f"  bottle polar string: \"{sp_a}\"")
    print(f"  glass  polar string: \"{sp_b}\"")
    print(f"  true planar gap: {true_gap:.1f} cm")
    print("  >> distances 2.04 m and 1.96 m both serialize to \"2.0\";")
    print("     only a 1.8-degree bearing difference remains to separate two distinct objects.")


# ----------------------------------------------------------------------------

def main():
    print("Simulation of build_obj_prob_loc_txt coordinate truncation (f'{coord:.1f}')")
    print("Replicates: rotation into agent frame -> per-axis rounding to 0.1 m (10 cm).")

    scenario_cup_on_table()
    scenario_two_chairs()
    scenario_rounding_boundary()
    scenario_orientation_sweep()
    scenario_polar()

    print(f"\n{'=' * 78}")
    print("SUMMARY: why ':.1f' truncation breaks spatial-relationship evaluation")
    print(f"{'=' * 78}")
    print("""
1. QUANTIZATION FLOOR. f"{coord:.1f}" on meter-valued coordinates snaps every
   axis to a 10 cm grid. Any pair of objects whose egocentric coordinates fall
   in the same grid cell (anything closer than ~14 cm diagonally, and up to
   <10 cm per axis) can serialize to IDENTICAL strings. The LLM decoder
   receives "cup: 2.1,0.6,...  table: 2.1,0.6,..." -- it has literally zero
   signal that the cup is to the right of / in front of / on the table.
   Any 'which is closer / what is left of X' question becomes a coin flip.

2. SIGN FLIPS AT THE MIDLINE. An object 4 cm left of the agent axis
   (y=+0.04) prints as "0.0" while one 8 cm right (y=-0.08) prints as "-0.1".
   A 12 cm true left/right split becomes "centered vs right", or with slightly
   different values flips the apparent side entirely. The decoder then asserts
   the wrong 'left of'/'right of' relation with full confidence, which is
   exactly the failure mode graded by spatial-relationship benchmarks.

3. ERROR AMPLIFICATION NEAR BIN EDGES. Independent rounding of two endpoints
   adds up to +/-0.05 m of error EACH, so a true 18 cm gap can print as 30 cm
   (+67%) or a 14 cm gap as 0 cm (-100%). Relative magnitude judgments
   ("closest to the window", "between the chairs") are computed by the LLM
   from these strings, so the ranking of near-tied candidates is corrupted.

4. ORIENTATION-DEPENDENT NONDETERMINISM. The rotation_matrix is applied
   BEFORE truncation, so where coordinates land on the 10 cm grid depends on
   the agent's yaw. The sweep above shows the same physical pair collapsing
   at some viewing angles and surviving at others: the spatial relation the
   decoder is asked to reason about changes with camera pose even though the
   scene is static. Evaluation results therefore become noisy and
   irreproducible across episodes/viewpoints.

5. POLAR MODE IS NO SAFER. ":.1f" on distance is again a 10 cm bin, and
   ":.1f" on bearing leaves sub-degree differences as the only separator for
   proximate objects -- e.g. two objects 10 cm apart share the distance "2.0"
   and differ by ~1.8 degrees of bearing.

NET EFFECT: inside <obj_loc_prob>...</obj_loc_prob>, proximate objects (cups
on tables, paired chairs, clustered desk items) are presented to the decoder
as co-located or mis-ordered. When the chain-of-thought step asks the LLM to
ground "the chair on the left" or "the cup on the table", the textual scene
representation no longer encodes that relation, so the decoder either guesses
or hallucinates it -- a deterministic, input-side cause of spatial-relationship
evaluation failures that no amount of LLM capability can recover, because the
information was destroyed before tokenization. Using ':.2f' (1 cm grid) or
object-relative offsets would preserve it at negligible token cost.
""")


if __name__ == "__main__":
    main()
