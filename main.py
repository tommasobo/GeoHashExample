#!/usr/bin/env python3
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

###############################################################################
# CONFIGURATION & GLOBALS
###############################################################################
MATCH_THRESHOLD = 0.2   # A scene point is considered matching if its distance to any transformed model point is below this.
random.seed(42)         # For reproducibility

# Fixed axis limits for all subplots (for example)
XMIN, XMAX = -3, 7
YMIN, YMAX = -1, 8

###############################################################################
# DETAILED CAT MODEL: 20 VERTICES (Approximate the ASCII Cat Face)
#
# The intended outline (vertices in order):
#
#   1. (-2.0, 4.0)       -> left ear tip
#   2. (-1.8, 4.2)       -> left ear top curve
#   3. (-1.5, 4.0)       -> left ear inner corner
#   4. (-1.2, 3.8)       -> left upper cheek
#   5. (-0.8, 4.0)       -> left forehead
#   6. (-0.5, 4.2)       -> near top center left
#   7. (0.0, 4.3)        -> top center
#   8. (0.5, 4.2)        -> near top center right
#   9. (0.8, 4.0)        -> right forehead
#  10. (1.2, 3.8)        -> right upper cheek
#  11. (1.5, 4.0)        -> right ear inner corner
#  12. (1.8, 4.2)        -> right ear top curve
#  13. (2.0, 4.0)        -> right ear tip
#  14. (2.2, 3.0)        -> right face high
#  15. (2.0, 2.0)        -> right face low
#  16. (1.5, 1.0)        -> right jaw
#  17. (0.0, 0.5)        -> chin
#  18. (-1.5, 1.0)       -> left jaw
#  19. (-2.0, 2.0)       -> left face low
#  20. (-2.2, 3.0)       -> left face high
###############################################################################
detailed_cat_face = [
    (-2.0, 4.0),
    (-1.8, 4.2),
    (-1.5, 4.0),
    (-1.2, 3.8),
    (-0.8, 4.0),
    (-0.5, 4.2),
    (0.0, 4.3),
    (0.5, 4.2),
    (0.8, 4.0),
    (1.2, 3.8),
    (1.5, 4.0),
    (1.8, 4.2),
    (2.0, 4.0),
    (2.2, 3.0),
    (2.0, 2.0),
    (1.5, 1.0),
    (0.0, 0.5),
    (-1.5, 1.0),
    (-2.0, 2.0),
    (-2.2, 3.0)
]

###############################################################################
# BINNING UTILS
# (These functions are O(1) per call and are used inside the triple–nested loops below.)
###############################################################################
def binCoord(value, binSize=0.001):
    return int(round(value / binSize))

def binPair(x, y, binSize=0.001):
    return (binCoord(x, binSize), binCoord(y, binSize))

###############################################################################
# BUILD T+R+S (SIMILARITY) HASH
# Complexity: O(n^3) over model points.
###############################################################################
def build_TRS_hash(model_points, model_id="Cat", binSize=0.001):
    """
    For each pair (p1, p2) in the model, define a basis by:
      - Translating so that the midpoint is (0,0)
      - Rotating so that the vector from p1 to p2 aligns with the x-axis
      - Scaling so that the distance between p1 and p2 becomes 1.
    Then, transform every other model point into that coordinate system and store its binned coordinate.
    """
    htable = defaultdict(list)
    n = len(model_points)
    
    def transform(p1, p2, p):
        mx = 0.5 * (p1[0] + p2[0])
        my = 0.5 * (p1[1] + p2[1])
        tx = p[0] - mx
        ty = p[1] - my
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-12:
            return None
        angle = math.atan2(dy, dx)
        cosA = math.cos(-angle)
        sinA = math.sin(-angle)
        rx = tx*cosA - ty*sinA
        ry = tx*sinA + ty*cosA
        scale_factor = 2.0 / dist  # normalize so that full distance becomes 2.
        rx *= scale_factor
        ry *= scale_factor
        return binPair(rx, ry, binSize)
    
    for i in range(n):
        for j in range(i+1, n):
            p1, p2 = model_points[i], model_points[j]
            dd = math.hypot(p2[0]-p1[0], p2[1]-p1[1])
            if dd < 1e-12:
                continue
            for k in range(n):
                if k not in (i, j):
                    binned = transform(p1, p2, model_points[k])
                    if binned:
                        htable[binned].append((model_id, (i, j)))
    return htable

###############################################################################
# MATCH SCENE T+R+S (HASH-BASED)
# Complexity: O(m^3) over scene points.
###############################################################################
def match_scene_TRS(scene_points, hash_table, binSize=0.001):
    """
    For each pair (s_i, s_j) in the scene, define a basis (using the same process as for the model),
    transform every other scene point into that coordinate system, and if the resulting binned coordinate exists in the hash table, cast a vote.
    Returns a dictionary: votes[(model_id, mod_basis, (i,j))] = count.
    """
    votes = defaultdict(int)
    m = len(scene_points)
    
    def transform(p1, p2, p):
        mx = 0.5 * (p1[0] + p2[0])
        my = 0.5 * (p1[1] + p2[1])
        tx = p[0] - mx
        ty = p[1] - my
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dist = math.hypot(dx, dy)
        if dist < 1e-12:
            return None
        angle = math.atan2(dy, dx)
        cosA = math.cos(-angle)
        sinA = math.sin(-angle)
        rx = tx*cosA - ty*sinA
        ry = tx*sinA + ty*cosA
        sf = 2.0 / dist
        rx *= sf
        ry *= sf
        return binPair(rx, ry, binSize)
    
    for i in range(m):
        for j in range(i+1, m):
            dd = math.hypot(scene_points[j][0]-scene_points[i][0],
                            scene_points[j][1]-scene_points[i][1])
            if dd < 1e-12:
                continue
            for k in range(m):
                if k not in (i, j):
                    binned = transform(scene_points[i], scene_points[j], scene_points[k])
                    if binned and binned in hash_table:
                        for (model_id, mod_basis) in hash_table[binned]:
                            votes[(model_id, mod_basis, (i, j))] += 1
    return votes

###############################################################################
# INVERSE TRANSFORM USING COMPLEX NUMBERS
# Complexity: O(n) per transform.
###############################################################################
def transform_model_to_scene_complex(model_pts, mod_basis, sc_basis, scene_pts):
    """
    Compute the similarity transform from model to scene using complex numbers.
    Represent each model point as a complex number.
    For a chosen basis pair in the model (M1, M2) and corresponding pair in the scene (S1, S2):
         A = (S2 - S1) / (M2 - M1)
         M_centroid = (M1 + M2)/2, S_centroid = (S1 + S2)/2.
    Then, for any model point M, the transformed point is:
         T(M) = S_centroid + A * (M - M_centroid)
    """
    (mb1, mb2) = mod_basis
    (sb1, sb2) = sc_basis
    M1 = complex(model_pts[mb1][0], model_pts[mb1][1])
    M2 = complex(model_pts[mb2][0], model_pts[mb2][1])
    S1 = complex(scene_pts[sb1][0], scene_pts[sb1][1])
    S2 = complex(scene_pts[sb2][0], scene_pts[sb2][1])
    M_centroid = (M1 + M2) / 2.0
    S_centroid = (S1 + S2) / 2.0
    A = (S2 - S1) / (M2 - M1)
    def transform_point(m):
        M = complex(m[0], m[1])
        S = S_centroid + A * (M - M_centroid)
        return (S.real, S.imag)
    return transform_point

###############################################################################
# HIGHLIGHT MATCHES (USING COMPLEX INVERSE TRANSFORM)
# Complexity: O(m*n) per scene.
###############################################################################
def highlight_matches(ax, model_pts, scene_pts, votes):
    """
    Use the best transform from hash votes (via complex numbers) to compute the transformed model points.
    Then, for each scene point, if its distance to any transformed model point is below MATCH_THRESHOLD,
    mark that scene point in green (others remain red).
    Returns the number of matched scene points.
    """
    best_key, best_val = max(votes.items(), key=lambda x: x[1]) if votes else (None, 0)
    if not best_key:
        return 0
    (model_id, mod_basis, sc_basis) = best_key
    transform_fn = transform_model_to_scene_complex(model_pts, mod_basis, sc_basis, scene_pts)
    if not transform_fn:
        return best_val
    
    transformed_model = [transform_fn(mp) for mp in model_pts]
    
    matched_indices = set()
    for idx, sp in enumerate(scene_pts):
        if any(math.hypot(sp[0]-tmp[0], sp[1]-tmp[1]) < MATCH_THRESHOLD for tmp in transformed_model):
            matched_indices.add(idx)
    
    matched_pts = [scene_pts[i] for i in matched_indices]
    unmatched_pts = [scene_pts[i] for i in range(len(scene_pts)) if i not in matched_indices]
    
    ax.scatter([p[0] for p in unmatched_pts], [p[1] for p in unmatched_pts],
               c='r', s=50, zorder=1)
    ax.scatter([p[0] for p in matched_pts], [p[1] for p in matched_pts],
               c='green', s=70, zorder=2)
    return len(matched_indices)

###############################################################################
# PLOTTING UTILITY
###############################################################################
def plot_points(ax, pts, color='r', marker='o', zorder=1):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.scatter(xs, ys, c=color, marker=marker, s=50, zorder=zorder)

###############################################################################
# MAIN DEMO: 4 Rows x 3 Columns Figure
###############################################################################
def main():
    # Use the detailed cat model.
    model_pts = detailed_cat_face
    
    # Build the T+R+S hash for the model.
    cat_hash = build_TRS_hash(model_pts, "Cat", binSize=0.001)
    
    # Create a 4-row x 3-column figure:
    # Row 1: Reference image with contour lines.
    # Row 2: Pure transformation scenes (Translation, Scale, Rotation).
    # Row 3: Mixed transformation scenes (T+R, T+R+S, Mirror).
    # Row 4: No-match scenes (Random scatter, Distorted cat, Incomplete cat).
    fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(16,16))
    for r in range(4):
        for c in range(3):
            axes[r][c].set_xlim(XMIN, XMAX)
            axes[r][c].set_ylim(YMIN, YMAX)
            axes[r][c].set_aspect('equal', 'box')
            axes[r][c].grid(True)
    
    ###########################################################################
    # ROW 1: Reference Cat Image with Contour Lines.
    ###########################################################################
    ax_ref = axes[0][1]
    ax_ref.set_title("Reference Cat (Detailed)")
    plot_points(ax_ref, model_pts, color='blue', marker='o', zorder=2)
    # Draw contour lines connecting the points in order.
    for i in range(len(model_pts)-1):
        ax_ref.plot([model_pts[i][0], model_pts[i+1][0]],
                   [model_pts[i][1], model_pts[i+1][1]],
                   c='blue', lw=2)
    # Close the contour.
    ax_ref.plot([model_pts[-1][0], model_pts[0][0]],
             [model_pts[-1][1], model_pts[0][1]],
             c='blue', lw=2)
    # Hide left and right subplots in row 1.
    axes[0][0].axis('off')
    axes[0][2].axis('off')
    
    ###########################################################################
    # ROW 2: Pure Transformation Scenes.
    ###########################################################################
    # Scene 1: Translation only.
    s1_desc = "Translation only: shift(+2, +3)"
    scene1 = [(x+2, y+3) for (x,y) in model_pts]
    
    # Scene 2: Scale only.
    s2_desc = "Scale only: scale x1.5"
    scale_val = 1.5
    scene2 = [(x*scale_val, y*scale_val) for (x,y) in model_pts]
    
    # Scene 3: Rotation only.
    s3_desc = "Rotation only: rotate(30°) no noise"
    angle = math.radians(30)
    cos_val, sin_val = math.cos(angle), math.sin(angle)
    scene3 = []
    for (x,y) in model_pts:
        rx = x*cos_val - y*sin_val
        ry = x*sin_val + y*cos_val
        scene3.append((rx, ry))
    
    pure_scenes = [(scene1, s1_desc), (scene2, s2_desc), (scene3, s3_desc)]
    
    ###########################################################################
    # ROW 3: Mixed Transformation Scenes.
    ###########################################################################
    # Scene 4: T+R.
    s4_desc = "T+R: shift(+2, +1), rotate(20°)"
    angle4 = math.radians(20)
    cos4, sin4 = math.cos(angle4), math.sin(angle4)
    scene4 = []
    for (x,y) in model_pts:
        rx = x*cos4 - y*sin4
        ry = x*sin4 + y*cos4
        scene4.append((rx+2, ry+1))
    
    # Scene 5: T+R+S.
    s5_desc = "T+R+S: shift(+2, +3), rotate(-45°), scale(0.8)"
    angle5 = math.radians(-45)
    cos5, sin5 = math.cos(angle5), math.sin(angle5)
    scale5 = 0.8
    scene5 = []
    for (x,y) in model_pts:
        sx = x*scale5
        sy = y*scale5
        rx = sx*cos5 - sy*sin5
        ry = sx*sin5 + sy*cos5
        scene5.append((rx+2, ry+3))
    
    # Scene 6: Mirror: flip x.
    s6_desc = "Mirror: flip x"
    scene6 = [(-x, y) for (x,y) in model_pts]
    
    mixed_scenes = [(scene4, s4_desc), (scene5, s5_desc), (scene6, s6_desc)]
    
    ###########################################################################
    # ROW 4: No-Match Scenes.
    ###########################################################################
    # Scene 7: Random scatter.
    s7_desc = "Random scatter: no coherent shape"
    scene7 = [(random.uniform(-2,3), random.uniform(-2,3)) for _ in range(12)]
    
    # Scene 8: Distorted cat: add high noise.
    s8_desc = "Distorted cat: high noise"
    scene8 = []
    for (x,y) in model_pts:
        scene8.append((x + random.uniform(-3,3), y + random.uniform(-3,3)))
    
    # Scene 9: Incomplete cat: only 3 of 20 points.
    s9_desc = "Incomplete cat: 3 of 20 points"
    scene9 = random.sample(model_pts, 3)
    
    no_match_scenes = [(scene7, s7_desc), (scene8, s8_desc), (scene9, s9_desc)]
    
    ###########################################################################
    # HELPER FUNCTION: Process a Scene.
    # Note: The triple–nested loops in build_TRS_hash and match_scene_TRS are O(n^3) and O(m^3) respectively.
    ###########################################################################
    def handle_scene(ax, sc_pts, desc):
        ax.set_title(desc)
        ax.scatter([p[0] for p in sc_pts],
                   [p[1] for p in sc_pts],
                   c='r', marker='o', s=50, zorder=1)
        votes = match_scene_TRS(sc_pts, cat_hash, binSize=0.001)
        num_matched = highlight_matches(ax, model_pts, sc_pts, votes)
        return len(sc_pts), num_matched
    
    # Process Pure Scenes (Row 2)
    for idx, (scene, desc) in enumerate(pure_scenes):
        ax = axes[1][idx]
        n_pts, num_matched = handle_scene(ax, scene, desc)
        print(f"[Pure Scene {idx+1}] {desc}: totalPoints={n_pts}, matchedPoints={num_matched}")
    
    # Process Mixed Scenes (Row 3)
    for idx, (scene, desc) in enumerate(mixed_scenes):
        ax = axes[2][idx]
        n_pts, num_matched = handle_scene(ax, scene, desc)
        print(f"[Mixed Scene {idx+4}] {desc}: totalPoints={n_pts}, matchedPoints={num_matched}")
    
    # Process No-Match Scenes (Row 4)
    for idx, (scene, desc) in enumerate(no_match_scenes):
        ax = axes[3][idx]
        n_pts, num_matched = handle_scene(ax, scene, desc)
        print(f"[No-Match Scene {idx+7}] {desc}: totalPoints={n_pts}, matchedPoints={num_matched}")
    
    plt.tight_layout()
    plt.show()
    plt.savefig("cat.png")

if __name__=="__main__":
    main()
