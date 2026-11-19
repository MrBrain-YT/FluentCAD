"""Match the inner boundaries of loft sections and build their voids."""
from bisect import bisect_right
from itertools import accumulate

import build123d as cad
import numpy as np
from scipy.optimize import linear_sum_assignment

from UI.model.guided_loft import guide_section_contact
from UI.model.loft_patch_continuity import support_meets_section


def _descriptors(loops):
    outer = loops[0]
    scale = outer.area ** .5
    return np.asarray([(*tuple((loop.center() - outer.center()) / scale), (loop.area / outer.area) ** .5)
                       for loop in loops])


def _match_holes(previous, current, fixed):
    left = [i for i in range(1, len(previous)) if i not in fixed]
    right = [i for i in range(1, len(current)) if i not in fixed.values()]
    if not left:
        return fixed
    a, b = _descriptors(previous)[left], _descriptors(current)[right]
    costs = np.sum((a[:, None] - b[None, :]) ** 2, axis=2)
    rows, columns = linear_sum_assignment(costs)
    minimum = costs[rows, columns].sum()
    for row, column in zip(rows, columns):
        alternative = costs.copy()
        alternative[row, column] = np.inf
        try:
            i, j = linear_sum_assignment(alternative)
        except ValueError:
            continue
        if alternative[i, j].sum() <= minimum + 1e-8:
            raise ValueError('Hole correspondence is ambiguous. Match the holes manually or add guides.')
    return {**fixed, **{left[i]: right[j] for i, j in zip(rows, columns)}}


def _guide_tracks(profiles, loops, guides):
    tracks = [[] for _ in guides]
    for profile, rims in zip(profiles, loops):
        wires = [face.outer_wire().order_edges() for face in rims]
        offsets = list(accumulate(map(len, wires), initial=0))
        perimeter = [edge for wire in wires for edge in wire]
        for guide, track in zip(guides, tracks):
            marker, _ = guide_section_contact(guide, profile, perimeter)
            track.append(bisect_right(offsets, marker) - 1)
    return tracks


def loft_with_holes(profiles, guides, ends, rim_ends, hole_matches, build_solid):
    loops = [[cad.Face(wire) for wire in (profile.outer_wire(), *profile.inner_wires())]
             for profile in profiles]
    count = len(loops[0])
    if any(len(rims) != count for rims in loops):
        raise ValueError('Loft sections must have the same number of holes.')
    tracks = _guide_tracks(profiles, loops, guides)
    correspondence = [list(range(count))]
    for level, (previous, current) in enumerate(zip(loops, loops[1:])):
        fixed = {0: 0, **hole_matches.get(level, {})}
        for track in tracks:
            source, target = track[level:level + 2]
            if (source in fixed and fixed[source] != target) or (target in fixed.values() and fixed.get(source) != target):
                raise ValueError('A Loft guide must follow the same boundary through all sections '
                                 'and respect manual hole matches.')
            fixed[source] = target
        match = _match_holes(previous, current, fixed)
        correspondence.append([match[index] for index in correspondence[-1]])
    guide_groups = [[] for _ in range(count)]
    for guide, track in zip(guides, tracks):
        guide_groups[track[0]].append(guide)
    conditions = [{} for _ in range(count)]
    for side, condition in (ends or {}).items():
        level = 0 if side == 'start' else -1
        matches = [i for i, index in enumerate(correspondence[level])
                   if support_meets_section(loops[level][index], condition[1])]
        if not matches:
            raise ValueError('The support face boundary must coincide with an end section rim.')
        for index in matches:
            conditions[index][side] = condition
    for side, end_conditions in rim_ends.items():
        level = 0 if side == 'start' else -1
        for index, condition in end_conditions.items():
            if not 0 <= index < count:
                raise ValueError('The selected Loft contour is missing.')
            if not support_meets_section(loops[level][index], condition[1]):
                raise ValueError('The support face boundary must coincide with the selected rim.')
            conditions[correspondence[level].index(index)][side] = condition
    solids = []
    for i in range(count):
        sections = [rims[order[i]] for rims, order in zip(loops, correspondence)]
        solids.append(build_solid(sections, guide_groups[i], conditions[i]))
    outer, *holes = solids
    result = outer.cut(*holes)
    if isinstance(result, list) or len(result.solids()) != 1 or not result.is_valid:
        raise ValueError('The Loft holes do not form one valid solid.')
    volume = outer.volume - sum(hole.volume for hole in holes)
    if volume <= 0 or abs(result.volume - volume) > max(1e-6, outer.volume * 1e-5):
        raise ValueError('The Loft holes overlap or leave the outer boundary between sections.')
    return result
