"""Stable references to the contours of a selected Loft section."""
import json
from collections import Counter

import build123d as cad

from UI.model.sketch_projection_curves import projectable_sketch_curves


def section_rims(sketch, profile):
    holes = profile.inner_wires()
    rims = [('outer', cad.Face(profile.outer_wire()))]
    if len(holes) == 1:
        return [*rims, ('only-hole', cad.Face(holes[0]))]
    if not holes:
        return rims
    curves = projectable_sketch_curves(sketch)
    counts = Counter(name for name, _ in curves)
    for wire in holes:
        tokens = set()
        for edge in wire.edges():
            samples = [cad.Vertex(*tuple(edge.position_at(t))) for t in (.2, .5, .8)]
            matches = [name for name, candidate in curves
                       if all(point.distance_to(candidate) <= 1e-6 for point in samples)]
            if len(matches) != 1 or counts[matches[0]] != 1:
                break
            tokens.add(matches[0])
        else:
            rims.append((json.dumps(sorted(tokens), separators=(',', ':')), cad.Face(wire)))
            continue
        rims.append((None, cad.Face(wire)))
    keys = Counter(key for key, _ in rims)
    return [(key if keys[key] == 1 else None, face) for key, face in rims]


def resolve_hole_matches(refs, section_count, rims_at):
    if not isinstance(refs, list):
        raise ValueError('Invalid Loft hole matches.')
    matches = {}
    for ref in refs:
        if (not isinstance(ref, list) or len(ref) != 3 or type(ref[0]) is not int
                or not 0 <= ref[0] < section_count - 1
                or any(not isinstance(key, str) or not key or key == 'outer' for key in ref[1:])):
            raise ValueError('A Loft hole match must reference holes in adjacent sections.')
        level, first, second = ref
        left = rims_at(level).get(first)
        right = rims_at(level + 1).get(second)
        if left is None or right is None:
            raise ValueError(f'A manually matched hole between sections {level + 1} and {level + 2} '
                             'is missing or ambiguous.')
        fixed = matches.setdefault(level, {})
        if left[0] in fixed or right[0] in fixed.values():
            raise ValueError('Each Loft hole can be matched only once per pair of sections.')
        fixed[left[0]] = right[0]
    return matches
