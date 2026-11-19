"""Resolve a saved sketch path and build a solid sweep."""
from collections import Counter

import build123d as cad

from UI.model.sketch_offset import order_offset_chain
from UI.model.sketch_projection_curves import projectable_sketch_curves
from UI.model.solid_operations import combine_solid


def sweep_curves(graph_results):
    for feature_id, sketch in graph_results.items():
        curves = projectable_sketch_curves(sketch)
        counts = Counter(name for name, _ in curves)
        for name, edge in curves:
            if counts[name] == 1 and edge.length > 1e-6:
                yield (feature_id, name), edge


def build_sketch_sweep(profile, path, results, base=None):
    if (not isinstance(path, list) or not 1 <= len(path) <= 1000
            or any(not isinstance(ref, list) or len(ref) != 2
                   or not all(isinstance(value, str) and value for value in ref) for ref in path)):
        raise ValueError('Select a chain of sketch segments.')
    refs = list(map(tuple, path))
    if len(set(refs)) != len(refs):
        raise ValueError('The same path segment was selected more than once.')
    curves = dict(sweep_curves({ref[0]: results[ref[0]] for ref in refs if ref[0] in results}))
    if any(ref not in curves for ref in refs):
        raise ValueError('A sweep path segment is missing or ambiguous.')
    edges = {ref: curves[ref] for ref in refs}
    try:
        ordered = order_offset_chain(edges, refs[0])
    except ValueError as exc:
        raise ValueError('Select a single connected sweep path without branches.') from exc
    segments = [edges[ref].reversed() if reverse else edges[ref] for ref, reverse in ordered]
    normal = profile.normal_at()
    origin = profile.center()

    def starts_on_profile(edge, end):
        return (abs((edge.position_at(end) - origin).dot(normal)) <= 1e-6
                and abs(edge.tangent_at(end).dot(normal)) > 1e-6)

    if not starts_on_profile(segments[0], 0):
        if not starts_on_profile(segments[-1], 1):
            raise ValueError('The sweep path must start or end on the profile plane and leave it at that point.')
        segments = [edge.reversed() for edge in reversed(segments)]
    wire = cad.Wire(segments)
    if not wire.is_valid or len(wire.edges()) != len(segments):
        raise ValueError('The selected segments do not form a valid sweep path.')
    try:
        solid = cad.sweep(profile, path=wire, transition=cad.Transition.RIGHT)
    except (ValueError, RuntimeError) as exc:
        raise ValueError('Could not sweep this profile along the selected path.') from exc
    if len(solid.solids()) != 1 or not solid.is_valid or solid.volume <= 1e-9:
        raise ValueError('The sweep did not produce one valid solid.')
    return combine_solid(solid, 'new_body', base)
