"""Conservative profile identity from sketch boundary provenance."""
import json
import numpy as np
import build123d as cad
from UI.model.profiles import indexed_sketch_edges
from UI.model.sketch_projection_curves import sketch_curve_name


def profile_keys(profiles, entities, tools, normal, linked_edges=(), corner_edges=(), corner_sources=None):
    candidates, source_tokens = [], {}
    for entity, tool in zip(entities, tools):
        if getattr(tool, 'is_construction', False): continue
        kind = entity.get('type')
        if kind not in ('line', 'circle', 'arc', 'ellipse', 'slot', 'spline', 'fit_spline', 'nurbs', 'periodic_nurbs'):
            continue
        references = entity.get('points', [entity.get('center')])
        if not isinstance(references, list) or not all(isinstance(name, str) for name in references): continue
        for ordinal, (segment, edge) in enumerate(indexed_sketch_edges(tool, normal)):
            index = segment if 'curve_spans' in entity else ordinal
            identity = json.dumps([kind, references, index], separators=(',', ':'), ensure_ascii=True)
            candidates.append((identity, edge, kind))
            if corner_sources:
                source_tokens[sketch_curve_name(entity, segment)] = identity
    for name, edge in linked_edges:
        identity = json.dumps(['linked', name], separators=(',', ':'))
        candidates.append((identity, edge, 'linked'))
    candidates.extend((name, edge, 'corner') for name, edge in corner_edges)
    keys = []
    for face in profiles:
        tokens, sides, matched_corners, valid = set(), {}, [], True
        center = np.asarray(tuple(face.center()), dtype=float)
        for edge in face.edges():
            samples = [cad.Vertex(*tuple(edge.position_at(t))) for t in (.2,.5,.8)]
            matches = [(name, source, kind) for name, source, kind in candidates
                       if all(vertex.distance_to(source) < 1e-5 for vertex in samples)]
            if len(matches) != 1:
                valid = False; break
            name, source, kind = matches[0]
            if kind == 'corner':
                whole = abs(edge.length - source.length) < 1e-5
                matched_corners.append((name, whole))
                continue
            tokens.add(name)
            if kind == 'line':
                a, b = np.asarray(tuple(source.position_at(0))), np.asarray(tuple(source.position_at(1)))
                direction = b-a
                value = float(np.dot(np.cross(direction, center-a), normal))
                scale = max(np.linalg.norm(direction), 1.)
                if abs(value) < 1e-8*scale:
                    valid = False; break
                sides[name] = 1 if value > 0 else -1
        for name, whole in matched_corners:
            # A complete corner connector replaces the old vertex on the same profile.
            if not whole or any(source_tokens.get(ref) not in tokens for ref in corner_sources[name]):
                tokens.add(json.dumps(['corner', name], separators=(',', ':')))
        keys.append(json.dumps([sorted(tokens), sorted(sides.items())], separators=(',', ':')) if valid and tokens else None)
    return [key if key is not None and keys.count(key) == 1 else None for key in keys]
