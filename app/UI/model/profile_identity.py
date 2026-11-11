"""Conservative profile identity from sketch boundary provenance."""
import json
import numpy as np
import build123d as cad
from UI.model.profiles import sketch_edges


def profile_keys(profiles, entities, tools, normal):
    candidates = []
    for entity, tool in zip(entities, tools):
        if getattr(tool, 'is_construction', False): continue
        kind = entity.get('type')
        if kind not in ('line', 'circle', 'arc', 'ellipse', 'slot', 'spline', 'fit_spline', 'nurbs'):
            continue
        references = entity.get('points', [entity.get('center')])
        if not isinstance(references, list) or not all(isinstance(name, str) for name in references): continue
        for index, edge in enumerate(sketch_edges([tool], normal)):
            identity = json.dumps([kind, references, index], separators=(',', ':'), ensure_ascii=True)
            candidates.append((identity, edge, kind))
    keys = []
    for face in profiles:
        tokens, sides, valid = set(), {}, True
        center = np.asarray(tuple(face.center()), dtype=float)
        for edge in face.edges():
            samples = [cad.Vertex(*tuple(edge.position_at(t))) for t in (.2,.5,.8)]
            matches = [(name, source, kind) for name, source, kind in candidates
                       if all(vertex.distance_to(source) < 1e-5 for vertex in samples)]
            if len(matches) != 1:
                valid = False; break
            name, source, kind = matches[0]
            tokens.add(name)
            if kind == 'line':
                a, b = np.asarray(tuple(source.position_at(0))), np.asarray(tuple(source.position_at(1)))
                direction = b-a
                value = float(np.dot(np.cross(direction, center-a), normal))
                scale = max(np.linalg.norm(direction), 1.)
                if abs(value) < 1e-8*scale:
                    valid = False; break
                sides[name] = 1 if value > 0 else -1
        keys.append(json.dumps([sorted(tokens), sorted(sides.items())], separators=(',', ':')) if valid and tokens else None)
    return [key if key is not None and keys.count(key) == 1 else None for key in keys]
