"""Screen-space picking of edges belonging to a visible CAD face."""
import numpy as np


def nearest_screen_edge(edges, candidate_indices, cursor, project, tolerance=12.0, samples=65):
    cursor = np.asarray(cursor, dtype=float)
    best_index, best_distance = None, float(tolerance)
    for index in candidate_indices:
        edge = edges[index]
        projected = np.asarray([project(tuple(edge.position_at(t)))[:2]
                                for t in np.linspace(0, 1, samples)], dtype=float)
        if not np.isfinite(projected).all():
            continue
        start, delta = projected[:-1], np.diff(projected, axis=0)
        denominator = np.sum(delta * delta, axis=1)
        numerator = np.sum((cursor - start) * delta, axis=1)
        fractions = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)
        fractions = np.clip(fractions, 0, 1)
        distances = np.linalg.norm(start + fractions[:, None] * delta - cursor, axis=1)
        distance = float(np.min(distances))
        if distance < best_distance:
            best_index, best_distance = index, distance
    return best_index


def face_edge_indices(shape, face_index):
    faces, edges = list(shape.faces()), list(shape.edges())
    if not 0 <= face_index < len(faces):
        return edges, []
    face_edges = list(faces[face_index].edges())
    return edges, [index for index, edge in enumerate(edges)
                   if any(edge.is_same(candidate) for candidate in face_edges)]
