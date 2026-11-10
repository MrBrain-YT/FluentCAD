"""Read-only measurements of the current solid and selected surface."""
import math


def measure_model(shape):
    if shape is None or not shape.solids():
        raise ValueError('Create or import a solid body first.')
    bounds = shape.bounding_box()
    dimensions = (bounds.max.X - bounds.min.X, bounds.max.Y - bounds.min.Y,
                  bounds.max.Z - bounds.min.Z)
    return {'volume': shape.volume, 'area': sum(face.area for face in shape.faces()),
            'dimensions': dimensions, 'bodies': len(shape.solids())}


def measure_face(face):
    if face is None:
        raise ValueError('Select a face or profile first.')
    center = tuple(face.center())
    return {'area': face.area, 'perimeter': sum(edge.length for edge in face.edges()),
            'center': center, 'geometry': face.geom_type.name}


def measure_edges(shape, indices):
    if shape is None:
        raise ValueError('Select one or more edges first.')
    edges = list(shape.edges())
    if not indices or any(type(index) is not int or not 0 <= index < len(edges) for index in indices):
        raise ValueError('Select one or more valid edges first.')
    chosen = [edges[index] for index in sorted(set(indices))]
    return {'count': len(chosen), 'length': sum(edge.length for edge in chosen),
            'types': sorted({edge.geom_type.name for edge in chosen})}
