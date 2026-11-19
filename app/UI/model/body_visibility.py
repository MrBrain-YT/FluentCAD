"""Conservative body identities for display state, independent of list order."""
import math
import numpy as np


def describe_body(body):
    box = body.bounding_box()
    return {'center': [float(box.center().X), float(box.center().Y), float(box.center().Z)],
            'size': [float(box.size.X), float(box.size.Y), float(box.size.Z)],
            'volume': float(body.volume), 'faces': len(body.faces()), 'edges': len(body.edges())}


def validate_body_reference(value):
    if not isinstance(value, dict) or set(value) != {'center', 'size', 'volume', 'faces', 'edges'}:
        raise ValueError('Invalid body visibility reference.')
    for key in ('center', 'size'):
        if not isinstance(value[key], list) or len(value[key]) != 3 or any(type(x) not in (int, float) or not math.isfinite(x) for x in value[key]):
            raise ValueError('Invalid body visibility reference.')
    if (type(value['volume']) not in (int, float) or not math.isfinite(value['volume']) or value['volume'] <= 0
            or any(type(value[key]) is not int or value[key] < 0 for key in ('faces', 'edges'))):
        raise ValueError('Invalid body visibility reference.')
    return value


def resolve_body_reference(reference, bodies):
    reference = validate_body_reference(reference)
    matches = []
    for index, body in enumerate(bodies):
        actual = describe_body(body)
        if (actual['faces'] != reference['faces'] or actual['edges'] != reference['edges']
                or not np.allclose(actual['center'], reference['center'], rtol=1e-7, atol=1e-7)
                or not np.allclose(actual['size'], reference['size'], rtol=1e-7, atol=1e-7)
                or not math.isclose(actual['volume'], reference['volume'], rel_tol=1e-7, abs_tol=1e-7)):
            continue
        matches.append(index)
    if len(matches) != 1:
        raise ValueError('Hidden body is missing or ambiguous after rebuild.')
    return matches[0]
