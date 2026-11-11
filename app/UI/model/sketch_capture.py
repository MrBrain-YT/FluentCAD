"""Convert interactive sketch tools into serializable parametric geometry."""
import math
import numpy as np


def capture_sketch(sketch, origin, x_axis, y_axis):
    origin, x_axis, y_axis = map(lambda v: np.asarray(v, dtype=float), (origin, x_axis, y_axis))
    points, entities = {}, []
    def local(point):
        delta = np.asarray(point, dtype=float) - origin
        return [float(np.dot(delta, x_axis)), float(np.dot(delta, y_axis))]
    for index, tool in enumerate(sketch.get('sketch_func', []), 1):
        kind = tool.type
        coords = list(getattr(tool, 'points', []))
        construction = bool(getattr(tool, 'is_construction', False))
        if kind == 'line':
            if len(coords) < 2: continue
            names = [f'L{index}.P{i}' for i in range(1, len(coords)+1)]
            points.update({name: local(p) for name, p in zip(names, coords)})
            entity = {'type':'line', 'points':names}
        elif kind == 'circle':
            if getattr(tool, 'center', None) is None or getattr(tool, 'radius', None) is None: continue
            name = f'C{index}.center'; points[name] = local(tool.center)
            entity = {'type':'circle', 'center':name, 'radius':float(tool.radius)}
        elif kind == 'curve':
            if len(coords) < 3: continue
            names = [f'A{index}.P{i}' for i in range(1,4)]
            points.update({name: local(p) for name,p in zip(names,coords[:3])})
            entity = {'type':'arc', 'points':names}
        elif kind == 'slot':
            if len(coords) < 3: continue
            names = [f'S{index}.P{i}' for i in range(1,4)]
            points.update({name: local(p) for name,p in zip(names,coords[:3])})
            entity = {'type':'slot', 'points':names}
        elif kind in ('spline','fit_spline','nurbs','periodic_nurbs'):
            if len(coords) < (4 if kind == 'spline' else 3): continue
            names = [f'SP{index}.P{i}' for i in range(1,len(coords)+1)]
            points.update({name:local(p) for name,p in zip(names,coords)})
            entity = {'type':kind, 'points':names}
            if kind in ('nurbs','periodic_nurbs'):
                entity.update(degree=tool.degree, weights=list(tool.weights))
                entity['knots' if kind == 'nurbs' else 'spans'] = list(tool.knots if kind == 'nurbs' else tool.spans)
        elif kind == 'ellipse':
            if getattr(tool, 'center', None) is None: continue
            axis = np.asarray(tool.major_axis_vector, dtype=float)
            name = f'E{index}.center'; points[name] = local(tool.center)
            entity = {'type':'ellipse','center':name,'radius_x':float(tool.major_radius),
                      'radius_y':float(tool.minor_radius),
                      'rotation':math.degrees(math.atan2(np.dot(axis,y_axis),np.dot(axis,x_axis)))}
            if not getattr(tool, 'is_closed', False):
                if len(coords) < 5: continue
                entity.update(is_closed=False, start_angle=math.degrees(float(tool.start_angle)),
                              end_angle=math.degrees(float(tool.end_angle)))
        elif kind == 'point':
            for i, coord in enumerate(coords, 1): points[f'P{index}.{i}'] = local(coord)
            continue
        else:
            if coords: raise ValueError(f'Sketch element {kind} is not supported in parametric history.')
            continue
        entity['construction'] = construction
        entities.append(entity)
    constraints = []
    def serializable(value):
        if isinstance(value, (list, tuple, np.ndarray)):
            return [serializable(item) for item in value]
        if isinstance(value, np.generic):
            return value.item()
        return value
    for constraint in sketch.get('constraints', []):
        entry = {'kind':constraint.kind, 'entities':serializable(constraint.entities)}
        if constraint.value is not None: entry['value'] = serializable(constraint.value)
        if constraint.driven: entry['driven'] = True
        constraints.append(entry)
    return {'points':points, 'entities':entities, 'constraints':constraints}
