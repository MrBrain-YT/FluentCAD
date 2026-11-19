"""Conservative operation-derived names for solids in a rebuilt feature graph."""
import numpy as np
import build123d as cad
import hashlib
import json
from UI.model.body_visibility import describe_body, resolve_body_reference
from UI.model.expressions import evaluate_expression


def body_names_for_feature(feature, result, results, names, parameters):
    if not hasattr(result, 'solids') or not result.solids():
        return ()
    output = list(result.solids())
    parent_id = feature.input_id
    if feature.kind in ('extrude', 'revolve', 'sweep', 'loft'):
        parent_id = feature.parameters.get('base_id')
    parent = results.get(parent_id)
    inputs = list(parent.solids()) if parent is not None and hasattr(parent, 'solids') else []
    input_names = names.get(parent_id, ())
    assigned = [None] * len(output)
    used = set()
    for i, body in enumerate(inputs):
        matches = [j for j, candidate in enumerate(output) if candidate.is_same(body)]
        if len(matches) == 1 and i < len(input_names) and input_names[i] is not None:
            assigned[matches[0]] = input_names[i]
            used.add(i)
    # Patterns carry an operation ID and a stable instance coordinate, not an
    # index in the resulting Open CASCADE solids list.
    args = feature.parameters
    if feature.kind == 'frozen_body' and not inputs:
        signatures = []
        for body in output:
            descriptor = describe_body(body)
            for key in ('center', 'size'):
                descriptor[key] = [round(value, 7) for value in descriptor[key]]
            descriptor['volume'] = round(descriptor['volume'], 7)
            signatures.append(hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode('utf-8')).hexdigest())
        return tuple(f'body:{feature.id}:frozen:{key}' if signatures.count(key) == 1 else None for key in signatures)
    if feature.kind in ('rectangular_pattern', 'circular_pattern') and inputs and len(input_names) == len(inputs):
        def number(value):
            return evaluate_expression(value, parameters) if isinstance(value, str) else float(value)
        try:
            pattern_inputs = list(zip(inputs, input_names))
            if 'body_face' in args:
                from UI.model.geometry_references import resolve_reference
                from UI.model.transforms import body_index_for_face
                selected_face = resolve_reference(args['body_face'], parent, parent_id, body_names=input_names)
                selected_body = body_index_for_face(parent, selected_face)
                if selected_body is None:
                    return tuple(assigned)
                pattern_inputs = [pattern_inputs[selected_body]]
            if feature.kind == 'rectangular_pattern':
                nx, ny = int(number(args['count_x'])), int(number(args['count_y']))
                transforms = [(f'{x},{y}', cad.Pos(x * number(args['spacing_x']), y * number(args['spacing_y'])))
                              for x in range(nx) for y in range(ny)]
                predicted = [(slot, body.moved(transform), name) for slot, transform in transforms
                             for body, name in pattern_inputs]
            else:
                count, angle = int(number(args['count'])), number(args['angle'])
                axis = cad.Axis(tuple(number(x) for x in args.get('center', (0, 0, 0))),
                                {'X': (1, 0, 0), 'Y': (0, 1, 0), 'Z': (0, 0, 1)}[args.get('axis', 'Z')])
                divisor = count if np.isclose(abs(angle), 360) else count - 1
                predicted = [(str(i), body.rotate(axis, angle * i / divisor), name)
                             for i in range(count) for body, name in pattern_inputs]
            for slot, expected, name in predicted:
                if name is None: continue
                try:
                    index = resolve_body_reference(describe_body(expected), output)
                except ValueError:
                    continue
                candidate_name = name if slot in ('0', '0,0') else f'{name}/pattern:{feature.id}:{slot}'
                if assigned[index] is None or assigned[index] == name:
                    assigned[index] = candidate_name
        except (ValueError, KeyError, ZeroDivisionError):
            pass
    elif feature.kind in ('move', 'scale', 'mirror') and inputs and len(input_names) == len(inputs):
        from UI.model.transforms import move_copy, scale_model, mirror_model
        def number(value):
            return evaluate_expression(value, parameters) if isinstance(value, str) else float(value)
        try:
            selected = list(range(len(inputs)))
            if 'body_face' in args:
                from UI.model.geometry_references import resolve_reference
                from UI.model.transforms import body_index_for_face
                face_index = resolve_reference(args['body_face'], parent, parent_id, body_names=input_names)
                body_index = body_index_for_face(parent, face_index)
                selected = [body_index] if body_index is not None else []
            for index in selected:
                name = input_names[index]
                if name is None: continue
                body = inputs[index]
                if feature.kind == 'move':
                    expected = move_copy(body, [number(x) for x in args.get('translation', (0, 0, 0))],
                                         [number(x) for x in args.get('rotation', (0, 0, 0))],
                                         [number(x) for x in args.get('pivot', (0, 0, 0))])
                elif feature.kind == 'scale':
                    expected = scale_model(body, number(args['factor']), [number(x) for x in args.get('center', (0, 0, 0))])
                else:
                    expected = mirror_model(body, args['plane'], number(args.get('offset', 0)), copy=False)
                try:
                    target = resolve_body_reference(describe_body(expected), output)
                except ValueError:
                    continue
                copy_default = feature.kind == 'mirror'
                candidate = f'{name}/copy:{feature.id}' if args.get('copy', copy_default) else name
                if assigned[target] is None or assigned[target] == candidate:
                    assigned[target] = candidate
        except (ValueError, KeyError):
            pass
    else:
        missing_inputs = [i for i in range(len(inputs)) if i not in used and i < len(input_names) and input_names[i] is not None]
        missing_outputs = [j for j, name in enumerate(assigned) if name is None]
        if len(missing_inputs) == len(missing_outputs) == 1:
            assigned[missing_outputs[0]] = input_names[missing_inputs[0]]
        elif not inputs and len(output) == 1:
            assigned[0] = f'body:{feature.id}'
        elif feature.kind in ('primitive', 'extrude', 'revolve', 'sweep', 'loft') and args.get('operation', 'new_body') == 'new_body' and len(missing_outputs) == 1 and len(used) == len(inputs):
            assigned[missing_outputs[0]] = f'body:{feature.id}'
    return tuple(assigned)
