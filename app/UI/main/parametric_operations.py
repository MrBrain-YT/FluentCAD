"""Translate scene operation controls into rebuildable history features."""
from UI.model.feature_graph import Feature, FeatureGraph
from UI.model.geometry_references import make_reference


def commit_parametric_operation(manager, dialog):
    graph = manager.parametric_graph
    source = next((item for item in reversed(graph.features)
                   if item.id in graph.results and hasattr(graph.results[item.id], 'solids')
                   and graph.results[item.id].solids()), None)
    if source is None:
        raise ValueError('No solid is available for this operation.')
    shape = graph.results[source.id]
    body_names = graph.body_names.get(source.id)
    scope = getattr(dialog, 'scope', None)
    selected_body = scope is not None and scope.currentData() == 'selected' and len(list(shape.solids())) > 1
    kind, c = dialog.kind, dialog.controls
    value = lambda key: c[key].value()
    point = lambda prefix: [value(prefix + axis) for axis in 'XYZ']
    if kind == 'move':
        params = {'translation': point('d'), 'rotation': point('r'), 'pivot': point('p'), 'copy': c['copy'].isChecked()}
    elif kind == 'scale':
        params = {'factor': value('factor'), 'center': point('p'), 'copy': c['copy'].isChecked()}
    elif kind == 'mirror':
        params = {'plane': c['plane'].currentText(), 'offset': value('offset'), 'copy': c['copy'].isChecked()}
    elif kind == 'rectangular':
        kind, params = 'rectangular_pattern', {'count_x': value('nx'), 'count_y': value('ny'), 'spacing_x': value('dx'), 'spacing_y': value('dy')}
    elif kind == 'circular':
        kind, params = 'circular_pattern', {'count': value('count'), 'angle': value('angle'), 'axis': c['axis'].currentText(), 'center': point('p')}
    elif kind in ('fillet', 'chamfer', 'fillet_selected', 'chamfer_selected'):
        selected = kind.endswith('_selected')
        kind = kind.replace('_selected', '')
        indices = dialog.edge_indices if selected else range(len(list(shape.edges())))
        params = {'edges': [make_reference(source.id, shape, 'edge', i, body_names) for i in indices],
                  'radius' if kind == 'fillet' else 'distance': value('value')}
    elif kind in ('shell', 'hole'):
        params = {'face': make_reference(source.id, shape, 'face', dialog.face_index, body_names)}
        if kind == 'shell': params['thickness'] = value('value')
        else: params.update(diameter=value('diameter'), depth=value('depth'), offset_u=value('offset_u'),
                            offset_v=value('offset_v'), through_all=c['through'].isChecked())
    elif kind == 'offset':
        params = {'distance': value('value')}
    else:
        raise ValueError(f'Unsupported parametric operation: {kind}')
    if selected_body:
        if kind not in ('move', 'scale', 'mirror', 'rectangular_pattern', 'circular_pattern', 'offset', 'shell', 'hole'):
            raise ValueError('This operation cannot yet target a single body in parametric history.')
        params['body_face'] = make_reference(source.id, shape, 'face', manager.selected_face_id, body_names)
    feature = Feature.create(dialog.windowTitle(), kind, params, source.id)
    data = graph.to_data()
    data['features'].append({'id': feature.id, 'name': feature.name, 'kind': feature.kind,
                             'parameters': feature.parameters, 'input_id': feature.input_id, 'suppressed': False})
    manager.install_parametric_graph(FeatureGraph.from_data(data))
