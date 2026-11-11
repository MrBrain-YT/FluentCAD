"""Lossless migration of snapshot geometry into a frozen graph feature."""
import base64
from copy import deepcopy
from UI.model.feature_graph import Feature, FeatureGraph


def encode_body(body):
    from UI.model.snapshot import _write_shape, MAX_ENTRY_BYTES
    data = _write_shape(body)
    if len(data) > MAX_ENTRY_BYTES: raise ValueError('Body is too large to convert.')
    return base64.b64encode(data).decode('ascii')


def decode_body(data):
    from UI.model.snapshot import _read_shape, MAX_ENTRY_BYTES
    if not isinstance(data, str) or len(data) > ((MAX_ENTRY_BYTES+2)//3)*4:
        raise ValueError('Invalid or oversized frozen body.')
    try: raw = base64.b64decode(data, validate=True)
    except (ValueError, TypeError) as exc: raise ValueError('Invalid frozen body encoding.') from exc
    return _read_shape(raw)


def convert_snapshot(current, undo, redo, timeline):
    if current is None or not current.solids(): raise ValueError('Open or create a solid body first.')
    backup = {'undo':[encode_body(body) if body is not None else None for body in undo],
              'redo':[encode_body(body) if body is not None else None for body in redo],
              'timeline':deepcopy(timeline)}
    feature = Feature.create('Imported snapshot', 'frozen_body', {'brep':encode_body(current), 'legacy_backup':backup})
    return FeatureGraph([feature])


def restore_snapshot(graph):
    from UI.model.timeline import validate_timeline
    feature = next((item for item in graph.features if item.kind == 'frozen_body' and 'legacy_backup' in item.parameters), None)
    if feature is None: raise ValueError('This graph has no legacy snapshot backup.')
    backup = feature.parameters['legacy_backup']
    if not isinstance(backup, dict): raise ValueError('Invalid legacy history backup.')
    undo, redo = backup.get('undo'), backup.get('redo')
    if not isinstance(undo, list) or not isinstance(redo, list) or len(undo) > 100 or len(redo) > 100:
        raise ValueError('Invalid legacy history backup.')
    timeline = validate_timeline(backup.get('timeline'), len(undo), len(redo))
    return (decode_body(feature.parameters['brep']),
            [decode_body(x) if x is not None else None for x in undo],
            [decode_body(x) if x is not None else None for x in redo], timeline)
