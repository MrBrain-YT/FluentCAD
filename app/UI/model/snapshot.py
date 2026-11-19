"""Versioned snapshots of solid geometry and session undo/redo history."""
import io
import json
import math
import os
import tempfile
import zipfile
from pathlib import Path

import build123d as cad
from OCP.BRep import BRep_Builder
from OCP.BRepTools import BRepTools
from OCP.TopAbs import TopAbs_COMPOUND, TopAbs_COMPSOLID, TopAbs_SOLID
from OCP.TopoDS import TopoDS, TopoDS_Shape
from UI.model.timeline import validate_timeline
from UI.model.feature_graph import FeatureGraph

MAX_ENTRY_BYTES = 128 * 1024 * 1024
MAX_TOTAL_BYTES = 256 * 1024 * 1024
MAX_HISTORY = 100


def _write_shape(shape):
    stream = io.BytesIO()
    if not cad.export_brep(shape, stream):
        raise ValueError('Unable to serialize solid geometry.')
    return stream.getvalue()


def _read_shape(data):
    raw = TopoDS_Shape()
    BRepTools.Read_s(raw, io.BytesIO(data), BRep_Builder())
    if raw.IsNull():
        raise ValueError('Invalid solid geometry in snapshot.')
    kind = raw.ShapeType()
    if kind == TopAbs_SOLID:
        shape = cad.Solid(TopoDS.Solid_s(raw))
    elif kind == TopAbs_COMPOUND:
        shape = cad.Compound(TopoDS.Compound_s(raw))
    elif kind == TopAbs_COMPSOLID:
        shape = cad.Compound(list(cad.Compound(raw).solids()))
    else:
        raise ValueError('Snapshot geometry is not a solid or compound.')
    if not shape.is_valid or not shape.solids():
        raise ValueError('Snapshot contains invalid or empty solid geometry.')
    return shape


def save_snapshot(path, current, undo, redo, planes=None, timeline=None, feature_graph=None, graph_history=None, visibility=None):
    if current is None:
        if feature_graph is None:
            raise ValueError('Create a solid before saving a model snapshot.')
    elif not current.solids():
        raise ValueError('The current model does not contain a solid.')
    if len(undo) > MAX_HISTORY or len(redo) > MAX_HISTORY:
        raise ValueError('The model history is too large to save.')
    path = Path(path)
    plane_data = validate_planes(planes or [])
    timeline_data = validate_timeline(timeline, len(undo), len(redo))
    manifest = {'format': 'FluentCAD-solid-snapshot', 'version': 1,
                'undo': [x is not None for x in undo], 'redo': [x is not None for x in redo],
                'planes': plane_data, 'timeline': timeline_data}
    if visibility is not None:
        manifest['visibility'] = validate_visibility(visibility)
    if feature_graph is not None:
        graph_data = feature_graph.to_data()
        rebuilt_graph = FeatureGraph.from_data(graph_data)
        if current is None:
            if rebuilt_graph.latest_solid is not None or undo or redo:
                raise ValueError('Inconsistent parametric document without a solid.')
            manifest['has_current'] = False
        manifest['feature_graph'] = graph_data
        manifest['graph_history'] = validate_graph_history(graph_history)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.fluentcad-', suffix='.tmp', dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('manifest.json', json.dumps(manifest))
            if current is not None:
                archive.writestr('current.brep', _write_shape(current))
            for name, shapes in [('undo', undo), ('redo', redo)]:
                for i, shape in enumerate(shapes):
                    if shape is not None:
                        archive.writestr(f'{name}/{i}.brep', _write_shape(shape))
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def validate_planes(planes):
    if not isinstance(planes, list) or len(planes) > 1000:
        raise ValueError('Invalid construction planes in snapshot.')
    result = []
    for plane in planes:
        if not isinstance(plane, dict) or plane.get('base') not in ('XY', 'XZ', 'YZ', 'face'):
            raise ValueError('Invalid construction plane in snapshot.')
        offset = plane.get('offset')
        if type(offset) not in (int, float) or not math.isfinite(offset) or abs(offset) > 1000000:
            raise ValueError('Invalid construction plane offset in snapshot.')
        item = {'base': plane['base'], 'offset': float(offset)}
        if plane['base'] == 'face':
            for key in ('origin', 'normal'):
                vector = plane.get(key)
                if (not isinstance(vector, (list, tuple)) or len(vector) != 3 or
                        any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)):
                    raise ValueError('Invalid construction plane coordinates in snapshot.')
                item[key] = [float(value) for value in vector]
            if sum(value * value for value in item['normal']) < 1e-18:
                raise ValueError('Invalid construction plane normal in snapshot.')
        result.append(item)
    return result


def validate_graph_history(history):
    if history is None:
        return {'undo': [], 'redo': []}
    if not isinstance(history, dict) or set(history) != {'undo', 'redo'}:
        raise ValueError('Invalid parametric history.')
    result = {}
    for key in ('undo', 'redo'):
        values = history[key]
        if not isinstance(values, list) or len(values) > MAX_HISTORY:
            raise ValueError('Invalid parametric history length.')
        result[key] = [FeatureGraph.from_data(value).to_data() for value in values]
    return result


def validate_visibility(value):
    if value is None:
        return {'hidden_bodies': [], 'visible_planes': [], 'visible_sketch': None, 'hidden_body_refs': []}
    if not isinstance(value, dict) or not {'hidden_bodies', 'visible_planes', 'visible_sketch'} <= set(value) or set(value) - {'hidden_bodies', 'visible_planes', 'visible_sketch', 'hidden_body_refs', 'hidden_body_ids'}:
        raise ValueError('Invalid visibility settings.')
    for key in ('hidden_bodies', 'visible_planes'):
        if not isinstance(value[key], list) or len(value[key]) > 1000 or any(type(x) is not int or x < 0 or x > 999 for x in value[key]):
            raise ValueError('Invalid visibility settings.')
    sketch = value['visible_sketch']
    if sketch is not None and (not isinstance(sketch, str) or len(sketch) > 128):
        raise ValueError('Invalid visibility settings.')
    from UI.model.body_visibility import validate_body_reference
    refs = value.get('hidden_body_refs', [])
    if not isinstance(refs, list) or len(refs) > 1000:
        raise ValueError('Invalid visibility settings.')
    result = {'hidden_bodies': sorted(set(value['hidden_bodies'])),
              'visible_planes': sorted(set(value['visible_planes'])), 'visible_sketch': sketch}
    if 'hidden_body_refs' in value:
        result['hidden_body_refs'] = [validate_body_reference(ref) for ref in refs]
    if 'hidden_body_ids' in value:
        ids = value['hidden_body_ids']
        if not isinstance(ids, list) or len(ids) > 1000 or any(not isinstance(item, str) or not item or len(item) > 4096 for item in ids):
            raise ValueError('Invalid visibility settings.')
        result['hidden_body_ids'] = list(dict.fromkeys(ids))
    return result


def load_snapshot(path, include_planes=False, include_timeline=False, include_feature_graph=False, include_graph_history=False, include_visibility=False):
    try:
        with zipfile.ZipFile(path, 'r') as archive:
            entries = archive.infolist()
            if (len(entries) > 2 * MAX_HISTORY + 2 or
                    sum(entry.file_size for entry in entries) > MAX_TOTAL_BYTES or
                    len({entry.filename for entry in entries}) != len(entries)):
                raise ValueError('Snapshot archive is too large or contains duplicate entries.')
            def read(name):
                info = archive.getinfo(name)
                if info.file_size > MAX_ENTRY_BYTES:
                    raise ValueError('Snapshot entry is too large.')
                return archive.read(info)
            manifest = json.loads(read('manifest.json'))
            if (not isinstance(manifest, dict) or manifest.get('format') != 'FluentCAD-solid-snapshot'
                    or manifest.get('version') != 1):
                raise ValueError('Unsupported model snapshot format or version.')
            plane_data = validate_planes(manifest.get('planes', []))
            has_current = manifest.get('has_current', True)
            if type(has_current) is not bool or has_current != ('current.brep' in archive.namelist()):
                raise ValueError('Invalid current geometry in model snapshot.')
            current = _read_shape(read('current.brep')) if has_current else None
            histories = []
            for name in ('undo', 'redo'):
                flags = manifest.get(name)
                if not isinstance(flags, list) or len(flags) > MAX_HISTORY or not all(type(x) is bool for x in flags):
                    raise ValueError('Invalid model history.')
                histories.append([_read_shape(read(f'{name}/{i}.brep')) if present else None
                                  for i, present in enumerate(flags)])
            timeline_data = validate_timeline(manifest.get('timeline'), len(histories[0]), len(histories[1]))
            graph = None
            graph_data = manifest.get('feature_graph')
            if graph_data is not None and (include_feature_graph or not has_current):
                graph = FeatureGraph.from_data(graph_data)
            if not has_current and (graph is None or graph.latest_solid is not None or any(histories)):
                raise ValueError('Inconsistent parametric document without a solid.')
            result = (current, histories[0], histories[1])
            if include_planes:
                result += (plane_data,)
            if include_timeline:
                result += (timeline_data,)
            if include_feature_graph:
                result += (graph,)
            if include_graph_history:
                result += (validate_graph_history(manifest.get('graph_history')),)
            if include_visibility:
                result += (validate_visibility(manifest['visibility']) if 'visibility' in manifest else None,)
            return result
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError('Invalid or damaged model snapshot.') from exc
