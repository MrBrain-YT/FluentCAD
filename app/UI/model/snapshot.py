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


def save_snapshot(path, current, undo, redo, planes=None, timeline=None, feature_graph=None, graph_history=None):
    if current is None or not current.solids():
        raise ValueError('Create a solid before saving a model snapshot.')
    if len(undo) > MAX_HISTORY or len(redo) > MAX_HISTORY:
        raise ValueError('The model history is too large to save.')
    path = Path(path)
    plane_data = validate_planes(planes or [])
    timeline_data = validate_timeline(timeline, len(undo), len(redo))
    manifest = {'format': 'FluentCAD-solid-snapshot', 'version': 1,
                'undo': [x is not None for x in undo], 'redo': [x is not None for x in redo],
                'planes': plane_data, 'timeline': timeline_data}
    if feature_graph is not None:
        graph_data = feature_graph.to_data()
        FeatureGraph.from_data(graph_data)
        manifest['feature_graph'] = graph_data
        manifest['graph_history'] = validate_graph_history(graph_history)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix='.fluentcad-', suffix='.tmp', dir=path.parent, delete=False) as file:
            temporary = Path(file.name)
        with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED) as archive:
            archive.writestr('manifest.json', json.dumps(manifest))
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
        result[key] = []
        for value in values:
            graph = FeatureGraph.from_data(value)
            if not any(hasattr(item, 'solids') and item.solids() for item in graph.results.values()):
                raise ValueError('Parametric history contains no active solid.')
            result[key].append(graph.to_data())
    return result


def load_snapshot(path, include_planes=False, include_timeline=False, include_feature_graph=False, include_graph_history=False):
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
            if manifest.get('format') != 'FluentCAD-solid-snapshot' or manifest.get('version') != 1:
                raise ValueError('Unsupported model snapshot format or version.')
            plane_data = validate_planes(manifest.get('planes', []))
            current = _read_shape(read('current.brep'))
            histories = []
            for name in ('undo', 'redo'):
                flags = manifest.get(name)
                if not isinstance(flags, list) or len(flags) > MAX_HISTORY or not all(type(x) is bool for x in flags):
                    raise ValueError('Invalid model history.')
                histories.append([_read_shape(read(f'{name}/{i}.brep')) if present else None
                                  for i, present in enumerate(flags)])
            timeline_data = validate_timeline(manifest.get('timeline'), len(histories[0]), len(histories[1]))
            result = (current, histories[0], histories[1])
            if include_planes:
                result += (plane_data,)
            if include_timeline:
                result += (timeline_data,)
            if include_feature_graph:
                graph_data = manifest.get('feature_graph')
                result += (FeatureGraph.from_data(graph_data) if graph_data is not None else None,)
            if include_graph_history:
                result += (validate_graph_history(manifest.get('graph_history')),)
            return result
    except (zipfile.BadZipFile, KeyError, json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise ValueError('Invalid or damaged model snapshot.') from exc
