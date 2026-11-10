"""Transactional feature graph for rebuilding supported solid operations.

References identify complete feature results, not faces or edges. This module is
independent from the legacy snapshot timeline until that data can be migrated.
"""
from dataclasses import dataclass, field, replace
from copy import deepcopy
import ast
import uuid
import math
import build123d as cad

from UI.model.expressions import evaluate_expression
from UI.model.solid_operations import build_primitive, build_extrusion, build_revolution
from UI.model.transforms import move_copy, scale_model, mirror_model, rectangular_pattern, circular_pattern
from UI.model.geometry_references import resolve_reference
from UI.model.parametric_sketch import ParametricSketch, build_sketch
from UI.model.surface_operations import fillet_edges, chamfer_edges, shell_face, hole_at_face_center, offset_solid


class RebuildError(ValueError):
    def __init__(self, message, feature_id=None):
        super().__init__(message)
        self.feature_id = feature_id


@dataclass(frozen=True)
class Feature:
    id: str
    name: str
    kind: str
    parameters: dict = field(default_factory=dict)
    input_id: str | None = None
    suppressed: bool = False

    @staticmethod
    def create(name, kind, parameters, input_id=None):
        return Feature(uuid.uuid4().hex, name, kind, deepcopy(parameters), input_id)


def resolve_parameters(expressions):
    expressions = dict(expressions)
    values, visiting = {}, set()
    def resolve(name):
        if name in values:
            return values[name]
        if name in visiting:
            raise RebuildError(f'Circular parameter dependency: {name}')
        if name not in expressions:
            raise RebuildError(f'Unknown parameter: {name}')
        if not name.isidentifier() or name.startswith('_'):
            raise RebuildError(f'Invalid parameter name: {name}')
        visiting.add(name)
        expression = str(expressions[name])
        try:
            tree = ast.parse(expression, mode='eval')
            dependencies = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
            variables = {dependency: resolve(dependency) for dependency in dependencies}
            value = evaluate_expression(expression, variables)
        except (SyntaxError, ValueError) as exc:
            if isinstance(exc, RebuildError):
                raise
            raise RebuildError(f'Invalid parameter {name}: {exc}') from exc
        finally:
            visiting.discard(name)
        values[name] = value
        return value
    for parameter in expressions:
        resolve(parameter)
    return values


class FeatureGraph:
    def __init__(self, features=(), parameters=None):
        self._features = tuple(deepcopy(tuple(features)))
        self._parameters = deepcopy(parameters or {})
        self.results = {}
        self.blocked = {}
        self.rebuild()

    @property
    def features(self):
        return deepcopy(self._features)

    @property
    def parameters(self):
        return deepcopy(self._parameters)

    @staticmethod
    def _build(features, parameters):
        values = resolve_parameters(parameters)
        results, blocked, seen = {}, {}, set()
        def number(value):
            if isinstance(value, str):
                return evaluate_expression(value, values)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('Numeric parameter must be finite.')
            return float(value)
        def count(value):
            numeric = number(value)
            if not numeric.is_integer() or numeric < 1 or numeric > 100:
                raise ValueError('Pattern count must be a whole number from 1 to 100.')
            return int(numeric)
        for feature in features:
            if not feature.id or feature.id in seen:
                raise RebuildError('Duplicate or empty feature identifier.', feature.id)
            seen.add(feature.id)
            if feature.suppressed:
                blocked[feature.id] = 'Suppressed'
                continue
            source = None
            if feature.input_id:
                if feature.input_id == feature.id or feature.input_id not in seen:
                    raise RebuildError('Dependency is missing or occurs later in the timeline.', feature.id)
                if feature.input_id not in results:
                    blocked[feature.id] = f'Dependency {feature.input_id} is suppressed or blocked.'
                    continue
                source = results[feature.input_id]
            try:
                args = feature.parameters
                if feature.kind == 'datum_plane':
                    origin = tuple(number(x) for x in args.get('origin', (0, 0, 0)))
                    normal = tuple(number(x) for x in args.get('normal', (0, 0, 1)))
                    result = cad.Plane(origin=origin, z_dir=normal)
                elif feature.kind == 'rectangle_sketch':
                    if not isinstance(source, cad.Plane):
                        raise ValueError('Rectangle sketch requires a datum plane.')
                    width, height = number(args['width']), number(args['height'])
                    if width <= 0 or height <= 0:
                        raise ValueError('Sketch dimensions must be positive.')
                    result = cad.Face.make_rect(width, height, source)
                elif feature.kind == 'planar_sketch':
                    if not isinstance(source, cad.Plane):
                        raise ValueError('Sketch requires a datum plane.')
                    result = build_sketch(source, args.get('entities'), number, args.get('points'), args.get('constraints'))
                elif feature.kind in ('extrude', 'revolve'):
                    if isinstance(source, ParametricSketch):
                        profile_index = args.get('profile_index', 0)
                        if type(profile_index) is not int or not 0 <= profile_index < len(source.profiles):
                            raise ValueError('Sketch profile index is not available.')
                        profile = source.profiles[profile_index]
                    elif isinstance(source, cad.Face):
                        profile = source
                    else:
                        raise ValueError('Operation requires a planar sketch profile.')
                    if feature.kind == 'extrude':
                        result = build_extrusion(profile, tuple(profile.normal_at()), number(args['distance']))
                    else:
                        origin = [number(x) for x in args.get('axis_origin', (0, 0, 0))]
                        axis = [number(x) for x in args.get('axis_direction', (0, 1, 0))]
                        result = build_revolution(profile, origin, axis, number(args.get('angle', 360)))
                elif feature.kind == 'primitive':
                    result = build_primitive(args['type'], [number(x) for x in args['dimensions']],
                        [number(x) for x in args.get('position', (0, 0, 0))],
                        args.get('operation', 'new_body'), source)
                elif feature.kind == 'move':
                    if source is None: raise ValueError('Move requires an input feature.')
                    result = move_copy(source, [number(x) for x in args.get('translation', (0, 0, 0))],
                        [number(x) for x in args.get('rotation', (0, 0, 0))],
                        [number(x) for x in args.get('pivot', (0, 0, 0))], bool(args.get('copy', False)))
                elif feature.kind == 'scale':
                    if source is None: raise ValueError('Scale requires an input feature.')
                    result = scale_model(source, number(args['factor']),
                        [number(x) for x in args.get('center', (0, 0, 0))], bool(args.get('copy', False)))
                elif feature.kind == 'mirror':
                    if source is None: raise ValueError('Mirror requires an input feature.')
                    result = mirror_model(source, args['plane'], number(args.get('offset', 0)), bool(args.get('copy', True)))
                elif feature.kind == 'rectangular_pattern':
                    if source is None: raise ValueError('Pattern requires an input feature.')
                    result = rectangular_pattern(source, count(args['count_x']), count(args['count_y']),
                                                 number(args['spacing_x']), number(args['spacing_y']))
                elif feature.kind == 'circular_pattern':
                    if source is None: raise ValueError('Pattern requires an input feature.')
                    result = circular_pattern(source, count(args['count']), number(args['angle']),
                                              args.get('axis', 'Z'), [number(x) for x in args.get('center', (0, 0, 0))])
                elif feature.kind in ('fillet', 'chamfer'):
                    if source is None: raise ValueError('Edge operation requires an input feature.')
                    refs = args['edges']
                    if not isinstance(refs, list) or not refs: raise ValueError('Select one or more edges.')
                    if any(ref.get('kind') != 'edge' for ref in refs): raise ValueError('Expected edge references.')
                    indices = [resolve_reference(ref, source, feature.input_id) for ref in refs]
                    result = fillet_edges(source, indices, number(args['radius'])) if feature.kind == 'fillet' else chamfer_edges(source, indices, number(args['distance']))
                elif feature.kind in ('shell', 'hole'):
                    if source is None: raise ValueError('Face operation requires an input feature.')
                    ref = args['face']
                    if ref.get('kind') != 'face': raise ValueError('Expected a face reference.')
                    face_index = resolve_reference(ref, source, feature.input_id)
                    result = shell_face(source, face_index, number(args['thickness'])) if feature.kind == 'shell' else hole_at_face_center(source, face_index, number(args['diameter']), number(args['depth']), number(args.get('offset_u', 0)), number(args.get('offset_v', 0)), bool(args.get('through_all', False)))
                elif feature.kind == 'offset':
                    if source is None: raise ValueError('Offset requires an input feature.')
                    result = offset_solid(source, number(args['distance']))
                else:
                    raise ValueError(f'Unsupported feature type: {feature.kind}')
            except Exception as exc:
                raise RebuildError(f'{feature.name}: {exc}', feature.id) from exc
            results[feature.id] = result
        return results, blocked

    def rebuild(self):
        results, blocked = self._build(self.features, self.parameters)
        self.results, self.blocked = results, blocked
        return results

    def _commit(self, features, parameters):
        features, parameters = tuple(deepcopy(features)), deepcopy(parameters)
        results, blocked = self._build(features, parameters)
        self._features, self._parameters = features, parameters
        self.results, self.blocked = results, blocked

    def to_data(self):
        return {'version': 1, 'parameters': deepcopy(self._parameters),
                'features': [dict(id=item.id, name=item.name, kind=item.kind,
                                  parameters=deepcopy(item.parameters), input_id=item.input_id,
                                  suppressed=item.suppressed) for item in self._features]}

    @classmethod
    def from_data(cls, data):
        if not isinstance(data, dict) or data.get('version') != 1:
            raise ValueError('Unsupported feature graph format.')
        features, parameters = data.get('features'), data.get('parameters')
        if not isinstance(features, list) or len(features) > 1000 or not isinstance(parameters, dict) or len(parameters) > 1000:
            raise ValueError('Invalid feature graph.')
        items = []
        for value in features:
            if not isinstance(value, dict) or set(value) != {'id', 'name', 'kind', 'parameters', 'input_id', 'suppressed'}:
                raise ValueError('Invalid feature entry.')
            if (not isinstance(value['id'], str) or not value['id'] or len(value['id']) > 128 or
                not isinstance(value['name'], str) or not value['name'].strip() or len(value['name']) > 128 or
                not isinstance(value['kind'], str) or len(value['kind']) > 128 or
                not isinstance(value['parameters'], dict) or type(value['suppressed']) is not bool or
                (value['input_id'] is not None and not isinstance(value['input_id'], str))):
                raise ValueError('Invalid feature data.')
            items.append(Feature(**deepcopy(value)))
        return cls(items, parameters)

    def set_parameters(self, parameters):
        self._commit(self.features, parameters)

    def edit(self, feature_id, **changes):
        if not changes or set(changes) - {'name', 'parameters', 'input_id', 'suppressed'}:
            raise ValueError('Unsupported feature changes.')
        if not any(feature.id == feature_id for feature in self.features):
            raise ValueError('Feature does not exist.')
        if 'name' in changes and (not isinstance(changes['name'], str) or not changes['name'].strip()):
            raise ValueError('Feature name must not be empty.')
        self._commit([replace(feature, **changes) if feature.id == feature_id else feature
                      for feature in self.features], self.parameters)

    def reorder(self, feature_id, position):
        features = list(self.features)
        index = next((i for i, feature in enumerate(features) if feature.id == feature_id), None)
        if index is None or type(position) is not int or not 0 <= position < len(features):
            raise ValueError('Invalid feature or position.')
        features.insert(position, features.pop(index))
        self._commit(features, self.parameters)
