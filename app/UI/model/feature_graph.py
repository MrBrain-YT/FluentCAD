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
from UI.model.transforms import move_copy, scale_model, mirror_model, rectangular_pattern, circular_pattern, body_index_for_face, apply_to_body
from UI.model.geometry_references import resolve_reference
from UI.model.operation_geometry import extrusion_target, revolution_axis
from UI.model.sketch_links import freeze_sketch_geometry, rebuild_sketch_link, restore_sketch_geometry, validate_sketch_links
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


def sketch_profile_index(sketch, selection):
    key = selection.get('profile_key')
    if key is not None:
        matches = [i for i, candidate in enumerate(sketch.profile_keys) if candidate == key]
        if len(matches) != 1:
            raise ValueError('Selected sketch profile is missing or ambiguous; reselect it.')
        return matches[0]
    if 'profile_count' in selection and selection['profile_count'] != len(sketch.profiles):
        raise ValueError('Sketch profile topology changed; reselect the profile.')
    index = selection.get('profile_index', 0)
    if type(index) is not int or not 0 <= index < len(sketch.profiles):
        raise ValueError('Sketch profile index is not available.')
    return index


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

    @property
    def latest_solid_id(self):
        for feature in reversed(self._features):
            result = self.results.get(feature.id)
            if hasattr(result, 'solids') and result.solids():
                return feature.id
        return None

    @property
    def latest_solid(self):
        return self.results.get(self.latest_solid_id)

    @staticmethod
    def _build(features, parameters):
        values = resolve_parameters(parameters)
        results, blocked, seen = {}, {}, set()
        body_names, link_issues = {}, {}
        all_ids = {feature.id for feature in features}
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
                full_source, body_index = source, None
                if 'body_face' in args:
                    if feature.kind not in ('move', 'scale', 'mirror', 'rectangular_pattern', 'circular_pattern', 'offset', 'shell', 'hole'):
                        raise ValueError('This operation does not support a body reference.')
                    face_index = resolve_reference(args['body_face'], source, feature.input_id, body_names=body_names.get(feature.input_id))
                    body_index = body_index_for_face(source, face_index)
                    if body_index is None:
                        raise ValueError('The referenced body is ambiguous or missing.')
                    source = list(source.solids())[body_index]
                if feature.kind == 'frozen_body':
                    if source is not None: raise ValueError('Frozen body cannot have an input.')
                    from UI.model.legacy_conversion import decode_body
                    result = decode_body(args['brep'])
                elif feature.kind == 'datum_plane':
                    origin = tuple(number(x) for x in args.get('origin', (0, 0, 0)))
                    normal = tuple(number(x) for x in args.get('normal', (0, 0, 1)))
                    if 'face' in args:
                        if source is None or not feature.input_id or args['face'].get('kind') != 'face':
                            raise ValueError('Face-supported plane requires a solid face reference.')
                        index = resolve_reference(args['face'], source, feature.input_id, body_names=body_names.get(feature.input_id))
                        face = list(source.faces())[index]
                        if face.geom_type != cad.GeomType.PLANE: raise ValueError('Sketch support must remain planar.')
                        origin, normal = tuple(face.center()), tuple(face.normal_at())
                    x_dir = tuple(number(x) for x in args['x_dir']) if 'x_dir' in args else None
                    result = cad.Plane(origin=origin, x_dir=x_dir, z_dir=normal)
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
                    links = validate_sketch_links(args.get('links', []))
                    dependencies = {link['source_id'] for link in links}
                    if feature.id in dependencies or (dependencies & all_ids) - seen:
                        raise ValueError('A sketch link points to a later feature or creates a cycle.')
                    inactive = {link['source_id'] for link in links
                                if link['source_id'] in blocked and 'fallback' not in link}
                    if inactive:
                        blocked[feature.id] = f'Linked source {min(inactive)} is suppressed or blocked.'
                        continue
                    linked = []
                    for link in links:
                        source_id = link['source_id']
                        try:
                            if source_id not in results:
                                raise ValueError('The source is missing or suppressed.')
                            geometry = rebuild_sketch_link(link, results[source_id], body_names.get(source_id), source)
                        except (ValueError, RuntimeError) as exc:
                            if 'fallback' not in link:
                                raise
                            geometry = restore_sketch_geometry(link['fallback'], source, link['id'])
                            link_issues.setdefault(feature.id, {})[link['id']] = str(exc)
                        else:
                            link['fallback'] = freeze_sketch_geometry(geometry, source)
                        linked.append(geometry)
                    detached = args.get('detached_links', [])
                    if not isinstance(detached, list) or len(detached) > 1000:
                        raise ValueError('Invalid detached sketch geometry.')
                    detached_ids = {link['id'] for link in links}
                    independent = []
                    for item in detached:
                        if (not isinstance(item, dict) or not isinstance(item.get('id'), str)
                                or not item['id'] or item['id'] in detached_ids):
                            raise ValueError('Invalid or duplicate detached sketch geometry.')
                        detached_ids.add(item['id'])
                        independent.append(restore_sketch_geometry(item.get('geometry'), source, item['id']))
                    result = build_sketch(source, args.get('entities'), number, args.get('points'), args.get('constraints'),
                                          linked, independent, offsets=args.get('offsets'), corners=args.get('corners'),
                                          mirrors=args.get('mirrors'), patterns=args.get('patterns'))
                elif feature.kind == 'loft':
                    from UI.model.loft_end_conditions import resolve_loft_support
                    from UI.model.loft_rims import resolve_hole_matches, section_rims
                    from UI.model.sketch_loft import build_sketch_loft
                    from UI.model.sketch_sweep import sweep_curves
                    sections = args.get('sections')
                    if (not isinstance(sections, list) or not 2 <= len(sections) <= 100
                            or any(not isinstance(section, dict) or not isinstance(section.get('sketch_id'), str)
                                   for section in sections)):
                        raise ValueError('Select two to 100 closed sketch profiles for Loft.')
                    dependencies = {section['sketch_id'] for section in sections}
                    guide_refs = args.get('guides', [])
                    if (not isinstance(guide_refs, list) or len(guide_refs) > 100
                            or any(not isinstance(ref, list) or len(ref) != 2
                                   or not all(isinstance(value, str) and value for value in ref) for ref in guide_refs)):
                        raise ValueError('Invalid Loft guide references.')
                    guide_keys = list(map(tuple, guide_refs))
                    if len(set(guide_keys)) != len(guide_keys):
                        raise ValueError('A Loft guide was selected more than once.')
                    dependencies.update(sketch_id for sketch_id, _ in guide_keys)
                    end_refs = args.get('ends', {})
                    rim_refs = args.get('rim_ends', {})
                    sides = {'start', 'end'}
                    if (not isinstance(end_refs, dict) or set(end_refs) - sides
                            or not isinstance(rim_refs, dict) or set(rim_refs) - sides
                            or (end_refs and rim_refs)
                            or any(not isinstance(rims, dict) or any(not isinstance(key, str) or not key
                                                                     for key in rims)
                                   for rims in rim_refs.values())):
                        raise ValueError('Invalid Loft end conditions.')
                    conditions = [*end_refs.values(),
                                  *(condition for rims in rim_refs.values() for condition in rims.values())]
                    for condition in conditions:
                        if not isinstance(condition, dict) or condition.get('mode') not in ('tangent', 'curvature'):
                            raise ValueError('Invalid Loft end condition.')
                        reference = condition.get('face')
                        if (not isinstance(reference, dict) or reference.get('kind') != 'face'
                                or not isinstance(reference.get('feature_id'), str)):
                            raise ValueError('Select a support face for each Loft end condition.')
                        dependencies.add(reference['feature_id'])
                    if args.get('base_id'):
                        dependencies.add(args['base_id'])
                    if feature.id in dependencies or dependencies - seen:
                        raise ValueError('Loft dependency is missing or occurs later in the timeline.')
                    if dependencies - results.keys():
                        blocked[feature.id] = 'A loft dependency is suppressed or blocked.'
                        continue
                    profiles = []
                    section_sketches = []
                    selected = set()
                    for section in sections:
                        sketch_id = section['sketch_id']
                        sketch = results[sketch_id]
                        if not isinstance(sketch, ParametricSketch):
                            raise ValueError('Loft sections must be planar sketch profiles.')
                        index = sketch_profile_index(sketch, section)
                        if (sketch_id, index) in selected:
                            raise ValueError('A loft profile was selected more than once.')
                        selected.add((sketch_id, index))
                        profiles.append(sketch.profiles[index])
                        section_sketches.append(sketch)
                    available = dict(sweep_curves({sketch_id: results[sketch_id] for sketch_id, _ in guide_keys}))
                    if any(key not in available for key in guide_keys):
                        raise ValueError('A Loft guide curve is missing or ambiguous.')
                    def resolve_condition(profile, condition):
                        reference = condition['face']
                        support_id = reference['feature_id']
                        support = results[support_id]
                        if not hasattr(support, 'solids') or not support.solids():
                            raise ValueError('A Loft support must reference a solid face.')
                        face = resolve_loft_support(profile, reference, support,
                                                    support_id, body_names.get(support_id))
                        return condition['mode'], face

                    rim_cache = {}

                    def rims_at(level):
                        if level not in rim_cache:
                            rims = section_rims(section_sketches[level], profiles[level])
                            rim_cache[level] = {key: (index, face) for index, (key, face) in enumerate(rims)
                                                if key is not None}
                        return rim_cache[level]

                    hole_matches = resolve_hole_matches(args.get('hole_matches', []), len(profiles), rims_at)
                    end_levels = {'start': 0, 'end': len(profiles) - 1}
                    ends = {side: resolve_condition(profiles[end_levels[side]], condition)
                            for side, condition in end_refs.items()}
                    rim_ends = {}
                    for side, refs in rim_refs.items():
                        if not refs:
                            continue
                        level = end_levels[side]
                        rims = rims_at(level)
                        if refs.keys() - rims.keys():
                            name = 'first' if side == 'start' else 'last'
                            raise ValueError(f'A supported contour of the {name} Loft section is missing or ambiguous.')
                        rim_ends[side] = {rims[key][0]: resolve_condition(rims[key][1], condition)
                                          for key, condition in refs.items()}
                    result = build_sketch_loft(profiles, results.get(args.get('base_id')),
                                               guides=[available[key] for key in guide_keys], ends=ends,
                                               rim_ends=rim_ends, hole_matches=hole_matches)
                elif feature.kind == 'sweep':
                    from UI.model.sketch_sweep import build_sketch_sweep
                    if not isinstance(source, ParametricSketch):
                        raise ValueError('Sweep requires a planar sketch profile.')
                    path = args.get('path')
                    if not isinstance(path, list) or any(not isinstance(ref, list) or len(ref) != 2
                            or not isinstance(ref[0], str) for ref in path):
                        raise ValueError('Select a chain of sketch segments.')
                    dependencies = {ref[0] for ref in path}
                    if args.get('base_id'):
                        dependencies.add(args['base_id'])
                    if feature.id in dependencies or dependencies - seen:
                        raise ValueError('Sweep dependency is missing or occurs later in the timeline.')
                    if dependencies - results.keys():
                        blocked[feature.id] = 'A sweep dependency is suppressed or blocked.'
                        continue
                    profile = source.profiles[sketch_profile_index(source, args)]
                    result = build_sketch_sweep(profile, path, results, results.get(args.get('base_id')))
                elif feature.kind in ('extrude', 'revolve'):
                    selections = args.get('profiles', [args])
                    if not isinstance(selections, list) or not selections or not all(isinstance(item, dict) for item in selections):
                        raise ValueError('Select at least one valid sketch profile.')
                    if feature.kind == 'extrude' and 'profile_faces' in args:
                        refs = args['profile_faces']
                        if (not hasattr(source, 'solids') or not source.solids() or not isinstance(refs, list)
                                or not refs or any(not isinstance(ref, dict) or ref.get('kind') != 'face' for ref in refs)):
                            raise ValueError('Select one or more planar faces of a solid.')
                        faces = list(source.faces())
                        indices = [resolve_reference(ref, source, feature.input_id, body_names=body_names.get(feature.input_id))
                                   for ref in refs]
                        if len(set(indices)) != len(indices):
                            raise ValueError('The same face was selected more than once.')
                        profiles = [faces[index] for index in indices]
                        normal = tuple(profiles[0].normal_at())
                    elif isinstance(source, ParametricSketch):
                        indices = [sketch_profile_index(source, selection) for selection in selections]
                        if len(set(indices)) != len(indices):
                            raise ValueError('The same profile was selected more than once.')
                        profiles = [source.profiles[index] for index in indices]
                        normal = tuple(source.plane.z_dir if 'profiles' in args or args.get('along_sketch_normal')
                                       else profiles[0].normal_at())
                    elif isinstance(source, cad.Face) and 'profiles' not in args:
                        profiles = [source]
                        normal = tuple(source.normal_at())
                    else:
                        raise ValueError('Operation requires a planar sketch profile.')
                    base_id = args.get('base_id')
                    if base_id:
                        if base_id == feature.id or base_id not in seen:
                            raise ValueError('Base body is missing or occurs later in the timeline.')
                        if base_id in blocked:
                            blocked[feature.id] = f'Base body {base_id} is suppressed or blocked.'
                            continue
                        if base_id not in results:
                            raise ValueError('Base body is unavailable.')
                    base = results.get(base_id)
                    if feature.kind == 'extrude':
                        extent = args.get('extent', 'distance')
                        target = None
                        if extent == 'to_object':
                            selection = args.get('target')
                            target_id = selection.get('feature_id') if selection else None
                            if target_id == feature.id or (target_id and target_id not in seen):
                                raise ValueError('Target plane is missing or occurs later in the timeline.')
                            if target_id in blocked:
                                blocked[feature.id] = f'Target plane {target_id} is suppressed or blocked.'
                                continue
                            target = extrusion_target(selection, base, base_id, body_names.get(base_id), results)
                        result = build_extrusion(profiles, normal, number(args.get('distance', 0)),
                            args.get('operation', 'new_body'), base, number(args.get('offset', 0)),
                            number(args.get('taper', 0)), bool(args.get('symmetric', False)),
                            number(args['second_distance']) if args.get('second_distance') is not None else None,
                            extent, target, bool(args.get('reverse', False)))
                    else:
                        if len(profiles) != 1:
                            raise ValueError('Select one profile for revolution.')
                        sketch_axis = args.get('sketch_axis')
                        if 'axis_selection' in args:
                            origin, axis = revolution_axis(args['axis_selection'], source if isinstance(source, ParametricSketch) else None,
                                                           base, base_id, body_names.get(base_id))
                        elif sketch_axis is None:
                            origin = [number(x) for x in args.get('axis_origin', (0, 0, 0))]
                            axis = [number(x) for x in args.get('axis_direction', (0, 1, 0))]
                        else:
                            if not isinstance(source, ParametricSketch) or sketch_axis not in ('horizontal', 'vertical'):
                                raise ValueError('Select a valid sketch axis.')
                            origin = tuple(source.plane.origin)
                            axis = tuple(source.plane.x_dir if sketch_axis == 'horizontal' else source.plane.y_dir)
                        result = build_revolution(profiles[0], origin, axis, number(args.get('angle', 360)),
                                                  args.get('operation', 'new_body'), base)
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
                    indices = [resolve_reference(ref, source, feature.input_id, body_names=body_names.get(feature.input_id)) for ref in refs]
                    result = fillet_edges(source, indices, number(args['radius'])) if feature.kind == 'fillet' else chamfer_edges(source, indices, number(args['distance']))
                elif feature.kind in ('shell', 'hole'):
                    if source is None: raise ValueError('Face operation requires an input feature.')
                    ref = args['face']
                    if ref.get('kind') != 'face': raise ValueError('Expected a face reference.')
                    if body_index is not None:
                        full_index = resolve_reference(ref, full_source, feature.input_id, body_names=body_names.get(feature.input_id))
                        selected_face = list(full_source.faces())[full_index]
                        face_index = next((i for i, face in enumerate(source.faces()) if face.is_same(selected_face)), -1)
                        if face_index < 0: raise ValueError('The selected face no longer belongs to the referenced body.')
                    else:
                        face_index = resolve_reference(ref, source, feature.input_id, body_names=body_names.get(feature.input_id))
                    result = shell_face(source, face_index, number(args['thickness'])) if feature.kind == 'shell' else hole_at_face_center(source, face_index, number(args['diameter']), number(args['depth']), number(args.get('offset_u', 0)), number(args.get('offset_v', 0)), bool(args.get('through_all', False)))
                elif feature.kind == 'offset':
                    if source is None: raise ValueError('Offset requires an input feature.')
                    result = offset_solid(source, number(args['distance']))
                else:
                    raise ValueError(f'Unsupported feature type: {feature.kind}')
                if body_index is not None:
                    result = apply_to_body(full_source, body_index, lambda _body: result)
            except Exception as exc:
                raise RebuildError(f'{feature.name}: {exc}', feature.id) from exc
            results[feature.id] = result
            from UI.model.body_lineage import body_names_for_feature
            body_names[feature.id] = body_names_for_feature(feature, result, results, body_names, values)
        return results, blocked, body_names, link_issues

    def rebuild(self):
        features = self.features
        self.results, self.blocked, self.body_names, self.link_issues = self._build(features, self.parameters)
        self._features = features
        return self.results

    def _commit(self, features, parameters):
        features, parameters = tuple(deepcopy(features)), deepcopy(parameters)
        results, blocked, body_names, link_issues = self._build(features, parameters)
        self._features, self._parameters = features, parameters
        self.results, self.blocked, self.body_names = results, blocked, body_names
        self.link_issues = link_issues

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
