"""Planar sketch topology, built with Open CASCADE's intersection engine."""
import numpy as np
import build123d as cad
from OCP.BOPAlgo import BOPAlgo_Splitter
from OCP.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCP.TopExp import TopExp_Explorer
from OCP.TopoDS import TopoDS
from UI.model.tools.Slot import slot_edges, polyline_slot_edges


def sketch_edges(tools, normal, tolerance=1e-6):
    edges = []
    for tool in tools:
        points = [tuple(float(v) for v in p) for p in tool.points]
        if tool.type == 'line':
            for start, end in zip(points, points[1:]):
                if np.linalg.norm(np.subtract(start, end)) > tolerance:
                    edges.append(cad.Edge.make_line(start, end))
        elif tool.type == 'slot':
            edges.extend(slot_edges(points, normal, tolerance))
        elif tool.type == 'polyline_slot':
            edges.extend(polyline_slot_edges(points, normal, tool.radius, tolerance))
        elif tool.type == 'arc_slot':
            from UI.model.arc_slot import arc_slot_edges
            edges.extend(arc_slot_edges(points, normal, tool.radius))
        elif tool.type == 'spline_slot':
            from UI.model.spline_slot import spline_slot_edges
            edges.extend(spline_slot_edges(points, tool.radius, tool.path_kind, tool.degree, tool.knots, tool.weights))
        elif tool.type == 'composite_path_slot':
            from UI.model.composite_slot import composite_slot_edges
            edges.extend(composite_slot_edges(tool.segments, tool.plane, tool.radius))
        elif tool.type == 'spline':
            if len(points) == 4: edges.append(cad.Edge.make_bezier(*points))
        elif tool.type == 'nurbs':
            from UI.model.nurbs import nurbs_edge
            if len(points) >= 3: edges.append(nurbs_edge(points, tool.degree, tool.knots, tool.weights))
        elif tool.type == 'periodic_nurbs':
            from UI.model.nurbs import periodic_nurbs_edge
            if len(points) >= 3: edges.append(periodic_nurbs_edge(points, tool.degree, tool.weights, tool.spans))
        elif tool.type == 'fit_spline':
            from UI.model.constraints import fit_spline
            if len(points) < 3: continue
            curve = fit_spline(points)
            for i in range(len(points) - 1):
                a, b = curve(i), curve(i + 1)
                edges.append(cad.Edge.make_bezier(tuple(a), tuple(a + curve(i, 1) / 3),
                                                  tuple(b - curve(i + 1, 1) / 3), tuple(b)))
        elif tool.type == 'curve' and len(points) >= 3:
            try:
                edges.append(cad.Edge.make_three_point_arc(points[0], points[2], points[1]))
            except (ValueError, RuntimeError):
                continue
        elif tool.type == 'circle' and tool.center is not None and tool.radius and tool.radius > tolerance:
            edges.append(cad.Edge.make_circle(tool.radius, cad.Plane(origin=tuple(tool.center), z_dir=tuple(normal))))
        elif tool.type == 'ellipse' and tool.center is not None and (tool.is_closed or len(points) >= 5):
            if min(tool.major_radius, tool.minor_radius) <= tolerance:
                continue
            x_axis = np.asarray(tool.major_axis_vector)
            major, minor = tool.major_radius, tool.minor_radius
            angle_shift = 0.0
            if minor > major:
                major, minor = minor, major
                x_axis = np.cross(normal, x_axis)
                angle_shift = -np.pi / 2
            plane = cad.Plane(origin=tuple(tool.center), x_dir=tuple(x_axis), z_dir=tuple(normal))
            if tool.is_closed:
                edges.append(cad.Edge.make_ellipse(major, minor, plane))
            else:
                start = np.degrees(tool.start_angle + angle_shift)
                end = np.degrees(tool.end_angle + angle_shift)
                edges.append(cad.Edge.make_ellipse(major, minor, plane, start_angle=start, end_angle=end))
    return edges


def _explore(shape, kind):
    explorer = TopExp_Explorer(shape, kind)
    result = []
    while explorer.More():
        if kind == TopAbs_EDGE:
            result.append(cad.Edge(TopoDS.Edge_s(explorer.Current())))
        else:
            result.append(cad.Face(TopoDS.Face_s(explorer.Current())))
        explorer.Next()
    return result


def split_edges(edges, tolerance=1e-6):
    if len(edges) < 2:
        return list(edges)
    splitter = BOPAlgo_Splitter()
    splitter.SetFuzzyValue(tolerance)
    for edge in edges:
        splitter.AddArgument(edge.wrapped)
    splitter.Perform()
    if splitter.HasErrors():
        raise ValueError('Open CASCADE failed to split sketch curves')
    return _explore(splitter.Shape(), TopAbs_EDGE)


def build_profiles(tools, normal, tolerance=1e-6):
    """Extract all bounded cells, including regions with holes, from a planar sketch."""
    edges = sketch_edges([tool for tool in tools if not getattr(tool, 'is_construction', False)], normal, tolerance)
    if not edges:
        return []
    if len(edges) == 1 and edges[0].is_closed:
        try:
            face = cad.Face(cad.Wire([edges[0]]))
            if face.is_valid and face.area > tolerance * tolerance:
                return [face]
        except (ValueError, RuntimeError):
            pass
    # Construct a generous planar frame around the sketch. The only face touching
    # its boundary is the exterior, which is larger than all bounded cells combined.
    samples = [np.array(tuple(edge.position_at(t))) for edge in edges for t in np.linspace(0, 1, 17)]
    origin = samples[0]
    frame_plane = cad.Plane(origin=tuple(origin), z_dir=tuple(normal))
    x_axis, y_axis = np.array(tuple(frame_plane.x_dir)), np.array(tuple(frame_plane.y_dir))
    local = np.array([[np.dot(p - origin, x_axis), np.dot(p - origin, y_axis)] for p in samples])
    low, high = local.min(axis=0), local.max(axis=0)
    midpoint = (low + high) / 2
    span = max(float(np.max(high - low)), 1.)
    center = origin + midpoint[0] * x_axis + midpoint[1] * y_axis
    frame = cad.Face.make_rect(span * 10, span * 10,
                               cad.Plane(origin=tuple(center), x_dir=tuple(x_axis), z_dir=tuple(normal)))
    splitter = BOPAlgo_Splitter()
    splitter.SetFuzzyValue(tolerance)
    splitter.AddArgument(frame.wrapped)
    for edge in edges:
        splitter.AddTool(edge.wrapped)
    splitter.Perform()
    if splitter.HasErrors():
        raise ValueError('Open CASCADE failed to construct sketch regions')
    faces = _explore(splitter.Shape(), TopAbs_FACE)
    if not faces:
        return []
    exterior = max(range(len(faces)), key=lambda i: faces[i].area)
    return [face for i, face in enumerate(faces)
            if i != exterior and face.is_valid and face.area > tolerance * tolerance]
