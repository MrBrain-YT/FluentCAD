"""Orthographic footprint boundaries of a solid in sketch coordinates."""
import build123d as cad
from OCP.BRepLib import BRepLib
from OCP.HLRAlgo import HLRAlgo_Projector
from OCP.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCP.IntCurvesFace import IntCurvesFace_ShapeIntersector
from OCP.gp import gp_Ax2, gp_Dir, gp_Lin, gp_Pnt

from UI.model.profiles import build_profiles


def _visible_edges(body, axis):
    algorithm = HLRBRep_Algo()
    algorithm.Add(body.wrapped)
    algorithm.Projector(HLRAlgo_Projector(gp_Ax2(gp_Pnt(), axis, gp_Dir(1, 0, 0))))
    algorithm.Update()
    algorithm.Hide()
    result = HLRBRep_HLRToShape(algorithm)
    edges = []
    for compound in (result.VCompound(), result.Rg1LineVCompound(), result.OutLineVCompound()):
        if not compound.IsNull():
            if not BRepLib.BuildCurves3d_s(compound, 1e-7):
                raise ValueError('Could not build the silhouette curves.')
            edges.extend(edge for edge in cad.Compound.cast(compound).edges() if edge.length > 1e-6)
    return edges


def _interior_point(face, tolerance):
    points, triangles = face.tessellate(tolerance, 0.1)
    if not triangles:
        raise ValueError('Could not classify a region of the silhouette.')
    # A centroid of a nondegenerate interior triangle also works for concave faces
    # and annuli, where the geometric center of the entire face can be outside.
    a, b, c = max(triangles, key=lambda t: (points[t[1]] - points[t[0]]).cross(points[t[2]] - points[t[0]]).length)
    return (points[a] + points[b] + points[c]) / 3


def project_body_silhouette(body, plane):
    local = plane.to_local_coords(body)
    axis = gp_Dir(0, 0, 1)
    edges = _visible_edges(local, axis)
    cells = build_profiles([], (0, 0, 1), linked_edges=edges)
    bounds = local.bounding_box()
    margin = max(bounds.diagonal * 0.1, 1.0)
    z = bounds.min.Z - margin
    length = bounds.max.Z - z + margin
    intersector = IntCurvesFace_ShapeIntersector()
    intersector.Load(local.wrapped, 1e-7)
    covered = []
    for face in cells:
        sample = _interior_point(face, max(face.bounding_box().diagonal * 1e-4, 1e-6))
        intersector.Perform(gp_Lin(gp_Pnt(sample.X, sample.Y, z), axis), 0, length)
        if not intersector.IsDone():
            raise ValueError('Could not classify a region of the silhouette.')
        if intersector.NbPnt():
            covered.append(face)
    if not covered:
        raise ValueError('The selected body has no closed silhouette on this plane.')
    footprint = covered[0] if len(covered) == 1 else covered[0].fuse(*covered[1:])
    if footprint is None:
        raise ValueError('Could not join the covered silhouette regions.')
    faces = footprint.faces()
    exteriors = [face.outer_wire() for face in faces]
    holes = [wire for face in faces for wire in face.inner_wires()]
    boundaries = []
    # HLR supplies no persistent identities for distinct holes. Sharing a token
    # prevents a profile in one of several similar holes from silently switching.
    # The loop counts also distinguish topology changes from resized boundaries.
    for kind, wires in (('exterior', exteriors), ('hole', holes)):
        name = f'{kind}-{len(wires)}'
        boundaries.extend((name, plane.from_local_coords(edge)) for wire in wires for edge in wire.edges())
    return boundaries
