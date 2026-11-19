"""Construct a solid through an ordered set of sketch profiles."""
import build123d as cad

from UI.model.solid_operations import combine_solid


def _loft_solid(profiles, guides, ends):
    from UI.model.guided_loft import guided_loft
    if ends:
        from UI.model.loft_end_conditions import constrained_loft
        from UI.model.loft_patch_continuity import PATCH_SUPPORTS
        try:
            return constrained_loft(profiles, ends, guides)
        except ValueError:
            if len(guides) < 2 or any(face.geom_type not in PATCH_SUPPORTS for _, face in ends.values()):
                raise
            return guided_loft(profiles, guides, ends)
    if guides:
        return guided_loft(profiles, guides)
    return cad.loft(profiles)


def build_sketch_loft(profiles, base=None, *, guides=(), ends=None, rim_ends=None, hole_matches=None):
    rim_ends = rim_ends or {}
    hole_matches = hole_matches or {}
    try:
        if hole_matches or ((guides or ends or rim_ends) and any(profile.inner_wires() for profile in profiles)):
            from UI.model.loft_holes import loft_with_holes
            solid = loft_with_holes(profiles, guides, ends, rim_ends, hole_matches, _loft_solid)
        else:
            conditions = dict(ends or {})
            for side, per_rim in rim_ends.items():
                if per_rim.keys() - {0}:
                    raise ValueError('The selected Loft contour is missing.')
                if per_rim:
                    conditions[side] = per_rim[0]
            solid = _loft_solid(profiles, guides, conditions)
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f'Could not loft the selected sections in this order. {exc}') from exc
    if len(solid.solids()) != 1 or not solid.is_valid or solid.volume <= 1e-9:
        raise ValueError('Loft did not produce one valid solid. Check the section order.')
    return combine_solid(solid, 'new_body', base)
