"""Export solid geometry, retaining CAD geometry in STEP."""
from pathlib import Path
import build123d as cad


def export_body(part, path, format_name):
    if part is None or not part.solids():
        raise ValueError('Create a solid body before exporting.')
    path = Path(path)
    if format_name == 'step':
        success = cad.export_step(part, path)
    elif format_name == 'stl':
        success = cad.export_stl(part, path)
    else:
        raise ValueError('Unsupported export format.')
    if not success:
        raise OSError('The model could not be exported.')
    return path
