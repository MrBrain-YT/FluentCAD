import sys


def smoke_test():
    """Check bundled CAD dependencies and resources without opening a window."""
    from pathlib import Path
    from build123d import Box
    from UI.main import content
    from UI.model.ModelConverter import b123d_to_pro_mesh

    if not (Path(content.__file__).resolve().parent.parent / 'images' / 'splash.png').is_file():
        raise RuntimeError('Bundled interface images are missing')
    mesh = b123d_to_pro_mesh(Box(1, 1, 1))
    if mesh.n_points == 0 or mesh.n_cells == 0:
        raise RuntimeError('CAD mesh smoke test failed')


if __name__ == '__main__':
    if sys.argv[1:] == ['--smoke-test']:
        smoke_test()
    else:
        from UI.main import open_window
        open_window()
