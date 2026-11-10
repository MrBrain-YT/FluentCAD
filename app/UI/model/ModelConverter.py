import numpy as np
import pyvista as pv

def b123d_to_pro_mesh(part, tol=0.001, ang=0.1):
    all_verts = []
    all_tris = []
    face_ids = []
    v_count = 0

    for i, face in enumerate(list(part.faces())):
        v, t = face.tessellate(tol, ang)
        if not v or not t:
            continue
            
        v_np = np.array([(pt.X, pt.Y, pt.Z) for pt in v], dtype=np.float32)
        t_np = np.array(t, dtype=np.int32) + v_count

        all_verts.append(v_np)
        vtk_tris = np.insert(t_np, 0, 3, axis=1).flatten()
        all_tris.append(vtk_tris)
        face_ids.extend([i] * len(t_np))
        v_count += len(v_np)

    mesh = pv.PolyData(np.concatenate(all_verts), np.concatenate(all_tris))
    mesh.cell_data["face_id"] = np.array(face_ids, dtype=np.int32)
    mesh.cell_data["original_cell_ids"] = np.arange(mesh.n_cells, dtype=np.int32)

    # Просто вычисляем нормали без переворота
    mesh.compute_normals(
        inplace=True,
        consistent_normals=True,
        auto_orient_normals=True,
        split_vertices=True,
        feature_angle=30.0
    )
    
    return mesh