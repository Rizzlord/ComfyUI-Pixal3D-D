import numpy as np
from tqdm import tqdm
import pyvista as pv


def _is_empty_mesh(vertices: np.ndarray, faces: np.ndarray) -> bool:
    return vertices is None or faces is None or len(vertices) == 0 or len(faces) == 0


def _has_nonfinite_vertices(vertices: np.ndarray) -> bool:
    return vertices is None or not np.isfinite(vertices).all()


def postprocess_mesh(
    vertices: np.array,
    faces: np.array,
    simplify: bool = False,
    simplify_ratio: float = 0.9,
    verbose: bool = False,
):
    """
    Postprocess a mesh by simplifying.

    Args:
        vertices (np.array): Vertices of the mesh. Shape (V, 3).
        faces (np.array): Faces of the mesh. Shape (F, 3).
        simplify (bool): Whether to simplify the mesh, using quadric edge collapse.
        simplify_ratio (float): Ratio of faces to keep after simplification.
        verbose (bool): Whether to print progress.
    """

    if verbose:
        tqdm.write(f'Before postprocess: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    if _is_empty_mesh(vertices, faces):
        if verbose:
            tqdm.write('Skipping postprocess: empty mesh')
        return vertices, faces

    # Simplify
    if simplify and simplify_ratio > 0:
        original_vertices = vertices
        original_faces = faces
        mesh = pv.PolyData(vertices, np.concatenate([np.full((faces.shape[0], 1), 3), faces], axis=1))
        mesh = mesh.decimate(simplify_ratio, progress_bar=verbose)
        vertices, faces = mesh.points, mesh.faces.reshape(-1, 4)[:, 1:]
        if _has_nonfinite_vertices(vertices):
            if verbose:
                tqdm.write('Decimate produced non-finite vertices, reverting to original mesh')
            vertices, faces = original_vertices, original_faces
        if verbose:
            tqdm.write(f'After decimate: {vertices.shape[0]} vertices, {faces.shape[0]} faces')

    return vertices, faces
