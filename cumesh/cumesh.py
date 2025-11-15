from typing import *
import torch
from tqdm import tqdm
from . import _C


class CuMesh:
    def __init__(self):
        self.cu_mesh = _C.CuMesh()

    def init(self, vertices: torch.Tensor, faces: torch.Tensor):
        """
        Initialize the CuMesh with vertices and faces.

        Args:
            vertices: a tensor of shape [V, 3] containing the vertex positions.
            faces: a tensor of shape [F, 3] containing the face indices.
        """
        assert vertices.ndim == 2 and vertices.shape[1] == 3, "Input vertices must be of shape [V, 3]"
        assert faces.ndim == 2 and faces.shape[1] == 3, "Input faces must be of shape [F, 3]"
        assert vertices.is_contiguous() and faces.is_contiguous(), "Input tensors must be contiguous"
        assert vertices.is_cuda and faces.is_cuda and vertices.device == faces.device, "Input tensors must both be on the same CUDA device"
        self.cu_mesh.init(vertices, faces)
        
    @property
    def num_vertices(self) -> int:
        return self.cu_mesh.num_vertices()
    
    @property
    def num_faces(self) -> int:
        return self.cu_mesh.num_faces()
    
    @property
    def num_edges(self) -> int:
        return self.cu_mesh.num_edges()
    
    @property
    def num_boundaries(self) -> int:
        return self.cu_mesh.num_boundaries()
    
    @property
    def num_conneted_components(self) -> int:
        return self.cu_mesh.num_conneted_components()
    
    @property
    def num_boundary_conneted_components(self) -> int:
        return self.cu_mesh.num_boundary_conneted_components()
    
    @property
    def num_boundary_loops(self) -> int:
        return self.cu_mesh.num_boundary_loops()

    def read(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Read the current vertices and faces from the CuMesh.

        Returns:
            A tuple of two tensors: the vertex positions and the face indices.
        """
        return self.cu_mesh.read()
    
    def read_edges(self) -> torch.Tensor:
        """
        Read the edges of the mesh from the CuMesh.

        Returns:
            A tensor of shape [E, 2] containing the edge indices.
        """
        return self.cu_mesh.read_edges()
    
    def read_boundaries(self) -> torch.Tensor:
        """
        Read the boundary edges of the mesh from the CuMesh.

        Returns:
            A tensor of shape [B] containing the boundary edge indices.
        """
        return self.cu_mesh.read_boundaries()
    
    
    def read_manifold_face_adjacency(self) -> torch.Tensor:
        """
        Read the manifold face adjacency from the CuMesh.

        Returns:
            A tensor of shape [M, 2] containing the manifold face adjacency.
        """
        return self.cu_mesh.read_manifold_face_adjacency()
    
    def read_manifold_boundary_adjacency(self) -> torch.Tensor:
        """
        Read the manifold boundary adjacency from the CuMesh.

        Returns:
            A tensor of shape [M, 2] containing the manifold boundary adjacency.
        """
        return self.cu_mesh.read_manifold_boundary_adjacency()
    
    def read_connected_components(self) -> Tuple[int, torch.Tensor]:
        """
        Read the connected component IDs for each face.

        Returns:
            A tuple of two values:
                - the number of connected components
                - a tensor of shape [F] containing the connected component ID for each face.
        """
        return self.cu_mesh.read_connected_components()
    
    def read_boundary_connected_components(self) -> Tuple[int, torch.Tensor]:
        """
        Read the connected component IDs for each boundary edge.

        Returns:
            A tuple of two values:
                - the number of connected components
                - a tensor of shape [E] containing the connected component ID for each boundary edge.
        """
        return self.cu_mesh.read_boundary_connected_components()
    
    def read_boundary_loops(self) -> Tuple[int, torch.Tensor, torch.Tensor]:
        """
        Read the boundary loops of the mesh.

        Returns:
            A tuple of three values:
                - the number of boundary loops
                - a tensor of shape [L] containing the indices of the boundary edges in each loop.
                - a tensor of shape [N_loops + 1] containing the offsets of the boundary edges in each loop.
        """
        return self.cu_mesh.read_boundary_loops()
        
    def get_vertex_face_adjacency(self):
        """
        Compute the vertex to face adjacency.
        """
        self.cu_mesh.get_vertex_face_adjacency()
        
    def get_edges(self):
        """
        Compute the edges of the mesh.
        """
        self.cu_mesh.get_edges()
        
    def get_edge_face_adjacency(self):
        """
        Compute the edge to face adjacency.
        """
        self.cu_mesh.get_edge_face_adjacency()
        
    def get_vertex_edge_adjacency(self):
        """
        Compute the vertex to edge adjacency.
        """
        self.cu_mesh.get_vertex_edge_adjacency()
        
    def get_boundary_info(self):
        """
        Compute the boundary information of the mesh.
        """
        self.cu_mesh.get_boundary_info()
        
    def get_vertex_boundary_adjacency(self):
        """
        Compute the vertex to boundary adjacency.
        """
        self.cu_mesh.get_vertex_boundary_adjacency()
        
    def get_manifold_face_adjacency(self):
        """
        Compute the manifold face adjacency.
        """
        self.cu_mesh.get_manifold_face_adjacency()
        
    def get_manifold_boundary_adjacency(self):
        """
        Compute the manifold boundary adjacency.
        """
        self.cu_mesh.get_manifold_boundary_adjacency()
        
    def get_connected_components(self):
        """
        Compute the connected components of the mesh.
        """
        self.cu_mesh.get_connected_components()
        
    def get_boundary_connected_components(self):
        """
        Compute the connected components of the boundary of the mesh.
        """
        self.cu_mesh.get_boundary_connected_components()
        
    def get_boundary_loops(self):
        """
        Compute the boundary loops of the mesh.
        """
        self.cu_mesh.get_boundary_loops()
        
    def remove_faces(self, face_mask: torch.Tensor):
        """
        Remove faces from the mesh.

        Args:
            face_mask: a boolean tensor of shape [F] indicating which faces to remove.
        """
        assert face_mask.ndim == 1 and face_mask.shape[0] == self.num_faces, "face_mask must be a boolean tensor of shape [F]"
        assert face_mask.is_contiguous() and face_mask.is_cuda, "face_mask must be a CUDA tensor"
        assert face_mask.dtype == torch.bool, "face_mask must be a boolean tensor"
        self.cu_mesh.remove_faces(face_mask)
    
    def remove_unreferenced_vertices(self):
        """
        Remove unreferenced vertices from the mesh.
        """
        self.cu_mesh.remove_unreferenced_vertices()
        
    def remove_duplicate_faces(self):
        """
        Remove duplicate faces from the mesh.
        """
        self.cu_mesh.remove_duplicate_faces()
        
    def fill_holes(self, max_hole_perimeter: float=3e-2):
        """
        Fill holes in the mesh.

        Args:
            max_hole_perimeter: the maximum perimeter of a hole to fill.
        """
        self.cu_mesh.fill_holes(max_hole_perimeter)
    
    def simplify(self, target_num_faces: int, verbose: bool=False, options: dict={}):
        """
        Simplifies the mesh using a fast approximation algorithm with gpu acceleration.

        Args:
            target_num_faces: the target number of faces to simplify to.
            verbose: whether to print the progress of the simplification.
            options: a dictionary of options for the simplification algorithm.
        """
        assert isinstance(target_num_faces, int) and target_num_faces > 0, "target_num_faces must be a positive integer"

        num_face = self.cu_mesh.num_faces()
        if num_face <= target_num_faces:
            return

        with tqdm(total=num_face-target_num_faces, desc="Simplifying", disable=not verbose) as pbar:
            thresh = options.get('thresh', 1e-8)
            lambda_edge_length = options.get('lambda_edge_length', 1e-2)
            lambda_skinny = options.get('lambda_skinny', 1e-3)
            while True:
                pbar.set_description(f"Simplifying [thres={thresh:.2e}]")
                
                new_num_vert, new_num_face = self.cu_mesh.simplify_step(lambda_edge_length, lambda_skinny, thresh, False)
                
                pbar.update(num_face - max(target_num_faces, new_num_face))

                if new_num_face <= target_num_faces:
                    break
                
                del_num_face = num_face - new_num_face
                if del_num_face / num_face < 1e-2:
                    thresh *= 10
                num_face = new_num_face
            