#include <torch/extension.h>
#include "cumesh.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    /* CUMESH */
    py::class_<cumesh::CuMesh>(m, "CuMesh")
        .def(py::init<>())
        .def("num_vertices", &cumesh::CuMesh::num_vertices)
        .def("num_faces", &cumesh::CuMesh::num_faces)
        .def("num_edges", &cumesh::CuMesh::num_edges)
        .def("num_boundaries", &cumesh::CuMesh::num_boundaries)
        .def("num_conneted_components", &cumesh::CuMesh::num_conneted_components)
        .def("num_boundary_conneted_components", &cumesh::CuMesh::num_boundary_conneted_components)
        .def("num_boundary_loops", &cumesh::CuMesh::num_boundary_loops)
        .def("init", &cumesh::CuMesh::init)
        .def("read", &cumesh::CuMesh::read)
        .def("read_edges", &cumesh::CuMesh::read_edges)
        .def("read_boundaries", &cumesh::CuMesh::read_boundaries)
        .def("read_manifold_face_adjacency", &cumesh::CuMesh::read_manifold_face_adjacency)
        .def("read_manifold_boundary_adjacency", &cumesh::CuMesh::read_manifold_boundary_adjacency)
        .def("read_connected_components", &cumesh::CuMesh::read_connected_components)
        .def("read_boundary_connected_components", &cumesh::CuMesh::read_boundary_connected_components)
        .def("read_boundary_loops", &cumesh::CuMesh::read_boundary_loops)
        .def("get_vertex_face_adjacency", &cumesh::CuMesh::get_vertex_face_adjacency)
        .def("get_edges", &cumesh::CuMesh::get_edges)
        .def("get_edge_face_adjacency", &cumesh::CuMesh::get_edge_face_adjacency)
        .def("get_vertex_edge_adjacency", &cumesh::CuMesh::get_vertex_edge_adjacency)
        .def("get_boundary_info", &cumesh::CuMesh::get_boundary_info)
        .def("get_vertex_boundary_adjacency", &cumesh::CuMesh::get_vertex_boundary_adjacency)
        .def("get_edge_is_manifold", &cumesh::CuMesh::get_edge_is_manifold)
        .def("get_vertex_is_manifold", &cumesh::CuMesh::get_vertex_is_manifold)
        .def("get_manifold_face_adjacency", &cumesh::CuMesh::get_manifold_face_adjacency)
        .def("get_manifold_boundary_adjacency", &cumesh::CuMesh::get_manifold_boundary_adjacency)
        .def("get_connected_components", &cumesh::CuMesh::get_connected_components)
        .def("get_boundary_connected_components", &cumesh::CuMesh::get_boundary_connected_components)
        .def("get_boundary_loops", &cumesh::CuMesh::get_boundary_loops)
        .def("remove_faces", &cumesh::CuMesh::remove_faces)
        .def("remove_unreferenced_vertices", &cumesh::CuMesh::remove_unreferenced_vertices)
        .def("remove_duplicate_faces", &cumesh::CuMesh::remove_duplicate_faces)
        .def("fill_holes", &cumesh::CuMesh::fill_holes)
        .def("simplify_step", &cumesh::CuMesh::simplify_step);
}