// pySMESH2 _core extension module.
//
// B1 milestone: this contains only the "Hello SMESHDS" smoke helper that proves the full
// static-link graph (KERNEL + Geom + SMDS/SMESHDS/SMESH core + StdMeshers + OCCT + VTK)
// resolves and runs. Tier-1 bindings (Shape/Mesh/viscous) land in B2/B3; the test helper is
// removed once those exist.

#include <pybind11/pybind11.h>

#include <BRepPrimAPI_MakeBox.hxx>
#include <TopoDS_Shape.hxx>

#include <SMESH_Gen.hxx>
#include <SMESH_Mesh.hxx>
#include <SMESHDS_Mesh.hxx>

namespace py = pybind11;

namespace {

// Construct an SMESH_Mesh on a unit box, inject one node, return the node count.
// Exercises SMESH_Gen -> SMESH_Mesh -> SMESHDS_Mesh and OCCT shape construction.
long long create_test_mesh() {
    SMESH_Gen gen;
    SMESH_Mesh* mesh = gen.CreateMesh(false);
    const TopoDS_Shape box = BRepPrimAPI_MakeBox(1.0, 1.0, 1.0).Shape();
    mesh->ShapeToMesh(box);

    SMESHDS_Mesh* meshDS = mesh->GetMeshDS();
    meshDS->AddNode(0.0, 0.0, 0.0);
    return static_cast<long long>(meshDS->NbNodes());
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "pySMESH2 native core (B1 skeleton).";
    m.def("create_test_mesh", &create_test_mesh,
          "Build a 1-node mesh on a unit box; return NbNodes() (B1 smoke test).");
}
