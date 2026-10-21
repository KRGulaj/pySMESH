// Copyright (C) 2007-2026  CEA, EDF, OPEN CASCADE
//
// Copyright (C) 2003-2007  OPEN CASCADE, EADS/CCR, LIP6, CEA/DEN,
// CEDRAT, EDF R&D, LEG, PRINCIPIA R&D, BUREAU VERITAS
//
// This library is free software; you can redistribute it and/or
// modify it under the terms of the GNU Lesser General Public
// License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
// Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public
// License along with this library; if not, write to the Free Software
// Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
//
// See http://www.salome-platform.org/ or email : webmaster.salome@opencascade.com
//
//  File    : GEOMAlgo_AlgoTools.hxx
//  Created :
//  Author  : Peter KURNEV

#ifndef _GEOMAlgo_AlgoTools_HeaderFile
#define _GEOMAlgo_AlgoTools_HeaderFile

#include <Standard.hxx>
#include <Standard_Macro.hxx>
#include <Standard_Boolean.hxx>
#include <IntTools_Context.hxx>
#include <Standard_Integer.hxx>

#include <gp_Pnt.hxx>
#include <gp_Pnt2d.hxx>

#include <Geom_Surface.hxx>

#include <TopAbs_ShapeEnum.hxx>

#include <TopoDS_Shape.hxx>
#include <TopoDS_Edge.hxx>
#include <TopoDS_Wire.hxx>
#include <TopoDS_Face.hxx>
#include <TopoDS_Compound.hxx>

#include <TopTools_ListOfShape.hxx>
#include <TopTools_IndexedDataMapOfShapeListOfShape.hxx>
#include <TopTools_ListOfShape.hxx>
#include <TopTools_IndexedDataMapOfShapeShape.hxx>

#include <GEOMAlgo_IndexedDataMapOfPassKeyShapeListOfShape.hxx>
#include <GEOMAlgo_ListOfCoupleOfShapes.hxx>
#include <GEOMAlgo_IndexedDataMapOfShapeIndexedMapOfShape.hxx>

//!  Auxiliary tools for Algorithms <br>
//=======================================================================
//class    : GEOMAlgo_AlgoTools
//purpose  :
//=======================================================================
namespace GEOMAlgo_AlgoTools
{
  Standard_EXPORT
    void FaceNormal (const TopoDS_Face& aF,
                     const Standard_Real U,
                     const Standard_Real V,
                     gp_Vec& aN);

  //! Computes a point <theP> inside the face <theF>. <br>
  //!          <theP2D> - 2D representation of <theP> <br>
  //!          on the surface of <theF> <br>
  //!          Returns 0 in case of success. <br>
  Standard_EXPORT
     Standard_Integer PntInFace(const TopoDS_Face& theF,
                                gp_Pnt& theP,
                                gp_Pnt2d& theP2D) ;

  //! Computes a set of points inside the face <theF>. <br>
  //!          Returns 0 in case of success. <br>
  Standard_EXPORT
     Standard_Integer PointCloudInFace(const TopoDS_Face& theF,
                                       const int          theNbPnts,
                                       TopoDS_Compound&   theCompound) ;

  Standard_EXPORT
     Standard_Boolean IsCompositeShape(const TopoDS_Shape& aS) ;


  Standard_EXPORT
     Standard_Integer RefineSDShapes
      (GEOMAlgo_IndexedDataMapOfPassKeyShapeListOfShape& aMSD,
       const Standard_Real aTol,
       const Handle(IntTools_Context)& aCtx) ;

  Standard_EXPORT
     Standard_Integer FindSDShapes(const TopTools_ListOfShape& aLE,
                                   const Standard_Real aTol,
                                   TopTools_IndexedDataMapOfShapeListOfShape& aMEE,
                                   const Handle(IntTools_Context)& aCtx) ;
  Standard_EXPORT
     Standard_Integer FindSDShapes(const TopoDS_Shape& aE1,
                                   const TopTools_ListOfShape& aLE,
                                   const Standard_Real aTol,
                                   TopTools_ListOfShape& aLESD,
                                   const Handle(IntTools_Context)& aCtx) ;

  Standard_EXPORT
     void PointOnShape(const TopoDS_Shape& aS,
                       gp_Pnt& aP3D) ;

  Standard_EXPORT
     void PointOnEdge(const TopoDS_Edge& aE,
                      gp_Pnt& aP3D) ;

  Standard_EXPORT
     void PointOnEdge(const TopoDS_Edge& aE,
                      const Standard_Real aT,
                      gp_Pnt& aP3D) ;

  Standard_EXPORT
     void PointOnFace(const TopoDS_Face& aF,
                      gp_Pnt& aP3D) ;

  Standard_EXPORT
     void PointOnFace(const TopoDS_Face& aF,
                      const Standard_Real aU,
                      const Standard_Real aV,
                      gp_Pnt& aP3D) ;

  Standard_EXPORT
     Standard_Boolean ProjectPointOnShape(const gp_Pnt& aP1,
                                          const TopoDS_Shape& aS,
                                          gp_Pnt& aP2,
                                          const Handle(IntTools_Context)& aCtx) ;

  Standard_EXPORT
    Standard_Boolean IsSplitToReverse1 (const TopoDS_Edge& aEF1,
                                        const TopoDS_Edge& aEF2,
                                        const Handle(IntTools_Context)& aCtx) ;
  Standard_EXPORT
    void RefinePCurveForEdgeOnFace(const TopoDS_Edge& aE,
                                   const TopoDS_Face& aF,
                                   const Standard_Real aUMin,
                                   const Standard_Real aUMax);

  Standard_EXPORT
    Standard_Boolean IsUPeriodic(const Handle(Geom_Surface) &aS);
  
  Standard_EXPORT
    void MakeContainer(const TopAbs_ShapeEnum theType,
                       TopoDS_Shape& theC);

  Standard_EXPORT
    Standard_Boolean IsSplitToReverse(const TopoDS_Edge& theSplit,
                                      const TopoDS_Edge& theEdge,
                                      const Handle(IntTools_Context)& theCtx) ;
  
  Standard_EXPORT
    Standard_Boolean IsSplitToReverse  (const TopoDS_Face& theFSp,
                                        const TopoDS_Face& theFSr,
                                        const Handle(IntTools_Context)& theCtx) ;
  
  Standard_EXPORT
    Standard_Boolean IsSplitToReverse  (const TopoDS_Shape& theSp,
                                        const TopoDS_Shape& theSr,
                                        const Handle(IntTools_Context)& theCtx) ;
  Standard_EXPORT
    Standard_Integer BuildPCurveForEdgeOnFace  (const TopoDS_Edge& aEold,
                                                const TopoDS_Edge& aEnew,
                                                const TopoDS_Face& aF,
                                                const Handle(IntTools_Context)& aCtx) ;

  //
  Standard_EXPORT
    void FindChains(const GEOMAlgo_ListOfCoupleOfShapes& aLCS,
                    GEOMAlgo_IndexedDataMapOfShapeIndexedMapOfShape& aMapChains);

  Standard_EXPORT
    void FindChains(const GEOMAlgo_IndexedDataMapOfShapeIndexedMapOfShape& aMCV,
                    GEOMAlgo_IndexedDataMapOfShapeIndexedMapOfShape& aMapChains);

  Standard_EXPORT
     void CopyShape(const TopoDS_Shape& aS,
                    TopoDS_Shape& aSC) ;
  
  Standard_EXPORT
     void CopyShape(const TopoDS_Shape& aS,
                    TopoDS_Shape& aSC,
                    TopTools_IndexedDataMapOfShapeShape& aMSS) ;
  
  Standard_EXPORT
     Standard_Integer FindRegion(const TopoDS_Shape& theShape,
                                 const TopoDS_Face& theStartFace,
                                 const TopTools_ListOfShape& theBoundaryEdges,
                                 TopTools_ListOfShape& theRegionFaces);

  Standard_EXPORT
     Standard_Integer FindRegion(const TopoDS_Shape& theShape,
                                 const TopoDS_Face& theStartFace,
                                 const TopoDS_Wire& theBoundary,
                                 TopTools_ListOfShape& theRegionFaces);

  /*!
   * \brief Get default deflection coefficient used for triangulation
   * \return default deflection value
   */
  Standard_EXPORT
     double DefaultDeflection();

  /*!
   * \brief Generate triangulation for \a theShape.
   *
   * \param theShape shape to be meshed.
   * \param theDeflection deflection coefficient to be used.
   * \param theForced if \c true, causes generation of mesh regardless it is already present in the shape.
   * \param theAngleDeflection angular deflection coefficient to be used.
   * \param isRelative if true, \a theDeflection is considered relative to \a theShape maximum axial dimension.
   * \param doPostCheck if true, check mesh generation result and return corresponding boolean value.
   * \retval bool Returns false in the following cases:
   *              1. The shape has neither faces nor edges, i.e. impossible to build triangulation or polygon.
   *              2. \a theForced is false and \a theShape has no mesh or has incomplete mesh.
   *              3. \a doPostCheck is true and mesh generation failed or produced an incomplete mesh.
   */
  Standard_EXPORT
     bool MeshShape(const TopoDS_Shape theShape,
                    const double theDeflection = DefaultDeflection(),
                    const bool theForced = true,
                    const double theAngleDeflection = 0.5,
                    const bool isRelative = true,
                    const bool doPostCheck = false);

  /*!
   * \brief Adjust tolerances of all edges of \a theShape so that
   *        the shape become valid if we check it with "exact" option.
   *
   *        Raise edge tolerances to maximum Curve On Surface distance.
   *
   *        Each edge of \a theShape is considered relative to
   *        all faces of \a theShape to which it belongs.
   *        Maximum deviation of its curve to all surfaces is MAXDEV.
   *        If current edge tolerance is below MAXDEV, it will be set to MAXDEV.
   *        If current edge tolerance exceeds MAXDEV, it will not be changed.
   * \retval bool Returns true if it changed tolerance of at least one edge.
   *
   * \param theShape shape to be modified.
   */
  Standard_EXPORT
    bool FixCurveOnSurfaceTolerances(const TopoDS_Shape& theShape);
};
#endif
