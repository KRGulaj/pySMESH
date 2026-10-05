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

//  NETGENPlugin : C++ implementation
// File      : NETGENPlugin_Hypothesis_2D_i.cxx
// Author    : Michael Sazonov (OCN)
// Date      : 03/04/2006
// Project   : SALOME
//=============================================================================
//
#include "NETGENPlugin_Hypothesis_2D_i.hxx"

#include <SMESH_Gen.hxx>
#include <SMESH_Gen_i.hxx>
#include <SMESH_Group_i.hxx>
#include <SMESH_Group.hxx>
#include <SMESH_PythonDump.hxx>

//=============================================================================
/*!
 *  NETGENPlugin_Hypothesis_2D_i::NETGENPlugin_Hypothesis_2D_i
 *
 *  Constructor
 */
//=============================================================================
NETGENPlugin_Hypothesis_2D_i::
NETGENPlugin_Hypothesis_2D_i (PortableServer::POA_ptr thePOA,
                              ::SMESH_Gen*            theGenImpl)
  : SALOME::GenericObj_i( thePOA ),
    SMESH_Hypothesis_i( thePOA ),
    NETGENPlugin_Hypothesis_i( thePOA, theGenImpl )
{
  if (myBaseImpl)
    delete (::NETGENPlugin_Hypothesis*)myBaseImpl;
  myBaseImpl = new ::NETGENPlugin_Hypothesis_2D (theGenImpl->GetANewId(),
                                                 theGenImpl);
}

//=============================================================================
/*!
 *  NETGENPlugin_Hypothesis_2D_i::~NETGENPlugin_Hypothesis_2D_i
 *
 *  Destructor
 */
//=============================================================================
NETGENPlugin_Hypothesis_2D_i::~NETGENPlugin_Hypothesis_2D_i()
{
}

//=============================================================================
/*!
 *  NETGENPlugin_Hypothesis_2D_i::GetImpl
 *
 *  Get implementation
 */
//=============================================================================
::NETGENPlugin_Hypothesis_2D* NETGENPlugin_Hypothesis_2D_i::GetImpl()
{
  return (::NETGENPlugin_Hypothesis_2D*)myBaseImpl;
}

//================================================================================
/*!
 * \brief Verify whether hypothesis supports given entity type 
  * \param type - dimension (see SMESH::Dimension enumeration)
  * \retval CORBA::Boolean - TRUE if dimension is supported, FALSE otherwise
 * 
 * Verify whether hypothesis supports given entity type (see SMESH::Dimension enumeration)
 */
//================================================================================  
CORBA::Boolean NETGENPlugin_Hypothesis_2D_i::IsDimSupported( SMESH::Dimension type )
{
  return type == SMESH::DIM_2D;
}


//=============================================================================
/*!
 *  NETGENPlugin_RemesherHypothesis_2D_i::NETGENPlugin_RemesherHypothesis_2D_i
 *
 *  Constructor
 */
//=============================================================================
NETGENPlugin_RemesherHypothesis_2D_i::
NETGENPlugin_RemesherHypothesis_2D_i (PortableServer::POA_ptr thePOA,
                                      ::SMESH_Gen*            theGenImpl)
  : SALOME::GenericObj_i( thePOA ),
    SMESH_Hypothesis_i( thePOA ),
    NETGENPlugin_Hypothesis_2D_i( thePOA, theGenImpl )
{
  myBaseImpl = new ::NETGENPlugin_RemesherHypothesis_2D (theGenImpl->GetANewId(),
                                                         theGenImpl);
}

//================================================================================
/*!
 * \brief Verify whether hypothesis supports given entity type
 * \param type - dimension (see SMESH::Dimension enumeration)
 * \retval CORBA::Boolean - TRUE if dimension is supported, FALSE otherwise
 *
 * Verify whether hypothesis supports given entity type (see SMESH::Dimension enumeration)
 */
//================================================================================
CORBA::Boolean NETGENPlugin_RemesherHypothesis_2D_i::IsDimSupported( SMESH::Dimension type )
{
  return type == SMESH::DIM_2D;
}

//=============================================================================
/*!
 *  NETGENPlugin_RemesherHypothesis_2D_i::GetImpl
 *
 *  Get implementation
 */
//=============================================================================
::NETGENPlugin_RemesherHypothesis_2D* NETGENPlugin_RemesherHypothesis_2D_i::GetImpl()
{
  return (::NETGENPlugin_RemesherHypothesis_2D*)myBaseImpl;
}

//================================================================================
/*!
 * \brief Set ridge angle
 */
//================================================================================

void NETGENPlugin_RemesherHypothesis_2D_i::SetRidgeAngle( CORBA::Double angle )
{
  if ( GetRidgeAngle() != angle )
  {
    GetImpl()->SetRidgeAngle( angle );

    SMESH::TPythonDump() << _this() << ".SetRidgeAngle( " << SMESH::TVar(angle) << " )";
  }
}

//================================================================================
/*!
 * \brief Return ridge angle
 */
//================================================================================

CORBA::Double NETGENPlugin_RemesherHypothesis_2D_i::GetRidgeAngle()
{
  return GetImpl()->GetRidgeAngle();
}

void NETGENPlugin_RemesherHypothesis_2D_i::SetEdgeCornerAngle(CORBA::Double angle )
{
  if ( GetEdgeCornerAngle() != angle )
  {
    GetImpl()->SetEdgeCornerAngle( angle );

    SMESH::TPythonDump() << _this() << ".SetEdgeCornerAngle( " << SMESH::TVar(angle) << " )";
  }
}

CORBA::Double NETGENPlugin_RemesherHypothesis_2D_i::GetEdgeCornerAngle()
{
  return GetImpl()->GetEdgeCornerAngle();
}

void NETGENPlugin_RemesherHypothesis_2D_i::SetChartAngle(CORBA::Double angle )
{
  if ( GetChartAngle() != angle )
  {
    GetImpl()->SetChartAngle( angle );

    SMESH::TPythonDump() << _this() << ".SetChartAngle( " << SMESH::TVar(angle) << " )";
  }
}

CORBA::Double NETGENPlugin_RemesherHypothesis_2D_i::GetChartAngle()
{
  return GetImpl()->GetChartAngle();
}

void NETGENPlugin_RemesherHypothesis_2D_i::SetOuterChartAngle(CORBA::Double angle )
{
  if ( GetOuterChartAngle() != angle )
  {
    GetImpl()->SetOuterChartAngle( angle );

    SMESH::TPythonDump() << _this() << ".SetOuterChartAngle( " << SMESH::TVar(angle) << " )";
  }
}

CORBA::Double NETGENPlugin_RemesherHypothesis_2D_i::GetOuterChartAngle()
{
  return GetImpl()->GetOuterChartAngle();
}

void NETGENPlugin_RemesherHypothesis_2D_i::SetRestHChartDistFactor(CORBA::Double f )
{
  if ( GetRestHChartDistFactor() != f )
  {
    GetImpl()->SetRestHChartDistFactor( f );

    SMESH::TPythonDump() << _this() << ".SetRestHChartDistFactor( " << SMESH::TVar(f) << " )";
  }
}

CORBA::Double NETGENPlugin_RemesherHypothesis_2D_i::GetRestHChartDistFactor()
{
  return GetImpl()->GetRestHChartDistFactor();
}

void NETGENPlugin_RemesherHypothesis_2D_i::SetRestHChartDistEnable(CORBA::Boolean enable )
{
  if ( GetRestHChartDistEnable() != enable )
  {
    GetImpl()->SetRestHChartDistEnable( enable );

    SMESH::TPythonDump() << _this() << ".SetRestHChartDistEnable( " << enable << " )";
  }
}

CORBA::Boolean NETGENPlugin_RemesherHypothesis_2D_i::GetRestHChartDistEnable()
{
  return GetImpl()->GetRestHChartDistEnable();
}

void NETGENPlugin_RemesherHypothesis_2D_i::SetRestHLineLengthFactor(CORBA::Double f )
{
  if ( GetRestHLineLengthFactor() != f )
  {
    GetImpl()->SetRestHLineLengthFactor( f );

    SMESH::TPythonDump() << _this() << ".SetRestHLineLengthFactor( " << SMESH::TVar(f) << " )";
  }
}

CORBA::Double NETGENPlugin_RemesherHypothesis_2D_i::GetRestHLineLengthFactor()
{
  return GetImpl()->GetRestHLineLengthFactor();
}

void NETGENPlugin_RemesherHypothesis_2D_i::SetRestHLineLengthEnable(CORBA::Boolean enable )
{
  if ( GetRestHLineLengthEnable() != enable )
  {
    GetImpl()->SetRestHLineLengthEnable( enable );

    SMESH::TPythonDump() << _this() << ".SetRestHLineLengthEnable( " << enable << " )";
  }
}

CORBA::Boolean NETGENPlugin_RemesherHypothesis_2D_i::GetRestHLineLengthEnable()
{
  return GetImpl()->GetRestHLineLengthEnable();
}

void NETGENPlugin_RemesherHypothesis_2D_i::SetRestHCloseEdgeFactor(CORBA::Double f )
{
  if ( GetRestHCloseEdgeFactor() != f )
  {
    GetImpl()->SetRestHCloseEdgeFactor( f );

    SMESH::TPythonDump() << _this() << ".SetRestHCloseEdgeFactor( " << SMESH::TVar(f) << " )";
  }
}

CORBA::Double NETGENPlugin_RemesherHypothesis_2D_i::GetRestHCloseEdgeFactor()
{
  return GetImpl()->GetRestHCloseEdgeFactor();
}

void NETGENPlugin_RemesherHypothesis_2D_i::SetRestHCloseEdgeEnable(CORBA::Boolean enable )
{
  if ( GetRestHCloseEdgeEnable() != enable )
  {
    GetImpl()->SetRestHCloseEdgeEnable( enable );

    SMESH::TPythonDump() << _this() << ".SetRestHCloseEdgeEnable( " << enable << " )";
  }
}

CORBA::Boolean NETGENPlugin_RemesherHypothesis_2D_i::GetRestHCloseEdgeEnable()
{
  return GetImpl()->GetRestHCloseEdgeEnable();
}

void NETGENPlugin_RemesherHypothesis_2D_i::SetRestHSurfCurvFactor(CORBA::Double f )
{
  if ( GetRestHSurfCurvFactor() != f )
  {
    GetImpl()->SetRestHSurfCurvFactor( f );

    SMESH::TPythonDump() << _this() << ".SetRestHSurfCurvFactor( " << SMESH::TVar(f) << " )";
  }
}

CORBA::Double NETGENPlugin_RemesherHypothesis_2D_i::GetRestHSurfCurvFactor()
{
  return GetImpl()->GetRestHSurfCurvFactor();
}

void NETGENPlugin_RemesherHypothesis_2D_i::SetRestHSurfCurvEnable(CORBA::Boolean enable )
{
  if ( GetRestHSurfCurvEnable() != enable )
  {
    GetImpl()->SetRestHSurfCurvEnable( enable );

    SMESH::TPythonDump() << _this() << ".SetRestHSurfCurvEnable( " << enable << " )";
  }
}

CORBA::Boolean NETGENPlugin_RemesherHypothesis_2D_i::GetRestHSurfCurvEnable()
{
  return GetImpl()->GetRestHSurfCurvEnable();
}

void NETGENPlugin_RemesherHypothesis_2D_i::SetRestHEdgeAngleFactor(CORBA::Double f )
{
  if ( GetRestHEdgeAngleFactor() != f )
  {
    GetImpl()->SetRestHEdgeAngleFactor( f );

    SMESH::TPythonDump() << _this() << ".SetRestHEdgeAngleFactor( " << SMESH::TVar(f) << " )";
  }
}

CORBA::Double NETGENPlugin_RemesherHypothesis_2D_i::GetRestHEdgeAngleFactor()
{
  return GetImpl()->GetRestHEdgeAngleFactor();
}

void NETGENPlugin_RemesherHypothesis_2D_i::SetRestHEdgeAngleEnable(CORBA::Boolean enable )
{
  if ( GetRestHEdgeAngleEnable() != enable )
  {
    GetImpl()->SetRestHEdgeAngleEnable( enable );

    SMESH::TPythonDump() << _this() << ".SetRestHEdgeAngleEnable( " << enable << " )";
  }
}

CORBA::Boolean NETGENPlugin_RemesherHypothesis_2D_i::GetRestHEdgeAngleEnable()
{
  return GetImpl()->GetRestHEdgeAngleEnable();
}

void NETGENPlugin_RemesherHypothesis_2D_i::SetRestHSurfMeshCurvFactor(CORBA::Double f )
{
  if ( GetRestHSurfMeshCurvFactor() != f )
  {
    GetImpl()->SetRestHSurfMeshCurvFactor( f );

    SMESH::TPythonDump() << _this() << ".SetRestHSurfMeshCurvFactor( " << SMESH::TVar(f) << " )";
  }
}

CORBA::Double NETGENPlugin_RemesherHypothesis_2D_i::GetRestHSurfMeshCurvFactor()
{
  return GetImpl()->GetRestHSurfMeshCurvFactor();
}

void NETGENPlugin_RemesherHypothesis_2D_i::SetRestHSurfMeshCurvEnable(CORBA::Boolean enable )
{
  if ( GetRestHSurfMeshCurvEnable() != enable )
  {
    GetImpl()->SetRestHSurfMeshCurvEnable( enable );

    SMESH::TPythonDump() << _this() << ".SetRestHSurfMeshCurvEnable( " << enable << " )";
  }
}

CORBA::Boolean NETGENPlugin_RemesherHypothesis_2D_i::GetRestHSurfMeshCurvEnable()
{
  return GetImpl()->GetRestHSurfMeshCurvEnable();
}

void NETGENPlugin_RemesherHypothesis_2D_i::SetKeepExistingEdges( CORBA::Boolean toKeep )
{
  if ( GetKeepExistingEdges() != toKeep )
  {
    GetImpl()->SetKeepExistingEdges( toKeep );

    SMESH::TPythonDump() << _this() << ".SetKeepExistingEdges( " << toKeep << " )";
  }
}

CORBA::Boolean NETGENPlugin_RemesherHypothesis_2D_i::GetKeepExistingEdges()
{
  return GetImpl()->GetKeepExistingEdges();
}

void NETGENPlugin_RemesherHypothesis_2D_i::SetMakeGroupsOfSurfaces( CORBA::Boolean toMake )
{
  if ( GetMakeGroupsOfSurfaces() != toMake )
  {
    GetImpl()->SetMakeGroupsOfSurfaces( toMake );

    SMESH::TPythonDump() << _this() << ".SetMakeGroupsOfSurfaces( " << toMake << " )";
  }
}

CORBA::Boolean NETGENPlugin_RemesherHypothesis_2D_i::GetMakeGroupsOfSurfaces()
{
  return GetImpl()->GetMakeGroupsOfSurfaces();
}

void
NETGENPlugin_RemesherHypothesis_2D_i::SetFixedEdgeGroup( SMESH::SMESH_GroupBase_ptr edgeGroup )
{
  const SMESH_Group * group = 0;
  if ( SMESH_GroupBase_i* group_i = SMESH::DownCast< SMESH_GroupBase_i* >( edgeGroup ))
  {
    if ( group_i->GetType() == SMESH::EDGE )
      group = group_i->GetSmeshGroup();
  }

  int id = group ? group->GetID() : -1;
  if ( id != GetImpl()->GetFixedEdgeGroupID() )
  {
    GetImpl()->SetFixedEdgeGroup( group );
    SMESH::TPythonDump() << _this() << ".SetFixedEdgeGroup( " << edgeGroup << " )";
  }
}

SMESH::SMESH_GroupBase_ptr
NETGENPlugin_RemesherHypothesis_2D_i::GetFixedEdgeGroup( SMESH::SMESH_Mesh_ptr mesh )
{
  SMESH::SMESH_GroupBase_var resGroup;
  if ( SMESH_Mesh_i* mesh_i = SMESH::DownCast< SMESH_Mesh_i* >( mesh ))
  {
    const std::map<int, SMESH::SMESH_GroupBase_ptr>& groups = mesh_i->getGroups();
    std::map<int, SMESH::SMESH_GroupBase_ptr>::const_iterator i_gr =
      groups.find( GetImpl()->GetFixedEdgeGroupID() );
    if ( i_gr != groups.end() && i_gr->second->GetType() == SMESH::EDGE )
      resGroup = SMESH::SMESH_GroupBase::_duplicate( i_gr->second );
  }
  return resGroup._retn();
}

void NETGENPlugin_RemesherHypothesis_2D_i::SetLoadMeshOnCancel( CORBA::Boolean toMake )
{
  if ( GetLoadMeshOnCancel() != toMake )
  {
    GetImpl()->SetLoadMeshOnCancel( toMake );

    SMESH::TPythonDump() << _this() << ".SetLoadMeshOnCancel( " << toMake << " )";
  }
}

CORBA::Boolean NETGENPlugin_RemesherHypothesis_2D_i::GetLoadMeshOnCancel()
{
  return GetImpl()->GetLoadMeshOnCancel();
}
