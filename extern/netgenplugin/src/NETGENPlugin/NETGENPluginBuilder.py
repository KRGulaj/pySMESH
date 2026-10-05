# Copyright (C) 2007-2026  CEA, EDF, OPEN CASCADE
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
#
# See http://www.salome-platform.org/ or email : webmaster.salome@opencascade.com
#

##
# @package NETGENPluginBuilder
# Python API for the NETGEN meshing plug-in module.

from salome.smesh.smesh_algorithm import Mesh_Algorithm
from salome.smesh.smeshBuilder import AssureGeomPublished, ParseParameters, IsEqual

# import NETGENPlugin module if possible
noNETGENPlugin = 0
try:
    import NETGENPlugin
except ImportError:
    noNETGENPlugin = 1
    pass

LIBRARY = "libNETGENEngine.so"

NETGEN_VERSION_MAJOR = NETGENPlugin.NETGEN_VERSION_MAJOR

#----------------------------
# Mesh algo type identifiers
#----------------------------

## Algorithm type: Netgen tetrahedron 3D algorithm, see NETGEN_3D_Algorithm
NETGEN_3D     = "NETGEN_3D"
NETGEN_3D_Remote     = "NETGEN_3D_Remote"
## Algorithm type: Netgen tetrahedron 1D-2D-3D algorithm, see NETGEN_1D2D3D_Algorithm
NETGEN_1D2D3D = "NETGEN_2D3D"
## Algorithm type: Netgen triangle 1D-2D algorithm, see NETGEN_1D2D_Algorithm
NETGEN_1D2D   = "NETGEN_2D"
## Algorithm type: Netgen triangle 2D algorithm, see NETGEN_2D_Only_Algorithm
NETGEN_2D     = "NETGEN_2D_ONLY"
NETGEN_2D_Remote = "NETGEN_2D_Remote"
## Algorithm type: Synonim of NETGEN_1D2D3D, see NETGEN_1D2D3D_Algorithm
NETGEN_FULL   = NETGEN_1D2D3D
## Algorithm type: Synonim of NETGEN_3D, see NETGEN_3D_Algorithm
NETGEN        = NETGEN_3D
## Algorithm type: Synonim of NETGEN_1D2D3D, see NETGEN_1D2D3D_Algorithm
FULL_NETGEN   = NETGEN_FULL

#----------------------------
# Hypothesis type enumeration
#----------------------------

## Hypothesis type enumeration: complex hypothesis
#  (full set of parameters can be specified),
#  see NETGEN_Algorithm.Parameters()
SOLE   = 0
## Hypothesis type enumeration: simple hypothesis
#  (only major parameters are specified),
#  see NETGEN_Algorithm.Parameters()
SIMPLE = 1

#----------------------
# Fineness enumeration
#----------------------

## Fineness enumeration: very coarse quality of mesh,
#  see NETGEN_Algorithm.SetFineness()
VeryCoarse = 0
## Fineness enumeration: coarse quality of mesh,
#  see NETGEN_Algorithm.SetFineness()
Coarse     = 1
## Fineness enumeration: moderate quality of mesh,
#  see NETGEN_Algorithm.SetFineness()
Moderate   = 2
## Fineness enumeration: fine quality of mesh,
#  see NETGEN_Algorithm.SetFineness()
Fine       = 3
## Fineness enumeration: very fine quality of mesh,
#  see NETGEN_Algorithm.SetFineness()
VeryFine   = 4
## Fineness enumeration: custom quality of mesh specified by other parameters),
#  see NETGEN_Algorithm.SetFineness()
Custom     = 5

#----------------------
# Algorithms
#----------------------

## Base of all NETGEN algorithms.
#
#  This class provides common methods for all algorithms implemented by NETGEN plugin.
#  @note This class must not be instantiated directly.
class NETGEN_Algorithm(Mesh_Algorithm):

    ## Private constructor
    #  @param mesh parent mesh object algorithm is assigned to
    #  @param geom geometry (shape/sub-shape) algorithm is assigned to;
    #              if it is @c 0 (default), the algorithm is assigned to the main shape
    def __init__(self, mesh, geom=0):
        Mesh_Algorithm.__init__(self)
        if noNETGENPlugin: print("Warning: NETGENPlugin module unavailable")
        if not mesh.GetMesh().HasShapeToMesh() and \
           self.meshMethod == "Triangle": # create a 2D remesher
            self.Create(mesh, geom, "NETGEN_Remesher_2D", LIBRARY)
        else:
            self.Create(mesh, geom, self.algoType, LIBRARY)
        self.params = None
        pass

    ## Sets @c MaxSize parameter
    #  @param theSize new value of the @c MaxSize parameter
    def SetMaxSize(self, theSize):
        if self.Parameters(): self.params.SetMaxSize(theSize)
        pass

    ## Sets @c MinSize parameter
    #  @param theSize new value of the @c MinSize parameter
    def SetMinSize(self, theSize):
        if self.Parameters(): self.params.SetMinSize(theSize)
        pass

    ## Sets @c Optimize flag
    #  @param theVal new value of the @c Optimize parameter
    def SetOptimize(self, theVal):
        if self.Parameters(): self.params.SetOptimize(theVal)
        pass

    ## Sets @c Fineness parameter
    #  @param theFineness new value of the @c Fineness parameter; it can be:
    #  @ref VeryCoarse, @ref Coarse, @ref Moderate, @ref Fine, @ref VeryFine or @ref Custom
    def SetFineness(self, theFineness):
        if self.Parameters(): self.params.SetFineness(theFineness)
        pass

    ## Sets @c GrowthRate parameter
    #  @param theRate new value of the @c GrowthRate parameter
    def SetGrowthRate(self, theRate):
        if self.Parameters(): self.params.SetGrowthRate(theRate)
        pass

    ## Sets @c NbThreads parameter
    #  @param theRate new value of the @c NbThreads parameter
    def SetNbThreads(self, theNumber):
        if self.Parameters(): self.params.SetNbThreads(theNumber)
        pass

    ## Creates meshing hypothesis according to the chosen algorithm type
    #  and initializes it with default parameters
    #  @param which hypothesis type; can be either @ref SOLE (default) or @ref SIMPLE
    #  @return hypothesis object
    def Parameters(self, which=SOLE):
        if self.algoType == NETGEN_1D2D:
            if which == SIMPLE:
                hypType = "NETGEN_SimpleParameters_2D"
            else:
                hypType = "NETGEN_Parameters_2D"
        elif self.algoType == NETGEN_1D2D3D:
            if which == SIMPLE:
                hypType = "NETGEN_SimpleParameters_3D"
            else:
                hypType = "NETGEN_Parameters"
        elif self.algoType in [NETGEN_2D, NETGEN_2D_Remote]:
            hypType = "NETGEN_Parameters_2D_ONLY"
        else:
            hypType = "NETGEN_Parameters_3D"

        if self.algo.GetName() == "NETGEN_Remesher_2D":
            hypType = "NETGEN_RemesherParameters_2D"

        if self.params and self.params.GetName() != hypType:
            self.mesh.RemoveHypothesis( self.params, self.geom )
            self.params = None
        if not self.params:
            self.params = self.Hypothesis(hypType, [], LIBRARY, UseExisting=0)

        return self.params

    ## Defines a file specifying size of elements at points and lines
    #  @param file name of the file
    def SetMeshSizeFile(self, file):
        self.Parameters().SetMeshSizeFile(file)
        pass

    ## Set size of elements on a shape
    #  @param shape - geometry
    #  @param size - element size
    def SetLocalSizeOnShape(self, shape, size ):
        self.Parameters().SetLocalSizeOnShape(shape, size)
        pass


    pass # end of NETGEN_Algorithm class


## Tetrahedron 1D-2D-3D algorithm.
#
#  It can be created by calling smeshBuilder.Mesh.Tetrahedron( smeshBuilder.NETGEN_1D2D3D, geom=0 ).
#  This algorithm generates all 1D (edges), 2D (faces) and 3D (volumes) elements
#  for given geometrical shape.
class NETGEN_1D2D3D_Algorithm(NETGEN_Algorithm):

    ## name of the dynamic method in smeshBuilder.Mesh class
    #  @internal
    meshMethod = "Tetrahedron"
    ## type of algorithm used with helper function in smeshBuilder.Mesh class
    #  @internal
    algoType   = NETGEN_1D2D3D
    ## doc string of the method
    #  @internal
    docHelper  = "Creates tetrahedron 3D algorithm for solids"

    ## Private constructor.
    #  @param mesh parent mesh object algorithm is assigned to
    #  @param geom geometry (shape/sub-shape) algorithm is assigned to;
    #              if it is @c 0 (default), the algorithm is assigned to the main shape
    def __init__(self, mesh, geom=0):
        NETGEN_Algorithm.__init__(self, mesh, geom)
        pass

    ## Sets @c SecondOrder flag
    #  @param theVal new value of the @c SecondOrder parameter
    def SetSecondOrder(self, theVal):
        if self.Parameters(): self.params.SetSecondOrder(theVal)
        pass

    ## Sets @c NbSegPerEdge parameter
    #  @param theVal new value of the @c NbSegPerEdge parameter
    def SetNbSegPerEdge(self, theVal):
        if self.Parameters(): self.params.SetNbSegPerEdge(theVal)
        pass

    ## Sets @c NbSegPerRadius parameter
    #  @param theVal new value of the @c NbSegPerRadius parameter
    def SetNbSegPerRadius(self, theVal):
        if self.Parameters(): self.params.SetNbSegPerRadius(theVal)
        pass

    ## Sets @c ChordalError parameter
    #  @param theVal new value of the @c ChordalError parameter
    def SetChordalError(self, theVal):
        if self.Parameters():
            self.params.SetChordalError(theVal)
            self.params.SetChordalErrorEnabled( theVal > 0 )
        pass

    ## Sets @c RidgeAngle parameter
    #  @param theVal new value of the @c RidgeAngle parameter
    def SetRidgeAngle(self, theVal):
        if self.Parameters():
            self.params.SetRidgeAngle(theVal)
        pass

    ## Sets @c QuadAllowed flag
    #  @param toAllow new value of the @c QuadAllowed parameter (@c True by default)
    def SetQuadAllowed(self, toAllow=True):
        if self.Parameters(): self.params.SetQuadAllowed(toAllow)
        pass
    ## Sets @c UseSurfaceCurvature flag
    #  @param toUse new value of the @c UseSurfaceCurvature parameter (@c True by default)
    def SetUseSurfaceCurvature(self, toUse=True):
        if self.Parameters(): self.params.SetUseSurfaceCurvature(toUse)
        pass
    ## Sets @c FuseEdges flag
    #  @param toFuse new value of the @c FuseEdges parameter (@c False by default)
    def SetFuseEdges(self, toFuse=False):
        if self.Parameters(): self.params.SetFuseEdges(toFuse)
        pass

    ## Sets number of segments overriding the value set by SetLocalLength()
    #  @param theVal new value of number of segments parameter
    def SetNumberOfSegments(self, theVal):
        self.Parameters(SIMPLE).SetNumberOfSegments(theVal)
        pass

    ## Sets number of segments overriding the value set by SetNumberOfSegments()
    #  @param theVal new value of local length parameter
    def SetLocalLength(self, theVal):
        self.Parameters(SIMPLE).SetLocalLength(theVal)
        pass

    ## Defines @c MaxElementArea parameter of @c NETGEN_SimpleParameters_3D hypothesis.
    #  Overrides value set by LengthFromEdges()
    #  @param area new value of @c MaxElementArea parameter
    def MaxElementArea(self, area):
        self.Parameters(SIMPLE).SetMaxElementArea(area)
        pass

    ## Defines @c LengthFromEdges parameter of @c NETGEN_SimpleParameters_3D hypothesis.
    #  Overrides value set by MaxElementArea()
    def LengthFromEdges(self):
        self.Parameters(SIMPLE).LengthFromEdges()
        pass

    ## Defines @c LengthFromFaces parameter of @c NETGEN_SimpleParameters_3D hypothesis.
    #  Overrides value set by MaxElementVolume()
    def LengthFromFaces(self):
        self.Parameters(SIMPLE).LengthFromFaces()
        pass

    ## Defines @c MaxElementVolume parameter of @c NETGEN_SimpleParameters_3D hypothesis.
    #  Overrides value set by LengthFromFaces()
    #  @param vol new value of @c MaxElementVolume parameter
    def MaxElementVolume(self, vol):
        self.Parameters(SIMPLE).SetMaxElementVolume(vol)
        pass

    pass # end of NETGEN_1D2D3D_Algorithm class


## Triangle NETGEN 1D-2D algorithm.
#
#  It can be created by calling smeshBuilder.Mesh.Triangle( smeshBuilder.NETGEN_1D2D, geom=0 )
#
#  This algorithm generates 1D (edges) and 2D (faces) elements
#  for given geometrical shape.
class NETGEN_1D2D_Algorithm(NETGEN_1D2D3D_Algorithm):

    ## name of the dynamic method in smeshBuilder.Mesh class
    #  @internal
    meshMethod = "Triangle"
    ## type of algorithm used with helper function in smeshBuilder.Mesh class
    #  @internal
    algoType   = NETGEN_1D2D
    ## doc string of the method
    #  @internal
    docHelper  = "Creates triangle 2D algorithm for faces"


    ## Private constructor.
    #  @param mesh parent mesh object algorithm is assigned to
    #  @param geom geometry (shape/sub-shape) algorithm is assigned to;
    #              if it is @c 0 (default), the algorithm is assigned to the main shape
    def __init__(self, mesh, geom=0):
        NETGEN_1D2D3D_Algorithm.__init__(self, mesh, geom)
        pass

    pass # end of NETGEN_1D2D_Algorithm class


## Triangle NETGEN 2D algorithm
#
#  It can be created by calling smeshBuilder.Mesh.Triangle( smeshBuilder.NETGEN_2D, geom=0 )
#
#  This algorithm generates only 2D (faces) elements for given geometrical shape
#  and, in contrast to NETGEN_1D2D_Algorithm class, should be used in conjunction
#  with other 1D meshing algorithm.
class NETGEN_2D_Only_Algorithm(NETGEN_Algorithm):

    ## name of the dynamic method in smeshBuilder.Mesh class
    #  @internal
    meshMethod = "Triangle"
    ## type of algorithm used with helper function in smeshBuilder.Mesh class
    #  @internal
    algoType = NETGEN_2D
    ## flag pointing whether this algorithm should be used by default in dynamic method
    #  of smeshBuilder.Mesh class
    isDefault  = True
    ## doc string of the method
    #  @internal
    docHelper  = "Creates triangle 2D algorithm for faces"

    isDefault = True

    ## Private constructor.
    #  @param mesh parent mesh object algorithm is assigned to
    #  @param geom geometry (shape/sub-shape) algorithm is assigned to;
    #              if it is @c 0 (default), the algorithm is assigned to the main shape
    def __init__(self, mesh, geom=0):
        NETGEN_Algorithm.__init__(self, mesh, geom)
        pass

    ## Defines @c MaxElementArea parameter of hypothesis basing on the definition of the
    #  maximum area of each triangle
    #  @param area maximum area value of each triangle
    #  @param UseExisting if \c True - searches for an existing hypothesis created with the
    #                     same parameters, else (default) - creates a new one
    #  @return hypothesis object
    def MaxElementArea(self, area, UseExisting=0):
        compFun = lambda hyp, args: IsEqual(hyp.GetMaxElementArea(), args[0])
        hyp = self.Hypothesis("MaxElementArea", [area], UseExisting=UseExisting,
                              CompareMethod=compFun)
        hyp.SetMaxElementArea(area)
        return hyp

    ## Defines @c LengthFromEdges hypothesis to build triangles
    #  based on the length of the edges taken from the wire
    #  @return hypothesis object
    def LengthFromEdges(self):
        hyp = self.Hypothesis("LengthFromEdges", UseExisting=1, CompareMethod=self.CompareEqualHyp)
        return hyp

    ## Sets @c UseSurfaceCurvature flag
    #  @param toUse new value of the @c UseSurfaceCurvature parameter (@c True by default)
    def SetUseSurfaceCurvature(self, toUse=True):
        if self.Parameters(): self.params.SetUseSurfaceCurvature(toUse)
        pass

    ## Sets @c QuadAllowed flag.
    #  @param toAllow new value of the @c QuadAllowed parameter (@c True by default)
    #  @return hypothesis object
    def SetQuadAllowed(self, toAllow=True):
        if not self.params:
            # use simple hyps
            hasSimpleHyps = False
            simpleHyps = ["QuadranglePreference","LengthFromEdges","MaxElementArea"]
            for hyp in self.mesh.GetHypothesisList( self.geom ):
                if hyp.GetName() in simpleHyps:
                    hasSimpleHyps = True
                    if hyp.GetName() == "QuadranglePreference":
                        if not toAllow: # remove QuadranglePreference
                            self.mesh.RemoveHypothesis( self.geom, hyp )
                        else:
                            return hyp
                        return None
                    pass
                pass
            if hasSimpleHyps:
                if toAllow: # add QuadranglePreference
                    return self.Hypothesis("QuadranglePreference", UseExisting=1, CompareMethod=self.CompareEqualHyp)
                return None
            pass
        self.Parameters().SetQuadAllowed( toAllow )
        return self.params

    pass # end of NETGEN_2D_Only_Algorithm class


## Tetrahedron 3D algorithm
#
#  It can be created by calling smeshBuilder.Mesh.Tetrahedron() or smeshBuilder.Mesh.Tetrahedron( smeshBuilder.NETGEN, geom=0 )
#
#  This algorithm generates only 3D (volumes) elements for given geometrical shape
#  and, in contrast to NETGEN_1D2D3D_Algorithm class, should be used in conjunction
#  with other 1D and 2D meshing algorithms.
class NETGEN_3D_Algorithm(NETGEN_Algorithm):

    ## name of the dynamic method in smeshBuilder.Mesh class
    #  @internal
    meshMethod = "Tetrahedron"
    ## type of algorithm used with helper function in smeshBuilder.Mesh class
    #  @internal
    algoType   = NETGEN
    ## flag pointing either this algorithm should be used by default in dynamic method
    #  of smeshBuilder.Mesh class
    #  @internal
    isDefault  = True
    ## doc string of the method
    #  @internal
    docHelper  = "Creates tetrahedron 3D algorithm for solids"

    ## Private constructor.
    #  @param mesh parent mesh object algorithm is assigned to
    #  @param geom geometry (shape/sub-shape) algorithm is assigned to;
    #              if it is @c 0 (default), the algorithm is assigned to the main shape
    def __init__(self, mesh, geom=0):
        NETGEN_Algorithm.__init__(self, mesh, geom)
        pass

    ## Defines @c MaxElementVolume hypothesis to specify the maximum volume value of each tetrahedron
    #  @param vol maximum volume value of each tetrahedron
    #  @param UseExisting if \c True - searches for the existing hypothesis created with
    #                   the same parameters, else (default) - creates a new one
    #  @return hypothesis object
    def MaxElementVolume(self, vol, UseExisting=0):
        compFun = lambda hyp, args: IsEqual(hyp.GetMaxElementVolume(), args[0])
        hyp = self.Hypothesis("MaxElementVolume", [vol], UseExisting=UseExisting,
                              CompareMethod=compFun)
        hyp.SetMaxElementVolume(vol)
        return hyp

    pass # end of NETGEN_3D_Algorithm class

## Tetrahedron 3D algorithm
#
#  It can be created by calling smeshBuilder.Mesh.Tetrahedron() or smeshBuilder.Mesh.Tetrahedron( smeshBuilder.NETGEN, geom=0 )
#
#  This algorithm generates only 3D (volumes) elements for given geometrical shape
#  and, in contrast to NETGEN_1D2D3D_Algorithm class, should be used in conjunction
#  with other 1D and 2D meshing algorithms.
class NETGEN_3D_Remote_Algorithm(NETGEN_3D_Algorithm):

    ## type of algorithm used with helper function in smeshBuilder.Mesh class
    #  @internal
    algoType   = NETGEN_3D_Remote
    ## flag pointing either this algorithm should be used by default in dynamic method
    #  of smeshBuilder.Mesh class
    #  @internal
    isDefault  = False
    ## doc string of the method
    #  @internal
    docHelper  = "Remotely Creates tetrahedron 3D algorithm for solids"

    ## Private constructor.
    #  @param mesh parent mesh object algorithm is assigned to
    #  @param geom geometry (shape/sub-shape) algorithm is assigned to;
    #              if it is @c 0 (default), the algorithm is assigned to the main shape
    def __init__(self, mesh, geom=0):
        NETGEN_3D_Algorithm.__init__(self, mesh, geom)
        pass

    pass # end of NETGEN_3D_Remote_Algorithm class

## Tetrahedron 2D algorithm
#
#  It can be created by calling smeshBuilder.Mesh.Triangle() or smeshBuilder.Mesh.Triangle( smeshBuilder.NETGEN, geom=0 )
#
class NETGEN_2D_Remote_Algorithm(NETGEN_2D_Only_Algorithm):

    ## type of algorithm used with helper function in smeshBuilder.Mesh class
    #  @internal
    algoType   = NETGEN_2D_Remote
    ## flag pointing either this algorithm should be used by default in dynamic method
    #  of smeshBuilder.Mesh class
    #  @internal
    isDefault  = False
    ## doc string of the method
    #  @internal
    docHelper  = "Remotely Creates triangles in face of solids"

    ## Private constructor.
    #  @param mesh parent mesh object algorithm is assigned to
    #  @param geom geometry (shape/sub-shape) algorithm is assigned to;
    #              if it is @c 0 (default), the algorithm is assigned to the main shape
    def __init__(self, mesh, geom=0):
        self.algoType = NETGEN_2D_Remote
        NETGEN_2D_Only_Algorithm.__init__(self, mesh, geom)
        pass

    pass # end of NETGEN_2D_Remote_Algorithm class


## Triangle (helper) 1D-2D algorithm
#
#  This is the helper class that is used just to allow creating of create NETGEN_1D2D algorithm
#  by calling smeshBuilder.Mesh.Triangle( smeshBuilder.NETGEN, geom=0 ); this is required for backward compatibility
#  with old Python scripts.
#
#  @note This class (and corresponding smeshBuilder.Mesh function) is obsolete;
#  use smeshBuilder.Mesh.Triangle( smeshBuilder.NETGEN_1D2D, geom=0 ) instead.
class NETGEN_1D2D_Algorithm_2(NETGEN_1D2D_Algorithm):

    ## name of the dynamic method in smeshBuilder.Mesh class
    #  @internal
    algoType = NETGEN

    ## Private constructor.
    #  @param mesh parent mesh object algorithm is assigned to
    #  @param geom geometry (shape/sub-shape) algorithm is assigned to;
    #              if it is @c 0 (default), the algorithm is assigned to the main shape
    def __init__(self, mesh, geom=0):
        self.algoType = NETGEN_1D2D
        NETGEN_1D2D_Algorithm.__init__(self,mesh, geom)
        pass

    pass # end of NETGEN_1D2D_Algorithm_2 class


## Tetrahedron (helper) 1D-2D-3D algorithm.
#
#  This is the helper class that is used just to allow creating of create NETGEN_1D2D3D
#  by calling smeshBuilder.Mesh.Netgen(); this is required for backward compatibility with old Python scripts.
#
#  @note This class (and corresponding smeshBuilder.Mesh function) is obsolete;
#  use smeshBuilder.Mesh.Tetrahedron( smeshBuilder.NETGEN_1D2D3D, geom=0 ) instead.
class NETGEN_1D2D3D_Algorithm_2(NETGEN_1D2D3D_Algorithm):

    ## name of the dynamic method in smeshBuilder.Mesh class
    #  @internal
    meshMethod = "Netgen"
    ## doc string of the method
    #  @internal
    docHelper  = "Deprecated, used only for compatibility! See Tetrahedron() method."

    ## Private constructor.
    #  @param mesh parent mesh object algorithm is assigned to
    #  @param geom geometry (shape/sub-shape) algorithm is assigned to;
    #              if it is @c 0 (default), the algorithm is assigned to the main shape
    def __init__(self, mesh, geom=0):
        NETGEN_1D2D3D_Algorithm.__init__(self,mesh, geom)
        pass

    pass # end of NETGEN_1D2D3D_Algorithm_2 class
