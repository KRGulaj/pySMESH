// SPDX-License-Identifier: LGPL-2.1-only
// Copyright (C) 2026 Kajetan R. Gulaj
// Created: 2026-10-21

// pySMESH shim for FreeCAD's src/boost_graph_adjacency_list.hpp, which PlaneGCS's GCS.cpp
// includes.
//
// FreeCAD's header includes <boost/graph/adjacency_list.hpp> with
// BOOST_ALLOW_DEPRECATED_HEADERS defined, because that Boost header includes headers that
// Boost marks as deprecated (since Boost 1.75), and Boost then prints a message for each.
// This shim does the same. The Boost.Graph headers are those of the build environment's
// Boost, as for every other Boost header that pySMESH compiles against.

#ifndef PYSMESH_PLANEGCS_BOOST_GRAPH_ADJACENCY_LIST_HPP
#define PYSMESH_PLANEGCS_BOOST_GRAPH_ADJACENCY_LIST_HPP

#define BOOST_ALLOW_DEPRECATED_HEADERS
#include <boost/graph/adjacency_list.hpp>
#undef BOOST_ALLOW_DEPRECATED_HEADERS

#endif  // PYSMESH_PLANEGCS_BOOST_GRAPH_ADJACENCY_LIST_HPP
