# Copyright (C) 2012-2026  CEA, EDF, OPEN CASCADE
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

SALOME_CONFIGURE_FILE(VERSION.in VERSION INSTALL ${RUN_SALOME_SCRIPTS})
SALOME_CONFIGURE_FILE(salomeContextUtils.py.in salomeContextUtils.py)
SALOME_CONFIGURE_FILE(bashrc .bashrc)

# ===============================================================
# Files to be installed
# ===============================================================

# These files are executable scripts
SET(SCRIPTS
  addToKillList.py
  killSalome.py
  killSalomeWithPort.py
  launchConfigureParser.py
  NSparam.py
  ORBConfigFile.py
  parseConfigFile.py
  PortManager.py
  runSalomeCommon.py
  runSalomeOnDemand.py
  runSession.py
  runRemote.py
  runTests.py
  ${CMAKE_CURRENT_BINARY_DIR}/salomeContextUtils.py
  salomeContext.py
  salome_utils.py
  searchFreePort.py
  server.py
  setenv.py
  python_test_driver.py
  ## For salome test with session
  #salome_test_driver.py
  )

CONFIGURE_FILE(test/CTestTestfile.cmake.in test/CTestTestfile.cmake @ONLY)

INSTALL(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/test
        DESTINATION ${RUN_SALOME_SCRIPTS})
INSTALL(PROGRAMS ${SCRIPTS} 
        DESTINATION ${RUN_SALOME_SCRIPTS})
INSTALL(FILES ${CMAKE_CURRENT_BINARY_DIR}/.bashrc DESTINATION ${ABS_APPLI_PATH})

IF( (NOT DEFINED $ENV{PYTHON_ROOT_DIR}) AND (NOT DEFINED $ENV{PYTHON_INSTALL_DIR}) AND (NOT DEFINED $ENV{PYTHONHOME}) )
  INSTALL(CODE "
  EXECUTE_PROCESS(COMMAND ln -s /usr/bin/python3 python WORKING_DIRECTORY ${CMAKE_INSTALL_PREFIX}/${RUN_SALOME_SCRIPTS} )
  ")
ENDIF()
