# Copyright (c) 2026, Trail of Bits, Inc.
#
# This source code is licensed in accordance with the terms specified in
# the LICENSE file found in the root directory of this source tree.

add_executable(patchir-dslc
    main.cpp
)

target_link_libraries(patchir-dslc
    PRIVATE
        LLVMSupport
        patchestry_settings
)

if(PATCHESTRY_INSTALL)
  install(TARGETS patchir-dslc
    DESTINATION ${CMAKE_INSTALL_BINDIR}
    COMPONENT patchestry-tools
  )
endif()
