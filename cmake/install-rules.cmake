install(
    TARGETS patchestry_exe
    RUNTIME COMPONENT patchestry_Runtime
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
