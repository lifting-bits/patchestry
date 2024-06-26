include(CTest)

add_executable(patchestry_test ghidra.cpp)
target_link_libraries(patchestry_test
    PRIVATE
        patchestry_ghidra
        patchestry_settings
)

add_test(NAME patchestry_test COMMAND patchestry_test)
