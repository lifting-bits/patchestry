# Parent project does not export its library target, so this CML implicitly
# depends on being added from it, i.e. the testing is done only from the build
# tree and is not feasible from an install location

project(patchestryTests LANGUAGES CXX)

# ---- Tests ----

add_executable(patchestry_test source/ghidra.cpp)
target_link_libraries(patchestry_test PRIVATE PatchestryGhidra)
target_compile_features(patchestry_test PRIVATE cxx_std_20)

add_test(NAME patchestry_test COMMAND patchestry_test)

# ---- End-of-file commands ----

add_folders(Test)
