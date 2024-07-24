function(add_patchestry_doc doc_filename output_file output_directory command)
  set(VAST_TARGET_DEFINITIONS ${doc_filename}.td)
  mlir_tablegen(${output_file}.md ${command} ${ARGN})
  set(GEN_DOC_FILE ${PATCHESTRY_BINARY_DIR}/docs/${output_directory}${output_file}.md)
  add_custom_command(
          OUTPUT ${GEN_DOC_FILE}
          COMMAND ${CMAKE_COMMAND} -E copy
                  ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md
                  ${GEN_DOC_FILE}
          DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/${output_file}.md)
  add_custom_target(${output_file}DocGen DEPENDS ${GEN_DOC_FILE})
  add_dependencies(patchestry-doc ${output_file}DocGen)
endfunction(add_patchestry_doc)

function(add_patchestry_dialect_with_doc dialect dialect_namespace)
    add_mlir_dialect(${dialect} ${dialect_namespace})
    add_patchestry_doc(${dialect} ${dialect}Dialect Dialects/ -gen-dialect-doc -dialect=${dialect_namespace})
endfunction(add_patchestry_dialect_with_doc)