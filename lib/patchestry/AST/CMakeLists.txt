# Copyright (c) 2024, Trail of Bits, Inc. This source code is licensed
# in accordance with the terms specified in the LICENSE file found
# in the root directory of this source tree.

add_library(patchestry_ast STATIC
  ASTConsumer.cpp
  PcodeLifter.cpp
  FunctionBuilder.cpp
  IntrinsicHandlers.cpp
  OperationBuilder.cpp
  OperationStmt.cpp
  TypeBuilder.cpp
  Utils.cpp
  SNode.cpp
  SNodeDebug.cpp
  ClangEmitter.cpp
  ClangEmitterCleanup.cpp
  ClangEmitterPostPasses.cpp
  CfgDotEmitter.cpp
  CGraph.cpp
  CGraphBuilder.cpp
  SNodePostPasses.cpp
  BuildSNodeFromRegion.cpp
  PcodeValidator.cpp
  CSourceUnit.cpp
)

add_library(patchestry::ast ALIAS patchestry_ast)

target_link_libraries(patchestry_ast
  PUBLIC
    clangFrontend
  PRIVATE
    LLVMSupport
    patchestry_ghidra
    patchestry_frontend
    clangAnalysis
    patchestry_settings
    glog::glog
)
