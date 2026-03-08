# Copyright (c) 2024, Trail of Bits, Inc. This source code is licensed
# in accordance with the terms specified in the LICENSE file found
# in the root directory of this source tree.

add_library(patchestry_ast STATIC
  ASTConsumer.cpp
  ASTPassManager.cpp
  FunctionBuilder.cpp
  IntrinsicHandlers.cpp
  ASTNormalizationPipeline.cpp
  NormalizationPipelineHelpers.cpp
  NormalizationCfgPasses.cpp
  NormalizationConditionalPasses.cpp
  NormalizationLoopPasses.cpp
  NormalizationSwitchPasses.cpp
  NormalizationCleanupPasses.cpp
  OperationBuilder.cpp
  OperationStmt.cpp
  TypeBuilder.cpp
  Utils.cpp
  SNode.cpp
  SNodeDebug.cpp
  SNodePassManager.cpp
  CfgBuilder.cpp
  SNodeBuilder.cpp
  DomTree.cpp
  LoopInfo.cpp
  StructuralPasses.cpp
  ClangEmitter.cpp
  CollapseStructure.cpp
)

add_library(patchestry::ast ALIAS patchestry_ast)

target_link_libraries(patchestry_ast
  PUBLIC
    clangFrontend
  PRIVATE
    LLVMSupport
    patchestry_settings
    glog::glog
)

target_include_directories(patchestry_ast
  PRIVATE
    "$<BUILD_INTERFACE:${RELLIC_INCLUDE_DIRS}>"
)
