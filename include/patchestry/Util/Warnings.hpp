/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 * All rights reserved.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#define PATCHESTRY_COMMON_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic ignored \"-Wsign-conversion\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wconversion\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wold-style-cast\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wunused-parameter\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wcast-align\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Woverloaded-virtual\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wctad-maybe-unsupported\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wdouble-promotion\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wshadow\"") \
  _Pragma( "GCC diagnostic ignored \"-Wunused-function\"") \
  _Pragma( "GCC diagnostic ignored \"-Wextra-semi\"")

#define PATCHESTRY_CLANG_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic ignored \"-Wambiguous-reversed-operator\"" )

#define PATCHESTRY_GCC_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic ignored \"-Wuseless-cast\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wnull-dereference\"" ) \
  _Pragma( "GCC diagnostic ignored \"-Wmaybe-uninitialized\"" )

#ifdef __clang__
#define PATCHESTRY_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic push" ) \
  PATCHESTRY_COMMON_RELAX_WARNINGS \
  PATCHESTRY_CLANG_RELAX_WARNINGS
#elif __GNUC__
#define PATCHESTRY_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic push" ) \
  PATCHESTRY_COMMON_RELAX_WARNINGS \
  PATCHESTRY_GCC_RELAX_WARNINGS
#else
#define PATCHESTRY_RELAX_WARNINGS \
  _Pragma( "GCC diagnostic push" ) \
  PATCHESTRY_COMMON_RELAX_WARNINGS
#endif

#define PATCHESTRY_UNRELAX_WARNINGS \
  _Pragma( "GCC diagnostic pop" )

