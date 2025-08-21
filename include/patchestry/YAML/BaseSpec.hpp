/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <string>

#include <llvm/Support/YAMLTraits.h>

namespace patchestry::passes {
    struct Metadata {
        std::string name;
        std::string description;
        std::string version;
        std::string author;
        std::string created;
        std::string organization;
    };
}

namespace llvm::yaml {
    // Parse Metadata
    template<>
    struct MappingTraits< patchestry::passes::Metadata >
    {
        static void mapping(IO &io, patchestry::passes::Metadata &metadata) {
            io.mapOptional("name", metadata.name);
            io.mapOptional("description", metadata.description);
            io.mapOptional("version", metadata.version);
            io.mapOptional("author", metadata.author);
            io.mapOptional("created", metadata.created);
            io.mapOptional("organization", metadata.organization);
        }
    };
}