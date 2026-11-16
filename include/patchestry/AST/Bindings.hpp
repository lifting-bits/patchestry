/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <llvm/Support/raw_ostream.h>

namespace patchestry::ghidra {
    struct Program;
} // namespace patchestry::ghidra

namespace patchestry::ast {

    /// One global or function the lifted translation unit refers to, tied back
    /// to what it is in the original binary.  The lifted code names things;
    /// the binary has addresses.  A reinsertion step needs both to check that
    /// a name resolves to the original object and not to a fresh copy.
    struct Binding
    {
        /// C identifier in the lifted translation unit.
        std::string name;
        /// Name the linker sees: the asm label when the declaration has one,
        /// otherwise `name`.
        std::string symbol;
        /// `object` or `function`.
        std::string kind;
        /// Where the entity comes from:
        ///   original     it has an address in the binary's memory
        ///   imported     Ghidra placed it in its EXTERNAL space: the binary
        ///                refers to it but does not contain it
        ///   patch-owned  the lifter introduced it (an intrinsic helper); it
        ///                has no address in the binary
        ///   unknown      the declaration is in the unit but nothing in the
        ///                program model accounts for it
        std::string origin;
        /// Ghidra key the entity was lifted from (`ram:20000010`); empty for
        /// patch-owned and unknown entities.
        std::string key;
        /// Address space and offset parsed from `key`; unset when the key is
        /// not of the form `<space>:<hex offset>`.
        std::string address_space;
        std::optional< uint64_t > address;
        /// Size in bytes the binary gives the entity: a global's recovered
        /// size, a function's address ranges.  0 when unknown.
        uint64_t size = 0;
        /// Objects only: size in bytes of the lifted type; 0 when the type is
        /// incomplete.
        uint64_t lifted_size = 0;
        /// Functions only: the unit holds a body for it, so reinsertion
        /// replaces the original code at `address`.
        bool defined = false;
    };

    /// A call in a lifted function body whose callee the program model does
    /// not name.  Nothing in the manifest binds its target: the address is
    /// computed at run time (or, for a tail call, Ghidra found no function at
    /// it), so a reinsertion step cannot check where the call goes.
    struct UnresolvedCall
    {
        /// Name of the function that contains the call.
        std::string function;
        /// Key of the call operation (`ram:0800182a:51:4`).
        std::string key;
        /// `indirect` for a CALLIND, `tail-call` for a tail call to an
        /// address that is not a known function.
        std::string kind;
        /// Call-site address parsed from `key`; unset when the key does not
        /// start with `<space>:<hex offset>`.
        std::string address_space;
        std::optional< uint64_t > address;
        /// Indirect calls: name of the global the pointer is read from, when
        /// the target is a global variable.  Empty otherwise.
        std::string through;
        /// Tail calls: the literal target address Ghidra recorded.
        std::string target_address;
    };

    /// Every indirect call and unresolved tail call in the program's function
    /// bodies, ordered by function name and operation key.
    std::vector< UnresolvedCall > CollectUnresolvedCalls(const ghidra::Program &program);

    /// Write the bindings as a JSON manifest, sorted for a stable diff.
    /// `unresolved_calls` goes into a list of its own, each entry with
    /// `"target": "unresolved"`.  `architecture`, `language` and `format` are
    /// copied from the program header and may be empty.
    void WriteBindings(
        llvm::raw_ostream &out, std::vector< Binding > bindings,
        const std::vector< UnresolvedCall > &unresolved_calls,
        const std::string &architecture, const std::string &language,
        const std::string &format
    );

} // namespace patchestry::ast
