/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/PatchDSL/Serialize.hpp>

#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Bytecode/BytecodeReader.h>
#include <mlir/Bytecode/BytecodeWriter.h>
#include <mlir/IR/Attributes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinAttributes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/Location.h>
#include <mlir/IR/OwningOpRef.h>
#include <mlir/Parser/Parser.h>

namespace patchestry::patchdsl {

    namespace passes   = patchestry::passes;
    namespace patch    = patchestry::passes::patch;
    namespace contract = patchestry::passes::contract;

    namespace {

        // ------------------------------------------------------------------
        //  Enum <-> string helpers.
        // ------------------------------------------------------------------

        llvm::StringRef modeToString(passes::PatchInfoMode m) {
            switch (m) {
                case passes::PatchInfoMode::APPLY_BEFORE: return "apply_before";
                case passes::PatchInfoMode::APPLY_AFTER:  return "apply_after";
                case passes::PatchInfoMode::REPLACE:      return "replace";
                case passes::PatchInfoMode::NONE:         return "none";
            }
            return "none";
        }
        passes::PatchInfoMode modeFromString(llvm::StringRef s) {
            if (s == "apply_before") return passes::PatchInfoMode::APPLY_BEFORE;
            if (s == "apply_after")  return passes::PatchInfoMode::APPLY_AFTER;
            if (s == "replace")      return passes::PatchInfoMode::REPLACE;
            return passes::PatchInfoMode::NONE;
        }

        llvm::StringRef matchKindToString(passes::MatchKind k) {
            switch (k) {
                case passes::MatchKind::OPERATION: return "operation";
                case passes::MatchKind::FUNCTION:  return "function";
                case passes::MatchKind::NONE:      return "none";
            }
            return "none";
        }
        passes::MatchKind matchKindFromString(llvm::StringRef s) {
            if (s == "operation") return passes::MatchKind::OPERATION;
            if (s == "function")  return passes::MatchKind::FUNCTION;
            return passes::MatchKind::NONE;
        }

        llvm::StringRef sourceToString(passes::ArgumentSourceType s) {
            switch (s) {
                case passes::ArgumentSourceType::OPERAND:      return "operand";
                case passes::ArgumentSourceType::VARIABLE:     return "variable";
                case passes::ArgumentSourceType::SYMBOL:       return "symbol";
                case passes::ArgumentSourceType::CONSTANT:     return "constant";
                case passes::ArgumentSourceType::RETURN_VALUE: return "return_value";
            }
            return "operand";
        }
        passes::ArgumentSourceType sourceFromString(llvm::StringRef s) {
            if (s == "operand")      return passes::ArgumentSourceType::OPERAND;
            if (s == "variable")     return passes::ArgumentSourceType::VARIABLE;
            if (s == "symbol")       return passes::ArgumentSourceType::SYMBOL;
            if (s == "constant")     return passes::ArgumentSourceType::CONSTANT;
            if (s == "return_value") return passes::ArgumentSourceType::RETURN_VALUE;
            return passes::ArgumentSourceType::OPERAND;
        }

        // ------------------------------------------------------------------
        //  Configuration -> DictionaryAttr.
        // ------------------------------------------------------------------

        mlir::Attribute strAttr(mlir::MLIRContext *ctx, llvm::StringRef s) {
            return mlir::StringAttr::get(ctx, s);
        }

        mlir::NamedAttribute nattr(
            mlir::MLIRContext *ctx, llvm::StringRef key, mlir::Attribute val
        ) {
            return { mlir::StringAttr::get(ctx, key), val };
        }

        mlir::DictionaryAttr metadataAttr(
            mlir::MLIRContext *ctx, const passes::Metadata &m
        ) {
            llvm::SmallVector< mlir::NamedAttribute > fields;
            fields.push_back(nattr(ctx, "name",        strAttr(ctx, m.name)));
            fields.push_back(nattr(ctx, "description", strAttr(ctx, m.description)));
            fields.push_back(nattr(ctx, "version",     strAttr(ctx, m.version)));
            fields.push_back(nattr(ctx, "author",      strAttr(ctx, m.author)));
            fields.push_back(nattr(ctx, "created",     strAttr(ctx, m.created)));
            fields.push_back(nattr(ctx, "organization", strAttr(ctx, m.organization)));
            return mlir::DictionaryAttr::get(ctx, fields);
        }

        mlir::DictionaryAttr targetAttr(
            mlir::MLIRContext *ctx, const passes::Target &t
        ) {
            llvm::SmallVector< mlir::NamedAttribute > fields;
            fields.push_back(nattr(ctx, "binary", strAttr(ctx, t.binary)));
            fields.push_back(nattr(ctx, "arch",   strAttr(ctx, t.arch)));
            return mlir::DictionaryAttr::get(ctx, fields);
        }

        mlir::DictionaryAttr parameterAttr(
            mlir::MLIRContext *ctx, const passes::Parameter &p
        ) {
            llvm::SmallVector< mlir::NamedAttribute > fields;
            fields.push_back(nattr(ctx, "name",        strAttr(ctx, p.name)));
            fields.push_back(nattr(ctx, "type",        strAttr(ctx, p.type)));
            fields.push_back(nattr(ctx, "description", strAttr(ctx, p.description)));
            return mlir::DictionaryAttr::get(ctx, fields);
        }

        mlir::DictionaryAttr patchSpecAttr(
            mlir::MLIRContext *ctx, const patch::PatchSpec &ps
        ) {
            llvm::SmallVector< mlir::NamedAttribute > fields;
            fields.push_back(nattr(ctx, "name",          strAttr(ctx, ps.name)));
            fields.push_back(nattr(ctx, "description",   strAttr(ctx, ps.description)));
            fields.push_back(nattr(ctx, "category",      strAttr(ctx, ps.category)));
            fields.push_back(nattr(ctx, "severity",      strAttr(ctx, ps.severity)));
            fields.push_back(nattr(ctx, "code_file",     strAttr(ctx, ps.code_file)));
            fields.push_back(nattr(ctx, "function_name", strAttr(ctx, ps.function_name)));
            llvm::SmallVector< mlir::Attribute > params;
            for (auto const &p : ps.parameters) {
                params.push_back(parameterAttr(ctx, p));
            }
            fields.push_back(
                nattr(ctx, "parameters", mlir::ArrayAttr::get(ctx, params))
            );
            return mlir::DictionaryAttr::get(ctx, fields);
        }

        mlir::DictionaryAttr librariesAttr(
            mlir::MLIRContext *ctx, const passes::Library &lib
        ) {
            llvm::SmallVector< mlir::Attribute > patch_attrs;
            for (auto const &p : lib.patches) {
                patch_attrs.push_back(patchSpecAttr(ctx, p));
            }
            llvm::SmallVector< mlir::NamedAttribute > fields;
            fields.push_back(nattr(ctx, "patches", mlir::ArrayAttr::get(ctx, patch_attrs)));
            return mlir::DictionaryAttr::get(ctx, fields);
        }

        mlir::DictionaryAttr argumentSourceAttr(
            mlir::MLIRContext *ctx, const patch::ArgumentSource &a
        ) {
            llvm::SmallVector< mlir::NamedAttribute > fields;
            fields.push_back(nattr(ctx, "name",   strAttr(ctx, a.name)));
            fields.push_back(nattr(ctx, "source", strAttr(ctx, sourceToString(a.source))));
            if (a.index) {
                fields.push_back(nattr(
                    ctx, "index",
                    mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                           static_cast< int64_t >(*a.index))
                ));
            }
            if (a.symbol) {
                fields.push_back(nattr(ctx, "symbol", strAttr(ctx, *a.symbol)));
            }
            if (a.value) {
                fields.push_back(nattr(ctx, "value", strAttr(ctx, *a.value)));
            }
            if (a.is_reference) {
                fields.push_back(nattr(
                    ctx, "is_reference",
                    mlir::BoolAttr::get(ctx, *a.is_reference)
                ));
            }
            return mlir::DictionaryAttr::get(ctx, fields);
        }

        mlir::DictionaryAttr actionAttr(
            mlir::MLIRContext *ctx, const patch::Action &a
        ) {
            llvm::SmallVector< mlir::NamedAttribute > fields;
            fields.push_back(nattr(ctx, "mode",        strAttr(ctx, modeToString(a.mode))));
            fields.push_back(nattr(ctx, "patch_id",    strAttr(ctx, a.patch_id)));
            fields.push_back(nattr(ctx, "description", strAttr(ctx, a.description)));
            llvm::SmallVector< mlir::Attribute > args;
            for (auto const &arg : a.arguments) {
                args.push_back(argumentSourceAttr(ctx, arg));
            }
            fields.push_back(nattr(ctx, "arguments", mlir::ArrayAttr::get(ctx, args)));
            return mlir::DictionaryAttr::get(ctx, fields);
        }

        mlir::DictionaryAttr functionContextAttr(
            mlir::MLIRContext *ctx, const passes::FunctionContext &fc
        ) {
            llvm::SmallVector< mlir::NamedAttribute > fields;
            fields.push_back(nattr(ctx, "name", strAttr(ctx, fc.name)));
            fields.push_back(nattr(ctx, "type", strAttr(ctx, fc.type)));
            return mlir::DictionaryAttr::get(ctx, fields);
        }

        mlir::DictionaryAttr matchConfigAttr(
            mlir::MLIRContext *ctx, const patch::MatchConfig &m
        ) {
            llvm::SmallVector< mlir::NamedAttribute > fields;
            fields.push_back(nattr(ctx, "name", strAttr(ctx, m.name)));
            fields.push_back(nattr(ctx, "kind", strAttr(ctx, matchKindToString(m.kind))));
            llvm::SmallVector< mlir::Attribute > fctx;
            for (auto const &c : m.function_context) {
                fctx.push_back(functionContextAttr(ctx, c));
            }
            fields.push_back(
                nattr(ctx, "function_context", mlir::ArrayAttr::get(ctx, fctx))
            );
            return mlir::DictionaryAttr::get(ctx, fields);
        }

        mlir::DictionaryAttr patchActionAttr(
            mlir::MLIRContext *ctx, const patch::PatchAction &pa
        ) {
            llvm::SmallVector< mlir::NamedAttribute > fields;
            fields.push_back(nattr(ctx, "id",          strAttr(ctx, pa.action_id)));
            fields.push_back(nattr(ctx, "description", strAttr(ctx, pa.description)));
            llvm::SmallVector< mlir::Attribute > matches;
            for (auto const &m : pa.match) {
                matches.push_back(matchConfigAttr(ctx, m));
            }
            fields.push_back(nattr(ctx, "match", mlir::ArrayAttr::get(ctx, matches)));
            llvm::SmallVector< mlir::Attribute > actions;
            for (auto const &a : pa.action) {
                actions.push_back(actionAttr(ctx, a));
            }
            fields.push_back(nattr(ctx, "action", mlir::ArrayAttr::get(ctx, actions)));
            return mlir::DictionaryAttr::get(ctx, fields);
        }

        mlir::DictionaryAttr metaPatchAttr(
            mlir::MLIRContext *ctx, const patch::MetaPatchConfig &mp
        ) {
            llvm::SmallVector< mlir::NamedAttribute > fields;
            fields.push_back(nattr(ctx, "name",        strAttr(ctx, mp.name)));
            fields.push_back(nattr(ctx, "description", strAttr(ctx, mp.description)));
            llvm::SmallVector< mlir::Attribute > opts;
            for (auto const &o : mp.optimization) {
                opts.push_back(strAttr(ctx, o));
            }
            fields.push_back(nattr(ctx, "optimization", mlir::ArrayAttr::get(ctx, opts)));
            llvm::SmallVector< mlir::Attribute > actions;
            for (auto const &pa : mp.patch_actions) {
                actions.push_back(patchActionAttr(ctx, pa));
            }
            fields.push_back(
                nattr(ctx, "patch_actions", mlir::ArrayAttr::get(ctx, actions))
            );
            return mlir::DictionaryAttr::get(ctx, fields);
        }

        // ------------------------------------------------------------------
        //  DictionaryAttr -> Configuration.
        // ------------------------------------------------------------------

        std::string getStr(mlir::DictionaryAttr d, llvm::StringRef key) {
            if (auto s = d.getAs< mlir::StringAttr >(key)) {
                return s.str();
            }
            return {};
        }

        mlir::ArrayAttr getArr(mlir::DictionaryAttr d, llvm::StringRef key) {
            return d.getAs< mlir::ArrayAttr >(key);
        }

        passes::Metadata metadataFromAttr(mlir::DictionaryAttr d) {
            passes::Metadata m;
            if (!d) {
                return m;
            }
            m.name         = getStr(d, "name");
            m.description  = getStr(d, "description");
            m.version      = getStr(d, "version");
            m.author       = getStr(d, "author");
            m.created      = getStr(d, "created");
            m.organization = getStr(d, "organization");
            return m;
        }

        passes::Target targetFromAttr(mlir::DictionaryAttr d) {
            passes::Target t;
            if (!d) {
                return t;
            }
            t.binary = getStr(d, "binary");
            t.arch   = getStr(d, "arch");
            return t;
        }

        passes::Parameter parameterFromAttr(mlir::DictionaryAttr d) {
            passes::Parameter p;
            if (!d) {
                return p;
            }
            p.name        = getStr(d, "name");
            p.type        = getStr(d, "type");
            p.description = getStr(d, "description");
            return p;
        }

        patch::PatchSpec patchSpecFromAttr(mlir::DictionaryAttr d) {
            patch::PatchSpec ps;
            if (!d) {
                return ps;
            }
            ps.name          = getStr(d, "name");
            ps.description   = getStr(d, "description");
            ps.category      = getStr(d, "category");
            ps.severity      = getStr(d, "severity");
            ps.code_file     = getStr(d, "code_file");
            ps.function_name = getStr(d, "function_name");
            if (auto params = getArr(d, "parameters")) {
                for (mlir::Attribute a : params) {
                    ps.parameters.push_back(parameterFromAttr(
                        mlir::dyn_cast< mlir::DictionaryAttr >(a)
                    ));
                }
            }
            return ps;
        }

        passes::Library libraryFromAttr(mlir::DictionaryAttr d) {
            passes::Library lib;
            if (!d) {
                return lib;
            }
            if (auto arr = getArr(d, "patches")) {
                for (mlir::Attribute a : arr) {
                    lib.patches.push_back(
                        patchSpecFromAttr(mlir::dyn_cast< mlir::DictionaryAttr >(a))
                    );
                }
            }
            return lib;
        }

        patch::ArgumentSource argSourceFromAttr(mlir::DictionaryAttr d) {
            patch::ArgumentSource a;
            if (!d) {
                return a;
            }
            a.name   = getStr(d, "name");
            a.source = sourceFromString(getStr(d, "source"));
            if (auto i = d.getAs< mlir::IntegerAttr >("index")) {
                a.index = static_cast< unsigned >(i.getInt());
            }
            if (auto s = d.getAs< mlir::StringAttr >("symbol")) {
                a.symbol = s.str();
            }
            if (auto s = d.getAs< mlir::StringAttr >("value")) {
                a.value = s.str();
            }
            if (auto b = d.getAs< mlir::BoolAttr >("is_reference")) {
                a.is_reference = b.getValue();
            }
            return a;
        }

        patch::Action actionFromAttr(mlir::DictionaryAttr d) {
            patch::Action a;
            if (!d) {
                return a;
            }
            a.mode        = modeFromString(getStr(d, "mode"));
            a.patch_id    = getStr(d, "patch_id");
            a.description = getStr(d, "description");
            if (auto args = getArr(d, "arguments")) {
                for (mlir::Attribute x : args) {
                    a.arguments.push_back(
                        argSourceFromAttr(mlir::dyn_cast< mlir::DictionaryAttr >(x))
                    );
                }
            }
            return a;
        }

        passes::FunctionContext functionContextFromAttr(mlir::DictionaryAttr d) {
            passes::FunctionContext fc;
            if (!d) {
                return fc;
            }
            fc.name = getStr(d, "name");
            fc.type = getStr(d, "type");
            return fc;
        }

        patch::MatchConfig matchConfigFromAttr(mlir::DictionaryAttr d) {
            patch::MatchConfig m;
            if (!d) {
                return m;
            }
            m.name = getStr(d, "name");
            m.kind = matchKindFromString(getStr(d, "kind"));
            if (auto arr = getArr(d, "function_context")) {
                for (mlir::Attribute x : arr) {
                    m.function_context.push_back(
                        functionContextFromAttr(mlir::dyn_cast< mlir::DictionaryAttr >(x))
                    );
                }
            }
            return m;
        }

        patch::PatchAction patchActionFromAttr(mlir::DictionaryAttr d) {
            patch::PatchAction pa;
            if (!d) {
                return pa;
            }
            pa.action_id   = getStr(d, "id");
            pa.description = getStr(d, "description");
            if (auto arr = getArr(d, "match")) {
                for (mlir::Attribute x : arr) {
                    pa.match.push_back(
                        matchConfigFromAttr(mlir::dyn_cast< mlir::DictionaryAttr >(x))
                    );
                }
            }
            if (auto arr = getArr(d, "action")) {
                for (mlir::Attribute x : arr) {
                    pa.action.push_back(
                        actionFromAttr(mlir::dyn_cast< mlir::DictionaryAttr >(x))
                    );
                }
            }
            return pa;
        }

        patch::MetaPatchConfig metaPatchFromAttr(mlir::DictionaryAttr d) {
            patch::MetaPatchConfig mp;
            if (!d) {
                return mp;
            }
            mp.name        = getStr(d, "name");
            mp.description = getStr(d, "description");
            if (auto opts = getArr(d, "optimization")) {
                for (mlir::Attribute x : opts) {
                    if (auto s = mlir::dyn_cast< mlir::StringAttr >(x)) {
                        mp.optimization.insert(s.str());
                    }
                }
            }
            if (auto arr = getArr(d, "patch_actions")) {
                for (mlir::Attribute x : arr) {
                    mp.patch_actions.push_back(
                        patchActionFromAttr(mlir::dyn_cast< mlir::DictionaryAttr >(x))
                    );
                }
            }
            return mp;
        }

    } // namespace

    llvm::Error WritePatchmod(
        const passes::Configuration &cfg,
        llvm::StringRef out_path,
        mlir::MLIRContext &context
    ) {
        mlir::OpBuilder builder(&context);
        auto module = mlir::ModuleOp::create(builder.getUnknownLoc());

        llvm::SmallVector< mlir::NamedAttribute > dsl_attrs;
        dsl_attrs.push_back(nattr(
            &context, "patchestry.dsl.api_version",
            strAttr(&context, cfg.api_version)
        ));
        dsl_attrs.push_back(nattr(
            &context, "patchestry.dsl.metadata", metadataAttr(&context, cfg.metadata)
        ));
        dsl_attrs.push_back(nattr(
            &context, "patchestry.dsl.target", targetAttr(&context, cfg.target)
        ));
        dsl_attrs.push_back(nattr(
            &context, "patchestry.dsl.libraries", librariesAttr(&context, cfg.libraries)
        ));
        llvm::SmallVector< mlir::Attribute > execution_order;
        for (auto const &s : cfg.execution_order) {
            execution_order.push_back(strAttr(&context, s));
        }
        dsl_attrs.push_back(nattr(
            &context, "patchestry.dsl.execution_order",
            mlir::ArrayAttr::get(&context, execution_order)
        ));
        llvm::SmallVector< mlir::Attribute > meta_patches;
        for (auto const &mp : cfg.meta_patches) {
            meta_patches.push_back(metaPatchAttr(&context, mp));
        }
        dsl_attrs.push_back(nattr(
            &context, "patchestry.dsl.meta_patches",
            mlir::ArrayAttr::get(&context, meta_patches)
        ));

        // Merge DSL attrs onto the module's attribute dictionary.
        llvm::SmallVector< mlir::NamedAttribute > module_attrs(
            module->getAttrs().begin(), module->getAttrs().end()
        );
        for (auto const &na : dsl_attrs) {
            module_attrs.push_back(na);
        }
        module->setAttrs(module_attrs);

        std::error_code ec;
        llvm::raw_fd_ostream os(out_path, ec, llvm::sys::fs::OF_None);
        if (ec) {
            module->erase();
            return llvm::createStringError(
                ec, "cannot open `%s` for writing", out_path.str().c_str()
            );
        }
        mlir::BytecodeWriterConfig cfg_bc;
        if (mlir::failed(mlir::writeBytecodeToFile(module, os, cfg_bc))) {
            module->erase();
            return llvm::createStringError(
                std::errc::io_error, "failed to write MLIR bytecode"
            );
        }
        os.flush();
        module->erase();
        return llvm::Error::success();
    }

    llvm::Expected< passes::Configuration >
    ReadPatchmod(llvm::StringRef in_path, mlir::MLIRContext &context) {
        auto buf_or = llvm::MemoryBuffer::getFile(in_path);
        if (std::error_code ec = buf_or.getError()) {
            return llvm::createStringError(
                ec, "cannot read `%s`: %s", in_path.str().c_str(), ec.message().c_str()
            );
        }
        llvm::SourceMgr sm;
        sm.AddNewSourceBuffer(std::move(*buf_or), llvm::SMLoc());
        mlir::ParserConfig pc(&context);
        mlir::OwningOpRef< mlir::ModuleOp > module =
            mlir::parseSourceFile< mlir::ModuleOp >(sm, pc);
        if (!module) {
            return llvm::createStringError(
                std::errc::invalid_argument,
                "failed to parse `%s` as MLIR bytecode", in_path.str().c_str()
            );
        }

        passes::Configuration cfg;
        auto attrs = module->getOperation()->getAttrDictionary();
        if (auto api = attrs.getAs< mlir::StringAttr >("patchestry.dsl.api_version")) {
            cfg.api_version = api.str();
        }
        if (auto md = attrs.getAs< mlir::DictionaryAttr >("patchestry.dsl.metadata")) {
            cfg.metadata = metadataFromAttr(md);
        }
        if (auto tg = attrs.getAs< mlir::DictionaryAttr >("patchestry.dsl.target")) {
            cfg.target = targetFromAttr(tg);
        }
        if (auto lb = attrs.getAs< mlir::DictionaryAttr >("patchestry.dsl.libraries")) {
            cfg.libraries = libraryFromAttr(lb);
        }
        if (auto eo = attrs.getAs< mlir::ArrayAttr >("patchestry.dsl.execution_order")) {
            for (mlir::Attribute x : eo) {
                if (auto s = mlir::dyn_cast< mlir::StringAttr >(x)) {
                    cfg.execution_order.push_back(s.str());
                }
            }
        }
        if (auto mp = attrs.getAs< mlir::ArrayAttr >("patchestry.dsl.meta_patches")) {
            for (mlir::Attribute x : mp) {
                cfg.meta_patches.push_back(
                    metaPatchFromAttr(mlir::dyn_cast< mlir::DictionaryAttr >(x))
                );
            }
        }
        return cfg;
    }

} // namespace patchestry::patchdsl
