/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <cstdlib>
#include <map>
#include <system_error>
#include <clang/Basic/TargetInfo.h>
#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <clang/CIR/LowerToLLVM.h>
#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/DIBuilder.h>
#include <llvm/IR/DebugInfoMetadata.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/LogicalResult.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/raw_ostream.h>
#include <mlir/Dialect/LLVMIR/Transforms/Passes.h>
#include <mlir/IR/BuiltinOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/InitAllDialects.h>
#include <mlir/Parser/Parser.h>
#include <mlir/Support/LLVM.h>
#include <mlir/Target/LLVMIR/Dialect/All.h>
#include <mlir/Target/LLVMIR/Import.h>

#include <patchestry/Dialect/Contracts/ContractsDialect.hpp>
#include <patchestry/Util/Log.hpp>

namespace {
    // Command line option for input file. Defaults to "-" (stdin)
    const llvm::cl::opt< std::string > input_filename(
        llvm::cl::Positional, llvm::cl::desc("<input filename>"), llvm::cl::init("-")
    );

    // Command line option for output file. Defaults to "-" (stdout)
    const llvm::cl::opt< std::string > output_filename(
        "o", llvm::cl::desc("Output filename"), llvm::cl::value_desc("filename"),
        llvm::cl::init("-")
    );

    // Command line option to specify target triple for compilation
    const llvm::cl::opt< std::string > target_triple(
        "target", llvm::cl::desc("Specify a target triple when compiling"), llvm::cl::init("")
    );

    // Command line flag to emit LLVM IR instead of bitcode
    const llvm::cl::opt< bool >
        emit_ll("S", llvm::cl::desc("Emit LLVM IR Representation"), llvm::cl::init(false));
} // namespace

namespace cir {
    namespace direct {
        // External declaration for registering CIR dialect translation
        extern void registerCIRDialectTranslation(mlir::DialectRegistry &registry);
    } // namespace direct
} // namespace cir

namespace {
    // Sets the target triple attribute on the MLIR module
    std::string prepareModuleTriple(mlir::ModuleOp &module) {
        if (!target_triple.empty()) {
            module->setAttr(
                cir::CIRDialect::getTripleAttrName(),
                mlir::StringAttr::get(module.getContext(), target_triple.getValue())
            );
            return target_triple.getValue();
        }
        return {};
    }

    // Configures data layout for the module based on target triple
    llvm::LogicalResult prepareModuleDataLayout(mlir::ModuleOp module, llvm::StringRef tt) {
        auto *context = module.getContext();

        llvm::Triple triple(tt);
        clang::TargetOptions target_options;
        target_options.Triple = tt;
        // FIXME: AllocateTarget is a big deal. Better make it a global state.
        auto target_info = clang::targets::AllocateTarget(llvm::Triple(tt), target_options);
        if (!target_info) {
            module.emitError() << "error: invalid target triple '" << tt << "'\n";
            return llvm::failure();
        }
        std::string layout_string = target_info->getDataLayoutString();

        context->loadDialect< mlir::DLTIDialect, mlir::LLVM::LLVMDialect >();
        mlir::DataLayoutSpecInterface dl_spec =
            mlir::translateDataLayout(llvm::DataLayout(layout_string), context);
        module->setAttr(
            mlir::DLTIDialect::kDataLayoutAttrName, mlir::cast< mlir::Attribute >(dl_spec)
        );

        return llvm::success();
    }

    // Prepares MLIR module for translation by setting up target triple and data layout
    llvm::LogicalResult prepareModuleForTranslation(mlir::ModuleOp module) {
        auto mod_triple =
            module->getAttrOfType< mlir::StringAttr >(cir::CIRDialect::getTripleAttrName());
        auto data_layout       = module->getAttr(mlir::DLTIDialect::kDataLayoutAttrName);
        bool has_target_option = target_triple.getNumOccurrences() > 0;

        if (!has_target_option && mod_triple && data_layout) {
            return llvm::success();
        }

        std::string triple;
        if (!has_target_option && mod_triple) {
            triple = mod_triple.getValue();
        } else {
            triple = prepareModuleTriple(module);
        }

        return prepareModuleDataLayout(module, triple);
    }

    // Structure to hold collected string attributes from MLIR operations
    struct OperationMetadata {
        std::string operation_name;
        std::string patchestry_operation;
        std::string static_contract;
        // MLIR location information
        std::string mlir_file;
        unsigned mlir_line = 0;
        unsigned mlir_column = 0;
    };

    // Helper function to serialize a PredicateAttr to a human-readable string
    std::string serializePredicateAttr(const ::contracts::PredicateAttr &pred) {
        std::string result;
        llvm::raw_string_ostream os(result);

        // Write predicate kind
        os << "kind=" << ::contracts::stringifyPredicateKind(pred.getKind());

        if (pred.getTarget()) {
            os << ", target=";
            auto target = pred.getTarget();
            os << ::contracts::stringifyTargetKind(target.getKind());
            if (target.getKind() == ::contracts::TargetKind::Arg) {
                os << "(" << target.getIndex() << ")";
            } else if (target.getKind() == ::contracts::TargetKind::Symbol
                       && target.getSymbol())
            {
                os << "(" << target.getSymbol().getValue() << ")";
            }
        }

        // Write relation if present
        if (pred.getRelation() != ::contracts::RelationKind::none) {
            os << ", relation=" << ::contracts::stringifyRelationKind(pred.getRelation());
        }

        // Write alignment if present
        if (pred.getAlign()) {
            os << ", align=" << pred.getAlign().getAlignment();
        }

        // Write expression if present
        if (pred.getExpr()) {
            os << ", expr=\"" << pred.getExpr().getValue() << "\"";
        }

        // Write range if present
        if (pred.getRange()) {
            os << ", range=[";
            if (pred.getRange().getMin()) {
                os << "min=" << pred.getRange().getMin().getValue();
            }
            if (pred.getRange().getMax()) {
                os << ", max=" << pred.getRange().getMax().getValue();
            }
            os << "]";
        }

        return os.str();
    }

    // Helper function to serialize a StaticContractAttr to a human-readable string
    std::string serializeStaticContract(const ::contracts::StaticContractAttr &contract) {
        std::string result;
        llvm::raw_string_ostream os(result);

        // Serialize preconditions
        auto preconditions = contract.getPreconditions();
        if (!preconditions.empty()) {
            os << "preconditions=[";
            bool first = true;
            for (auto attr : preconditions) {
                if (auto preAttr = mlir::dyn_cast< ::contracts::PreconditionAttr >(attr)) {
                    if (!first) {
                        os << "; ";
                    }
                    first = false;
                    os << "{id=\"" << preAttr.getId().getValue() << "\", "
                       << serializePredicateAttr(preAttr.getPred()) << "}";
                }
            }
            os << "]";
        }

        // Serialize postconditions
        auto postconditions = contract.getPostconditions();
        if (!postconditions.empty()) {
            if (!preconditions.empty()) {
                os << ", ";
            }
            os << "postconditions=[";
            bool first = true;
            for (auto attr : postconditions) {
                if (auto postAttr = mlir::dyn_cast< ::contracts::PostconditionAttr >(attr)) {
                    if (!first) {
                        os << "; ";
                    }
                    first = false;
                    os << "{id=" << postAttr.getId().getValue() << ", "
                       << serializePredicateAttr(postAttr.getPred()) << "}";
                }
            }
            os << "]";
        }

        return os.str();
    }

    // Collects string attributes from MLIR operations
    std::vector< OperationMetadata > collectAttributes(mlir::ModuleOp module) {
        std::vector< OperationMetadata > metadata_list;

        // Get module name for fallback when mlir_file is empty or "-"
        std::string module_name = module.getName() ? module.getName()->str() : "module";

        module.walk([&](mlir::Operation *op) {
            OperationMetadata metadata;
            metadata.operation_name = op->getName().getStringRef().str();

            // Extract MLIR location information
            auto loc = op->getLoc();
            if (auto file_loc = mlir::dyn_cast< mlir::FileLineColLoc >(loc)) {
                metadata.mlir_file   = file_loc.getFilename().str();
                // Replace empty or "-" with module name
                if (metadata.mlir_file.empty() || metadata.mlir_file == "-") {
                    metadata.mlir_file = module_name;
                }
                metadata.mlir_line   = file_loc.getLine();
                metadata.mlir_column = file_loc.getColumn();
            } else if (auto fused_loc = mlir::dyn_cast< mlir::FusedLoc >(loc)) {
                // For fused locations, try to get the first file location
                for (auto sub_loc : fused_loc.getLocations()) {
                    if (auto file_loc = mlir::dyn_cast< mlir::FileLineColLoc >(sub_loc)) {
                        metadata.mlir_file   = file_loc.getFilename().str();
                        // Replace empty or "-" with module name
                        if (metadata.mlir_file.empty() || metadata.mlir_file == "-") {
                            metadata.mlir_file = module_name;
                        }
                        metadata.mlir_line   = file_loc.getLine();
                        metadata.mlir_column = file_loc.getColumn();
                        break;
                    }
                }
            }

            // Check for patchestry_operation attribute
            if (auto attr = op->getAttrOfType<mlir::StringAttr>("patchestry_operation")) {
                metadata.patchestry_operation = attr.getValue().str();
                metadata_list.push_back(metadata);
                LOG(INFO) << "Found patchestry_operation attribute: "
                          << metadata.patchestry_operation
                          << " on operation: " << metadata.operation_name;
                if (!metadata.mlir_file.empty()) {
                    LOG(INFO) << " at " << metadata.mlir_file << ":" << metadata.mlir_line << ":"
                              << metadata.mlir_column;
                }
                LOG(INFO) << "\n";
            }

            if (auto attr =
                    op->getAttrOfType< ::contracts::StaticContractAttr >("contract.static"))
            {
                metadata.static_contract = serializeStaticContract(attr);
                metadata_list.push_back(metadata);
                LOG(INFO) << "Found contract.static attribute on operation: "
                          << metadata.operation_name
                          << "\nContract details: " << metadata.static_contract << "\n";
                if (!metadata.mlir_file.empty()) {
                    LOG(INFO) << " at " << metadata.mlir_file << ":" << metadata.mlir_line
                              << ":" << metadata.mlir_column;
                }
                LOG(INFO) << "\nStatic contract details: " << metadata.static_contract << "\n";
            }
        });

        return metadata_list;
    }

    // Structure to uniquely identify a source location
    struct LocationKey {
        std::string file;
        unsigned line;
        unsigned column;

        bool operator<(const LocationKey &other) const {
            if (file != other.file) {
                return file < other.file;
            }
            if (line != other.line) {
                return line < other.line;
            }
            return column < other.column;
        }

        bool operator==(const LocationKey &other) const {
            return file == other.file && line == other.line && column == other.column;
        }
    };

    // Embeds collected string attributes as debug information in LLVM IR
    void embedDebugInformation(
        llvm::Module &module, const std::vector<OperationMetadata> &metadata_list
    ) {
        if (metadata_list.empty()) {
            LOG(INFO) << "No string attributes to embed as debug information\n";
            return;
        }

        llvm::LLVMContext &context = module.getContext();

        // Build a map from source location to metadata
        // This allows us to match LLVM instructions with their original MLIR operations
        std::map<LocationKey, const OperationMetadata *> location_to_metadata;

        // Also build a map from line number only (for fallback matching)
        std::multimap<unsigned, const OperationMetadata *> line_to_metadata;

        for (const auto &metadata : metadata_list) {
            if (!metadata.mlir_file.empty() && metadata.mlir_line > 0) {
                LocationKey key{ metadata.mlir_file, metadata.mlir_line, metadata.mlir_column };
                location_to_metadata[key] = &metadata;
                line_to_metadata.insert({ metadata.mlir_line, &metadata });

                LOG(INFO) << "Registered metadata for location: " << metadata.mlir_file << ":"
                          << metadata.mlir_line << ":" << metadata.mlir_column
                          << " op=" << metadata.operation_name;
                if (!metadata.patchestry_operation.empty()) {
                    LOG(INFO) << " patchestry_op=" << metadata.patchestry_operation;
                }
                if (!metadata.static_contract.empty()) {
                    LOG(INFO) << " static_contract=" << metadata.static_contract;
                }
                LOG(INFO) << "\n";
            }
        }

        unsigned matched_count      = 0;
        unsigned total_instructions = 0;

        // Iterate through all LLVM instructions
        for (auto &func : module.functions()) {
            for (auto &bb : func) {
                for (auto &inst : bb) {
                    total_instructions++;

                    // Get the current debug location of the instruction (set during lowering)
                    auto current_loc = inst.getDebugLoc();
                    if (!current_loc) {
                        continue; // Skip instructions without debug info
                    }

                    // Extract location information from the LLVM instruction
                    unsigned llvm_line = current_loc.getLine();
                    unsigned llvm_col  = current_loc.getCol();
                    llvm::StringRef llvm_file =
                        current_loc->getFile() ? current_loc->getFile()->getFilename() : "";

                    if (llvm_file.empty() || llvm_line == 0) {
                        continue;
                    }

                    // if llvm_file == "-" then use the source file name
                    if (llvm_file == "-") {
                        llvm_file = module.getSourceFileName();
                    }

                    // Try to find matching metadata based on location
                    LocationKey key{ llvm_file.str(), llvm_line, llvm_col };
                    auto it = location_to_metadata.find(key);

                    const OperationMetadata *matched_metadata = nullptr;

                    if (it != location_to_metadata.end()) {
                        // Exact match found (file:line:column)
                        matched_metadata = it->second;
                        LOG(INFO) << "Exact match found for " << llvm_file.str() << ":"
                                  << llvm_line << ":" << llvm_col << "\n";
                    } else {
                        // Try matching by file and line only (ignore column)
                        for (auto &[loc_key, metadata] : location_to_metadata) {
                            if (loc_key.file == llvm_file.str() && loc_key.line == llvm_line) {
                                matched_metadata = metadata;
                                LOG(INFO) << "Line match found for " << llvm_file.str() << ":"
                                          << llvm_line << "\n";
                                break;
                            }
                        }
                    }

                    // If we found matching metadata, attach custom patchestry metadata
                    if (matched_metadata) {
                        if (!matched_metadata->patchestry_operation.empty()) {
                            llvm::MDNode *md_node = llvm::MDNode::get(
                                context,
                                { llvm::MDString::get(context, "patchestry_operation"),
                                  llvm::MDString::get(
                                      context, matched_metadata->patchestry_operation
                                  ) }
                            );
                            inst.setMetadata("patchestry", md_node);
                        }

                        LOG(INFO)
                            << "Attached patchestry_operation metadata '"
                            << matched_metadata->patchestry_operation << "' to instruction at "
                            << llvm_file.str() << ":" << llvm_line << ":" << llvm_col << "\n";

                        if (!matched_metadata->static_contract.empty()) {
                            llvm::MDNode *contract_node = llvm::MDNode::get(
                                context,
                                { llvm::MDString::get(context, "static_contract"),
                                  llvm::MDString::get(
                                      context, matched_metadata->static_contract
                                  ) }
                            );
                            inst.setMetadata("static_contract", contract_node);

                            LOG(INFO)
                                << "Attached static contract metadata to instruction at "
                                << llvm_file.str() << ":" << llvm_line << ":" << llvm_col
                                << "\nContract: " << matched_metadata->static_contract << "\n";
                        }

                        llvm::MDNode *mlir_loc_node = llvm::MDNode::get(
                            context,
                            { llvm::MDString::get(context, "mlir_location"),
                              llvm::MDString::get(context, matched_metadata->mlir_file),
                              llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                                  llvm::Type::getInt32Ty(context), matched_metadata->mlir_line
                              )),
                              llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                                  llvm::Type::getInt32Ty(context), matched_metadata->mlir_column
                              )) }
                        );
                        inst.setMetadata("mlir_loc", mlir_loc_node);

                        LOG(INFO) << "Attached MLIR location metadata to instruction at "
                                  << llvm_file.str() << ":" << llvm_line << ":" << llvm_col
                                  << "\nMLIR location: " << matched_metadata->mlir_file << ":"
                                  << matched_metadata->mlir_line << ":"
                                  << matched_metadata->mlir_column << "\n";
                    }
                }
                matched_count++;
            }
        }

        LOG(INFO) << "Embedded " << matched_count << " out of " << metadata_list.size()
                  << " metadata entries (total instructions: " << total_instructions << ")\n";
    }

    // Writes LLVM module to file either as bitcode or LLVM IR
    llvm::LogicalResult
    writeBitcodeToFile(llvm::Module &module, llvm::StringRef output_filename) {
        std::error_code ec;
        llvm::raw_fd_ostream os(output_filename, ec, llvm::sys::fs::OF_None);
        if (ec) {
            LOG(ERROR) << "Error opening " << output_filename << ": " << ec.message() << "\n";
            return llvm::failure();
        }
        if (emit_ll) {
            module.print(os, nullptr);
        } else {
            llvm::WriteBitcodeToFile(module, os);
        }
        return llvm::success();
    }
} // namespace

// Main function for the patchir-cir2llvm tool
int main(int argc, char **argv) {
    llvm::InitLLVM y(argc, argv);
    llvm::cl::ParseCommandLineOptions(argc, argv, "patchir-cir2llvm driver\n");

    mlir::DialectRegistry registry;
    registry.insert< mlir::DLTIDialect, mlir::func::FuncDialect >();

    mlir::registerAllDialects(registry);
    mlir::registerAllToLLVMIRTranslations(registry);

    registry.insert< cir::CIRDialect >();
    cir::direct::registerCIRDialectTranslation(registry);

    registry.insert< ::contracts::ContractsDialect >();

    mlir::MLIRContext context(registry);
    auto file_or_err = llvm::MemoryBuffer::getFileOrSTDIN(input_filename);
    if (auto err = file_or_err.getError()) {
        LOG(ERROR) << "Error opening file: " << input_filename << "\n";
        return EXIT_FAILURE;
    }

    auto module =
        mlir::parseSourceString< mlir::ModuleOp >(file_or_err.get()->getBuffer(), &context);
    if (!module) {
        LOG(ERROR) << "Error parsing mlir module\n";
        return EXIT_FAILURE;
    }

    if (mlir::failed(prepareModuleForTranslation(module.get()))) {
        return EXIT_FAILURE;
    }

    // Collect string attributes from MLIR operations before lowering
    LOG(INFO) << "Collecting string attributes from MLIR module\n";
    auto metadata_list = collectAttributes(module.get());

    llvm::LLVMContext llvm_context;
    auto llvm_module = cir::direct::lowerDirectlyFromCIRToLLVMIR(*module, llvm_context);
    if (!llvm_module) {
        LOG(ERROR) << "Failed to lower cir to llvm\n";
        return EXIT_FAILURE;
    }

    // Embed collected string attributes as debug information in LLVM IR
    LOG(INFO) << "Embedding string attributes as debug information\n";
    embedDebugInformation(*llvm_module, metadata_list);

    auto err = writeBitcodeToFile(*llvm_module, output_filename);
    if (mlir::failed(err)) {
        LOG(ERROR) << "Failed to write bitcode to file\n";
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
