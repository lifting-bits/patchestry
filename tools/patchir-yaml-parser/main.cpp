/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <filesystem>
#include <iostream>

#include <llvm/Support/CommandLine.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/raw_ostream.h>

#include <patchestry/Util/Log.hpp>
#include <patchestry/YAML/ConfigurationFile.hpp>
#include <patchestry/YAML/YAMLParser.hpp>

using namespace patchestry;

namespace {
    // Command line options
    llvm::cl::opt< std::string > InputFile(
        llvm::cl::Positional, llvm::cl::desc("<input YAML file>"),
        llvm::cl::value_desc("filename"), llvm::cl::Required
    );

    llvm::cl::opt< std::string > OutputFile(
        "o", llvm::cl::desc("Output file (default: stdout)"), llvm::cl::value_desc("filename"),
        llvm::cl::init("")
    );

    llvm::cl::opt< bool > Validate(
        "validate", llvm::cl::desc("Validate YAML file structure only"), llvm::cl::init(false)
    );

    llvm::cl::opt< bool > Pretty(
        "pretty", llvm::cl::desc("Pretty print the parsed configuration"), llvm::cl::init(false)
    );

    llvm::cl::opt< bool > Serialize(
        "serialize", llvm::cl::desc("Serialize back to YAML and output"), llvm::cl::init(false)
    );

} // namespace

// Pretty print patch configuration
void prettyPrint(const passes::Configuration &config) {
    llvm::outs() << "\n=== Patchestry Configuration ===\n";

    llvm::outs() << "apiVersion: " << config.api_version << "\n";

    // Print metadata
    llvm::outs() << "Metadata:\n";
    llvm::outs() << "  Name: " << config.metadata.name << "\n";
    llvm::outs() << "  Description: " << config.metadata.description << "\n";
    llvm::outs() << "  Version: " << config.metadata.version << "\n";
    llvm::outs() << "  Author: " << config.metadata.author << "\n";
    llvm::outs() << "  Organization: " << config.metadata.organization << "\n";
    llvm::outs() << "\n";

    // Print target
    llvm::outs() << "Target:\n";
    llvm::outs() << "  Binary: " << config.target.binary << "\n";
    llvm::outs() << "  Architecture: " << config.target.arch << "\n";
    llvm::outs() << "\n";

    llvm::outs() << "Execution Order:\n";
    for (const auto &order : config.execution_order) {
        llvm::outs() << "  " << order << "\n";
    }
    llvm::outs() << "\n";

    // Print meta patches
    llvm::outs() << "Meta Patches:\n";
    for (const auto &meta_patch : config.meta_patches) {
        llvm::outs() << "  Name: " << meta_patch.name << "\n";
        llvm::outs() << "  Description: " << meta_patch.description << "\n";
        llvm::outs() << "  Optimization: ";
        for (const auto &opt : meta_patch.optimization) {
            llvm::outs() << opt << " ";
        }
        llvm::outs() << "\n";
    }

    // Print patches
    for (size_t i = 0; i < config.libraries.patches.size(); ++i) {
        const auto &patch = config.libraries.patches[i];
        llvm::outs() << "Patch " << (i + 1) << ": " << patch.name << "\n";

        llvm::outs() << "  Patch:\n";
        if (!patch.code_file.empty()) {
            llvm::outs() << "    File: " << patch.code_file << "\n";
        }

        if (!patch.function_name.empty()) {
            llvm::outs() << "    Function: " << patch.function_name << "\n";
        }

        // Print contracts
        llvm::outs() << "  Contracts:\n";
        for (const auto &contract : config.libraries.contracts) {
            llvm::outs() << "    Name: " << contract.name << "\n";
        }

        llvm::outs() << "\n";
    }
}

// Write output to file or stdout
void writeOutput(const std::string &content, const std::string &filename) {
    if (filename.empty()) {
        llvm::outs() << content;
    } else {
        std::error_code ec;
        llvm::raw_fd_ostream output(filename, ec);
        if (ec) {
            LOG(ERROR) << "Failed to open output file: " << filename << " - " << ec.message();
            return;
        }
        output << content;
    }
}

int main(int argc, char **argv) {
    llvm::InitLLVM init(argc, argv);
    llvm::cl::ParseCommandLineOptions(
        argc, argv, "YAML Parser to verify configuration files\n"
    );

    yaml::YAMLParser parser;

    // Parse the input file
    ConfigurationFile::getInstance().set_file_path(InputFile.getValue());
    auto file_path = llvm::sys::path::filename(InputFile.getValue()).str();
    auto config    = yaml::utils::loadConfiguration(file_path);
    if (!config) {
        LOG(ERROR) << "\nFailed to parse YAML file: " << file_path << "\n";
        return 1;
    }

    LOG(INFO) << "\nSuccessfully parsed YAML file: " << InputFile.getValue() << "\n";

    // Handle validation only
    if (Validate) {
        bool isValid = yaml::utils::validateConfiguration(*config);
        if (isValid) {
            llvm::outs() << "\nYAML file is valid\n";
            return 0;
        } else {
            llvm::outs() << "\nYAML file validation failed\n";
            return 1;
        }
    }

    // Handle different output modes
    if (Pretty) {
        prettyPrint(*config);
    }

    if (Serialize) {
        std::string yaml = parser.serialize_to_string< passes::Configuration >(*config);
        if (!yaml.empty()) {
            writeOutput(yaml, OutputFile.getValue());
        } else {
            LOG(ERROR) << "\nFailed to serialize configuration\n";
            return 1;
        }
    }

    // Default action: show basic info
    if (!Pretty && !Serialize) {
        llvm::outs() << "Number of patches: " << config->libraries.patches.size() << "\n";

        llvm::outs() << "\nPatch names:\n";
        for (const auto &patch : config->libraries.patches) {
            llvm::outs() << "  - " << patch.name << "\n";
        }
    }

    return 0;
}
