/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <clang/CIR/Dialect/IR/CIRDialect.h>
#include <clang/CIR/Dialect/IR/CIRTypes.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/Types.h>

#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

namespace patchestry {
    namespace utils {
        namespace {
            mlir::Type
            lookup_or_create_struct_type(mlir::MLIRContext *context, std::string struct_name) {
                // Create an opaque/incomplete struct type with the given name
                // This represents a forward-declared struct where we don't know the fields
                llvm::SmallVector< mlir::Type, 0 > fields; // Empty fields for opaque struct

                // Create the struct identifier string for CIR
                std::string struct_id = "struct_" + struct_name;

                // Use CIR's StructType::get with proper signature
                auto structType = cir::StructType::get(
                    context, fields, mlir::StringAttr::get(context, struct_id),
                    /*packed=*/false,
                    /*padded=*/false, cir::StructType::RecordKind::Struct,
                    cir::ASTRecordDeclInterface{} // Empty AST interface for now
                );

                return structType;
            }

            mlir::Type
            lookup_or_create_union_type(mlir::MLIRContext *context, std::string union_name) {
                // Similar to struct, create an opaque union type
                llvm::SmallVector< mlir::Type, 0 > fields;
                std::string union_id = "union_" + union_name;

                auto union_type = cir::StructType::get(
                    context, fields, mlir::StringAttr::get(context, union_id),
                    /*packed=*/false,
                    /*padded=*/false, cir::StructType::RecordKind::Union,
                    cir::ASTRecordDeclInterface{} // Empty AST interface
                );

                return union_type;
            }
        } // namespace

        /// Convert C-like type names to CIR types
        mlir::Type convertCTypesToCIRTypes(mlir::MLIRContext *context, std::string type_name) {
            mlir::OpBuilder builder(context);

            // Remove whitespace and normalize
            std::string normalized_type = type_name;
            normalized_type = std::regex_replace(normalized_type, std::regex("\\s+"), "");

            // Handle pointer types (e.g., "int*", "char*", "void*")
            if (normalized_type.back() == '*') {
                std::string base_type = normalized_type.substr(0, normalized_type.length() - 1);
                mlir::Type base_cir_type = convertCTypesToCIRTypes(context, base_type);
                if (base_cir_type) {
                    return cir::PointerType::get(base_cir_type);
                }
                return nullptr;
            }

            // Handle array types (e.g., "int[10]", "char[256]")
            std::regex array_regex(R"((\w+)\[(\d+)\])");
            std::smatch array_match;
            if (std::regex_match(normalized_type, array_match, array_regex)) {
                std::string element_type = array_match[1].str();
                uint64_t array_size = 0;
                try {
                    array_size = static_cast< uint64_t >(std::stoull(array_match[2].str()));
                } catch (const std::invalid_argument&) {
                    return nullptr;
                } catch (const std::out_of_range&) {
                    return nullptr;
                }

                mlir::Type element_cir_type = convertCTypesToCIRTypes(context, element_type);
                if (element_cir_type) {
                    return cir::ArrayType::get(context, element_cir_type, array_size);
                }
                return nullptr;
            }

            // Handle function pointer types (e.g., "int(*)(int,int)")
            std::regex func_ptr_regex(R"((\w+)\(\*\)\((.*)\))");
            std::smatch func_match;
            if (std::regex_match(normalized_type, func_match, func_ptr_regex)) {
                std::string return_type = func_match[1].str();
                std::string param_types = func_match[2].str();

                mlir::Type return_cir_type = convertCTypesToCIRTypes(context, return_type);
                if (!return_cir_type) {
                    return nullptr;
                }

                llvm::SmallVector< mlir::Type, 4 > param_cir_types;
                if (!param_types.empty() && param_types != "void") {
                    // Parse parameter types separated by commas
                    std::regex param_regex(R"([^,]+)");
                    std::sregex_iterator iter(
                        param_types.begin(), param_types.end(), param_regex
                    );
                    std::sregex_iterator end;

                    for (; iter != end; ++iter) {
                        std::string param_type = iter->str();
                        mlir::Type param_cir_type =
                            convertCTypesToCIRTypes(context, param_type);
                        if (!param_cir_type) {
                            return nullptr;
                        }
                        param_cir_types.push_back(param_cir_type);
                    }
                }

                auto func_type = cir::FuncType::get(param_cir_types, return_cir_type);
                return cir::PointerType::get(func_type);
            }

            // Basic integer types
            if (normalized_type == "void") {
                return cir::VoidType::get(context);
            }
            if (normalized_type == "char" || normalized_type == "signed char") {
                return cir::IntType::get(context, 8, true);
            }
            if (normalized_type == "unsigned char" || normalized_type == "uint8_t") {
                return cir::IntType::get(context, 8, false);
            }
            if (normalized_type == "short" || normalized_type == "signed short"
                || normalized_type == "short int")
            {
                return cir::IntType::get(context, 16, true);
            }
            if (normalized_type == "unsigned short" || normalized_type == "unsigned short int"
                || normalized_type == "uint16_t")
            {
                return cir::IntType::get(context, 16, false);
            }
            if (normalized_type == "int" || normalized_type == "signed int"
                || normalized_type == "signed")
            {
                return cir::IntType::get(context, 32, true);
            }
            if (normalized_type == "unsigned int" || normalized_type == "unsigned"
                || normalized_type == "uint32_t")
            {
                return cir::IntType::get(context, 32, false);
            }
            if (normalized_type == "long" || normalized_type == "signed long"
                || normalized_type == "long int")
            {
                return cir::IntType::get(context, sizeof(long) * 8, true); // Use platform-dependent size
            }
            if (normalized_type == "unsigned long" || normalized_type == "unsigned long int"
                || normalized_type == "uint64_t")
            {
                return cir::IntType::get(context, sizeof(long) * 8, false); // Use platform-dependent size
            }
            if (normalized_type == "long long" || normalized_type == "signed long long"
                || normalized_type == "long long int")
            {
                return cir::IntType::get(context, 64, true);
            }
            if (normalized_type == "unsigned long long"
                || normalized_type == "unsigned long long int")
            {
                return cir::IntType::get(context, 64, false);
            }

            // Standard integer types
            if (normalized_type == "int8_t") {
                return cir::IntType::get(context, 8, true);
            }
            if (normalized_type == "int16_t") {
                return cir::IntType::get(context, 16, true);
            }
            if (normalized_type == "int32_t") {
                return cir::IntType::get(context, 32, true);
            }
            if (normalized_type == "int64_t") {
                return cir::IntType::get(context, 64, true);
            }
            if (normalized_type == "size_t") {
                constexpr unsigned size_t_bits = sizeof(size_t) * 8;
                return cir::IntType::get(context, size_t_bits, false);
            }
            if (normalized_type == "uintptr_t") {
                constexpr unsigned uintptr_t_bits = sizeof(uintptr_t) * 8;
                return cir::IntType::get(context, uintptr_t_bits, false);
            }
            if (normalized_type == "ptrdiff_t") {
                constexpr unsigned ptrdiff_t_bits = sizeof(ptrdiff_t) * 8;
                return cir::IntType::get(context, ptrdiff_t_bits, true);
            }
            if (normalized_type == "ssize_t") {
                // ssize_t is typically the same size as ptrdiff_t
                constexpr unsigned ssize_t_bits = sizeof(ptrdiff_t) * 8;
                return cir::IntType::get(context, ssize_t_bits, true);
            }
            if (normalized_type == "intptr_t") {
                constexpr unsigned intptr_t_bits = sizeof(intptr_t) * 8;
                return cir::IntType::get(context, intptr_t_bits, true);
            }

            // Floating point types
            if (normalized_type == "float") {
                return cir::SingleType::get(context);
            }
            if (normalized_type == "double") {
                return cir::DoubleType::get(context);
            }
            if (normalized_type == "long double") {
                // For long double, we need to provide an underlying type (typically double)
                auto double_type = cir::DoubleType::get(context);
                return cir::LongDoubleType::get(context, double_type);
            }

            // Boolean type
            if (normalized_type == "bool" || normalized_type == "_Bool") {
                return cir::BoolType::get(context);
            }

            // Handle const/volatile qualifiers by stripping them
            if (normalized_type.find("const") == 0) {
                std::string base_type = normalized_type.substr(5);
                return convertCTypesToCIRTypes(context, base_type);
            }
            if (normalized_type.find("volatile") == 0) {
                std::string base_type = normalized_type.substr(8);
                return convertCTypesToCIRTypes(context, base_type);
            }

            // Handle struct types (e.g., "struct my_struct", "struct device")
            if (normalized_type.find("struct ") == 0) {
                std::string struct_name = normalized_type.substr(7);
                return lookup_or_create_struct_type(context, struct_name);
            }

            // Handle union types (e.g., "union my_union", "my_union_t")
            if (normalized_type.find("union ") == 0) {
                std::string union_name = normalized_type.substr(6);
                return lookup_or_create_union_type(context, union_name);
            }

            // Generic struct/type lookup for other naming patterns
            return lookup_or_create_struct_type(context, normalized_type);
        }

        /// Convert CIR type back to C-like type name string
        std::string convertCIRTypesToCTypes(mlir::Type cir_type) {
            if (auto void_type = mlir::dyn_cast< cir::VoidType >(cir_type)) {
                return "void";
            }

            if (auto int_type = mlir::dyn_cast< cir::IntType >(cir_type)) {
                unsigned width = int_type.getWidth();
                bool is_signed = int_type.isSigned();

                switch (width) {
                    case 8:
                        return is_signed ? "int8_t" : "uint8_t";
                    case 16:
                        return is_signed ? "int16_t" : "uint16_t";
                    case 32:
                        return is_signed ? "int32_t" : "uint32_t";
                    case 64:
                        return is_signed ? "int64_t" : "uint64_t";
                    default:
                        return is_signed ? ("int" + std::to_string(width) + "_t")
                                         : ("uint" + std::to_string(width) + "_t");
                }
            }

            if (auto ptr_type = mlir::dyn_cast< cir::PointerType >(cir_type)) {
                mlir::Type pointee_type       = ptr_type.getPointee();
                std::string pointee_type_name = convertCIRTypesToCTypes(pointee_type);
                return pointee_type_name + "*";
            }

            if (auto array_type = mlir::dyn_cast< cir::ArrayType >(cir_type)) {
                mlir::Type element_type       = array_type.getEltType();
                uint64_t size                 = array_type.getSize();
                std::string element_type_name = convertCIRTypesToCTypes(element_type);
                return element_type_name + "[" + std::to_string(size) + "]";
            }

            if (auto float_type = mlir::dyn_cast< cir::SingleType >(cir_type)) {
                return "float";
            }

            if (auto double_type = mlir::dyn_cast< cir::DoubleType >(cir_type)) {
                return "double";
            }

            if (auto long_double_type = mlir::dyn_cast< cir::LongDoubleType >(cir_type)) {
                return "long double";
            }

            if (auto bool_type = mlir::dyn_cast< cir::BoolType >(cir_type)) {
                return "bool";
            }

            if (auto struct_type = mlir::dyn_cast< cir::StructType >(cir_type)) {
                // Extract struct name from the CIR struct type
                if (auto name_attr = struct_type.getName()) {
                    return "struct " + name_attr.getValue().str();
                }
                return "struct <unnamed>";
            }

            if (auto func_type = mlir::dyn_cast< cir::FuncType >(cir_type)) {
                std::string return_type_name =
                    convertCIRTypesToCTypes(func_type.getReturnType());
                std::string result = return_type_name + "(";

                auto inputs = func_type.getInputs();
                for (size_t i = 0; i < inputs.size(); ++i) {
                    if (i > 0) {
                        result += ", ";
                    }
                    result += convertCIRTypesToCTypes(inputs[i]);
                }
                if (inputs.empty()) {
                    result += "void";
                }
                result += ")";
                return result;
            }

            // For unsupported types, return the MLIR type string representation
            std::string typeStr;
            llvm::raw_string_ostream os(typeStr);
            cir_type.print(os);
            return os.str();
        }
    } // namespace utils
} // namespace patchestry
