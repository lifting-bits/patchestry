/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include "llvm/Support/JSON.h"

namespace patchestry::ghidra {
    using JsonArray  = llvm::json::Array;
    using JsonObject = llvm::json::Object;
    using JsonValue  = llvm::json::Value;

    struct VarnodeType
    {
        enum Kind {
            VT_INVALID = 0,
            VT_BOOLEAN,
            VT_INTEGER,
            VT_FLOAT,
            VT_CHAR,
            VT_POINTER,
            VT_FUNCTION,
            VT_ARRAY,
            VT_STRUCT,
            VT_UNION,
            VT_ENUM,
            VT_TYPEDEF,
            VT_UNDEFINED,
            VT_VOID
        };

        static VarnodeType::Kind convertToKind(const std::string &kind) {
            static const std::unordered_map< std::string, VarnodeType::Kind > kind_map = {
                {     "bool",   VT_BOOLEAN},
                {  "integer",   VT_INTEGER},
                {    "float",     VT_FLOAT},
                {  "pointer",   VT_POINTER},
                { "function",  VT_FUNCTION},
                {    "array",     VT_ARRAY},
                {   "struct",    VT_STRUCT},
                {    "union",     VT_UNION},
                {     "enum",      VT_ENUM},
                {  "typedef",   VT_TYPEDEF},
                {"undefined", VT_UNDEFINED},
                {     "void",      VT_VOID}
            };

            // if kind is not present in the map, return vt_invalid
            auto iter = kind_map.find(kind);
            return iter != kind_map.end() ? iter->second : VT_INVALID;
        }

        VarnodeType() = default;

        VarnodeType(std::string &name, Kind kind, uint32_t size)
            : kind(kind), size(size), key({}), name(name) {}

        VarnodeType(const VarnodeType &)                = default;
        VarnodeType &operator=(const VarnodeType &)     = default;
        VarnodeType(VarnodeType &&) noexcept            = default;
        VarnodeType &operator=(VarnodeType &&) noexcept = default;
        virtual ~VarnodeType()                          = default;

        void set_key(const std::string &key) { this->key = key; }

        Kind kind{};
        uint32_t size{};
        std::string key;
        std::string name;
    };

    // BuiltinType
    struct BuiltinType : public VarnodeType
    {
        BuiltinType(std::string &name, Kind kind, uint32_t size)
            : VarnodeType(name, kind, size) {}
    };

    // ArrayType
    struct ArrayType : public VarnodeType
    {
        ArrayType(std::string name, Kind kind, uint32_t size)
            : VarnodeType(name, kind, size), num_elements(0), element_type(nullptr) {}

        uint32_t get_element_count(void) const { return num_elements; }

        std::shared_ptr< VarnodeType > get_element_type(void) const { return element_type; }

        void set_element_type(const std::shared_ptr< VarnodeType > &element) {
            element_type = element;
        }

        void set_element_count(uint32_t count) { num_elements = count; }

      private:
        uint32_t num_elements;
        std::shared_ptr< VarnodeType > element_type;
    };

    // PointerType
    struct PointerType : public VarnodeType
    {
        PointerType(
            std::string name, Kind kind, uint32_t size,
            std::shared_ptr< VarnodeType > pointee = nullptr
        )
            : VarnodeType(name, kind, size), pointee_type(std::move(pointee)) {}

        std::shared_ptr< VarnodeType > get_pointee_type() const { return pointee_type; }

        void set_pointee_type(const VarnodeType &pointee) {
            pointee_type = std::make_shared< VarnodeType >(pointee);
        }

        void set_pointee_type(const std::shared_ptr< VarnodeType > &pointee) {
            pointee_type = pointee;
        }

      private:
        std::shared_ptr< VarnodeType > pointee_type;
    };

    // TypedefType
    struct TypedefType : public VarnodeType
    {
        TypedefType(
            std::string name, Kind kind, uint32_t size,
            std::shared_ptr< VarnodeType > base = nullptr
        )
            : VarnodeType(name, kind, size), base_type(std::move(base)) {}

        std::shared_ptr< VarnodeType > get_base_type() const { return base_type; }

        void set_base_type(const std::shared_ptr< VarnodeType > &base) { base_type = base; }

      private:
        std::shared_ptr< VarnodeType > base_type;
    };

    // UndefinedType
    struct UndefinedType : public VarnodeType
    {
        UndefinedType(std::string name, Kind kind, uint32_t size)
            : VarnodeType(name, kind, size) {}
    };

    // FunctionType
    struct FunctionType : public VarnodeType
    {
        FunctionType(std::string name, Kind kind, uint32_t size)
            : VarnodeType(name, kind, size) {}
    };

    // EnumType
    struct EnumType : public VarnodeType
    {
        EnumType(std::string name, Kind kind, uint32_t size) : VarnodeType(name, kind, size) {}
    };

    // CompositeType
    struct CompositeType : public VarnodeType
    {
        struct Component
        {
            std::string name;
            uint32_t offset;
            std::shared_ptr< VarnodeType > type;
        };

        CompositeType(std::string name, Kind kind, uint32_t size)
            : VarnodeType(name, kind, size) {}

        void add_components(std::string &name, const VarnodeType &type, uint32_t offset) {
            components.emplace_back(
                Component(name, offset, std::make_shared< VarnodeType >(type))
            );
        }

        std::vector< Component > get_components(void) const { return components; }

      private:
        std::vector< Component > components;
    };

} // namespace patchestry::ghidra
