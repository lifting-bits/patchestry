/*
 * Copyright (c) 2024, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include <llvm/Support/JSON.h>

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
            VT_WIDECHAR,
            VT_POINTER,
            VT_FUNCTION,
            VT_ARRAY,
            VT_STRUCT,
            VT_UNION,
            VT_ENUM,
            VT_TYPEDEF,
            VT_UNDEFINED,
            VT_VOID,
            VT_BITFIELD,
            VT_STRING
        };

        static VarnodeType::Kind ConvertToKind(const std::string &kind) {
            static const std::unordered_map< std::string, VarnodeType::Kind > kind_map = {
                {   "invalid",   VT_INVALID },
                {   "boolean",   VT_BOOLEAN },
                {   "integer",   VT_INTEGER },
                {     "float",     VT_FLOAT },
                {   "pointer",   VT_POINTER },
                {  "function",  VT_FUNCTION },
                {     "array",     VT_ARRAY },
                {    "struct",    VT_STRUCT },
                {     "union",     VT_UNION },
                {      "enum",      VT_ENUM },
                {   "typedef",   VT_TYPEDEF },
                { "undefined", VT_UNDEFINED },
                {      "void",      VT_VOID },
                {     "wchar",  VT_WIDECHAR },
                {  "bitfield",  VT_BITFIELD },
                {    "string",    VT_STRING }
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

        void SetKey(const std::string &key) { this->key = key; }

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

        bool is_signed = false;
    };

    // ArrayType
    struct ArrayType : public VarnodeType
    {
        ArrayType(std::string name, Kind kind, uint32_t size)
            : VarnodeType(name, kind, size), num_elements(0), element_type(nullptr) {}

        uint32_t GetElementCount(void) const { return num_elements; }

        std::shared_ptr< VarnodeType > GetElementType(void) const { return element_type; }

        void SetElementType(const std::shared_ptr< VarnodeType > &element) {
            element_type = element;
        }

        void SetElementCount(uint32_t count) { num_elements = count; }

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

        std::shared_ptr< VarnodeType > GetPointeeType() const { return pointee_type; }

        void SetPointeeType(const VarnodeType &pointee) {
            pointee_type = std::make_shared< VarnodeType >(pointee);
        }

        void SetPointeeType(const std::shared_ptr< VarnodeType > &pointee) {
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

        std::shared_ptr< VarnodeType > GetBaseType() const { return base_type; }

        void SetBaseType(const std::shared_ptr< VarnodeType > &base) { base_type = base; }

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

        std::string return_type_key;
        std::vector< std::string > param_type_keys;
        bool is_variadic = false;
        bool is_noreturn = false;
    };

    // EnumType
    struct EnumType : public VarnodeType
    {
        struct Constant
        {
            std::string name;
            int64_t value;
        };

        EnumType(std::string name, Kind kind, uint32_t size) : VarnodeType(name, kind, size) {}

        void AddConstant(std::string constant_name, int64_t constant_value) {
            constants.emplace_back(Constant{ std::move(constant_name), constant_value });
        }

        const std::vector< Constant > &GetConstants() const { return constants; }

      private:
        std::vector< Constant > constants;
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

        void AddComponents(std::string &name, const VarnodeType &type, uint32_t offset) {
            components.emplace_back(
                Component(name, offset, std::make_shared< VarnodeType >(type))
            );
        }

        std::vector< Component > GetComponents(void) const { return components; }

      private:
        std::vector< Component > components;
    };

    // BitFieldType
    struct BitFieldType : public VarnodeType
    {
        BitFieldType(std::string name, Kind kind, uint32_t size)
            : VarnodeType(name, kind, size) {}

        void SetBaseType(const std::shared_ptr< VarnodeType > &base) { base_type_ = base; }

        std::shared_ptr< VarnodeType > GetBaseType() const { return base_type_; }

        uint32_t bit_offset = 0;
        uint32_t bit_size   = 0;

      private:
        std::shared_ptr< VarnodeType > base_type_;
    };

    // StringType — represents a string data type (e.g. char[] with charset info)
    struct StringType : public VarnodeType
    {
        StringType(std::string name, Kind kind, uint32_t size)
            : VarnodeType(name, kind, size) {}

        std::string charset;
    };

} // namespace patchestry::ghidra
