/*
 * Copyright (c) 2026, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#include <patchestry/Ghidra/Target.hpp>

#include <algorithm>
#include <cctype>
#include <sstream>
#include <unordered_map>
#include <vector>

#include <llvm/ADT/StringRef.h>
#include <llvm/TargetParser/Triple.h>

#include <patchestry/Util/Log.hpp>

namespace patchestry::ghidra {
    namespace {
        llvm::Triple::SubArchType getSubArch(const std::string &variant) {
            static const std::unordered_map< std::string, llvm::Triple::SubArchType >
                variantMap = {
                    // ARM 32-bit variants
                    {          "v4t",            llvm::Triple::ARMSubArch_v4t },
                    {           "v5",             llvm::Triple::ARMSubArch_v5 },
                    {         "v5te",           llvm::Triple::ARMSubArch_v5te },
                    {           "v6",             llvm::Triple::ARMSubArch_v6 },
                    {         "v6t2",           llvm::Triple::ARMSubArch_v6t2 },
                    {          "v6k",            llvm::Triple::ARMSubArch_v6k },
                    {          "v6m",            llvm::Triple::ARMSubArch_v6m },
                    {           "v7",             llvm::Triple::ARMSubArch_v7 },
                    {          "v7a",             llvm::Triple::ARMSubArch_v7 },
                    {          "v7r",             llvm::Triple::ARMSubArch_v7 },
                    {          "v7m",            llvm::Triple::ARMSubArch_v7m },
                    {         "v7em",           llvm::Triple::ARMSubArch_v7em },
                    {           "v8",             llvm::Triple::ARMSubArch_v8 },
                    {          "v8a",             llvm::Triple::ARMSubArch_v8 },
                    {          "v8r",            llvm::Triple::ARMSubArch_v8r },
                    {          "v8m",   llvm::Triple::ARMSubArch_v8m_baseline },
                    {        "v8.1m", llvm::Triple::ARMSubArch_v8_1m_mainline },

                    // Thumb variants
                    {    "v4t_thumb",            llvm::Triple::ARMSubArch_v4t },
                    {   "v5te_thumb",           llvm::Triple::ARMSubArch_v5te },
                    {     "v6_thumb",             llvm::Triple::ARMSubArch_v6 },
                    {   "v6t2_thumb",           llvm::Triple::ARMSubArch_v6t2 },
                    {     "v7_thumb",             llvm::Triple::ARMSubArch_v7 },
                    {    "v7m_thumb",            llvm::Triple::ARMSubArch_v7m },

                    // Cortex-M specific variants
                    {       "Cortex",            llvm::Triple::ARMSubArch_v7m },

                    // AArch64 variants (ARM 64-bit)
                    {          "v8A",     llvm::Triple::AArch64SubArch_arm64e }, // Generic v8A
                    { "AppleSilicon",     llvm::Triple::AArch64SubArch_arm64e },
            };
            auto it = variantMap.find(variant);
            if (it != variantMap.end()) { return it->second; }
            return llvm::Triple::NoSubArch;
        }

    } // namespace

    std::string targetTriple(
        const std::string &lang, VariantPolicy policy, std::optional< std::string > architecture
    ) {
        llvm::Triple target_triple;

        // Utility function to split the language identifier (lang) string
        auto split_language = [](const std::string &lang_id,
                                 char delim = ':') -> std::vector< std::string > {
            std::vector< std::string > tokens;
            std::stringstream ss(lang_id);
            std::string token;

            while (std::getline(ss, token, delim)) { tokens.push_back(token); }
            return tokens;
        };

        // Ghidra export lang id in the format - arch:endianess:size:variant
        auto lang_vec = split_language(lang);
        if (lang_vec.size() < 3) {
            LOG(
                ERROR
            ) << "Error: Invalid language format. Expected 'arch:endianess:size:variant'.\n";
            return "";
        }

        const std::string &arch = architecture ? *architecture : lang_vec[0];
        int bit_size            = 0;
        if (llvm::StringRef(lang_vec[2]).getAsInteger(10, bit_size)) {
            LOG(ERROR) << "Invalid bit size in language id: " << lang_vec[2] << "\n";
            return "";
        }
        auto is_le = (lang_vec[1] == "LE");

        auto is_equal = [&](std::string astr, std::string bstr) -> bool {
            // transform both the string to lower-case and compare
            std::ranges::transform(astr, astr.begin(), [](unsigned char c) {
                return static_cast< char >(std::toupper(c));
            });
            std::ranges::transform(bstr, bstr.begin(), [](unsigned char c) {
                return static_cast< char >(std::toupper(c));
            });
            return astr == bstr;
        };

        if (is_equal(arch, "x86") || is_equal(arch, "x86-64")) {
            target_triple.setArch(bit_size == 32U ? llvm::Triple::x86 : llvm::Triple::x86_64);
        } else if (is_equal(arch, "ARM") || is_equal(arch, "AARCH64")) {
            target_triple.setArch(
                bit_size == 32U ? (is_le ? llvm::Triple::arm : llvm::Triple::armeb)
                                : (is_le ? llvm::Triple::aarch64 : llvm::Triple::aarch64_be),
                policy == VariantPolicy::Preserve && lang_vec.size() > 3
                    ? getSubArch(lang_vec[3])
                    : llvm::Triple::NoSubArch
            );
        }

        else if (is_equal(arch, "MIPS"))
        {
            target_triple.setArch(
                bit_size == 32U ? (is_le ? llvm::Triple::mipsel : llvm::Triple::mips)
                                : (is_le ? llvm::Triple::mips64el : llvm::Triple::mips64)
            );
        } else if (is_equal(arch, "POWERPC")) {
            target_triple.setArch(
                bit_size == 32U ? (is_le ? llvm::Triple::ppcle : llvm::Triple::ppc)
                                : (is_le ? llvm::Triple::ppc64le : llvm::Triple::ppc64)
            );
        } else {
            target_triple.setArch(llvm::Triple::UnknownArch);
        }

        target_triple.setVendor(llvm::Triple::UnknownVendor);
        target_triple.setOS(llvm::Triple::Linux);

        // Set environment (for specific cases)
        if (is_equal(arch, "ARM") && bit_size == 32) {
            target_triple.setEnvironment(llvm::Triple::GNUEABIHF); // Hard float ABI
        }

        return target_triple.str();
    }

} // namespace patchestry::ghidra
