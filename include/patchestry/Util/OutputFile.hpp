/*
 * Copyright (c) 2025, Trail of Bits, Inc.
 *
 * This source code is licensed in accordance with the terms specified in
 * the LICENSE file found in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <system_error>

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Process.h>
#include <llvm/Support/raw_ostream.h>

namespace patchestry::util {

    /// Create or truncate `path`, hand a stream on it to `print`, then flush
    /// and close the file.  Returns the first open, write or close error.
    ///
    /// The descriptor is opened and closed here instead of by
    /// `raw_fd_ostream`.  When the process was started with stdin, stdout or
    /// stderr closed, `open()` hands the file one of the descriptors 0 to 2.
    /// LLVM never takes ownership of those: `raw_fd_ostream::close()` asserts
    /// before it flushes, and the destructor leaves the descriptor open, so
    /// errors that only show up at close (ENOSPC, EDQUOT, NFS) are lost.
    template< typename Print >
    std::error_code write_output_file(
        const std::string &path, llvm::sys::fs::OpenFlags flags, Print &&print
    ) {
        int fd = -1;
        if (auto ec = llvm::sys::fs::openFileForWrite(
                path, fd, llvm::sys::fs::CD_CreateAlways, flags
            ))
        {
            return ec;
        }

        std::error_code write_ec;
        {
            llvm::raw_fd_ostream out(fd, /*shouldClose=*/false);
            print(out);
            out.flush();
            write_ec = out.error();
            // The destructor aborts on a stream that still carries an error.
            out.clear_error();
        }

        auto close_ec = llvm::sys::Process::SafelyCloseFileDescriptor(fd);
        return write_ec ? write_ec : close_ec;
    }

} // namespace patchestry::util
