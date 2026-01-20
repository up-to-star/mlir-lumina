#pragma once

#include <filesystem>
#include <system_error>

#include "llvm/Support/SMLoc.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OwningOpRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/MemoryBuffer.h"

namespace mlir::utils::file {
template <typename OpTy = Operation*>
inline llvm::LogicalResult PrintToFile(OpTy op, const char* file) {
    std::error_code error_code;
    auto file_dir = std::filesystem::path(file).parent_path();
    if (!std::filesystem::exists(file_dir)) {
        if (!std::filesystem::create_directory(file_dir)) {
            llvm::outs() << "Failed to create directory: " << file_dir << "\n";
            return llvm::failure();
        }
    }
    llvm::raw_fd_ostream file_stream(file, error_code);
    op->print(file_stream);
    llvm::outs() << "print " << op->getName() << "to " << file << "\n";
    return llvm::success();
}

template <typename OpTy = Operation*>
inline llvm::LogicalResult ParseFile(mlir::MLIRContext& context,
                                     mlir::OwningOpRef<OpTy>& module,
                                     const char* file) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileorErr =
        llvm::MemoryBuffer::getFileOrSTDIN(file);
    if (std::error_code ec = fileorErr.getError()) {
        llvm::outs() << "Failed to open file: " << file << " " << ec.message()
                     << "\n";
        return llvm::failure();
    }
    llvm::SourceMgr source_mgr;
    source_mgr.AddNewSourceBuffer(std::move(*fileorErr), llvm::SMLoc());
    module = mlir::parseSourceFile<OpTy>(source_mgr, &context);
    if (!module) {
        llvm::outs() << "Failed to parse file: " << file << "\n";
        return llvm::failure();
    }
    return llvm::success();
}

template <class OpTy = Operation*>
inline llvm::LogicalResult ParseStr(mlir::MLIRContext& context,
                                    mlir::OwningOpRef<mlir::ModuleOp>& module,
                                    const char* str) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getMemBuffer(str, "mlir_module");
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::outs() << "load ir string error!\n";
        return llvm::failure();
    }
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    module = mlir::parseSourceFile<OpTy>(sourceMgr, {&context, false});
    if (!module) {
        llvm::outs() << "parse ir string fatal error!";
        return llvm::failure();
    }
    return llvm::success();
}
}  // namespace mlir::utils::file