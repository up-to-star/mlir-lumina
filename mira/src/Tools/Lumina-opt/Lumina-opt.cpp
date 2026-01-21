#include "Dialect/Lumina/IR/LuminaDialect.h"
#include "Dialect/Lumina/Transforms/Passes.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/SourceMgr.h"

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/InitAllExtensions.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Config/mlir-config.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir-c/Debug.h"

int main(int argc, char* argv[]) {
    mlir::registerAllPasses();
    mlir::DialectRegistry registry;
    mlir::registerAllDialects(registry);
    registry.insert<mlir::lumina::LuminaDialect>();
    mlir::registerAllExtensions(registry);
    mlir::lumina::registerLuminaOptPasses();

    return mlir::asMainReturnCode(mlir::MlirOptMain(
        argc, argv, "lumina modular optimizer driver\n", registry));
}