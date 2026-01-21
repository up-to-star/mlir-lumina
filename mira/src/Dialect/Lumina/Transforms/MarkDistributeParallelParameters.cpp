#include "Dialect/Lumina/IR/LuminaDialect.h"
#include "Dialect/Lumina/IR/LuminaAttrs.h"
#include "Dialect/Lumina/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/raw_ostream.h"
#include "Utils/Key.h"

#define DEBUG_TYPE "mark-distribute-parallel-parameters"

namespace mlir::lumina {
#define GEN_PASS_DEF_MARKDISTRIBUTEPARALLELPARAMETERSPASS
#include "Dialect/Lumina/Transforms/Passes.h.inc"
}  // namespace mlir::lumina

using namespace ::mlir;
using namespace ::mlir::lumina;

struct MarkDistributeParallelParametersPass
    : ::mlir::lumina::impl::MarkDistributeParallelParametersPassBase<
          MarkDistributeParallelParametersPass> {
    using MarkDistributeParallelParametersPassBase<
        MarkDistributeParallelParametersPass>::
        MarkDistributeParallelParametersPassBase;
    void runOnOperation() override;
};

void MarkDistributeParallelParametersPass::runOnOperation() {
    llvm::outs() << "run in: " << getPassName() << "\n";
    auto module = getOperation();
    llvm::outs() << "root op: " << module->getName() << "\n";
    llvm::outs() << "DPNums: " << DPNums << "\n";
    llvm::outs() << "TPNums: " << TPNums << "\n";
    llvm::outs() << "EPNums: " << EPNums << "\n";

    if (TPNums != 1) {
        llvm::errs() << "TPNums not supported currently\n";
    }
    if (DPNums != 1) {
        auto dp_attr = DataParallelismAttr::get(&getContext(), DPNums);
        module.walk(
            [&dp_attr](func::FuncOp op) { op->setAttr(KDPAttrName, dp_attr); });
    }
    llvm::outs() << "run out: " << getPassName() << "\n";
}