#include "Dialect/Lumina/IR/LuminaDialect.h"
#include "Dialect/Lumina/IR/LuminaAttrs.h"
#include "Dialect/Lumina/Transforms/Passes.h"
#include "Interfaces/DistributeParallelismInterfaces.h"
#include "Utils/Key.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorHandling.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#define DEBUG_TYPE "apply-distribute-transform"

namespace mlir::lumina {
#define GEN_PASS_DEF_APPLYDISTRIBUTETRANSFORMPASS
#include "Dialect/Lumina/Transforms/Passes.h.inc"
}  // namespace mlir::lumina

using namespace ::mlir;
using namespace ::mlir::lumina;

struct ApplyDistributeTransformPass
    : ::mlir::lumina::impl::ApplyDistributeTransformPassBase<
          ApplyDistributeTransformPass> {
    using ApplyDistributeTransformPassBase<
        ApplyDistributeTransformPass>::ApplyDistributeTransformPassBase;
    void runOnOperation() override;
};

void ApplyDistributeTransformPass::runOnOperation() {
    llvm::outs() << "run in: " << getPassName() << "\n";
    auto func = getOperation();
    llvm::outs() << "root op: " << func->getName() << "\n";
    auto dp_attr = llvm::dyn_cast_or_null<mlir::DistributeParallelAttr>(
        func->getAttr(KDPAttrName));
    if (!dp_attr) {
        llvm::errs() << "DP_attr not found\n";
    }
    func->walk([&dp_attr](mlir::Operation* op) {
        if (auto dis_op =
                llvm::dyn_cast_or_null<mlir::DistributeParallelOp>(op)) {
            if (dis_op.applyDistributeParallelism(dp_attr).succeeded()) {
                llvm::outs()
                    << "Apply DataParallelism to " << op->getName() << "\n";
            }
        }
    });
    llvm::outs() << "run out: " << getPassName() << "\n";
}

std::unique_ptr<::mlir::Pass>
mlir::lumina::createApplyDistributeTransformPass() {
    return std::make_unique<ApplyDistributeTransformPass>();
}
