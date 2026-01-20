#pragma once

#include <memory>
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

namespace mlir::lumina {
void populateBufferCastOpCanonicalizationPatterns(RewritePatternSet& patterns);

void populateDeviceRegionFusionPatterns(RewritePatternSet& patterns);

std::unique_ptr<::mlir::Pass> createApplyDistributeTransformPass();

#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "Dialect/Lumina/Transforms/Passes.h.inc"
}  // namespace mlir::lumina