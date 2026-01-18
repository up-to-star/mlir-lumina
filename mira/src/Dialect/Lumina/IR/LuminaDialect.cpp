#include "Dialect/Lumina/IR/LuminaDialect.h"
#include "Dialect/Lumina/IR/LuminaDialect.cpp.inc"
#include "llvm/Support/raw_ostream.h"

namespace mlir::lumina {
void LuminaDialect::initialize() {
    llvm::outs() << "Initializing Lumina dialect\n"
                 << this->getDialectNamespace() << "\n";
    registerType();
}

LuminaDialect::~LuminaDialect() {
    llvm::outs() << "Destroying Lumina dialect\n";
}

void LuminaDialect::sayHello() {
    llvm::outs() << "Hello from Lumina dialect\n";
}
}  // namespace mlir::lumina