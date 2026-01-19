#include "Interfaces/DistributeParallelismInterfaces.h"

#include "Interfaces/DistributeParallelismAttrInterfaces.cpp.inc"
#include "Interfaces/DistributeParallelismOpInterfaces.cpp.inc"
#include "Dialect/Lumina/IR/LuminaTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Operation.h"