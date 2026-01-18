#include "Dialect/Lumina/IR/LuminaOps.h"
#include "Dialect/Lumina/IR/LuminaDialect.h"
#include "Dialect/Lumina/IR/LuminaTypes.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/Casting.h"

#define GET_OP_CLASSES
#include "Dialect/Lumina/IR/LuminaOps.cpp.inc"

namespace mlir::lumina {
void LuminaDialect::registerOps() {
    llvm::outs() << "Registering Lumina ops\n";
    addOperations<
#define GET_OP_LIST
#include "Dialect/Lumina/IR/LuminaOps.cpp.inc"
        >();
}

llvm::LogicalResult GetTensorOp::verify() {
    auto device_id = getDeviceId();
    auto buffer = getBuffer();
    if (isa<BlockArgument>(buffer)) {
        auto buffer_type = cast<BufferType>(buffer.getType());
        auto device_ids = buffer_type.getDevices();
        for (auto id : device_ids) {
            if (id == device_id) {
                return llvm::success();
            }
        }
        return llvm::failure();
    }

    auto buffer_op = llvm::cast_or_null<BufferOp>(buffer.getDefiningOp());
    if (!buffer_op) {
        return llvm::failure();
    }
    for (auto tensor : buffer_op.getTensors()) {
        auto tensor_type = cast_or_null<LMTensorType>(tensor.getType());
        if (!tensor_type) {
            return llvm::failure();
        }
        if (device_id == tensor_type.getDeviceId()) {
            if (tensor_type != getType()) {
                return llvm::failure();
            }
            return llvm::success();
        }
    }
    return llvm::failure();
}

llvm::LogicalResult BufferOp::verify() {
    auto tensors = getTensors();
    auto devices = cast<BufferType>(getType()).getDevices();
    if (tensors.size() == 0) {
        return llvm::failure();
    }
    for (auto [index, device_id, tensor] : llvm::enumerate(devices, tensors)) {
        auto tensor_type = cast_or_null<LMTensorType>(tensor.getType());
        if (device_id != tensor_type.getDeviceId()) {
            return llvm::failure();
        }
    }
    return llvm::success();
}

llvm::LogicalResult SoftmaxOp::verify() {
    auto axis = getAxis();
    if (axis < 0) {
        return llvm::failure();
    }
    auto input_type = cast<LMTensorType>(getInput().getType());
    if (axis >= input_type.getShape().size()) {
        return llvm::failure();
    }
    return llvm::success();
}
}  // namespace mlir::lumina