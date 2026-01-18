#include <set>

#include "Dialect/Lumina/IR/LuminaDialect.h"
#include "Dialect/Lumina/IR/LuminaTypes.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Support/LLVM.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "Dialect/Lumina/IR/LuminaTypes.cpp.inc"

namespace mlir::lumina {
void LuminaDialect::registerType() {
    llvm::outs() << "Registering Lumina Types\n";
    addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/Lumina/IR/LuminaTypes.cpp.inc"
        >();
}

llvm::LogicalResult LMTensorType::verify(
    ::llvm::function_ref< ::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<int64_t> shape, Type elementType, int64_t device_id) {
    if (device_id < 0) {
        return emitError() << "Invalid device id";
    }
    if (!elementType.isIntOrFloat()) {
        return emitError() << "Invalid element type";
    }
    return success();
}

Type LMTensorType::parse(::mlir::AsmParser& parser) {
    // 首先尝试解析 < 符号，如果失败则返回空类型
    if (parser.parseLess()) {
        return Type();
    }
    // 解析维度列表，如果失败则返回空类型
    SmallVector<int64_t, 4> dimensions;
    if (parser.parseDimensionList(dimensions, true, true)) {
        return Type();
    }
    auto typeLoc = parser.getCurrentLocation();
    // 解析元素类型，如果失败则返回空类型
    Type elementType;
    if (parser.parseType(elementType)) {
        return Type();
    }
    if (parser.parseComma()) {
        return Type();
    }
    // 解析设备ID，如果失败则返回空类型
    int device_id = 0;
    if (parser.parseInteger(device_id)) {
        if (parser.parseGreater()) {
            return Type();
        }
    }
    return parser.getChecked<LMTensorType>(parser.getContext(), dimensions,
                                           elementType, device_id);
}

void LMTensorType::print(::mlir::AsmPrinter& printer) const {
    printer << "<";
    for (int64_t dim : getShape()) {
        if (dim < 0) {
            printer << "?" << "x";
        } else {
            printer << dim << "x";
        }
    }
    printer.printType(getElementType());
    printer << ",";
    printer << getDeviceId();
    printer << ">";
}

llvm::LogicalResult BufferType::verify(
    ::llvm::function_ref< ::mlir::InFlightDiagnostic()> emitError,
    ::llvm::ArrayRef<int64_t> devices) {
    if (std::set(devices.begin(), devices.end()).size() != devices.size()) {
        return emitError() << "Duplicate device ids";
    }
    for (auto id : devices) {
        if (id < 0) {
            return emitError() << "Invalid device id";
        }
    }
    return llvm::success();
}
}  // namespace mlir::lumina