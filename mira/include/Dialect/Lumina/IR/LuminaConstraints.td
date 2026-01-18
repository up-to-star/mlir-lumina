include "mlir/IR/CommonTypeConstraints.td"
include "mlir/IR/CommonAttrConstraints.td"

def AnyLMTensor : Type<And<[CPred<"::mlir::isa<::mlir::lumina::LMTensorType>($_self)">,
                            CPred<"::mlir::cast<::mlir::lumina::LMTensorType>($_self).getShape().size() > 0">]>>;

def AnyBuffer : Type<And<[CPred<"::mlir::isa<::mlir::lumina::BufferType>($_self)">,
                          CPred<"::mlir::cast<::mlir::lumina::BufferType>($_self).getDevices().size() > 0">]>>;