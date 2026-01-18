include "Dialect/Lumina/IR/LuminaAttrs.td"
include "Dialect/Lumina/IR/LuminaConstraints.td"

class Lumina_Op<string mnemonic, list<Trait> traits = []>
    : Op<LuminaDialect, mnemonic, traits> {
    let summary = cppNamespace#opName#"Op";

    let description = "$_name op";
}

class Lumina_UnaryOp<string mnemonic, Type OperandType, Type resultType = OperandType, list<Trait> traits = [], dag attributes = (ins)> : 
    Lumina_Op<mnemonic, traits#[]> {
    let arguments = !con((ins
        OperandType: $input),
        attributes
    );   

    let results = (outs
        resultType: $result);
}

class Lumina_BinaryOp<string mnemonic, Type OperandType, Type resultType = OperandType, list<Trait> traits = [], dag attributes = (ins)> :
    Lumina_Op<mnemonic, traits#[]> {
    let arguments = !con((ins
        OperandType: $lhs,
        OperandType: $rhs),
        attributes
    );
    let results = (outs
        resultType: $result);
}

def Lumina_ConstOp : Lumina_Op<"const", []> {
    let arguments = (ins
        ElementsAttr: $value);

    let results = (outs
        AnyLMTensor: $result);
}

def Lumina_SoftmaxOp : Lumina_UnaryOp<"softmax", AnyLMTensor, AnyLMTensor, [], (ins I64Attr: $axis)> {
    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "::mlir::Value": $input, "int64_t": $axis),
            [{
                $_state.addOperands(input);
                $_state.getOrAddProperties<Properties>().axis = $_builder.getIntegerAttr(odsBuilder.getIntegerType(64, true), axis);
                $_state.addTypes(input.getType());
            }]>
    ];
}

def Lumina_ExpOp : Lumina_UnaryOp<"exp", AnyLMTensor> {
    let builders = [
        OpBuilder<(ins "::mlir::Value": $input), [{
            $_state.addOperands(input);
            $_state.addTypes(input.getType());
        }]>
    ];
}

def Lumina_AddOp : Lumina_BinaryOp<"add", AnyLMTensor>;
def Lumina_SubOp : Lumina_BinaryOp<"sub", AnyLMTensor>;
def Lumina_MulOp : Lumina_BinaryOp<"mul", AnyLMTensor>;
def Lumina_DivOp : Lumina_BinaryOp<"div", AnyLMTensor>;

def Lumina_AllToAllOp : Lumina_Op<"all_to_all", []> {
    let arguments = (ins
        AnyBuffer: $input,
        AnyBuffer: $output
    );
}

def Lumina_BufferOp : Lumina_Op<"buffer", []> {
    let description = "将多个device_id的tensor合并成一个buffer";
    let arguments = (ins
        Variadic<AnyLMTensor>: $tensors
    );
    let results = (outs
        AnyBuffer: $result);

    let hasVerifier = 1;
    let builders = [
        OpBuilder<(ins "::mlir::ValueRange": $tensors),
        [{
            $_state.addOperands(tensors);
            ::llvm::SmallVector<int64_t> devices;
            for (auto tensor : tensors) {
                auto tensor_type = llvm::cast<::mlir::lumina::LMTensorType>(tensor.getType());
                devices.push_back(tensor_type.getDeviceId());
            }
            $_state.addTypes(::mlir::lumina::BufferType::get($_state.getContext(), devices));
        }]>
    ];
}

def Lumina_GetTensorOp : Lumina_Op<"get_tensor", []> {
    let description = "从buffer中获取tensor";
    let arguments = (ins
        AnyBuffer: $buffer,
        I64Attr: $device_id
    );
    let results = (outs
        AnyLMTensor: $result);

    let hasVerifier = 1;
}

def Lumina_PrintOp : Lumina_Op<"print", []> {
    let arguments = (ins
        AnyType: $input);
}