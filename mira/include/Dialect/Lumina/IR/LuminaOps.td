include "Dialect/Lumina/IR/LuminaAttrs.td"
include "Dialect/Lumina/IR/LuminaConstraints.td"

class Lumina_Op<string mnemonic, list<Trait> traits = []>
    : Op<LuminaDialect, mnemonic, traits> {
    let summary = cppNamespace#opName#"Op";

    let description = "$_name op";
}

class Lumina_UnaryOp<string mnemonic, Type OperandType, Type resultType = OperandType, list<Trait> traits = [], dag attributes = (ins)> : 
    Lumina_Op<mnemonic,[DeclareOpInterfaceMethods<DistributeParallelOp>] # traits> {
    let arguments = !con((ins
        OperandType: $input),
        attributes
    );   

    let results = (outs
        resultType: $result);
}

class Lumina_BinaryOp<string mnemonic, Type OperandType, Type resultType = OperandType, list<Trait> traits = [], dag attributes = (ins)> :
    Lumina_Op<mnemonic, [DeclareOpInterfaceMethods<DistributeParallelOp>] # traits> {
    let arguments = !con((ins
        OperandType: $lhs,
        OperandType: $rhs),
        attributes
    );
    let results = (outs
        resultType: $result);
}

class Lumina_ElewiseUnaryOp<string mnemonic, Type OperandType, Type resultType = OperandType, list<Trait> traits = [], dag attributes = (ins)> :
    Lumina_UnaryOp<mnemonic, OperandType, resultType, [DeclareOpInterfaceMethods<SupportedDataParallelismOp>] # traits, attributes>;

class Lumina_ElewiseBinaryOp<string mnemonic, Type OperandType, Type resultType = OperandType, list<Trait> traits = [], dag attributes = (ins)> :
    Lumina_BinaryOp<mnemonic, OperandType, resultType, [DeclareOpInterfaceMethods<SupportedDataParallelismOp>] # traits, attributes>;

def Lumina_ConstOp : Lumina_Op<"const", []> {
    let arguments = (ins
        ElementsAttr: $value);

    let results = (outs
        AnyLMTensor: $result);
}

def Lumina_SoftmaxOp : Lumina_UnaryOp<"softmax", AnyLMTensor, AnyLMTensor, 
                    [DeclareOpInterfaceMethods<SupportedDataParallelismOp, ["applyDataParallelism", "supportedDataParallelism"]>], (ins I64Attr: $axis)> {
    let hasVerifier = 1;

    let builders = [
        OpBuilder<(ins "::mlir::Value": $input, "int64_t": $axis),
            [{
                $_state.addOperands(input);
                $_state.getOrAddProperties<Properties>().axis = $_builder.getI64IntegerAttr(axis);
                $_state.addTypes(input.getType());
            }]>
    ];
}

def Lumina_ExpOp : Lumina_ElewiseUnaryOp<"exp", AnyLMTensor> {
    let builders = [
        OpBuilder<(ins "::mlir::Value": $input), [{
            $_state.addOperands(input);
            $_state.addTypes(input.getType());
        }]>
    ];
}

def Lumina_AddOp : Lumina_ElewiseBinaryOp<"add", AnyLMTensor>;
def Lumina_SubOp : Lumina_ElewiseBinaryOp<"sub", AnyLMTensor>;
def Lumina_MulOp : Lumina_ElewiseBinaryOp<"mul", AnyLMTensor>;
def Lumina_DivOp : Lumina_ElewiseBinaryOp<"div", AnyLMTensor>;

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

def Lumina_BufferCastOp : Lumina_Op<"buffer_cast", []> {
    let description = "对数据切分的标记";
    let arguments = (ins
       Variadic<LMTensorOrBuffer>: $inputs,
       DistributeParallelAttr: $distribute_attr);
    let results = (outs
        Variadic<LMTensorOrBuffer>: $outputs);
    let hasVerifier = 1;
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