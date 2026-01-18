include "mlir/IR/DialectBase.td"
include "mlir/IR/Traits.td"
include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/BuiltinTypeInterfaces.td"
include "Dialect/Lumina/IR/LuminaDialect.td"


class LuminaType<string name, string typeMnemonic, list<Trait> traits = [], string baseCppClass = "::mlir::Type">
     : TypeDef<LuminaDialect, name, traits, baseCppClass> {
        let mnemonic = typeMnemonic;
        let typeName = dialect.name # "." # typeMnemonic;
     }

def Lumina_TensorType : LuminaType<"LMTensor", "lm_tensor", []> {
    let summary = "Lumina Tensor Type";

    let description = "defines a tensor type with a given element type and shape and card_id";

    let parameters = (ins
        ArrayRefParameter<"int64_t">: $shape,
        "Type": $elementType,
        "int64_t": $device_id);

    let genStorageClass = 1;

    let hasStorageCustomConstructor = 0;

    let builders = [
        TypeBuilder<(ins
        "::mlir::ArrayRef<int64_t>": $shape,
        "::mlir::Type": $elementType), [{
            return $_get(elementType.getContext(), shape, elementType, 0);
        }]>
    ];

    let hasCustomAssemblyFormat = 1;

    // let assemblyFormat = "`<`$shape`,` $elementType`,` $device_id `>`";

    let skipDefaultBuilders = 0;

    let genVerifyDecl = 1;

    // let extraClassDeclaration = [{
    //     mlir::ShapedType cloneWith(std::optional<llvm::ArrayRef<int64_t>> shape, mlir::Type type);
    //     using ::mlir::ShapedType::Trait<LMTensorType>::getElementTypeBitWidth;
    //     using ::mlir::ShapedType::Trait<LMTensorType>::getRank;
    //     using ::mlir::ShapedType::Trait<LMTensorType>::getNumElements;
    //     using ::mlir::ShapedType::Trait<LMTensorType>::isDynamicDim;
    //     using ::mlir::ShapedType::Trait<LMTensorType>::hasStaticShape;
    //     using ::mlir::ShapedType::Trait<LMTensorType>::getNumDynamicDims;
    //     using ::mlir::ShapedType::Trait<LMTensorType>::getDimSize;
    //     using ::mlir::ShapedType::Trait<LMTensorType>::getDynamicDimIndex;

    //     LMTensorType clone(::mlir::ArrayRef<int64_t> shape, int64_t device_id) const;
    //     LMTensorType clone(::mlir::ArrayRef<int64_t> shape, ::mlir::Type elementType) const;
    //     LMTensorType clone(::mlir::ArrayRef<int64_t> shape) const;
    //     LMTensorType clone(::mlir::Type elementType) const;
    //     LMTensorType clone() const;
    // }];

}

def Lumina_BufferType : LuminaType<"Buffer", "buffer", []> {
    let summary = "Lumina Buffer Type";

    let description = "defines a buffer type";

    let parameters = (ins
        ArrayRefParameter<"int64_t">: $devices
    );

    let genStorageClass = 1;

    let hasStorageCustomConstructor = 0;

    let assemblyFormat = "`<`$devices`>`";

    let skipDefaultBuilders = 0;

    let genVerifyDecl = 1;
}