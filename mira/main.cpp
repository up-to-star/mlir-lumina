#include <iostream>
#include <memory>
#include "Dialect/Lumina/IR/LuminaDialect.h"
#include "Dialect/Lumina/IR/LuminaEnums.h"
#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "Dialect/Lumina/IR/LuminaTypes.h"
#include "Dialect/Lumina/IR/LuminaAttrs.h"

void test_dialect() {
    mlir::DialectRegistry registry;
    mlir::MLIRContext context(registry);
    auto lumina_dialect =
        context.getLoadedDialect<mlir::lumina::LuminaDialect>();
    lumina_dialect->sayHello();
}

void test_builtin_types() {
    // 文件定义：llvm-project/mlir/include/mlir/IR/BuiltinTypes.td
    auto context = std::make_unique<mlir::MLIRContext>();

    // 浮点数，每种位宽和标准定义一个
    auto f32 = mlir::Float32Type::get(context.get());
    llvm::outs() << "F32类型 :\t";
    f32.dump();

    auto bf16 = mlir::BFloat16Type::get(context.get());
    llvm::outs() << "BF16类型 :\t";
    bf16.dump();

    // Index 类型，机器相关的整数类型
    auto index = mlir::IndexType::get(context.get());
    llvm::outs() << "Index 类型 :\t";
    index.dump();

    // 整数类型, 参数: 位宽&&有无符号
    auto i32 = mlir::IntegerType::get(context.get(), 32);
    llvm::outs() << "I32 类型 :\t";
    i32.dump();
    auto ui16 =
        mlir::IntegerType::get(context.get(), 16, mlir::IntegerType::Unsigned);
    llvm::outs() << "UI16 类型 :\t";
    ui16.dump();

    // 张量类型,表示的是数据，不会有内存的布局信息。
    auto static_tensor = mlir::RankedTensorType::get({1, 2, 3}, f32);
    llvm::outs() << "静态F32 张量类型 :\t";
    static_tensor.dump();
    // 动态张量
    auto dynamic_tensor =
        mlir::RankedTensorType::get({mlir::ShapedType::kDynamic, 2, 3}, f32);
    llvm::outs() << "动态F32 张量类型 :\t";
    dynamic_tensor.dump();

    // Memref类型：表示内存
    auto basic_memref = mlir::MemRefType::get({1, 2, 3}, f32);
    llvm::outs() << "静态F32 内存类型 :\t";
    basic_memref.dump();
    // 带有布局信息的内存

    auto stride_layout_memref = mlir::MemRefType::get(
        {1, 2, 3}, f32,
        mlir::StridedLayoutAttr::get(context.get(), 1, {6, 3, 1}));
    llvm::outs() << "连续附带布局信息的 F32 内存类型 :\t";
    stride_layout_memref.dump();
    // 使用affine 表示布局信息的内存
    auto affine_memref = mlir::MemRefType::get(
        {1, 2, 3}, f32,
        mlir::StridedLayoutAttr::get(context.get(), 1, {6, 3, 1})
            .getAffineMap());
    llvm::outs() << "连续附带 affine 布局信息的 F32 内存类型 :\t";
    affine_memref.dump();
    // 动态连续附带 affine 布局信息的内存
    auto dynamic_affine_memref = mlir::MemRefType::get(
        {mlir::ShapedType::kDynamic, 2, 3}, f32,
        mlir::StridedLayoutAttr::get(context.get(), 1,
                                     {mlir::ShapedType::kDynamic, 3, 1})
            .getAffineMap());
    llvm::outs() << "连续附带 affine 布局信息的动态 F32 内存类型 :\t";
    dynamic_affine_memref.dump();
    // 具有内存层级信息的内存
    auto L1_memref = mlir::MemRefType::get(
        {mlir::ShapedType::kDynamic, 2, 3}, f32,
        mlir::StridedLayoutAttr::get(context.get(), 1,
                                     {mlir::ShapedType::kDynamic, 3, 1})
            .getAffineMap(),
        1);
    llvm::outs() << "处于L1层级的 F32 内存类型 :\t";
    L1_memref.dump();
    // gpu 私有内存层级的内存
    context->getOrLoadDialect<mlir::gpu::GPUDialect>();
    auto gpu_memref = mlir::MemRefType::get(
        {mlir::ShapedType::kDynamic, 2, 3}, f32,
        mlir::StridedLayoutAttr::get(context.get(), 1,
                                     {mlir::ShapedType::kDynamic, 3, 1})
            .getAffineMap(),
        mlir::gpu::AddressSpaceAttr::get(context.get(),
                                         mlir::gpu::AddressSpace::Private));
    llvm::outs()
        << "连续附带 affine 布局信息的动态 F32 Gpu Private内存类型 :\t";
    gpu_memref.dump();

    // 向量类型,定长的一段内存
    auto vector_type = mlir::VectorType::get(3, f32);
    llvm::outs() << "F32 1D向量类型 :\t";
    vector_type.dump();

    auto vector_2D_type = mlir::VectorType::get({3, 3}, f32);
    llvm::outs() << "F32 2D向量类型 :\t";
    vector_2D_type.dump();
}

void test_mytypes() {
    mlir::DialectRegistry registry;
    mlir::MLIRContext context(registry);
    auto dialect = context.getOrLoadDialect<mlir::lumina::LuminaDialect>();
    dialect->sayHello();
    mlir::lumina::LMTensorType lm_tensor = mlir::lumina::LMTensorType::get(
        &context, {1, 2, 3}, mlir::Float32Type::get(&context), 3);
    llvm::outs() << "LMTensor 类型 :\t";
    lm_tensor.dump();
    mlir::lumina::LMTensorType dy_lm_tensor = mlir::lumina::LMTensorType::get(
        &context, {mlir::ShapedType::kDynamic, 2, 3},
        mlir::Float32Type::get(&context), 3);
    llvm::outs() << "动态LMTensor 类型 :\t";
    dy_lm_tensor.dump();
}

void test_builtin_attr() {
    auto context = std::make_unique<mlir::MLIRContext>();
    context->getOrLoadDialect<mlir::lumina::LuminaDialect>();

    // float attr
    auto f32_attr =
        mlir::FloatAttr::get(mlir::Float32Type::get(context.get()), 2);
    llvm::outs() << "F32 属性 :\t";
    f32_attr.dump();

    // integer attr
    auto i32_attr =
        mlir::IntegerAttr::get(mlir::IntegerType::get(context.get(), 32), 10);
    llvm::outs() << "I32 属性 :\t";
    i32_attr.dump();

    // stridelayout attr
    auto stride_layout_attr =
        mlir::StridedLayoutAttr::get(context.get(), 1, {6, 3, 1});
    llvm::outs() << "StrideLayout 属性 :\t";
    stride_layout_attr.dump();

    // string attr
    auto str_attr = mlir::StringAttr::get(context.get(), "hello MLIR!");
    llvm::outs() << "String 属性 :\t";
    str_attr.dump();

    // strref attr 符号 attribute
    auto str_ref_attr = mlir::SymbolRefAttr::get(str_attr);
    llvm::outs() << "SymbolRef 属性 :\t";
    str_ref_attr.dump();

    // type attr
    auto type_attr = mlir::TypeAttr::get(mlir::lumina::LMTensorType::get(
        context.get(), {1, 2, 3}, mlir::Float32Type::get(context.get())));
    llvm::outs() << "Type 属性 :\t";
    type_attr.dump();

    // unit attr
    auto unit_attr = mlir::UnitAttr::get(context.get());
    llvm::outs() << "Unit 属性 :\t";
    unit_attr.dump();

    // dense element attr
    auto i64_arr = mlir::DenseI64ArrayAttr::get(context.get(), {1, 2, 3});
    llvm::outs() << "I64 数组属性 :\t";
    i64_arr.dump();

    auto dense_attr = mlir::DenseElementsAttr::get(
        mlir::RankedTensorType::get({2, 2},
                                    mlir::Float32Type::get(context.get())),
        llvm::ArrayRef<float>{1, 2, 3, 4});
    llvm::outs() << "Dense 属性 :\t";
    dense_attr.dump();
}

void test_myattrs() {
    mlir::DialectRegistry registry;
    mlir::MLIRContext context(registry);
    auto dialect = context.getOrLoadDialect<mlir::lumina::LuminaDialect>();
    auto nchw = mlir::lumina::Layout::NCHW;
    llvm::outs() << "NCHW: " << mlir::lumina::stringifyLayout(nchw) << "\n";
    auto nchw_attr = mlir::lumina::LayoutAttr::get(&context, nchw);
    llvm::outs() << "NCHW 属性 :\t";
    nchw_attr.dump();
    llvm::outs() << "NCHW is channelLast: " << nchw_attr.isChannelLast()
                 << "\n";

    auto dp_attr = mlir::lumina::DataParallelismAttr::get(&context, 2);
    llvm::outs() << "DataParallelism 属性 :\t";
    dp_attr.dump();
}

int main() {
    std::cout << "testing dialect" << std::endl;
    test_dialect();
    std::cout << "----------------------------------" << std::endl;
    std::cout << "testing builtin types" << std::endl;
    test_builtin_types();
    std::cout << "----------------------------------" << std::endl;
    std::cout << "testing my types" << std::endl;
    test_mytypes();
    std::cout << "----------------------------------" << std::endl;
    std::cout << "testing builtin attr" << std::endl;
    test_builtin_attr();
    std::cout << "----------------------------------" << std::endl;
    std::cout << "testing my attr" << std::endl;
    test_myattrs();
    std::cout << "----------------------------------" << std::endl;
    return 0;
}