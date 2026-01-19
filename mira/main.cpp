#include <iostream>
#include <memory>
#include "Dialect/Lumina/IR/LuminaDialect.h"
#include "Dialect/Lumina/IR/LuminaEnums.h"
#include "Interfaces/DistributeParallelismInterfaces.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "Dialect/Lumina/IR/LuminaTypes.h"
#include "Dialect/Lumina/IR/LuminaAttrs.h"
#include "Dialect/Lumina/IR/LuminaOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Support/LLVM.h"

static const char* const KEntryPoint = "main";
static const char* const KDPAttrName = "dp_attr";

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

void test_ops() {
    mlir::DialectRegistry registry;
    mlir::MLIRContext context(registry);
    context.getOrLoadDialect<mlir::lumina::LuminaDialect>();
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = builder.create<mlir::ModuleOp>(loc, "lumina");
    builder.setInsertionPointToStart(module.getBody());
    auto f32 = mlir::Float32Type::get(&context);
    auto shape = mlir::SmallVector<int64_t>({2, 2});
    auto const_value_1 =
        mlir::SmallVector<llvm::APFloat>(4, llvm::APFloat((float)1));
    auto const_value_2 =
        mlir::SmallVector<llvm::APFloat>(4, llvm::APFloat((float)2));
    auto tensor_type_1 =
        mlir::lumina::LMTensorType::get(&context, shape, f32, 0);
    auto tensor_type_2 =
        mlir::lumina::LMTensorType::get(&context, shape, f32, 1);
    auto const_1 = builder.create<mlir::lumina::ConstOp>(
        loc, tensor_type_1,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                     const_value_1));
    auto const_2 = builder.create<mlir::lumina::ConstOp>(
        loc, tensor_type_1,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                     const_value_1));
    auto const_3 = builder.create<mlir::lumina::ConstOp>(
        loc, tensor_type_2,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                     const_value_2));
    auto const_4 = builder.create<mlir::lumina::ConstOp>(
        loc, tensor_type_2,
        mlir::DenseElementsAttr::get(mlir::RankedTensorType::get(shape, f32),
                                     const_value_2));

    llvm::outs() << "Const tensor in device 0:\n";
    const_1->dump();
    llvm::outs() << "Const tensor in device 1:\n";
    const_3->dump();
    auto buffer_op = builder.create<mlir::lumina::BufferOp>(
        loc, mlir::ValueRange({const_1, const_3}));
    llvm::outs() << "BufferOp:\n";
    buffer_op->dump();

    auto get_tensor_op_1 = mlir::lumina::GetTensorOp::create(
        builder, loc, tensor_type_1, buffer_op, 0);
    auto get_tensor_op_2 = mlir::lumina::GetTensorOp::create(
        builder, loc, tensor_type_2, buffer_op, 1);
    llvm::outs() << "GetTensorOp:\n";
    get_tensor_op_1->dump();
    get_tensor_op_2->dump();

    auto softmax_op =
        mlir::lumina::SoftmaxOp::create(builder, loc, get_tensor_op_1, 1);
    llvm::outs() << "SoftmaxOp:\n";
    softmax_op->dump();
    auto exp_op = mlir::lumina::ExpOp::create(builder, loc, get_tensor_op_2);
    llvm::outs() << "ExpOp:\n";
    exp_op->dump();

    auto out_buffer_op = mlir::lumina::BufferOp::create(
        builder, loc, mlir::ValueRange({const_2, const_4}));
    auto all_to_all_op = mlir::lumina::AllToAllOp::create(
        builder, loc, buffer_op, out_buffer_op);
    llvm::outs() << "AllToAllOp:\n";
    all_to_all_op->dump();
}

mlir::ModuleOp getModule(mlir::OpBuilder& builder) {
    auto loc = builder.getUnknownLoc();
    auto context = builder.getContext();
    auto module = builder.create<mlir::ModuleOp>(loc, "Lumina");

    builder.setInsertionPointToStart(module.getBody());
    auto f32 = mlir::Float32Type::get(context);
    auto dy_dim = 128;
    auto dy_shape = mlir::SmallVector<int64_t>({dy_dim, dy_dim, 24});
    auto dy_tensor_type =
        mlir::lumina::LMTensorType::get(context, dy_shape, f32, 0);
    auto func_type =
        mlir::FunctionType::get(context, {dy_tensor_type}, {dy_tensor_type});
    auto func = builder.create<mlir::func::FuncOp>(loc, KEntryPoint, func_type);
    func->setAttr(KDPAttrName,
                  mlir::lumina::DataParallelismAttr::get(context, 2));

    auto block = func.addEntryBlock();
    builder.setInsertionPointToStart(block);
    // Softmax Op
    mlir::Value softmax_op =
        builder.create<mlir::lumina::SoftmaxOp>(loc, block->getArgument(0), 1);
    softmax_op = builder.create<mlir::lumina::SoftmaxOp>(loc, softmax_op, 1);
    builder.create<mlir::func::ReturnOp>(loc, mlir::ValueRange{softmax_op});
    return module;
}

void test_interface() {
    mlir::DialectRegistry registry;
    mlir::MLIRContext context(registry);
    context.getOrLoadDialect<mlir::lumina::LuminaDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    auto f32 = mlir::Float32Type::get(&context);
    auto dim = mlir::ShapedType::kDynamic;
    auto shape = mlir::SmallVector<int64_t>({dim, dim, 24});
    auto tensor_type = mlir::lumina::LMTensorType::get(&context, shape, f32, 0);
    auto shaped_type = mlir::cast<mlir::ShapedType>(tensor_type);
    llvm::outs() << "tensor type: \t";
    tensor_type.dump();
    llvm::outs() << "shaped type: \t";
    shaped_type.dump();
    auto cloned_type = shaped_type.clone(f32);
    llvm::outs() << "cloned type: \t";
    cloned_type.dump();

    auto dp_attr = mlir::lumina::DataParallelismAttr::get(&context, 2);
    llvm::outs() << dp_attr.getAbstractAttribute().getName()
                 << " has mlir::DataParallelAttr interface: "
                 << dp_attr.getAbstractAttribute().hasInterface(
                        mlir::DistributeParallelAttr::getInterfaceID())
                 << "\n";
    llvm::outs()
        << dp_attr.getAbstractAttribute().getName()
        << " has mlir::DataParallelAttr interface: "
        << dp_attr.hasPromiseOrImplementsInterface<mlir::DataParallelAttr>()
        << "\n";
    mlir::OpBuilder builder(&context);
    auto loc = builder.getUnknownLoc();
    auto module = getModule(builder);
    module.dump();
    module.walk([](mlir::func::FuncOp func) {
        if (auto dp_attr = llvm::dyn_cast_or_null<mlir::DistributeParallelAttr>(
                func->getAttr(KDPAttrName))) {
            func.walk([&](mlir::Operation* op) {
                if (auto dis_op =
                        llvm::dyn_cast_or_null<mlir::DistributeParallelOp>(
                            op)) {
                    if (dis_op.applyDistributeParallelism(dp_attr)
                            .succeeded()) {
                        llvm::outs() << "Apply DataParallelism to "
                                     << op->getName() << "\n";
                        op->erase();
                    }
                }
            });
        }
    });
    module->dump();
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
    std::cout << "testing ops" << std::endl;
    test_ops();
    std::cout << "----------------------------------" << std::endl;
    std::cout << "testing interface" << std::endl;
    test_interface();
    std::cout << "----------------------------------" << std::endl;
    return 0;
}