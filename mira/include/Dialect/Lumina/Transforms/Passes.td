include "mlir/Pass/PassBase.td"

def MarkDistributeParallelParametersPass : Pass<"mark-distribute-parallel-parameters", "::mlir::ModuleOp"> {
    let summary = "Mark parameters as parallelizable.";
    let description = [{
        "标记全局并行参数";
    }];
    let dependentDialects = ["::mlir::lumina::LuminaDialect", "::mlir::tensor::TensorDialect"];
    let options = [
        Option<"DPNums", "DP", "std::int64_t", /*default=*/"1", "DPNums des">,
        Option<"TPNums", "TP", "std::int64_t", /*default=*/"1", "TPNums des">,
    ];
    let statistics = [
        Statistic<"EPNums", "ep-nums", "number of EP">
    ];
}

def ApplyDistributeTransformPass : Pass<"apply-distribute-transforms", "::mlir::func::FuncOp"> {
    let summary = "根据标记的并行参数进行变换";
    let description = [{
        "根据标记的并行参数进行变换";
    }];
    let dependentDialects = ["::mlir::lumina::LuminaDialect", "::mlir::tensor::TensorDialect"];
    let constructor = "mlir::lumina::createApplyDistributeTransformPass()";
}

def DeviceRegionFusionPass : Pass<"device-region-fusion", "::mlir::func::FuncOp"> {
    let summary = "相同device的计算合并";
    let description = [{
        "相同device的计算合并";
    }];
    let dependentDialects = ["::mlir::lumina::LuminaDialect", "::mlir::tensor::TensorDialect"];
}