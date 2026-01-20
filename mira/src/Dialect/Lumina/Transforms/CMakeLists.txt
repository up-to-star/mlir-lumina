add_mlir_dialect_library(MLIRLuminaTransforms
    ApplyDistributeTransform.cpp
    MarkDistributeParallelParameters.cpp

    DEPENDS
    MLIRLuminaPassesIncGen

    LINK_LIBS PUBLIC
    MLIRLuminaDialect
)