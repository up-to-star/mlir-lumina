set(LIBS
    ${dialect_libs}
    ${conversion_libs}
    ${extension_libs}

    MLIRAffineAnalysis
    MLIRAnalysis
    MLIRCastInterfaces
    MLIRDialect
    MLIROptLib
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRTransformUtils
    MLIRSupport
    MLIRIR
    MLIRFuncDialect
    MLIRTensorDialect
    MLIRArithDialect
    MLIRRegisterAllPasses
    MLIRRegisterAllDialects
    MLIRRegisterAllExtensions

    MLIRToLLVMIRTranslationRegistration
    MLIRLuminaDialect
    MLIRLuminaTransforms
    MLIRCAPIDebug
)

add_mlir_tool(Lumina-opt
    Lumina-opt.cpp

    DEPENDS 
    ${LIBS}
)
target_link_libraries(Lumina-opt PRIVATE ${LIBS})
llvm_update_compile_flags(Lumina-opt)
mlir_check_all_link_libraries(Lumina-opt)
export_executable_symbols_for_plugins(Lumina-opt)