add_custom_target(Ksc)
set_target_properties(Ksc PROPERTIES FOLDER Examples)

macro(add_ksc_tool name)
  add_dependencies(Ksc ${name})
  add_llvm_example(${name} ${ARGN})
endmacro(add_ksc_tool name)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_ksc_tool(ksc-mlir
  main.cpp
  test.cpp
  src/MLIR.cpp
  src/Parser.cpp
  include/AST.h
  include/MLIR.h
  include/Parser.h
  )

include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
target_link_libraries(ksc-mlir
  PRIVATE
    ${dialect_libs}
    ${conversion_libs}
    MLIRAnalysis
    MLIRCallInterfaces
    MLIRExecutionEngine
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSideEffects
    MLIRTargetLLVMIR
    MLIRTransforms
    )
