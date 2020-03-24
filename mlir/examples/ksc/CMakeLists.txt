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
target_link_libraries(ksc-mlir
  PRIVATE
    MLIRAllDialects
    MLIRAffineToStandard
    MLIRAnalysis
    MLIRIR
    MLIRLoopToStandard
    MLIRParser
    MLIRPass
    MLIRStandardToLLVM
    MLIRTargetLLVMIR
    MLIRTransforms
    )
