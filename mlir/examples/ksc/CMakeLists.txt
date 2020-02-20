add_custom_target(Ksc)
set_target_properties(Ksc PROPERTIES FOLDER Examples)

macro(add_ksc_chapter name)
  add_dependencies(Ksc ${name})
  add_llvm_example(${name} ${ARGN})
endmacro(add_ksc_chapter name)

set(LLVM_LINK_COMPONENTS
  Core
  Support
  )

add_ksc_chapter(ksc-test
  test.cpp
  src/MLIR.cpp
  src/Parser.cpp
  include/AST.h
  include/MLIR.h
  include/Parser.h
  include/SymbolTable.h
  )

include_directories(include/)
include_directories(${CMAKE_CURRENT_BINARY_DIR})
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include/)
target_link_libraries(ksc-test
  PRIVATE
    MLIRAllDialects
    MLIRAffineToStandard
    MLIRAnalysis
    MLIRExecutionEngine
    MLIRIR
    MLIRLoopToStandard
    MLIRParser
    MLIRPass
    MLIRStandardToLLVM
    MLIRTargetLLVMIR
    MLIRTransforms
    )
