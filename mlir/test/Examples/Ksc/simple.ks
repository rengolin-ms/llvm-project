; RUN: ksc-mlir AST %s | FileCheck %s --check-prefix=AST
; RUN: ksc-mlir MLIR %s 2>&1 | FileCheck %s --check-prefix=MLIR
; RUN: ksc-mlir LLVM %s 2>&1 | FileCheck %s --check-prefix=LLVM

; AST: Block:
; MLIR: module {
; LLVM: ModuleID = 'LLVMDialectModule'
; LLVM: source_filename = "LLVMDialectModule"


(edef print Float (Float))
; AST-NEXT: Declaration:
; AST-NEXT: name [print]
; AST-NEXT: type [Float]
; AST-NEXT: Types: [ Float ]
; MLIR: func @print(f64) -> f64
; LLVM: declare double @print(double %0)

(def fun Integer ((x : Integer) (y : Float))
; AST-NEXT: Definition:
; AST-NEXT:   name [fun]
; AST-NEXT:   type [Integer]
; AST-NEXT:   Arguments:
; AST-NEXT:     Variable:
; AST-NEXT:       name [x]
; AST-NEXT:       type [Integer]
; AST-NEXT:     Variable:
; AST-NEXT:       name [y]
; AST-NEXT:       type [Float]
; MLIR: func @fun(%arg0: i64, %arg1: f64) -> i64 {
; LLVM: define i64 @fun(i64 %0, double %1) {


                 ((mul@ff y 1.5) (add@ii x 10)))
; AST-NEXT:     Implementation:
; AST-NEXT:       Block:
; AST-NEXT:         Operation:
; AST-NEXT:           name [mul@ff]
; AST-NEXT:           type [Float]
; AST-NEXT:           Variable:
; AST-NEXT:             name [y]
; AST-NEXT:             type [Float]
; AST-NEXT:           Literal:
; AST-NEXT:             value [1.5]
; AST-NEXT:             type [Float]
; AST-NEXT:         Operation:
; AST-NEXT:           name [add@ii]
; AST-NEXT:           type [Integer]
; AST-NEXT:           Variable:
; AST-NEXT:             name [x]
; AST-NEXT:             type [Integer]
; AST-NEXT:           Literal:
; AST-NEXT:             value [10]
; AST-NEXT:             type [Integer]
; MLIR-NEXT:  %cst = constant 1.500000e+00 : f64
; MLIR-NEXT:  %{{.*}} = mulf %arg1, %cst : f64
; MLIR-NEXT:  %c10{{.*}} = constant 10 : i64
; MLIR-NEXT:  %[[add:[0-9]+]] = addi %arg0, %c10{{.*}} : i64
; MLIR-NEXT:  return %[[add]] : i64
; LLVM-NEXT:  %{{.*}} = fmul double %1, 1.500000e+00
; LLVM-NEXT:  [[add:[0-9]+]] = add i64 %0, 10
; LLVM-NEXT:  ret i64 %[[add]]


(def main Integer () (fun 42 10.0) ; comment
; AST-NEXT:   Definition:
; AST-NEXT:     name [main]
; AST-NEXT:     type [Integer]
; AST-NEXT:     Arguments:
; AST-NEXT:     Implementation:
; AST-NEXT:       Block:
; AST-NEXT:         Operation:
; AST-NEXT:           name [fun]
; AST-NEXT:           type [Integer]
; AST-NEXT:           Literal:
; AST-NEXT:             value [42]
; AST-NEXT:             type [Integer]
; AST-NEXT:           Literal:
; AST-NEXT:             value [10.0]
; AST-NEXT:             type [Float]
; MLIR:       func @main() -> i64 {
; MLIR-NEXT:    %c42{{.*}} = constant 42 : i64
; MLIR-NEXT:    %cst = constant 1.000000e+01 : f64
; MLIR-NEXT:    %[[fun:[0-9]+]] = call @fun(%c42{{.*}}, %cst) : (i64, f64) -> i64
; MLIR-NEXT:    return %[[fun]] : i64
; LLVM:       define i64 @main() {
; LLVM-NEXT:    %[[fun:[0-9]+]] = call i64 @fun(i64 42, double 1.000000e+01)
; LLVM-NEXT:    ret i64 %[[fun]]
