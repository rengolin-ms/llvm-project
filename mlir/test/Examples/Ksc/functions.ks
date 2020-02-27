; RUN: ksc-mlir AST %s | FileCheck %s --check-prefix=AST
; RUN: ksc-mlir MLIR %s 2>&1 | FileCheck %s --check-prefix=MLIR
; RUN: ksc-mlir LLVM %s 2>&1 | FileCheck %s --check-prefix=LLVM

; Definition without declaration
(edef fun Integer (Integer))
; AST:       Declaration:
; AST-NEXT:    name [fun]
; AST-NEXT:    type [Integer]
; AST-NEXT:    Types: [ Integer ]
; MLIR: func @fun(i64) -> i64
; LLVM: declare i64 @fun(i64 %0)

; Definition with declaration and use
(edef foo Float (Float))
; The AST retains both, MLIR/LLVM deduplicates
; AST:       Declaration:
; AST-NEXT:    name [foo]
; AST-NEXT:    type [Float]
; AST-NEXT:    Types: [ Float ]
; MLIR-NOT: func @foo(f64) -> f64
; LLVM-NOT: declare double @foo(double %0)

(def foo Float ((x : Float)) x)
; AST:       Definition:
; AST-NEXT:    name [foo]
; AST-NEXT:    type [Float]
; AST-NEXT:    Arguments:
; AST-NEXT:      Variable:
; AST-NEXT:        name [x]
; AST-NEXT:        type [Float]
; AST-NEXT:    Implementation:
; AST-NEXT:      Block:
; AST-NEXT:        Variable:
; AST-NEXT:          name [x]
; AST-NEXT:          type [Float]
; MLIR:       func @foo(%arg0: f64) -> f64 {
; MLIR-NEXT:    return %arg0 : f64
; MLIR-NEXT:  }
; LLVM:       define double @foo(double %0) {
; LLVM-NEXT:    ret double %0
; LLVM-NEXT:  }

; Direct declaration with use
(def bar Integer ((y : Integer)) (add@ii y 40))
; AST:       Definition:
; AST-NEXT:    name [bar]
; AST-NEXT:    type [Integer]
; AST-NEXT:    Arguments:
; AST-NEXT:      Variable:
; AST-NEXT:        name [y]
; AST-NEXT:        type [Integer]
; AST-NEXT:    Implementation:
; AST-NEXT:      Block:
; AST-NEXT:        Operation:
; AST-NEXT:          name [add@ii]
; AST-NEXT:          type [Integer]
; AST-NEXT:          Variable:
; AST-NEXT:            name [y]
; AST-NEXT:            type [Integer]
; AST-NEXT:          Literal:
; AST-NEXT:            value [40]
; AST-NEXT:            type [Integer]
; MLIR:       func @bar(%arg0: i64) -> i64 {
; MLIR-NEXT:    %c40_i64 = constant 40 : i64
; MLIR-NEXT:    %0 = addi %arg0, %c40_i64 : i64
; MLIR-NEXT:    return %0 : i64
; MLIR-NEXT:  }
; LLVM:       define i64 @bar(i64 %0) {
; LLVM-NEXT:    %2 = add i64 %0, 40
; LLVM-NEXT:    ret i64 %2
; LLVM-NEXT:  }

; Single variable can be bare
(def baz Integer (z : Integer) (add@ii z 50))
; AST:       Definition:
; AST-NEXT:    name [baz]
; AST-NEXT:    type [Integer]
; AST-NEXT:    Arguments:
; AST-NEXT:      Variable:
; AST-NEXT:        name [z]
; AST-NEXT:        type [Integer]
; AST-NEXT:    Implementation:
; AST-NEXT:      Block:
; AST-NEXT:        Operation:
; AST-NEXT:          name [add@ii]
; AST-NEXT:          type [Integer]
; AST-NEXT:          Variable:
; AST-NEXT:            name [z]
; AST-NEXT:            type [Integer]
; AST-NEXT:          Literal:
; AST-NEXT:            value [50]
; AST-NEXT:            type [Integer]
; MLIR:       func @baz(%arg0: i64) -> i64 {
; MLIR-NEXT:    %c50_i64 = constant 50 : i64
; MLIR-NEXT:    %0 = addi %arg0, %c50_i64 : i64
; MLIR-NEXT:    return %0 : i64
; MLIR-NEXT:  }
; LLVM:       define i64 @baz(i64 %0) {
; LLVM-NEXT:    %2 = add i64 %0, 50
; LLVM-NEXT:    ret i64 %2
; LLVM-NEXT:  }

; Main, testing calls to functions
(def main Integer () (
; AST:       Definition:
; AST-NEXT:    name [main]
; AST-NEXT:    type [Integer]
; AST-NEXT:    Arguments:
; AST-NEXT:    Implementation:
; AST-NEXT:      Block:
; MLIR: func @main() -> i64 {
; LLVM: define i64 @main() {

(foo 10.0)
; AST-NEXT:  Operation:
; AST-NEXT:    name [foo]
; AST-NEXT:    type [Float]
; AST-NEXT:    Literal:
; AST-NEXT:      value [10.0]
; AST-NEXT:      type [Float]
; MLIR-NEXT:  %cst = constant 1.000000e+01 : f64
; MLIR-NEXT:  %0 = call @foo(%cst) : (f64) -> f64

(bar (fun 30))
; AST-NEXT:  Operation:
; AST-NEXT:    name [bar]
; AST-NEXT:    type [Integer]
; AST-NEXT:    Operation:
; AST-NEXT:      name [fun]
; AST-NEXT:      type [Integer]
; AST-NEXT:      Literal:
; AST-NEXT:        value [30]
; AST-NEXT:        type [Integer]
; MLIR-NEXT:  %c30_i64 = constant 30 : i64
; MLIR-NEXT:  %1 = call @fun(%c30_i64) : (i64) -> i64
; MLIR-NEXT:  %2 = call @bar(%1) : (i64) -> i64
; LLVM-NEXT:  %1 = call i64 @fun(i64 30)
; LLVM-NEXT:  %2 = add i64 %1, 40

))
; AST does not return anything
; MLIR: return %2 : i64
; LLVM: ret i64 %2
