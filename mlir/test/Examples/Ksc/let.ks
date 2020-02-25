; RUN: ksc-mlir AST %s | FileCheck %s --check-prefix=AST
; RUN: ksc-mlir MLIR %s 2>&1 | FileCheck %s --check-prefix=MLIR
; RUN: ksc-mlir LLVM %s 2>&1 | FileCheck %s --check-prefix=LLVM

; Forward declaration, for use below
(edef fun Integer (Integer))
; AST:       Declaration:
; AST-NEXT:    name [fun]
; AST-NEXT:    type [Integer]
; AST-NEXT:    Types: [ Integer ]
; MLIR

; Generally, Knossos does not need a main function, but to generate MLIR/LLVM
; we do, so instead of adding machinery to wrap the global context into a main
; function, we declare in the test, here. If in the future we add that
; functionality, this will not break, either.

(def main Integer () (
; AST:       Definition:
; AST-NEXT:    name [main]
; AST-NEXT:    type [Integer]
; AST-NEXT:    Arguments:
; AST-NEXT:    Implementation:
; MLIR: func @main() -> i64 {
; LLVM: define i64 @main() {

; Return the value of x
(let (x 10) x)
; AST:       Let:
; AST-NEXT:    type [Integer]
; AST-NEXT:    Variable:
; AST-NEXT:      name [x]
; AST-NEXT:      type [Integer]
; AST-NEXT:      Literal:
; AST-NEXT:        value [10]
; AST-NEXT:        type [Integer]
; AST-NEXT:    Variable:
; AST-NEXT:      name [x]
; AST-NEXT:      type [Integer]
; MLIR generation folds the variable if constant
; MLIR: %c10_i64 = constant 10 : i64
; LLVM optimises it away


; Call a function with y, return the value
(let (y 20) (add@ii y 30))
; AST:       Let:
; AST-NEXT:    type [Integer]
; AST-NEXT:    Variable:
; AST-NEXT:      name [y]
; AST-NEXT:      type [Integer]
; AST-NEXT:      Literal:
; AST-NEXT:        value [20]
; AST-NEXT:        type [Integer]
; AST-NEXT:    Operation:
; AST-NEXT:      name [add@ii]
; AST-NEXT:      type [Integer]
; AST-NEXT:      Variable:
; AST-NEXT:        name [y]
; AST-NEXT:        type [Integer]
; AST-NEXT:      Literal:
; AST-NEXT:        value [30]
; AST-NEXT:        type [Integer]
; MLIR generation folds the variable if constant
; MLIR:      %c20_i64 = constant 20 : i64
; MLIR-NEXT: %c30_i64 = constant 30 : i64
; MLIR-NEXT: %0 = addi %c20_i64, %c30_i64 : i64
; LLVM optimises it away

; Return the value of z, expanded from a function call
(let (z (fun 10)) z)
; AST:       Let:
; AST-NEXT:    type [Integer]
; AST-NEXT:    Variable:
; AST-NEXT:      name [z]
; AST-NEXT:      type [Integer]
; AST-NEXT:      Operation:
; AST-NEXT:        name [fun]
; AST-NEXT:        type [Integer]
; AST-NEXT:        Literal:
; AST-NEXT:          value [10]
; AST-NEXT:          type [Integer]
; AST-NEXT:    Variable:
; AST-NEXT:      name [z]
; AST-NEXT:      type [Integer]
; MLIR generation creates SSA value if not constant
; MLIR:       %c10_i64_0 = constant 10 : i64
; MLIR-NEXT:  %1 = call @fun(%c10_i64_0) : (i64) -> i64
; LLVM:       %1 = call i64 @fun(i64 10)

))
; AST does not return anything
; MLIR: return %1 : i64
; LLVM: ret i64 %1
