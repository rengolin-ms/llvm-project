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

(def main Integer (argc : Integer) (
; AST:       Definition:
; AST-NEXT:    name [main]
; AST-NEXT:    type [Integer]
; AST-NEXT:    Arguments:
; AST-NEXT:      Variable:
; AST-NEXT:        name [argc]
; AST-NEXT:        type [Integer]
; AST-NEXT:          Implementation:
; MLIR: func @main(%arg0: i64) -> i64 {
; LLVM: define i64 @main(i64 %0) {

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
; LLVM:       %2 = call i64 @fun(i64 10)

; Nested lets (ex1)
(let (l1 (mul@ii argc argc))
; AST-NEXT:  Let:
; AST-NEXT:    type [Integer]
; AST-NEXT:    Variable:
; AST-NEXT:      name [l1]
; AST-NEXT:      type [Integer]
; AST-NEXT:      Operation:
; AST-NEXT:        name [mul@ii]
; AST-NEXT:        type [Integer]
; AST-NEXT:        Variable:
; AST-NEXT:          name [argc]
; AST-NEXT:          type [Integer]
; AST-NEXT:        Variable:
; AST-NEXT:          name [argc]
; AST-NEXT:          type [Integer]
; MLIR-NEXT:  %2 = muli %arg0, %arg0 : i64
; LLVM optimises it away

     (let (l2 (add@ii argc l1))
; AST-NEXT:    Let:
; AST-NEXT:      type [Integer]
; AST-NEXT:      Variable:
; AST-NEXT:        name [l2]
; AST-NEXT:        type [Integer]
; AST-NEXT:        Operation:
; AST-NEXT:          name [add@ii]
; AST-NEXT:          type [Integer]
; AST-NEXT:          Variable:
; AST-NEXT:            name [argc]
; AST-NEXT:            type [Integer]
; AST-NEXT:          Variable:
; AST-NEXT:            name [l1]
; AST-NEXT:            type [Integer]
; MLIR-NEXT:  %3 = addi %arg0, %2 : i64
; LLVM optimises it away

          (mul@ii l1 l2)))
; AST-NEXT:      Operation:
; AST-NEXT:        name [mul@ii]
; AST-NEXT:        type [Integer]
; AST-NEXT:        Variable:
; AST-NEXT:          name [l1]
; AST-NEXT:          type [Integer]
; AST-NEXT:        Variable:
; AST-NEXT:          name [l2]
; AST-NEXT:          type [Integer]
; MLIR-NEXT:  %4 = muli %2, %3 : i64
; LLVM optimises it away

; Multiple bind lets (ex2)
(let ((i argc) (j 20) (k 30)) (add@ii (mul@ii i j) k))
; AST-NEXT:  Let:
; AST-NEXT:    type [Integer]
; AST-NEXT:    Variable:
; AST-NEXT:      name [i]
; AST-NEXT:      type [Integer]
; AST-NEXT:      Variable:
; AST-NEXT:        name [argc]
; AST-NEXT:        type [Integer]
; AST-NEXT:    Variable:
; AST-NEXT:      name [j]
; AST-NEXT:      type [Integer]
; AST-NEXT:      Literal:
; AST-NEXT:        value [20]
; AST-NEXT:        type [Integer]
; AST-NEXT:    Variable:
; AST-NEXT:      name [k]
; AST-NEXT:      type [Integer]
; AST-NEXT:      Literal:
; AST-NEXT:        value [30]
; AST-NEXT:        type [Integer]
; AST-NEXT:    Operation:
; AST-NEXT:      name [add@ii]
; AST-NEXT:      type [Integer]
; AST-NEXT:      Operation:
; AST-NEXT:        name [mul@ii]
; AST-NEXT:        type [Integer]
; AST-NEXT:        Variable:
; AST-NEXT:          name [i]
; AST-NEXT:          type [Integer]
; AST-NEXT:        Variable:
; AST-NEXT:          name [j]
; AST-NEXT:          type [Integer]
; AST-NEXT:      Variable:
; AST-NEXT:        name [k]
; AST-NEXT:        type [Integer]
; MLIR-NEXT:  %c20_i64_1 = constant 20 : i64
; MLIR-NEXT:  %c30_i64_2 = constant 30 : i64
; MLIR-NEXT:  %5 = muli %arg0, %c20_i64_1 : i64
; MLIR-NEXT:  %6 = addi %5, %c30_i64_2 : i64
; LLVM-NEXT:  %3 = mul i64 %0, 20
; LLVM-NEXT:  %4 = add i64 %3, 30

))
; AST does not return anything
; MLIR: return %6 : i64
; LLVM: ret i64 %4
