; RUN: ksc-mlir AST %s | FileCheck %s --check-prefix=AST
; RUN: ksc-mlir MLIR %s 2>&1 | FileCheck %s --check-prefix=MLIR
; RUN: ksc-mlir LLVM %s 2>&1 | FileCheck %s --check-prefix=LLVM

; TODO: Implement unary comparison operations

(edef eq Bool (Integer Integer))
(edef foo Integer (Integer))
(edef bar Integer (Integer))

; AST: Block:
; MLIR: module {
; LLVM: ModuleID = 'LLVMDialectModule'
; LLVM: source_filename = "LLVMDialectModule"

(def main Integer () (
; AST:       Definition:
; AST-NEXT:    name [main]
; AST-NEXT:    type [Integer]
; AST-NEXT:    Arguments:
; AST-NEXT:    Implementation:
; AST-NEXT:    Block:
; MLIR:      func @main() -> i64 {
; LLVM:      define i64 @main() {

; All literals, LLVM does not support, so we only lower the right block
  (if true 10 20)
; AST-NEXT:  Condition:
; AST-NEXT:    Literal:
; AST-NEXT:      value [true]
; AST-NEXT:      type [Bool]
; AST-NEXT:    True branch:
; AST-NEXT:      Literal:
; AST-NEXT:        value [10]
; AST-NEXT:        type [Integer]
; AST-NEXT:    False branch:
; AST-NEXT:      Literal:
; AST-NEXT:        value [20]
; AST-NEXT:        type [Integer]
; MLIR-NEXT:    %c10{{.*}} = constant 10 : i64
; LLVM optimises constants away

  (if false 10 20)
; AST-NEXT:  Condition:
; AST-NEXT:    Literal:
; AST-NEXT:      value [false]
; AST-NEXT:      type [Bool]
; AST-NEXT:    True branch:
; AST-NEXT:      Literal:
; AST-NEXT:        value [10]
; AST-NEXT:        type [Integer]
; AST-NEXT:    False branch:
; AST-NEXT:      Literal:
; AST-NEXT:        value [20]
; AST-NEXT:        type [Integer]
; MLIR-NEXT:    %c20{{.*}} = constant 20 : i64
; LLVM optimises constants away

; All constant expressions
  (if (eq 10 20) (foo 30) (bar 40))
; AST:       Condition:
; AST-NEXT:    Operation:
; AST-NEXT:      name [eq]
; AST-NEXT:      type [Bool]
; AST-NEXT:      Literal:
; AST-NEXT:        value [10]
; AST-NEXT:        type [Integer]
; AST-NEXT:      Literal:
; AST-NEXT:        value [20]
; AST-NEXT:        type [Integer]
; AST-NEXT:    True branch:
; AST-NEXT:      Operation:
; AST-NEXT:        name [foo]
; AST-NEXT:        type [Integer]
; AST-NEXT:        Literal:
; AST-NEXT:          value [30]
; AST-NEXT:          type [Integer]
; AST-NEXT:    False branch:
; AST-NEXT:      Operation:
; AST-NEXT:        name [bar]
; AST-NEXT:        type [Integer]
; AST-NEXT:        Literal:
; AST-NEXT:          value [40]
; AST-NEXT:          type [Integer]

; MLIR-NEXT:    %c30{{.*}} = constant 30 : i64
; MLIR-NEXT:    %[[foo1:[0-9]+]] = call @foo(%c30{{.*}}) : (i64) -> i64
; MLIR-NEXT:    %c40{{.*}} = constant 40 : i64
; MLIR-NEXT:    %[[bar1:[0-9]+]] = call @bar(%c40{{.*}}) : (i64) -> i64
; MLIR-NEXT:    %c10{{.*}} = constant 10 : i64
; MLIR-NEXT:    %c20{{.*}} = constant 20 : i64
; MLIR-NEXT:    %[[eq1:[0-9]+]] = call @eq(%c10{{.*}}, %c20{{.*}}) : (i64, i64) -> i1
; MLIR-NEXT:    %[[sel1:[0-9]+]] = select %[[eq1]], %[[foo1]], %[[bar1]] : i64

; LLVM-NEXT:    %[[foo1:[0-9]+]] = call i64 @foo(i64 30)
; LLVM-NEXT:    %[[bar1:[0-9]+]] = call i64 @bar(i64 40)
; LLVM-NEXT:    %[[eq1:[0-9]+]] = call i1 @eq(i64 10, i64 20)
; LLVM-NEXT:    %[[sel1:[0-9]+]] = select i1 %[[eq1]], i64 %[[foo1]], i64 %[[bar1]]


; Inside let, with variables
  (let (x (foo 50)) (if (eq x 60) (foo 70) (bar 80)))
; AST:       Condition:
; AST-NEXT:    Operation:
; AST-NEXT:      name [eq]
; AST-NEXT:      type [Bool]
; AST-NEXT:      Variable:
; AST-NEXT:        name [x]
; AST-NEXT:        type [Integer]
; AST-NEXT:      Literal:
; AST-NEXT:        value [60]
; AST-NEXT:        type [Integer]
; AST-NEXT:    True branch:
; AST-NEXT:      Operation:
; AST-NEXT:        name [foo]
; AST-NEXT:        type [Integer]
; AST-NEXT:        Literal:
; AST-NEXT:          value [70]
; AST-NEXT:          type [Integer]
; AST-NEXT:    False branch:
; AST-NEXT:      Operation:
; AST-NEXT:        name [bar]
; AST-NEXT:        type [Integer]
; AST-NEXT:        Literal:
; AST-NEXT:          value [80]
; AST-NEXT:          type [Integer]

; MLIR-NEXT:    %c50{{.*}} = constant 50 : i64
; MLIR-NEXT:    %[[foo_cond:[0-9]+]] = call @foo(%c50{{.*}}) : (i64) -> i64
; MLIR-NEXT:    %c70{{.*}} = constant 70 : i64
; MLIR-NEXT:    %[[foo2:[0-9]+]] = call @foo(%c70{{.*}}) : (i64) -> i64
; MLIR-NEXT:    %c80{{.*}} = constant 80 : i64
; MLIR-NEXT:    %[[bar2:[0-9]+]] = call @bar(%c80{{.*}}) : (i64) -> i64
; MLIR-NEXT:    %c60{{.*}} = constant 60 : i64
; MLIR-NEXT:    %[[eq2:[0-9]+]] = call @eq(%[[foo_cond]], %c60{{.*}}) : (i64, i64) -> i1
; MLIR-NEXT:    %[[sel2:[0-9]+]] = select %[[eq2]], %[[foo2]], %[[bar2]] : i64

; LLVM-NEXT:    %[[foo_cond:[0-9]+]] = call i64 @foo(i64 50)
; LLVM-NEXT:    %[[foo2:[0-9]+]] = call i64 @foo(i64 70)
; LLVM-NEXT:    %[[bar2:[0-9]+]] = call i64 @bar(i64 80)
; LLVM-NEXT:    %[[eq2:[0-9]+]] = call i1 @eq(i64 %[[foo_cond]], i64 60)
; LLVM-NEXT:    %[[sel2:[0-9]+]] = select i1 %[[eq2]], i64 %[[foo2]], i64 %[[bar2]]


))
; AST does not return anything
; MLIR: return %[[sel2]] : i64
; LLVM: ret i64 %[[sel2]]
