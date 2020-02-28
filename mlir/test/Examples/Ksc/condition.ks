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
; AST-NOT:   Condition:
; AST-NEXT:    Literal:
; AST-NEXT:      value [10]
; AST-NEXT:      type [Integer]
; MLIR-NEXT:    %c10{{.*}} = constant 10 : i64
; LLVM optimises constants away

  (if false 10 20)
; AST-NOT:   Condition:
; AST-NEXT:    Literal:
; AST-NEXT:      value [20]
; AST-NEXT:      type [Integer]
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

; MLIR-NEXT:    %c10{{.*}} = constant 10 : i64
; MLIR-NEXT:    %c20{{.*}} = constant 20 : i64
; MLIR-NEXT:    %[[eq:[0-9]+]] = call @eq(%c10{{.*}}, %c20{{.*}}) : (i64, i64) -> i1
; MLIR-NEXT:    cond_br %[[eq]], ^[[if:bb[0-9]+]], ^[[else:bb[0-9]+]]
; MLIR-NEXT:  ^[[if]]:
; MLIR-NEXT:    %c30{{.*}} = constant 30 : i64
; MLIR-NEXT:    %[[foo:[0-9]+]] = call @foo(%c30{{.*}}) : (i64) -> i64
; MLIR-NEXT:    br ^[[latch:bb[0-9]+]](%[[foo]] : i64)
; MLIR-NEXT:  ^[[else]]:
; MLIR-NEXT:    %c40{{.*}} = constant 40 : i64
; MLIR-NEXT:    %[[bar:[0-9]+]] = call @bar(%c40{{.*}}) : (i64) -> i64
; MLIR-NEXT:    br ^[[latch]](%[[bar]] : i64)
; MLIR-NEXT:  ^[[latch]](%[[last:[0-9]+]]: i64):

; LLVM-NEXT:    %[[eq:[0-9]+]] = call i1 @eq(i64 10, i64 20)
; LLVM-NEXT:    br i1 %[[eq]], label %[[if:[0-9]+]], label %[[else:[0-9]+]]
; LLVM:       [[if]]:
; LLVM-NEXT:    %[[foo:[0-9]+]] = call i64 @foo(i64 30)
; LLVM-NEXT:    br label %[[latch:[0-9]+]]
; LLVM:       [[else]]:
; LLVM-NEXT:    %[[bar:[0-9]+]] = call i64 @bar(i64 40)
; LLVM-NEXT:    br label %[[latch]]
; LLVM:       [[latch]]:
; LLVM-NEXT:    %[[last:[0-9]+]] = phi i64 [ %[[foo]], %[[if]] ], [ %[[bar]], %[[else]] ]


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
; MLIR-NEXT:    %[[fooeq:[0-9]+]] = call @foo(%c50{{.*}}) : (i64) -> i64
; MLIR-NEXT:    %c60{{.*}} = constant 60 : i64
; MLIR-NEXT:    %[[eq:[0-9]+]] = call @eq(%[[fooeq]], %c60{{.*}}) : (i64, i64) -> i1
; MLIR-NEXT:    cond_br %[[eq]], ^[[if:bb[0-9]+]], ^[[else:bb[0-9]+]]
; MLIR-NEXT:  ^[[if]]:
; MLIR-NEXT:    %c70{{.*}} = constant 70 : i64
; MLIR-NEXT:    %[[foo:[0-9]+]] = call @foo(%c70{{.*}}) : (i64) -> i64
; MLIR-NEXT:    br ^[[latch:bb[0-9]+]](%[[foo]] : i64)
; MLIR-NEXT:  ^[[else]]:
; MLIR-NEXT:    %c80{{.*}} = constant 80 : i64
; MLIR-NEXT:    %[[bar:[0-9]+]] = call @bar(%c80{{.*}}) : (i64) -> i64
; MLIR-NEXT:    br ^[[latch]](%[[bar]] : i64)
; MLIR-NEXT:  ^[[latch]](%[[last:[0-9]+]]: i64):

; LLVM-NEXT:    %[[fooeq:[0-9]+]] = call i64 @foo(i64 50)
; LLVM-NEXT:    %[[eq:[0-9]+]] = call i1 @eq(i64 %[[fooeq]], i64 60)
; LLVM-NEXT:    br i1 %[[eq]], label %[[if:[0-9]+]], label %[[else:[0-9]+]]
; LLVM:       [[if]]:
; LLVM-NEXT:    %[[foo:[0-9]+]] = call i64 @foo(i64 70)
; LLVM-NEXT:    br label %[[latch:[0-9]+]]
; LLVM:       [[else]]:
; LLVM-NEXT:    %[[bar:[0-9]+]] = call i64 @bar(i64 80)
; LLVM-NEXT:    br label %[[latch]]
; LLVM:       [[latch]]:
; LLVM-NEXT:    %[[last:[0-9]+]] = phi i64 [ %[[foo]], %[[if]] ], [ %[[bar]], %[[else]] ]


))
; AST does not return anything
; MLIR: return %[[last]] : i64
; LLVM: ret i64 %[[last]]
