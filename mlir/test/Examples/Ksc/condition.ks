; RUN: ksc-mlir AST %s | FileCheck %s --check-prefix=AST
; RUN: ksc-mlir MLIR %s 2>&1 | FileCheck %s --check-prefix=MLIR
; R-UN: ksc-mlir LLVM %s 2>&1 | FileCheck %s --check-prefix=LLVM

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
; MLIR-NEXT:    %c10_i64 = constant 10 : i64
; LLVM optimises this away

  (if false 10 20)
; AST-NOT:   Condition:
; AST-NEXT:    Literal:
; AST-NEXT:      value [20]
; AST-NEXT:      type [Integer]
; MLIR-NEXT:    %c20_i64 = constant 20 : i64
; LLVM optimises this away

  ; All expressions
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
; MLIR-NEXT:    %c10_i64_0 = constant 10 : i64
; MLIR-NEXT:    %c20_i64_1 = constant 20 : i64
; MLIR-NEXT:    %0 = call @eq(%c10_i64_0, %c20_i64_1) : (i64, i64) -> i1
; MLIR-NEXT:    cond_br %0, ^bb1, ^bb2
; MLIR-NEXT:  ^bb1:	// pred: ^bb0
; MLIR-NEXT:    %c30_i64 = constant 30 : i64
; MLIR-NEXT:    %1 = call @foo(%c30_i64) : (i64) -> i64
; MLIR-NEXT:    br ^bb3(%1 : i64)
; MLIR-NEXT:  ^bb2:	// pred: ^bb0
; MLIR-NEXT:    %c40_i64 = constant 40 : i64
; MLIR-NEXT:    %2 = call @bar(%c40_i64) : (i64) -> i64
; MLIR-NEXT:    br ^bb3(%2 : i64)
; MLIR-NEXT:  ^bb3(%3: i64):	// 2 preds: ^bb1, ^bb2
; LLVM:  %1 = call i1 @eq(i64 10, i64 20)
; LLVM:  br i1 %1, label %2, label %4
; LLVM:
; LLVM:2:                                                ; preds = %0
; LLVM:  %3 = call i64 @foo(i64 30)
; LLVM:  br label %6
; LLVM:
; LLVM:4:                                                ; preds = %0
; LLVM:  %5 = call i64 @bar(i64 40)
; LLVM:  br label %6
; LLVM:
; LLVM:6:                                                ; preds = %4, %2



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
; MLIR-NEXT:    %c50_i64 = constant 50 : i64
; MLIR-NEXT:    %4 = call @foo(%c50_i64) : (i64) -> i64
; MLIR-NEXT:    %c60_i64 = constant 60 : i64
; MLIR-NEXT:    %5 = call @eq(%4, %c60_i64) : (i64, i64) -> i1
; MLIR-NEXT:    cond_br %5, ^bb4, ^bb5
; MLIR-NEXT:  ^bb4:	// pred: ^bb3
; MLIR-NEXT:    %c70_i64 = constant 70 : i64
; MLIR-NEXT:    %6 = call @foo(%c70_i64) : (i64) -> i64
; MLIR-NEXT:    br ^bb6(%6 : i64)
; MLIR-NEXT:  ^bb5:	// pred: ^bb3
; MLIR-NEXT:    %c80_i64 = constant 80 : i64
; MLIR-NEXT:    %7 = call @bar(%c80_i64) : (i64) -> i64
; MLIR-NEXT:    br ^bb6(%7 : i64)
; MLIR-NEXT:  ^bb6(%8: i64):	// 2 preds: ^bb4, ^bb5
; LLVM:  %7 = call i64 @foo(i64 50)
; LLVM:  %8 = call i1 @eq(i64 %7, i64 60)
; LLVM:  br i1 %8, label %9, label %11
; LLVM:
; LLVM:9:                                                ; preds = %6
; LLVM:  %10 = call i64 @foo(i64 70)
; LLVM:  br label %13
; LLVM:
; LLVM:11:                                               ; preds = %6
; LLVM:  %12 = call i64 @bar(i64 80)
; LLVM:  br label %13
; LLVM:
; LLVM:13:                                               ; preds = %11, %9
; LLVM:  %14 = phi i64 [ %10, %9 ], [ %12, %11 ]


))
; AST does not return anything
; MLIR: return %8 : i64
; LLVM: ret i64 %14
