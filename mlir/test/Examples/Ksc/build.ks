; RUN: ksc-mlir MLIR %s 2>&1 | FileCheck %s --check-prefix=MLIR
; RUN: ksc-mlir LLVM %s 2>&1 | FileCheck %s --check-prefix=LLVM

(def main Integer (argc : Integer) (
; MLIR: func @main(%arg0: i64) -> i64 {
; LLVM: define i64 @main(i64 %0) {

; Creating a 10-element vector of integers, from 0 to 9
(let (v (build 10 (lam (i : Integer) (add@ii i argc)))) (add@ii (size v) (index 5 v)))
; MLIR:   %[[vec:[0-9]+]] = alloc() : memref<10xi64>
; MLIR:   %[[zero:[ci_0-9]+]] = constant 0 : i64
; MLIR:   %[[ten:[ci_0-9]+]] = constant 10 : i64
; MLIR:   br ^[[headBB:bb[0-9]+]](%c0_i64 : i64)
; MLIR: ^[[headBB]](%[[ivHead:[0-9]+]]: i64):	// 2 preds: ^bb0, ^[[bodyBB:bb[0-9]+]]
; MLIR:   %[[cond:[0-9]+]] = cmpi "slt", %[[ivHead]], %[[ten]] : i64
; MLIR:   cond_br %[[cond]], ^[[bodyBB]](%[[ivHead]] : i64), ^[[tailBB:bb[0-9]+]]
; MLIR: ^[[bodyBB]](%[[ivBody:[0-9]+]]: i64):	// pred: ^[[headBB]]
; MLIR:   %[[expr:[0-9]+]] = addi %[[ivBody]], %arg0 : i64
; MLIR:   %[[idxW:[0-9]+]] = index_cast %[[ivBody]] : i64 to index
; MLIR:   store %[[expr]], %[[vec]][%[[idxW]]] : memref<10xi64>
; MLIR:   %[[one:[ci_0-9]+]] = constant 1 : i64
; MLIR:   %[[incr:[0-9]+]] = addi %[[ivBody]], %[[one]] : i64
; MLIR:   br ^[[headBB]](%[[incr]] : i64)
; MLIR: ^[[tailBB]]: // pred: ^[[headBB]]
; MLIR:   %[[dim:[0-9]+]] = dim %[[vec]], 0 : memref<10xi64>
; MLIR:   %[[dimcast:[0-9]+]] = index_cast %[[dim]] : index to i64
; MLIR:   %[[five:[ci_0-9]+]] = constant 5 : i64
; MLIR:   %[[idxR:[0-9]+]] = index_cast %c5_i64 : i64 to index
; MLIR:   %[[idx:[0-9]+]] = load %[[vec]][%[[idxR]]] : memref<10xi64>
; MLIR:   %[[ret:[0-9]+]] = addi %[[dimcast]], %[[idx]] : i64
; MLIR:   return %[[ret]] : i64

; LLVM:   %[[vec:[0-9]+]] = call i8* @malloc(i64 mul (i64 ptrtoint (i64* getelementptr (i64, i64* null, i64 1) to i64), i64 10))
; LLVM:   br label %[[headBB:[0-9]+]]
; LLVM: [[headBB]]:                                                ; preds = %[[bodyBB:[0-9]+]], %1
; LLVM:   %[[headPHI:[0-9]+]] = phi i64 [ %[[incr:[0-9]+]], %[[bodyBB]] ], [ 0, %1 ]
; LLVM:   %[[cond:[0-9]+]] = icmp slt i64 %[[headPHI]], 10
; LLVM:   br i1 %[[cond]], label %[[bodyBB]], label %[[tailBB:[0-9]+]]
; LLVM: [[bodyBB]]:                                               ; preds = %[[headBB]]
; LLVM:   %[[bodyPHI:[0-9]+]] = phi i64 [ %[[headPHI]], %[[headBB]] ]
; LLVM:   %[[expr:[0-9]+]] = add i64 %[[bodyPHI]], %0
; LLVM:   %[[ptrW:[0-9]+]] = getelementptr i64
; LLVM:   store i64 %[[expr]], i64* %[[ptrW]]
; LLVM:   %[[incr]] = add i64 %[[bodyPHI]], 1
; LLVM:   br label %[[headBB]]
; LLVM: [[tailBB]]:                                               ; preds = %[[headBB]]
; LLVM:   %[[ptrR:[0-9]+]] = getelementptr i64, i64* %{{.*}}, i64 5
; LLVM:   %[[idx:[0-9]+]] = load i64, i64* %[[ptrR]]
; LLVM:   %[[ret:[0-9]+]] = add i64 10, %[[idx]]
; LLVM:   ret i64 %[[ret]]

))
