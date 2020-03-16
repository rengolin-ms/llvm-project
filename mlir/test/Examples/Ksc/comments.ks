; RUN: ksc-mlir AST %s | FileCheck %s --check-prefix=AST
; RUN: ksc-mlir MLIR %s 2>&1 | FileCheck %s --check-prefix=MLIR
; RUN: ksc-mlir LLVM %s 2>&1 | FileCheck %s --check-prefix=LLVM

; AST: Block:
; MLIR: module {
; LLVM: ModuleID = 'LLVMDialectModule'
; LLVM: source_filename = "LLVMDialectModule"

; This file should only have the main function returning 42.
; Everything else are comments of various types.

; Single line comment

#| single line multi-line comment |#

#| single #| line #| nested |# multi-line |# comment |#

#|
   multi line multi-line comment
|#

#| multi
  #| line
    #| nested
    |# multi-line
  |# comment
|#

(def main Integer #| inline multi-line comment |# () (42) ; Mid-line comment
; AST-NEXT:   Definition:
; AST-NEXT:     name [main]
; AST-NEXT:     type [Integer]
; AST-NEXT:     Arguments:
; AST-NEXT:     Implementation:
; AST-NEXT:       Block:
; AST-NEXT:         Literal:
; AST-NEXT:           value [42]
; AST-NEXT:           type [Integer]
; MLIR:       func @main() -> i64 {
; MLIR-NEXT:    %c42{{.*}} = constant 42 : i64
; MLIR-NEXT:    return %c42{{.*}} : i64
; LLVM:       define i64 @main() {
; LLVM-NEXT:    ret i64 42
