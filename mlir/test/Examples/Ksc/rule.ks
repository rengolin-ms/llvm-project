; RUN: ksc-mlir AST %s | FileCheck %s --check-prefix=AST
; Rules do not yet work on MLIR because they need a specific dialect
; Rules will never be lowered to LLVM IR

; Rules are global objects, do no need a main

(rule "mul2" (v : Float) (mul@ff v 2.0) (add@ff v v))
; AST:  Rule:
; AST:    name [mul2]
; AST:    type [none]
; AST:    Variable:
; AST:      Variable:
; AST:        name [v]
; AST:        type [Float]
; AST:    Pattern:
; AST:      Operation:
; AST:        name [mul@ff]
; AST:        type [Float]
; AST:        Variable:
; AST:          name [v]
; AST:          type [Float]
; AST:        Literal:
; AST:          value [2.0]
; AST:          type [Float]
; AST:    Result:
; AST:      Operation:
; AST:        name [add@ff]
; AST:        type [Float]
; AST:        Variable:
; AST:          name [v]
; AST:          type [Float]
; AST:        Variable:
; AST:          name [v]
; AST:          type [Float]

