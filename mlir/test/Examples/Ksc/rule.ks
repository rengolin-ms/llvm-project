; RUN: ksc-mlir AST %s | FileCheck %s --check-prefix=AST
; Rules do not yet work on MLIR because they need a specific dialect
; Rules will never be lowered to LLVM IR

; Rules are global objects, do no need a main

(rule "mul2" (v : Float) (mul@ff v 2.0) (add@ff v v))
; AST:  Rule:
; AST-NEXT:    name [mul2]
; AST-NEXT:    type [none]
; AST-NEXT:    Variable:
; AST-NEXT:      Variable:
; AST-NEXT:        name [v]
; AST-NEXT:        type [Float]
; AST-NEXT:    Pattern:
; AST-NEXT:      Operation:
; AST-NEXT:        name [mul@ff]
; AST-NEXT:        type [Float]
; AST-NEXT:        Variable:
; AST-NEXT:          name [v]
; AST-NEXT:          type [Float]
; AST-NEXT:        Literal:
; AST-NEXT:          value [2.0]
; AST-NEXT:          type [Float]
; AST-NEXT:    Result:
; AST-NEXT:      Operation:
; AST-NEXT:        name [add@ff]
; AST-NEXT:        type [Float]
; AST-NEXT:        Variable:
; AST-NEXT:          name [v]
; AST-NEXT:          type [Float]
; AST-NEXT:        Variable:
; AST-NEXT:          name [v]
; AST-NEXT:          type [Float]
