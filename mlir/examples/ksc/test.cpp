#include "MLIR.h"
#include "Parser.h"
#include <iostream>

using namespace Knossos::AST;
using namespace Knossos::MLIR;
using namespace std;

static int verbose = 0;

// ======================================================= Helpers

Expr::Ptr parse(const string &code) {
  Parser p(code);
  if (verbose > 2)
    cout << " -- Tokens\n";
  p.tokenise();
  if (verbose > 2)
    p.getRootToken()->dump();
  if (verbose > 1)
    cout << " -- AST\n";
  p.parse();
  if (verbose > 1)
    p.getRootNode()->dump();
  return p.moveRoot();
}

void build(const string &code, bool fromMLIR=false, bool emitLLVM=false) {
  Expr::Ptr tree;
  if (!fromMLIR)
    tree = parse(code);

  Generator g;
  if (verbose > 0)
    cout << " -- MLIR\n";
  mlir::ModuleOp module;
  if (fromMLIR)
    module = g.build(code);
  else
    module = g.build(tree.get());
  if (verbose > 0) {
    module.dump();
    cout << endl;
  }

  if (emitLLVM) {
    if (verbose > 0)
      cout << " -- LLVM\n";
    auto llvm = g.emitLLVM();
    if (verbose > 0) {
      llvm->dump();
      cout << endl;
    }
  }
}

// ======================================================= Lexer test

void test_lexer() {
  cout << "\n == test_lexer\n";
  Lexer l("(def f1 Integer ((x : Integer) (y : Integer)) (add@ii x y))");
  auto root = l.lex();
  if (verbose > 2) {
    cout << " -- Tokens\n";
    root->dump();
  }

  // Root can have many exprs, here only one
  assert(root->isValue == false);
  const Token *tok = root->getChild(0);
  // Kind is not a value and has 5 sub-exprs
  assert(tok->isValue == false);
  assert(tok->size() == 5);
  // Which are:
  assert(tok->getChild(0)->getValue() == "def");
  assert(tok->getChild(1)->getValue() == "f1");
  assert(tok->getChild(2)->getValue() == "Integer");
  const Token *args = tok->getChild(3);
  const Token *arg0 = args->getChild(0);
  assert(arg0->getChild(0)->getValue() == "x");
  assert(arg0->getChild(1)->getValue() == ":");
  assert(arg0->getChild(2)->getValue() == "Integer");
  const Token *arg1 = args->getChild(1);
  assert(arg1->getChild(0)->getValue() == "y");
  assert(arg1->getChild(1)->getValue() == ":");
  assert(arg1->getChild(2)->getValue() == "Integer");
  const Token *impl = tok->getChild(4);
  assert(impl->getChild(0)->getValue() == "add@ii");
  assert(impl->getChild(1)->getValue() == "x");
  assert(impl->getChild(2)->getValue() == "y");
  cout << "    OK\n";
}

// ======================================================= Parser test

void test_parser_block() {
  cout << "\n == test_parser_block\n";
  const Expr::Ptr tree = parse("(10.0 42 \"\" \" \" \"Hello\" \"Hello world\")");

  // Root can have many exprs, here only one
  Block* root = llvm::dyn_cast<Block>(tree.get());
  assert(root);
  // Kind is Block and has 1 sub-expr
  Block* block = llvm::dyn_cast<Block>(root->getOperand(0));
  assert(block);
  // Block has 6 literals
  Literal* op0 = llvm::dyn_cast<Literal>(block->getOperand(0));
  assert(op0);
  assert(op0->getValue() == "10.0");
  assert(op0->getType() == Type::Float);
  Literal* op1 = llvm::dyn_cast<Literal>(block->getOperand(1));
  assert(op1);
  assert(op1->getValue() == "42");
  assert(op1->getType() == Type::Integer);
  Literal* op2 = llvm::dyn_cast<Literal>(block->getOperand(2));
  assert(op2);
  assert(op2->getValue() == "");
  assert(op2->getType() == Type::String);
  Literal* op3 = llvm::dyn_cast<Literal>(block->getOperand(3));
  assert(op3);
  assert(op3->getValue() == " ");
  assert(op3->getType() == Type::String);
  Literal* op4 = llvm::dyn_cast<Literal>(block->getOperand(4));
  assert(op4);
  assert(op4->getValue() == "Hello");
  assert(op4->getType() == Type::String);
  Literal* op5 = llvm::dyn_cast<Literal>(block->getOperand(5));
  assert(op5);
  assert(op5->getValue() == "Hello world");
  assert(op5->getType() == Type::String);
  cout << "    OK\n";
}

void test_parser_let() {
  cout << "\n == test_parser_let\n";
  const Expr::Ptr tree = parse("(let (x 10) (add@ii x 10))");

  // Root can have many exprs, here only one
  Block* root = llvm::dyn_cast<Block>(tree.get());
  assert(root);
  // Kind is Let
  Let* def = llvm::dyn_cast<Let>(root->getOperand(0));
  assert(def);
  // Let has two parts: variable definitions and expression
  assert(def->getType() == Type::Integer);
  Variable* x = llvm::dyn_cast<Variable>(def->getVariable(0));
  assert(x->getType() == Type::Integer);
  Operation* expr = llvm::dyn_cast<Operation>(def->getExpr());
  assert(expr);
  assert(expr->getName() == "add@ii");
  assert(expr->getType() == Type::Integer);
  auto var = llvm::dyn_cast<Variable>(expr->getOperand(0));
  assert(var);
  assert(var->getName() == "x");
  assert(var->getType() == Type::Integer);
  auto lit = llvm::dyn_cast<Literal>(expr->getOperand(1));
  assert(lit);
  assert(lit->getValue() == "10");
  assert(lit->getType() == Type::Integer);
  cout << "    OK\n";
}

void test_parser_decl() {
  cout << "\n == test_parser_decl\n";
  const Expr::Ptr tree = parse("(edef fun Float (Integer String Bool))");

  // Root can have many exprs, here only one
  Block* root = llvm::dyn_cast<Block>(tree.get());
  assert(root);
  // Kind is Declaration
  Declaration* decl = llvm::dyn_cast<Declaration>(root->getOperand(0));
  assert(decl);
  // Declaration has 3 parts: name, return type, arg types decl
  assert(decl->getName() == "fun");
  assert(decl->getType() == Type::Float);
  assert(decl->getArgTypes()[0] == Type::Integer);
  assert(decl->getArgTypes()[1] == Type::String);
  assert(decl->getArgTypes()[2] == Type::Bool);
  cout << "    OK\n";
}

void test_parser_def() {
  cout << "\n == test_parser_def\n";
  const Expr::Ptr tree =
      parse("(def fun Integer ((x : Integer) (y : Integer)) (add@ii x 10))");

  // Root can have many exprs, here only one
  Block* root = llvm::dyn_cast<Block>(tree.get());
  assert(root);
  // Kind is Definition
  Definition* def = llvm::dyn_cast<Definition>(root->getOperand(0));
  assert(def);
  // Definition has 4 parts: name, return type, arg types def, expr
  assert(def->getName() == "fun");
  assert(def->getType() == Type::Integer);
  assert(def->size() == 2);
  Variable* x = llvm::dyn_cast<Variable>(def->getArgument(0));
  assert(x->getType() == Type::Integer);
  Block* body = llvm::dyn_cast<Block>(def->getImpl());
  assert(body);
  Operation* expr = llvm::dyn_cast<Operation>(body->getOperand(0));
  assert(expr);
  assert(expr->getName() == "add@ii");
  assert(expr->getType() == Type::Integer);
  auto var = llvm::dyn_cast<Variable>(expr->getOperand(0));
  assert(var);
  assert(var->getName() == "x");
  assert(var->getType() == Type::Integer);
  auto lit = llvm::dyn_cast<Literal>(expr->getOperand(1));
  assert(lit);
  assert(lit->getValue() == "10");
  assert(lit->getType() == Type::Integer);
  cout << "    OK\n";
}

void test_parser_decl_def_use() {
  cout << "\n == test_parser_decl_def_use\n";
  const Expr::Ptr tree = parse("(edef fun Integer (Integer))"
                               "(def fun Integer ((x : Integer)) (add@ii x 10))"
                               "(def main Integer () (add@ii (fun 10) 10)");

  // Root can have many exprs, here only 3
  Block* root = llvm::dyn_cast<Block>(tree.get());
  assert(root);
  // We're interested in the main call only
  Definition* main = llvm::dyn_cast<Definition>(root->getOperand(2));
  assert(main);
  assert(main->getName() == "main");
  assert(main->getType() == Type::Integer);
  assert(main->size() == 0);
  // And its implementation
  Block* body = llvm::dyn_cast<Block>(main->getImpl());
  assert(body);
  Operation* impl = llvm::dyn_cast<Operation>(body->getOperand(0));
  assert(impl);
  assert(impl->getName() == "add@ii");
  assert(impl->getType() == Type::Integer);
  // Arg1 is a call to fun
  Operation* call = llvm::dyn_cast<Operation>(impl->getOperand(0));
  assert(call);
  assert(call->getName() == "fun");
  assert(call->getType() == Type::Integer);
  auto arg0 = llvm::dyn_cast<Literal>(call->getOperand(0));
  assert(arg0);
  assert(arg0->getValue() == "10");
  assert(arg0->getType() == Type::Integer);
  // Arg2 is just a literal
  auto lit = llvm::dyn_cast<Literal>(impl->getOperand(1));
  assert(lit);
  assert(lit->getValue() == "10");
  assert(lit->getType() == Type::Integer);
  cout << "    OK\n";
}

void test_parser_cond() {
  cout << "\n == test_parser_cond\n";
  const Expr::Ptr tree = parse("(edef fun Integer (Integer))"
                               "(def fun Integer ((x : Integer)) (add@ii x 10))"
                               "(if (true) (fun 10) (add@ii 10 10))");

  // Root can have many exprs, here only 3
  Block* root = llvm::dyn_cast<Block>(tree.get());
  assert(root);
  // We're interested in the conditional only
  Condition* cond = llvm::dyn_cast<Condition>(root->getOperand(2));
  assert(cond);
  // Condition block is Bool true
  Block* c = llvm::dyn_cast<Block>(cond->getCond());
  assert(c);
  auto condVal = llvm::dyn_cast<Literal>(c->getOperand(0));
  assert(condVal);
  assert(condVal->getValue() == "true");
  assert(condVal->getType() == Type::Bool);
  // If block is "fun" call
  Operation* call = llvm::dyn_cast<Operation>(cond->getIfBlock());
  assert(call);
  assert(call->getName() == "fun");
  assert(call->getType() == Type::Integer);
  auto arg = llvm::dyn_cast<Literal>(call->getOperand(0));
  assert(arg);
  assert(arg->getValue() == "10");
  assert(arg->getType() == Type::Integer);
  // Else block is an "add" op
  Operation* expr = llvm::dyn_cast<Operation>(cond->getElseBlock());
  assert(expr);
  assert(expr->getName() == "add@ii");
  assert(expr->getType() == Type::Integer);
  auto op0 = llvm::dyn_cast<Literal>(expr->getOperand(0));
  assert(op0);
  assert(op0->getValue() == "10");
  assert(op0->getType() == Type::Integer);
  auto op1 = llvm::dyn_cast<Literal>(expr->getOperand(1));
  assert(op1);
  assert(op1->getValue() == "10");
  assert(op1->getType() == Type::Integer);
  cout << "    OK\n";
}

void test_parser_build() {
  cout << "\n == test_parser_build\n";
  const Expr::Ptr tree = parse("(build 10 (lam (i : Integer) (add@ii i i))))");

  // Root can have many exprs, here only one
  Block* root = llvm::dyn_cast<Block>(tree.get());
  assert(root);
  // Kind is Build
  Build* build = llvm::dyn_cast<Build>(root->getOperand(0));
  assert(build);
  // Build has three parts: range definition, induction variable and loop body
  assert(build->getType() == Type::Vector);
  assert(build->getType().getSubType() == Type::Integer);
  auto range = llvm::dyn_cast<Literal>(build->getRange());
  assert(range);
  assert(range->getValue() == "10");
  assert(range->getType() == Type::Integer);
  Variable* v = llvm::dyn_cast<Variable>(build->getVariable());
  assert(v->getType() == Type::Integer);
  Operation* expr = llvm::dyn_cast<Operation>(build->getExpr());
  assert(expr);
  assert(expr->getName() == "add@ii");
  assert(expr->getType() == Type::Integer);
  auto var = llvm::dyn_cast<Variable>(expr->getOperand(0));
  assert(var);
  assert(var->getName() == "i");
  assert(var->getType() == Type::Integer);
  var = llvm::dyn_cast<Variable>(expr->getOperand(1));
  assert(var);
  assert(var->getName() == "i");
  assert(var->getType() == Type::Integer);
  cout << "    OK\n";
}

void test_parser_index() {
  cout << "\n == test_parser_index\n";
  const Expr::Ptr tree = parse("(index 5 (build 10 (lam (i : Integer) (add@ii i i))))");

  // Root can have many exprs, here only one
  Block* root = llvm::dyn_cast<Block>(tree.get());
  assert(root);
  // Kind is Build
  Index* index = llvm::dyn_cast<Index>(root->getOperand(0));
  assert(index);
  // Build has two parts: index definition and vector
  assert(index->getType() == Type::Integer);
  auto i = llvm::dyn_cast<Literal>(index->getIndex());
  assert(i);
  assert(i->getValue() == "5");
  assert(i->getType() == Type::Integer);
  Expr* v = index->getVariable();
  assert(v->getType() == Type::Vector);
  assert(v->getType().getSubType() == Type::Integer);
  cout << "    OK\n";
}

// ======================================================= MLIR / LLVM IR tests

void test_llvm_ir() {
  cout << "\n == test_llvm_ir_from_ksc\n";
  build("(edef print Float (Float))"
        "(def fun Integer ((x : Integer) (y : Float))"
        "                 ((mul@ff y 1.5) (add@ii x 10)))"
        "(def main Integer () (fun 42 10.0)",
        /*from MLIR=*/false, /*to LLVM IR=*/true);
  cout << "    OK\n";

  cout << "\n == test_llvm_ir_from_mlir\n";
  build(
"module {"
"  func @print(f64) -> f64"
"  func @fun(%arg0: i64, %arg1: f64) -> i64 {"
"    %0 = \"std.constant\"() {value = 1.500000e+00 : f64} : () -> f64"
"    %1 = \"std.mulf\"(%arg1, %0) : (f64, f64) -> f64"
"    %2 = \"std.constant\"() {value = 10 : i64} : () -> i64"
"    %3 = \"std.addi\"(%arg0, %2) : (i64, i64) -> i64"
"    \"std.return\"(%3) : (i64) -> ()"
"  }"
"  func @main() -> i64 {"
"    %0 = \"std.constant\"() {value = 42 : i64} : () -> i64"
"    %1 = \"std.constant\"() {value = 1.000000e+01 : f64} : () -> f64"
"    %2 = \"std.call\"(%0, %1) {callee = @fun} : (i64, f64) -> i64"
"    \"std.return\"(%2) : (i64) -> ()"
"  }"
"}", /*from MLIR=*/true, /*to LLVM IR=*/true);
  cout << "    OK\n";
}


int test_all(int v=0) {
  verbose = v;

  test_lexer();

  test_parser_block();
  test_parser_let();
  test_parser_decl();
  test_parser_def();
  test_parser_decl_def_use();
  test_parser_cond();
  test_parser_build();
  test_parser_index();

  test_llvm_ir();

  cout << "\nAll tests OK\n";
  return 0;
}
