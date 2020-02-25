#include "MLIR.h"
#include "Parser.h"
#include <iostream>
#include <fstream>

using namespace Knossos::AST;
using namespace Knossos::MLIR;
using namespace std;

// Hack to get tests working
// FIXME: Use gtest or similar
int test_all(int v=0);

// Simple action based table
enum class Action {
  NONE,
  TEST,
  EMIT_AST,
  EMIT_MLIR,
  EMIT_LLVM
};

void help() {
  cout << "Unit Test Syntax: ksc-mlir TEST [-v(v(v))]\n";
  cout << " Compiler Syntax: ksc-mlir AST|MLIR|LLVM <filename.ks>\n";
}

int main(int argc, char **argv) {
  // Command line options
  if (argc < 2) {
    help();
    return 1;
  }

  // Action
  Action action = Action::NONE;
  string aStr(argv[1]);
  if (aStr == "TEST")
    action = Action::TEST;
  else if (aStr == "AST")
    action = Action::EMIT_AST;
  else if (aStr == "MLIR")
    action = Action::EMIT_MLIR;
  else if (aStr == "LLVM")
    action = Action::EMIT_LLVM;
  if (action == Action::NONE) {
    cout << "Invalid action!\n";
    help();
    return 1;
  }

  // FIXME: registering dialects must happen before building the context
  // Create a more logical API that doesn't require it to be done by the caller
  mlir::registerAllDialects();

  // Unit tests
  // FIXME: Use gtest or similar
  if (action == Action::TEST) {
    int v=0;
    if (argc > 2) {
      string arg(argv[2]);
      if (arg == "-v")
        v = 1;
      else if (arg == "-vv")
        v = 2;
      else if (arg == "-vvv")
        v = 3;
    }
    return test_all(v);
  }

  // Filename
  if (argc < 3) {
    help();
    return 1;
  }
  string filename(argv[2]);
  ifstream file(filename);
  if (!file.is_open()) {
    cout << "Invalid filename!\n";
    help();
    return 1;
  }

  // Read whole file into a string
  string code((istreambuf_iterator<char>(file)),istreambuf_iterator<char>());

  // Parse and output AST if requested
  Parser p(code);
  p.parse();
  if (action == Action::EMIT_AST) {
    p.getRootNode()->dump();
    return 0;
  }

  // Call generator and print output (MLIR/LLVM)
  Generator g;
  auto module = g.build(p.getRootNode());
  if (action == Action::EMIT_MLIR) {
    module.dump();
  } else if (action == Action::EMIT_LLVM) {
    g.emitLLVM()->dump();
  }

  return 0;
}
