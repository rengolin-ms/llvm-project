/* Copyright Microsoft Corp. 2020 */
#ifndef _PARSER_H_
#define _PARSER_H_

#include <cassert>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "AST.h"

namespace Knossos {
namespace AST {

//================================================ Tokeniser / Lexer

/// A token that has either value or children
/// Values are literals, variables, names, reserved words, types
/// Non-Values are lets, def/decl, ops, calls, control flow
///
/// Do not confuse with "continuation values", those are higher level.
struct Token {
  using Ptr = std::unique_ptr<Token>;
  Token(std::string str) : isValue(true), value(str) {}
  Token() : isValue(false) {}

  void addChild(Token::Ptr tok) {
    assert(!isValue && "Can't add children to values");
    children.push_back(std::move(tok));
  }
  llvm::ArrayRef<Ptr> getChildren() const {
    assert(!isValue && "No children in a value token");
    return children;
  }
  llvm::StringRef getValue() const {
    assert(isValue && "Not a value token");
    return value;
  }
  const Token *getChild(size_t idx) const {
    assert(!isValue && "No children in a value token");
    assert(idx < children.size() && "Offset error");
    return children[idx].get();
  }

  const Token *getHead() const {
    assert(!isValue && "No children in a value token");
    assert(children.size() > 0 && "No head");
    return children[0].get();
  }
  llvm::ArrayRef<Ptr> getTail() const {
    assert(!isValue && "No children in a value token");
    assert(children.size() > 1 && "No tail");
    return llvm::ArrayRef<Ptr>(children).slice(1);
  }

  const bool isValue;
  size_t size() const { return children.size(); }

  void dump(size_t tab = 0) const;

private:
  std::string value;
  std::vector<Ptr> children;
};

/// Tokenise the text into recursive tokens grouped by parenthesis.
///
/// The Lexer will pass the ownership of the Tokens to the Parser.
class Lexer {
  std::string code;
  size_t len;
  Token::Ptr root;
  size_t multiLineComments;

  /// Build a tree of tokens
  size_t lexToken(Token *tok, size_t pos);

public:
  Lexer(std::string &&code)
      : code(code), len(code.size()), root(new Token()), multiLineComments(0) {
    assert(len > 0 && "Empty code?");
  }

  Token::Ptr lex() {
    lexToken(root.get(), 0);
    assert(multiLineComments == 0);
    return std::move(root);
  }
};

//================================================ Parse Tokens into Nodes

/// Identify each token as an AST node and build it.
/// The parser will take ownership of the Tokens.
class Parser {
  Token::Ptr rootT;
  Expr::Ptr rootE;
  Lexer lex;

  // Lookup table of reserved operations and keywords
  // Don't use StringRef, as data() doesn't need to be null terminated.
  // FIXME: Knossos declares all of those, we need a runtime library
  const std::set<std::string> reservedOps{
      "add@ii", "sub@ii", "mul@ii", "div@ii",
      "add@ff", "sub@ff", "mul@ff", "div@ff"
  };
  bool isReservedOp(std::string name) const {
    return reservedOps.find(name) != reservedOps.end();
  }
  // Note: "get$i$N" is not on the list, as it needs special parsing
  const std::set<std::string > reservedWords{
      "let",   "edef",   "def",   "if", "build",
      "index", "size", "tuple", "fold",  "rule"
  };
  bool isReservedWord(std::string name) const {
    // FIXME: This should really be a regex
    if (llvm::StringRef(name).startswith("get$"))
        return true;
    return reservedWords.find(name) != reservedWords.end();
  }
  /// Simple symbol table for parsing only (no validation)
  struct Symbols {
    std::map<std::string, Expr*> Symbols;
    bool exists(std::string name) {
      return Symbols.find(name) != Symbols.end();
    }
    bool add(std::string name, Expr* val) {
      auto result = Symbols.insert({name, val});
      return result.second;
    }
    Expr* get(std::string name) {
      if (exists(name))
        return Symbols[name];
      return nullptr;
    }
  };
  Symbols functions;
  Symbols variables;
  Symbols rules;

  // Build AST nodes from Tokens
  Expr::Ptr parseToken(const Token *tok);
  // Specific Token parsers
  Type parseType(const Token *tok);
  Expr::Ptr parseBlock(const Token *tok);
  Expr::Ptr parseValue(const Token *tok);
  Expr::Ptr parseCall(const Token *tok);
  Expr::Ptr parseOperation(const Token *tok);
  Expr::Ptr parseVariable(const Token *tok);
  Expr::Ptr parseLet(const Token *tok);
  Expr::Ptr parseDecl(const Token *tok);
  Expr::Ptr parseDef(const Token *tok);
  Expr::Ptr parseCond(const Token *tok);
  Expr::Ptr parseBuild(const Token *tok);
  Expr::Ptr parseIndex(const Token *tok);
  Expr::Ptr parseSize(const Token *tok);
  Expr::Ptr parseTuple(const Token *tok);
  Expr::Ptr parseGet(const Token *tok);
  Expr::Ptr parseFold(const Token *tok);
  Expr::Ptr parseRule(const Token *tok);

public:
  Parser(std::string code)
      : rootT(nullptr), rootE(nullptr), lex(std::move(code)) {}

  void tokenise() {
    assert(!rootT && "Won't overwrite root token");
    rootT = lex.lex();
  }
  void parse() {
    assert(!rootE && "Won't overwrite root node");
    if (!rootT) tokenise();
    rootE = parseBlock(rootT.get());
  }
  const Token* getRootToken() {
    return rootT.get();
  }
  const Expr* getRootNode() {
    return rootE.get();
  }
  Expr::Ptr moveRoot() {
    return std::move(rootE);
  }
};

} // namespace AST
} // namespace Knossos
#endif /// _PARSER_H_
