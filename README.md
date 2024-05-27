# LISP Interpreter

This project is a LISP interpreter implemented in Python. It includes three main components: a tokenizer, a parser, and an evaluator, which allow the interpreter to process and execute LISP programs.

## Components

### 1. Tokenizer
The tokenizer takes a string input and converts it into a list of tokens. Tokens are the meaningful units in the syntax of the LISP programming language. The tokenizer handles:
- Identifying symbols, numbers, and keywords
- Managing parentheses to define structure

### 2. Parser
The parser takes the list of tokens produced by the tokenizer and generates a structured representation of the program.

### 3. Evaluator
The evaluator takes the output from the parser and executes the program. The evaluator handles:
- Interpreting the meaning of each expression and form
- Performing computations and function calls
- Managing environment and scope for variable bindings

## Usage

Here is a simple example of how to use the interpreter:

```
(define x 10)
(+ x 5)
```

1. **Tokenization**:
   ```
   ['(', 'define', 'x', '10', ')', '(', '+', 'x', '5', ')']
   ```

2. **Parsing**:
   ```
   (define x 10)
   (+ x 5)
   ```

3. **Evaluation**:
   ```
   15
   ```

To get started, please **clone the repository** and **run lab.py**
Then, **input your LISP program** and see the result.
