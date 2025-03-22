# rlox2

A dynamic programming language interpreter with a focus on simplicity and
extensibility. rlox2 features lexical scoping, first-class functions, closures,
and a rich standard library.

## Features

- Dynamic typing with support for numbers, strings, booleans, lists, and
  functions
- Lexical scoping with proper closure support and block expressions
- Expression-oriented design where blocks evaluate to their last expression
- Rich standard library with I/O, math, and list manipulation functions
- REPL with syntax highlighting, history, and multiline editing
- Error messages with colorful formatting and precise location reporting
- First-class functions and closures
- List data structure with comprehensive operations
- String manipulation utilities
- Built-in random number generation

## Installation

Make sure you have Rust installed on your system. Then:

```bash
git clone https://github.com/dbolivar25/rlox2.git
cd rlox2
cargo build --release # or cargo install --path .
```

The executable will be available at `target/release/rlox2`.

## Usage

rlox2 provides three main modes of operation:

### Running Files

```bash
rlox2 run path/to/script.rlox
```

### REPL Mode

```bash
rlox2 repl
```

The REPL provides:

- Command history (accessible with up/down arrows)
- Syntax highlighting
- Multi-line editing support
- Proper error reporting with location information

### Syntax Checking

```bash
rlox2 check path/to/script.rlox
```

## Language Syntax

### Variables and Basic Types

```rust
// Numbers
let x = 42;
let pi = 3.14159;

// Strings
let message = "Hello, World!";

// Booleans
let flag = true;
let empty = false;

// Lists
let numbers = [1, 2, 3, 4, 5];

// Nil
let nothing = nil;
```

### Block Expressions and Semicolons

In rlox2, blocks evaluate to their last expression, and all expressions except
the last in a block must be terminated with a semicolon. If a block has no final
expression without a terminating semicolon, it implicitly evaluates to nil:

```rust
// Block evaluating to 30
let result = {
    let x = 10;     // Semicolon required
    let y = 20;     // Semicolon required
    x + y           // Last expression, no semicolon needed
};

// Block evaluating to nil
let result = {
    let x = 10;     // Semicolon required
    let y = 20;     // All expressions terminated with semicolons
};                  // Implicitly evaluates to nil
```

### Control Flow

```rust
// If expressions
if x > 0 {
    println("Positive");     // Semicolon needed if there are more expressions
    x                       // Last expression, no semicolon needed
} else {
    println("Non-positive");
    -x
}

// While loops
let i = 0;
while i < 10 {
    println(i);        // Semicolon required
    i = i + 1         // Last expression, no semicolon needed
}
```

### Functions and Closures

```rust
// Function with block body
let greet = fn(name) {
    println("Hello, " <> name <> "!");   // Semicolon required
    "Greeting completed"               // Return value, no semicolon needed
};

// Recursive function using 'rec'
let rec fibonacci = fn(n) {
    if n <= 1 {
        n
    } else {
        fibonacci(n - 1) + fibonacci(n - 2)
    }
};

// Closure example
let make_counter = {
    let count = 0;
    fn() {
        println(count);     // Semicolon required
        count = count + 1   // Last expression
    }
};
```

### Built-in Functions

#### I/O Functions

- `print(value)`: Print a value without a newline
- `println(value)`: Print a value with a newline
- `input(prompt)`: Read a line of input with a prompt

#### List Operations

- `len(list)`: Get the length of a list or string
- `push(list, item)`: Add an item to a list
- `get(list, index)`: Get an item at a specific index
- `slice(list, start, end)`: Get a sublist
- `<>`: List concatenation operator

#### Math Functions

- `sqrt(number)`: Square root
- `random()`: Random number between 0 and 1
- `random_range(min, max)`: Random number in range
- `random_int_range(min, max)`: Random integer in range
- `abs(number)`: Absolute value
- `round(number)`: Round to nearest integer
- `floor(number)`: Round down
- `ceil(number)`: Round up

#### String Operations

- `split(string, delimiter)`: Split string into list
- `trim(string)`: Remove whitespace
- `replace(string, pattern, replacement)`: Replace substring
- `string(value)`: Convert value to string
- `<>`: String concatenation operator

## Error Handling

rlox2 provides detailed error messages with location information:

```
Runtime Error: at line 1, column 12:
    let x = 1 + "2";
                ^
Operands must be numbers
```

## Examples

For complete example programs demonstrating rlox2's features, including
implementations of:

- Magic 8-Ball
- Binary Search
- Merge Sort
- Closures and Scoping

Check out the `examples/` directory in the repository.
