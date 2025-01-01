use crate::error::Error;
use crate::parser::{BinaryOp, Expr, Literal, UnaryOp};
use crate::{environment::Environment, error::Result};
use std::{
    fmt::{self, Debug, Display, Formatter},
    rc::Rc,
};

#[derive(Clone)]
pub enum Value {
    Number(f64),
    String(String),
    Boolean(bool),
    List(Vec<Value>),
    Callable(Callable),
    Nil,
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Value::Number(a), Value::Number(b)) => {
                const EPSILON: f64 = 1e-10;
                (a - b).abs() < EPSILON
            }
            (Value::String(a), Value::String(b)) => a == b,
            (Value::Boolean(a), Value::Boolean(b)) => a == b,
            (Value::List(a), Value::List(b)) => a == b,
            (Value::Nil, Value::Nil) => true,
            (Value::Callable(_), Value::Callable(_)) => false,
            _ => false,
        }
    }
}

impl Eq for Value {}

impl Display for Value {
    fn fmt(&self, f: &mut Formatter) -> fmt::Result {
        match self {
            Value::Number(n) => {
                if n.fract() == 0.0 {
                    write!(f, "{}", *n as i64)
                } else {
                    write!(f, "{}", n)
                }
            }
            Value::String(s) => write!(f, "{}", s),
            Value::Boolean(b) => write!(f, "{}", b),
            Value::List(items) => {
                write!(f, "[")?;
                for (i, item) in items.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", item)?;
                }
                write!(f, "]")
            }
            Value::Callable(c) => write!(f, "{}", c),
            Value::Nil => write!(f, "nil"),
        }
    }
}

impl Debug for Value {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if let Value::String(s) = self {
            write!(f, "\"{}\"", s)
        } else {
            write!(f, "{}", self)
        }
    }
}

#[derive(Clone)]
pub enum Callable {
    Function {
        params: Vec<String>,
        body: Box<Expr>,
        closure: Environment,
    },
    BuiltIn {
        name: String,
        arity: usize,
        func: Rc<dyn Fn(Vec<Value>) -> Result<Value>>,
    },
}

impl Display for Callable {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Callable::Function { params, .. } => {
                write!(f, "fn({})", params.join(", "))
            }
            Callable::BuiltIn { name, .. } => {
                write!(f, "<built-in {}>", name)
            }
        }
    }
}

impl Debug for Callable {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self)
    }
}

impl PartialEq for Callable {
    fn eq(&self, _other: &Self) -> bool {
        false
    }
}

impl Eq for Callable {}

pub fn evaluate(expr: &Expr, env: &mut Environment) -> Result<Value> {
    match expr {
        Expr::Literal(lit) => evaluate_literal(lit, env),
        Expr::Unary { operator, right } => evaluate_unary(operator, right, env),
        Expr::Binary {
            left,
            operator,
            right,
        } => evaluate_binary(left, operator, right, env),
        Expr::Grouping(expr) => evaluate(expr, env),
        Expr::Let {
            name,
            initializer,
            recursive,
        } => {
            if *recursive {
                // For recursive bindings, we need to:
                // 1. Create placeholder in current scope
                // 2. Evaluate initializer with access to placeholder
                // 3. Update the binding with the real value
                match evaluate(initializer, env)? {
                    Value::Callable(Callable::Function {
                        params,
                        body,
                        closure,
                    }) => {
                        // Create new closure that includes the function itself
                        let mut func_env = closure.clone();
                        let func = Value::Callable(Callable::Function {
                            params: params.clone(),
                            body: body.clone(),
                            closure: func_env.clone(),
                        });

                        // Add the function to its own environment
                        func_env.insert(name.clone(), func.clone());

                        // Add to current environment
                        env.insert(name.clone(), func.clone());
                        Ok(func)
                    }
                    _ => Err(Error::Runtime {
                        message: "rec keyword can only be used with function definitions"
                            .to_string(),
                        line: 0,
                        column: 0,
                        context: String::new(),
                    }),
                }
            } else {
                // Non-recursive binding behaves as before
                let value = evaluate(initializer, env)?;
                env.insert(name.clone(), value.clone());
                Ok(value)
            }
        }
        Expr::Variable(name) => env.get(name).clone().ok_or_else(|| Error::Runtime {
            message: format!("Undefined variable '{}'", name),
            line: 0,
            column: 0,
            context: String::new(),
        }),
        Expr::Assign { name, value } => {
            let new_value = evaluate(value, env)?;
            // Try to update existing value
            match env.get(name) {
                Some(_) => {
                    env.update_or(name, new_value.clone());
                    Ok(new_value)
                }
                None => Err(Error::Runtime {
                    message: format!("Cannot assign to undefined variable '{}'", name),
                    line: 0,
                    column: 0,
                    context: String::new(),
                }),
            }
        }
        Expr::Block(expressions) => {
            let mut result = Value::Nil;
            let mut block_env = env.extend(); // Create new scope

            for expr in expressions {
                result = evaluate(expr, &mut block_env)?;
            }
            Ok(result)
        }
        Expr::If {
            condition,
            then_branch,
            else_branch,
        } => {
            let cond_val = evaluate(condition, env)?;
            match cond_val {
                Value::Boolean(true) => evaluate(then_branch, env),
                Value::Boolean(false) => evaluate(else_branch, env),
                _ => Err(Error::Runtime {
                    message: "Condition must evaluate to a boolean".to_string(),
                    line: 0,
                    column: 0,
                    context: String::new(),
                }),
            }
        }
        Expr::While { condition, body } => {
            let mut result = Value::Nil;
            loop {
                match evaluate(condition, env)? {
                    Value::Boolean(true) => {
                        result = evaluate(body, env)?;
                    }
                    Value::Boolean(false) => break,
                    _ => {
                        return Err(Error::Runtime {
                            message: "Condition must evaluate to a boolean".to_string(),
                            line: 0,
                            column: 0,
                            context: String::new(),
                        })
                    }
                }
            }
            Ok(result)
        }
        Expr::Call { callee, arguments } => {
            let callee_val = evaluate(callee, env)?;
            let mut evaluated_args = Vec::with_capacity(arguments.len());

            for arg in arguments {
                evaluated_args.push(evaluate(arg, env)?);
            }

            match callee_val {
                Value::Callable(callable) => match callable {
                    Callable::Function {
                        params,
                        body,
                        closure,
                    } => {
                        if params.len() != evaluated_args.len() {
                            return Err(Error::Runtime {
                                message: format!(
                                    "Expected {} arguments but got {}",
                                    params.len(),
                                    evaluated_args.len()
                                ),
                                line: 0,
                                column: 0,
                                context: String::new(),
                            });
                        }

                        // Create new environment extending the closure's environment
                        let mut call_env = closure.extend();

                        // Bind parameters to arguments
                        for (param, arg) in params.iter().zip(evaluated_args) {
                            call_env.insert(param.clone(), arg);
                        }

                        // Evaluate function body in the new environment
                        evaluate(&body, &mut call_env)
                    }
                    Callable::BuiltIn { func, arity, .. } => {
                        if arity != evaluated_args.len() {
                            return Err(Error::Runtime {
                                message: format!(
                                    "Expected {} arguments but got {}",
                                    arity,
                                    evaluated_args.len()
                                ),
                                line: 0,
                                column: 0,
                                context: String::new(),
                            });
                        }
                        func(evaluated_args)
                    }
                },
                _ => Err(Error::Runtime {
                    message: "Can only call functions and built-ins".to_string(),
                    line: 0,
                    column: 0,
                    context: String::new(),
                }),
            }
        }
    }
}

fn evaluate_literal(lit: &Literal, env: &mut Environment) -> Result<Value> {
    match lit {
        Literal::Number(n) => Ok(Value::Number(*n)),
        Literal::String(s) => Ok(Value::String(s.clone())),
        Literal::Boolean(b) => Ok(Value::Boolean(*b)),
        Literal::Function { params, body } => Ok(Value::Callable(Callable::Function {
            params: params.clone(),
            body: body.clone(),
            closure: env.clone(), // Capture current environment
        })),
        Literal::List(items) => {
            let mut values = Vec::with_capacity(items.len());
            for item in items {
                values.push(evaluate(item, env)?);
            }
            Ok(Value::List(values))
        }
        Literal::Nil => Ok(Value::Nil),
    }
}

fn evaluate_unary(operator: &UnaryOp, right: &Expr, env: &mut Environment) -> Result<Value> {
    let right_val = evaluate(right, env)?;

    match operator {
        UnaryOp::Negate => match right_val {
            Value::Number(n) => Ok(Value::Number(-n)),
            _ => Err(Error::Runtime {
                message: "Operand must be a number".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        },
        UnaryOp::Not => match right_val {
            Value::Boolean(b) => Ok(Value::Boolean(!b)),
            _ => Err(Error::Runtime {
                message: "Operand must be a boolean".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        },
    }
}

fn evaluate_binary(
    left: &Expr,
    operator: &BinaryOp,
    right: &Expr,
    env: &mut Environment,
) -> Result<Value> {
    let left_val = evaluate(left, env)?;
    let right_val = evaluate(right, env)?;

    match operator {
        BinaryOp::Add => match (&left_val, &right_val) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a + b)),
            (Value::String(a), Value::String(b)) => Ok(Value::String(format!("{}{}", a, b))),
            _ => Err(Error::Runtime {
                message: "Operands must be two numbers or two strings".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        },
        BinaryOp::Subtract => match (&left_val, &right_val) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a - b)),
            _ => Err(Error::Runtime {
                message: "Operands must be numbers".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        },
        BinaryOp::Multiply => match (&left_val, &right_val) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Number(a * b)),
            _ => Err(Error::Runtime {
                message: "Operands must be numbers".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        },
        BinaryOp::Divide => match (&left_val, &right_val) {
            (Value::Number(a), Value::Number(b)) => {
                if *b == 0.0 {
                    Err(Error::Runtime {
                        message: "Division by zero".to_string(),
                        line: 0,
                        column: 0,
                        context: String::new(),
                    })
                } else {
                    Ok(Value::Number(a / b))
                }
            }
            _ => Err(Error::Runtime {
                message: "Operands must be numbers".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        },
        BinaryOp::Modulo => match (&left_val, &right_val) {
            (Value::Number(a), Value::Number(b)) => {
                if *b == 0.0 {
                    Err(Error::Runtime {
                        message: "Modulo by zero".to_string(),
                        line: 0,
                        column: 0,
                        context: String::new(),
                    })
                } else {
                    Ok(Value::Number(a % b))
                }
            }
            _ => Err(Error::Runtime {
                message: "Operands must be numbers".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        },
        BinaryOp::Equal => Ok(Value::Boolean(left_val == right_val)),
        BinaryOp::NotEqual => Ok(Value::Boolean(left_val != right_val)),
        BinaryOp::Less => match (&left_val, &right_val) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Boolean(a < b)),
            _ => Err(Error::Runtime {
                message: "Operands must be numbers".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        },
        BinaryOp::LessEqual => match (&left_val, &right_val) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Boolean(a <= b)),
            _ => Err(Error::Runtime {
                message: "Operands must be numbers".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        },
        BinaryOp::Greater => match (&left_val, &right_val) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Boolean(a > b)),
            _ => Err(Error::Runtime {
                message: "Operands must be numbers".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        },
        BinaryOp::GreaterEqual => match (&left_val, &right_val) {
            (Value::Number(a), Value::Number(b)) => Ok(Value::Boolean(a >= b)),
            _ => Err(Error::Runtime {
                message: "Operands must be numbers".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        },
        BinaryOp::Concat => match (&left_val, &right_val) {
            (Value::String(a), Value::String(b)) => Ok(Value::String(format!("{}{}", a, b))),
            _ => Err(Error::Runtime {
                message: "Operands must be strings".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        },
        BinaryOp::And => match left_val {
            Value::Boolean(false) => Ok(Value::Boolean(false)),
            Value::Boolean(true) => match right_val {
                Value::Boolean(_) => Ok(right_val),
                _ => Err(Error::Runtime {
                    message: "Operands must be booleans".to_string(),
                    line: 0,
                    column: 0,
                    context: String::new(),
                }),
            },
            _ => Err(Error::Runtime {
                message: "Operands must be booleans".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        },
        BinaryOp::Or => match left_val {
            Value::Boolean(true) => Ok(Value::Boolean(true)),
            Value::Boolean(false) => match right_val {
                Value::Boolean(_) => Ok(right_val),
                _ => Err(Error::Runtime {
                    message: "Operands must be booleans".to_string(),
                    line: 0,
                    column: 0,
                    context: String::new(),
                }),
            },
            _ => Err(Error::Runtime {
                message: "Operands must be booleans".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::environment::Environment;
    use crate::parser::parse;
    use crate::stdlib::create_standard_env;
    use crate::tokenizer::tokenize;

    #[test]
    fn test_basic_arithmetic() -> Result<()> {
        let mut env = Environment::new();

        // Test addition
        let source = b"2 + 3";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        let result = evaluate(&expr, &mut env)?;
        assert_eq!(result, Value::Number(5.0));

        // Test variable reassignment
        let source = b"let x = 42; x = 24; x";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        let result = evaluate(&expr, &mut env)?;
        assert_eq!(result, Value::Number(24.0));

        Ok(())
    }

    #[test]
    fn test_function_definition_and_call() -> Result<()> {
        let mut env = Environment::new();

        // Test function definition and call
        let source = b"let add = fn(x, y) { x + y }; add(3, 4)";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        let result = evaluate(&expr, &mut env)?;
        assert_eq!(result, Value::Number(7.0));

        // Test closure capturing
        let source = b"let x = 1; let adder = fn(y) { x + y }; let x = 2; adder(3)";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        let result = evaluate(&expr, &mut env)?;
        assert_eq!(result, Value::Number(4.0)); // Should be 4 (1 + 3) not 5 (2 + 3)

        Ok(())
    }

    #[test]
    fn test_built_in_functions() -> Result<()> {
        let mut env = create_standard_env();

        // Test println (should return nil)
        let source = b"println(\"hello\")";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        let result = evaluate(&expr, &mut env)?;
        assert_eq!(result, Value::Nil);

        // Test list operations
        let source = b"len([1, 2, 3])";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        let result = evaluate(&expr, &mut env)?;
        assert_eq!(result, Value::Number(3.0));

        Ok(())
    }

    #[test]
    fn test_lexical_scoping() -> Result<()> {
        let mut env = Environment::new();

        // Test that inner scope doesn't affect outer scope
        let source = b"
            let x = 1;
            {
                let x = 2;
                x
            };
            x
        ";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        let result = evaluate(&expr, &mut env)?;
        assert_eq!(result, Value::Number(1.0));

        Ok(())
    }

    #[test]
    fn test_recursion() -> Result<()> {
        let mut env = Environment::new();

        // Test factorial function
        let source = b"
            let rec factorial = fn(n) {
                if n <= 1 {
                    1
                } else {
                    n * factorial(n - 1)
                }
            };
            factorial(5)
        ";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        let result = evaluate(&expr, &mut env)?;
        assert_eq!(result, Value::Number(120.0));

        Ok(())
    }

    #[test]
    fn test_error_handling() -> Result<()> {
        let mut env = Environment::new();

        // Test undefined variable
        let source = b"nonexistent";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        assert!(evaluate(&expr, &mut env).is_err());

        // Test type error
        let source = b"1 + true";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        assert!(evaluate(&expr, &mut env).is_err());

        // Test division by zero
        let source = b"1 / 0";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        assert!(evaluate(&expr, &mut env).is_err());

        Ok(())
    }

    #[test]
    fn test_variable_binding() -> Result<()> {
        let mut env = Environment::new();

        // Test variable declaration and use
        let source = b"let x = 42; x";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        let result = evaluate(&expr, &mut env)?;
        assert_eq!(result, Value::Number(42.0));

        // Test variable reassignment
        let source = b"let x = 10; x = 20; x";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        let result = evaluate(&expr, &mut env)?;
        assert_eq!(result, Value::Number(20.0));

        // Test multiple variables
        let source = b"let x = 1; let y = 2; x + y";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        let result = evaluate(&expr, &mut env)?;
        assert_eq!(result, Value::Number(3.0));

        // Test variable scoping
        let source = b"
            let x = 1;
            {
                let x = 2;
                x
            };
            x
        ";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        let result = evaluate(&expr, &mut env)?;
        assert_eq!(result, Value::Number(1.0));

        // Test undefined variable error
        let source = b"x";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        assert!(evaluate(&expr, &mut env).is_err());

        // Test assignment to undefined variable error
        let source = b"x = 10";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        assert!(evaluate(&expr, &mut env).is_err());

        Ok(())
    }

    #[test]
    fn test_function_calls() -> Result<()> {
        let mut env = create_standard_env();

        // Test simple function definition and call
        let source = b"
            let add = fn(a, b) { a + b };
            add(2, 3)
        ";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        let result = evaluate(&expr, &mut env)?;
        assert_eq!(result, Value::Number(5.0));

        // Test closure capturing
        let source = b"
            let x = 10;
            let adder = fn(y) { x + y };
            let x = 20;  // This shouldn't affect the closure
            adder(5)
        ";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        let result = evaluate(&expr, &mut env)?;
        assert_eq!(result, Value::Number(15.0)); // Should be 15 (10 + 5), not 25 (20 + 5)

        // Test recursive function
        let source = b"
            let rec fib = fn(n) {
                if n <= 1 {
                    n
                } else {
                    fib(n - 1) + fib(n - 2)
                }
            };
            fib(6)
        ";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        let result = evaluate(&expr, &mut env)?;
        assert_eq!(result, Value::Number(8.0)); // 6th Fibonacci number

        // Test higher-order function
        let source = b"
            let apply_twice = fn(f, x) { f(f(x)) };
            let add_one = fn(x) { x + 1 };
            apply_twice(add_one, 5)
        ";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        let result = evaluate(&expr, &mut env)?;
        assert_eq!(result, Value::Number(7.0)); // 5 + 1 + 1 = 7

        // Test wrong argument count error
        let source = b"
            let f = fn(x) { x };
            f(1, 2)
        ";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        assert!(evaluate(&expr, &mut env).is_err());

        Ok(())
    }

    #[test]
    fn test_control_flow() -> Result<()> {
        let mut env = Environment::new();

        // Test if-else with blocks
        let source = b"
            if true {
                let x = 1;
                x + 1
            } else {
                let x = 2;
                x + 2
            }
        ";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        let result = evaluate(&expr, &mut env)?;
        assert_eq!(result, Value::Number(2.0));

        // Test while loop with accumulator
        let source = b"
            let sum = 0;
            let i = 1;
            while i <= 5 {
                sum = sum + i;
                i = i + 1
            };
            sum
        ";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        let result = evaluate(&expr, &mut env)?;
        assert_eq!(result, Value::Number(15.0)); // 1 + 2 + 3 + 4 + 5 = 15

        // Test nested control flow
        let source = b"
            let result = 0;
            let i = 0;
            while i < 3 {
                if i == 1 {
                    result = result + 10
                } else {
                    result = result + 1
                };
                i = i + 1
            };
            result
        ";
        let tokens = tokenize(source)?;
        let expr = parse(source, &tokens)?;
        let result = evaluate(&expr, &mut env)?;
        assert_eq!(result, Value::Number(12.0)); // 1 + 10 + 1 = 12

        Ok(())
    }

    #[test]
    fn test_value_debug() {
        let num = Value::Number(42.0);
        let string = Value::String("hello".to_string());
        let boolean = Value::Boolean(true);
        let list = Value::List(vec![num.clone(), string.clone()]);

        assert_eq!(format!("{:?}", num), "Number(42.0)");
        assert_eq!(format!("{:?}", string), "String(\"hello\")");
        assert_eq!(format!("{:?}", boolean), "Boolean(true)");
        assert_eq!(
            format!("{:?}", list),
            "List([Number(42.0), String(\"hello\")])"
        );
    }

    #[test]
    fn test_value_display() {
        let num = Value::Number(42.0);
        let string = Value::String("hello".to_string());
        let boolean = Value::Boolean(true);
        let list = Value::List(vec![num.clone(), string.clone()]);

        assert_eq!(format!("{}", num), "42");
        assert_eq!(format!("{}", string), "hello");
        assert_eq!(format!("{}", boolean), "true");
        assert_eq!(format!("{}", list), "[42, hello]");
    }

    #[test]
    fn test_callable_debug() {
        let built_in = Callable::BuiltIn {
            name: "test".to_string(),
            arity: 1,
            func: Rc::new(|_| Ok(Value::Nil)),
        };

        let debug_str = format!("{:?}", built_in);
        assert!(debug_str.contains("BuiltIn"));
        assert!(debug_str.contains("test"));
        assert!(debug_str.contains("1"));
    }

    #[test]
    fn test_value_equality() {
        assert_eq!(Value::Number(42.0), Value::Number(42.0));
        assert_eq!(
            Value::String("hello".to_string()),
            Value::String("hello".to_string())
        );
        assert_eq!(Value::Boolean(true), Value::Boolean(true));
        assert_eq!(Value::Nil, Value::Nil);

        assert_ne!(Value::Number(42.0), Value::Number(43.0));
        assert_ne!(
            Value::String("hello".to_string()),
            Value::String("world".to_string())
        );
        assert_ne!(Value::Boolean(true), Value::Boolean(false));
    }
}
