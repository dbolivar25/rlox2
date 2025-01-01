use crate::environment::Environment;
use crate::error::Error;
use crate::runtime::{Callable, Value};
use rand::Rng;
use std::rc::Rc;
use std::time::{SystemTime, UNIX_EPOCH};

pub fn create_standard_env() -> Environment {
    let mut env = Environment::new();

    // Helper macro to define built-in functions with error handling
    macro_rules! define_builtin {
        ($name:expr, $arity:expr, $func:expr) => {
            env.insert(
                $name.to_string(),
                Value::Callable(Callable::BuiltIn {
                    name: $name.to_string(),
                    arity: $arity,
                    func: Rc::new($func),
                }),
            );
        };
    }

    // I/O functions with proper error handling
    define_builtin!("print", 1, |args| {
        print!(
            "{}",
            args.first().ok_or_else(|| Error::Runtime {
                message: "print requires 1 argument".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            })?
        );
        Ok(Value::Nil)
    });

    define_builtin!("println", 1, |args| {
        println!(
            "{}",
            args.first().ok_or_else(|| Error::Runtime {
                message: "println requires 1 argument".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            })?
        );
        Ok(Value::Nil)
    });

    // Type functions
    define_builtin!("type", 1, |args| {
        let arg = args.first().ok_or_else(|| Error::Runtime {
            message: "type requires 1 argument".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        let type_name = match arg {
            Value::Number(_) => "number",
            Value::String(_) => "string",
            Value::Boolean(_) => "boolean",
            Value::Callable(_) => "callable",
            Value::List(_) => "list",
            Value::Nil => "nil",
        };
        Ok(Value::String(type_name.to_string()))
    });

    // List operations
    define_builtin!("len", 1, |args| {
        let arg = args.first().ok_or_else(|| Error::Runtime {
            message: "len requires 1 argument".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        match arg {
            Value::List(list) => Ok(Value::Number(list.len() as f64)),
            Value::String(s) => Ok(Value::Number(s.len() as f64)),
            _ => Err(Error::Runtime {
                message: "len() takes a list or string argument".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        }
    });

    define_builtin!("push", 2, |args| {
        let list = args.first().ok_or_else(|| Error::Runtime {
            message: "push requires 2 arguments".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        let value = args.get(1).ok_or_else(|| Error::Runtime {
            message: "push requires 2 arguments".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        if let Value::List(list) = list {
            let mut new_list = list.clone();
            new_list.push(value.clone());
            Ok(Value::List(new_list))
        } else {
            Err(Error::Runtime {
                message: "First argument to push() must be a list".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            })
        }
    });

    // Math functions
    define_builtin!("sqrt", 1, |args| {
        let arg = args.first().ok_or_else(|| Error::Runtime {
            message: "sqrt requires 1 argument".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        match arg {
            Value::Number(n) if *n >= 0.0 => Ok(Value::Number(n.sqrt())),
            _ => Err(Error::Runtime {
                message: "sqrt() takes a non-negative number argument".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        }
    });

    // Random number generation
    define_builtin!("random", 0, |_args| {
        Ok(Value::Number(rand::thread_rng().gen()))
    });

    define_builtin!("random_range", 2, |args| {
        let min = args.first().ok_or_else(|| Error::Runtime {
            message: "random_range requires 2 arguments".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        let max = args.get(1).ok_or_else(|| Error::Runtime {
            message: "random_range requires 2 arguments".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        match (min, max) {
            (Value::Number(min), Value::Number(max)) => {
                if min >= max {
                    Err(Error::Runtime {
                        message: "random_range: min must be less than max".to_string(),
                        line: 0,
                        column: 0,
                        context: String::new(),
                    })
                } else {
                    Ok(Value::Number(rand::thread_rng().gen_range(*min..*max)))
                }
            }
            _ => Err(Error::Runtime {
                message: "random_range requires two numbers".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        }
    });

    // String utilities
    define_builtin!("split", 2, |args| {
        let string = args.first().ok_or_else(|| Error::Runtime {
            message: "split requires 2 arguments".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        let delimiter = args.get(1).ok_or_else(|| Error::Runtime {
            message: "split requires 2 arguments".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        match (string, delimiter) {
            (Value::String(s), Value::String(delim)) => {
                let parts: Vec<Value> = s
                    .split(delim)
                    .map(|s| Value::String(s.to_string()))
                    .collect();
                Ok(Value::List(parts))
            }
            _ => Err(Error::Runtime {
                message: "split requires two strings".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        }
    });

    // Advanced list operations
    define_builtin!("map", 2, |args| {
        let list = args.first().ok_or_else(|| Error::Runtime {
            message: "map requires 2 arguments".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        let func = args.get(1).ok_or_else(|| Error::Runtime {
            message: "map requires 2 arguments".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        match (list, func) {
            (Value::List(list), Value::Callable(func)) => {
                let mut results = Vec::with_capacity(list.len());
                for item in list {
                    let result = match func {
                        Callable::Function { .. } => {
                            return Err(Error::Runtime {
                                message: "User-defined functions not supported in map yet"
                                    .to_string(),
                                line: 0,
                                column: 0,
                                context: String::new(),
                            })
                        }
                        Callable::BuiltIn { func, .. } => func(vec![item.clone()])?,
                    };
                    results.push(result);
                }
                Ok(Value::List(results))
            }
            _ => Err(Error::Runtime {
                message: "map requires a list and a function".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        }
    });

    define_builtin!("filter", 2, |args| {
        let list = args.first().ok_or_else(|| Error::Runtime {
            message: "filter requires 2 arguments".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        let func = args.get(1).ok_or_else(|| Error::Runtime {
            message: "filter requires 2 arguments".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        match (list, func) {
            (Value::List(list), Value::Callable(func)) => {
                let mut results = Vec::new();
                for item in list {
                    let result = match func {
                        Callable::Function { .. } => {
                            return Err(Error::Runtime {
                                message: "User-defined functions not supported in filter yet"
                                    .to_string(),
                                line: 0,
                                column: 0,
                                context: String::new(),
                            })
                        }
                        Callable::BuiltIn { func, .. } => func(vec![item.clone()])?,
                    };
                    if let Value::Boolean(true) = result {
                        results.push(item.clone());
                    }
                }
                Ok(Value::List(results))
            }
            _ => Err(Error::Runtime {
                message: "filter requires a list and a function".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        }
    });

    define_builtin!("reduce", 3, |args| {
        let list = args.first().ok_or_else(|| Error::Runtime {
            message: "reduce requires 3 arguments".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        let func = args.get(1).ok_or_else(|| Error::Runtime {
            message: "reduce requires 3 arguments".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        let initial = args.get(2).ok_or_else(|| Error::Runtime {
            message: "reduce requires 3 arguments".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        match (list, func) {
            (Value::List(list), Value::Callable(func)) => {
                let mut acc = initial.clone();
                for item in list {
                    acc = match func {
                        Callable::Function { .. } => {
                            return Err(Error::Runtime {
                                message: "User-defined functions not supported in reduce yet"
                                    .to_string(),
                                line: 0,
                                column: 0,
                                context: String::new(),
                            })
                        }
                        Callable::BuiltIn { func, .. } => func(vec![acc, item.clone()])?,
                    };
                }
                Ok(acc)
            }
            _ => Err(Error::Runtime {
                message: "reduce requires a list and a function".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        }
    });

    define_builtin!("reverse", 1, |args| {
        let arg = args.first().ok_or_else(|| Error::Runtime {
            message: "reverse requires 1 argument".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        match arg {
            Value::List(list) => {
                let mut reversed = list.clone();
                reversed.reverse();
                Ok(Value::List(reversed))
            }
            Value::String(s) => Ok(Value::String(s.chars().rev().collect())),
            _ => Err(Error::Runtime {
                message: "reverse requires a list or string".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        }
    });

    // String manipulation
    define_builtin!("trim", 1, |args| {
        let arg = args.first().ok_or_else(|| Error::Runtime {
            message: "trim requires 1 argument".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        match arg {
            Value::String(s) => Ok(Value::String(s.trim().to_string())),
            _ => Err(Error::Runtime {
                message: "trim requires a string".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        }
    });

    define_builtin!("replace", 3, |args| {
        let string = args.first().ok_or_else(|| Error::Runtime {
            message: "replace requires 3 arguments".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        let pattern = args.get(1).ok_or_else(|| Error::Runtime {
            message: "replace requires 3 arguments".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        let replacement = args.get(2).ok_or_else(|| Error::Runtime {
            message: "replace requires 3 arguments".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        match (string, pattern, replacement) {
            (Value::String(s), Value::String(from), Value::String(to)) => {
                Ok(Value::String(s.replace(from, to)))
            }
            _ => Err(Error::Runtime {
                message: "replace requires three strings".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        }
    });

    // Math utilities
    define_builtin!("round", 1, |args| {
        let arg = args.first().ok_or_else(|| Error::Runtime {
            message: "round requires 1 argument".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        match arg {
            Value::Number(n) => Ok(Value::Number(n.round())),
            _ => Err(Error::Runtime {
                message: "round requires a number".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        }
    });

    define_builtin!("abs", 1, |args| {
        let arg = args.first().ok_or_else(|| Error::Runtime {
            message: "abs requires 1 argument".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        match arg {
            Value::Number(n) => Ok(Value::Number(n.abs())),
            _ => Err(Error::Runtime {
                message: "abs requires a number".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        }
    });

    define_builtin!("min", 2, |args| {
        let a = args.first().ok_or_else(|| Error::Runtime {
            message: "min requires 2 arguments".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        let b = args.get(1).ok_or_else(|| Error::Runtime {
            message: "min requires 2 arguments".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        match (a, b) {
            (Value::Number(x), Value::Number(y)) => Ok(Value::Number(x.min(*y))),
            _ => Err(Error::Runtime {
                message: "min requires two numbers".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        }
    });

    define_builtin!("max", 2, |args| {
        let a = args.first().ok_or_else(|| Error::Runtime {
            message: "max requires 2 arguments".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        let b = args.get(1).ok_or_else(|| Error::Runtime {
            message: "max requires 2 arguments".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        match (a, b) {
            (Value::Number(x), Value::Number(y)) => Ok(Value::Number(x.max(*y))),
            _ => Err(Error::Runtime {
                message: "max requires two numbers".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        }
    });

    // Type conversion
    define_builtin!("number", 1, |args| {
        let arg = args.first().ok_or_else(|| Error::Runtime {
            message: "number requires 1 argument".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        match arg {
            Value::String(s) => s
                .parse::<f64>()
                .map(Value::Number)
                .map_err(|_| Error::Runtime {
                    message: "Could not convert string to number".to_string(),
                    line: 0,
                    column: 0,
                    context: String::new(),
                }),
            Value::Number(n) => Ok(Value::Number(*n)),
            _ => Err(Error::Runtime {
                message: "number requires a string or number".to_string(),
                line: 0,
                column: 0,
                context: String::new(),
            }),
        }
    });

    define_builtin!("string", 1, |args| {
        let arg = args.first().ok_or_else(|| Error::Runtime {
            message: "string requires 1 argument".to_string(),
            line: 0,
            column: 0,
            context: String::new(),
        })?;

        Ok(Value::String(arg.to_string()))
    });

    // Time functions
    define_builtin!("clock", 0, |_args| {
        let start = SystemTime::now();
        let since_epoch = start
            .duration_since(UNIX_EPOCH)
            .map_err(|e| Error::Runtime {
                message: format!("Time error: {}", e),
                line: 0,
                column: 0,
                context: String::new(),
            })?;
        Ok(Value::Number(since_epoch.as_secs_f64()))
    });

    env
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;

    #[test]
    fn test_stdlib_functions() -> Result<()> {
        let env = create_standard_env();

        // Test that standard functions exist
        assert!(matches!(
            env.get(&"print".to_string()),
            Some(Value::Callable(_))
        ));
        assert!(matches!(
            env.get(&"clock".to_string()),
            Some(Value::Callable(_))
        ));
        assert!(matches!(
            env.get(&"len".to_string()),
            Some(Value::Callable(_))
        ));
        assert!(matches!(
            env.get(&"type".to_string()),
            Some(Value::Callable(_))
        ));

        // Test len function
        if let Some(Value::Callable(Callable::BuiltIn { func, .. })) = env.get(&"len".to_string()) {
            let result = func(vec![Value::List(vec![
                Value::Number(1.0),
                Value::Number(2.0),
            ])])?;
            assert_eq!(result, Value::Number(2.0));
        }

        // Test type function
        if let Some(Value::Callable(Callable::BuiltIn { func, .. })) = env.get(&"type".to_string())
        {
            let result = func(vec![Value::Number(42.0)])?;
            assert_eq!(result, Value::String("number".to_string()));
        }

        Ok(())
    }
}
