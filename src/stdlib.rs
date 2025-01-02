use crate::environment::Environment;
use crate::runtime::{Callable, Value};
use rand::Rng;
use std::io::{self, Write};
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
            args.first()
                .ok_or_else(|| "print requires 1 argument".to_string())?
        );
        Ok(Value::Nil)
    });

    define_builtin!("println", 1, |args| {
        println!(
            "{}",
            args.first()
                .ok_or_else(|| "println requires 1 argument".to_string())?
        );
        Ok(Value::Nil)
    });

    define_builtin!("input", 1, |args| {
        let prompt = args
            .first()
            .ok_or_else(|| "input requires 1 argument".to_string())?;

        // Print the prompt and flush stdout to ensure it appears before input
        match prompt {
            Value::String(s) => {
                print!("{}", s);
                io::stdout()
                    .flush()
                    .map_err(|e| format!("IO error: {}", e))?;
            }
            _ => return Err("input prompt must be a string".to_string()),
        }

        // Read user input
        let mut input = String::new();
        io::stdin()
            .read_line(&mut input)
            .map_err(|e| format!("Failed to read input: {}", e))?;

        // Trim the trailing newline and return
        Ok(Value::String(input.trim_end().to_string()))
    });

    // Type functions
    define_builtin!("type", 1, |args| {
        let arg = args
            .first()
            .ok_or_else(|| "type requires 1 argument".to_string())?;

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
        let arg = args
            .first()
            .ok_or_else(|| "len requires 1 argument".to_string())?;

        match arg {
            Value::List(list) => Ok(Value::Number(list.len() as f64)),
            Value::String(s) => Ok(Value::Number(s.len() as f64)),
            _ => Err("len() takes a list or string argument".to_string()),
        }
    });

    define_builtin!("push", 2, |args| {
        let list = args
            .first()
            .ok_or_else(|| "push requires 2 arguments".to_string())?;

        let value = args
            .get(1)
            .ok_or_else(|| "push requires 2 arguments".to_string())?;

        if let Value::List(list) = list {
            let mut new_list = list.clone();
            new_list.push(value.clone());
            Ok(Value::List(new_list))
        } else {
            Err("First argument to push() must be a list".to_string())
        }
    });

    define_builtin!("get", 2, |args| {
        let list = args
            .first()
            .ok_or_else(|| "get requires 2 arguments".to_string())?;

        let index = args
            .get(1)
            .ok_or_else(|| "get requires 2 arguments".to_string())?;

        match (list, index) {
            (Value::List(list), Value::Number(idx)) => {
                let i = idx.trunc() as usize;
                if i < list.len() {
                    Ok(list[i].clone())
                } else {
                    Err(format!(
                        "Index {} out of bounds for list of length {}",
                        i,
                        list.len()
                    ))
                }
            }
            (Value::List(_), _) => Err("Second argument to get() must be a number".to_string()),
            _ => Err("First argument to get() must be a list".to_string()),
        }
    });

    define_builtin!("slice", 3, |args| {
        let list = args
            .first()
            .ok_or_else(|| "slice requires 3 arguments".to_string())?;

        let start = args
            .get(1)
            .ok_or_else(|| "slice requires 3 arguments".to_string())?;

        let end = args
            .get(2)
            .ok_or_else(|| "slice requires 3 arguments".to_string())?;

        match (list, start, end) {
            (Value::List(list), Value::Number(start), Value::Number(end)) => {
                let start = start.trunc() as usize;
                let end = end.trunc() as usize;
                if start <= end && end <= list.len() {
                    Ok(Value::List(list[start..end].to_vec()))
                } else {
                    Err(format!(
                        "Invalid slice range {}..{} for list of length {}",
                        start,
                        end,
                        list.len()
                    ))
                }
            }
            _ => Err("slice requires a list and two numbers".to_string()),
        }
    });

    // Math functions
    define_builtin!("sqrt", 1, |args| {
        let arg = args
            .first()
            .ok_or_else(|| "sqrt requires 1 argument".to_string())?;

        match arg {
            Value::Number(n) if *n >= 0.0 => Ok(Value::Number(n.sqrt())),
            _ => Err("sqrt() takes a non-negative number argument".to_string()),
        }
    });

    // Random number generation
    define_builtin!("random", 0, |_args| {
        Ok(Value::Number(rand::thread_rng().gen()))
    });

    define_builtin!("random_range", 2, |args| {
        let min = args
            .first()
            .ok_or_else(|| "random_range requires 2 arguments".to_string())?;

        let max = args
            .get(1)
            .ok_or_else(|| "random_range requires 2 arguments".to_string())?;

        match (min, max) {
            (Value::Number(min), Value::Number(max)) => {
                if min >= max {
                    Err("random_range: min must be less than max".to_string())
                } else {
                    Ok(Value::Number(rand::thread_rng().gen_range(*min..*max)))
                }
            }
            _ => Err("random_range requires two numbers".to_string()),
        }
    });

    define_builtin!("random_int_range", 2, |args| {
        let min = args
            .first()
            .ok_or_else(|| "random_range requires 2 arguments".to_string())?;

        let max = args
            .get(1)
            .ok_or_else(|| "random_range requires 2 arguments".to_string())?;

        match (min, max) {
            (Value::Number(min), Value::Number(max)) => {
                if min >= max {
                    Err("random_range: min must be less than max".to_string())
                } else {
                    Ok(Value::Number(
                        rand::thread_rng().gen_range(*min..*max).trunc(),
                    ))
                }
            }
            _ => Err("random_range requires two numbers".to_string()),
        }
    });

    // String utilities
    define_builtin!("split", 2, |args| {
        let string = args
            .first()
            .ok_or_else(|| "split requires 2 arguments".to_string())?;

        let delimiter = args
            .get(1)
            .ok_or_else(|| "split requires 2 arguments".to_string())?;

        match (string, delimiter) {
            (Value::String(s), Value::String(delim)) => {
                let parts: Vec<Value> = s
                    .split(delim)
                    .map(|s| Value::String(s.to_string()))
                    .collect();
                Ok(Value::List(parts))
            }
            _ => Err("split requires two strings".to_string()),
        }
    });

    // Advanced list operations
    define_builtin!("map", 2, |args| {
        let list = args
            .first()
            .ok_or_else(|| "map requires 2 arguments".to_string())?;

        let func = args
            .get(1)
            .ok_or_else(|| "map requires 2 arguments".to_string())?;

        match (list, func) {
            (Value::List(list), Value::Callable(func)) => {
                let mut results = Vec::with_capacity(list.len());
                for item in list {
                    let result = match func {
                        Callable::Function { .. } => {
                            return Err(
                                "User-defined functions not supported in map yet".to_string()
                            )
                        }
                        Callable::BuiltIn { func, .. } => func(vec![item.clone()])?,
                    };
                    results.push(result);
                }
                Ok(Value::List(results))
            }
            _ => Err("map requires a list and a function".to_string()),
        }
    });

    define_builtin!("filter", 2, |args| {
        let list = args
            .first()
            .ok_or_else(|| "filter requires 2 arguments".to_string())?;

        let func = args
            .get(1)
            .ok_or_else(|| "filter requires 2 arguments".to_string())?;

        match (list, func) {
            (Value::List(list), Value::Callable(func)) => {
                let mut results = Vec::new();
                for item in list {
                    let result = match func {
                        Callable::Function { .. } => {
                            return Err(
                                "User-defined functions not supported in filter yet".to_string()
                            )
                        }
                        Callable::BuiltIn { func, .. } => func(vec![item.clone()])?,
                    };
                    if let Value::Boolean(true) = result {
                        results.push(item.clone());
                    }
                }
                Ok(Value::List(results))
            }
            _ => Err("filter requires a list and a function".to_string()),
        }
    });

    define_builtin!("reduce", 3, |args| {
        let list = args
            .first()
            .ok_or_else(|| "reduce requires 3 arguments".to_string())?;

        let func = args
            .get(1)
            .ok_or_else(|| "reduce requires 3 arguments".to_string())?;

        let initial = args
            .get(2)
            .ok_or_else(|| "reduce requires 3 arguments".to_string())?;

        match (list, func) {
            (Value::List(list), Value::Callable(func)) => {
                let mut acc = initial.clone();
                for item in list {
                    acc = match func {
                        Callable::Function { .. } => {
                            return Err(
                                "User-defined functions not supported in reduce yet".to_string()
                            )
                        }
                        Callable::BuiltIn { func, .. } => func(vec![acc, item.clone()])?,
                    };
                }
                Ok(acc)
            }
            _ => Err("reduce requires a list and a function".to_string()),
        }
    });

    define_builtin!("reverse", 1, |args| {
        let arg = args
            .first()
            .ok_or_else(|| "reverse requires 1 argument".to_string())?;

        match arg {
            Value::List(list) => {
                let mut reversed = list.clone();
                reversed.reverse();
                Ok(Value::List(reversed))
            }
            Value::String(s) => Ok(Value::String(s.chars().rev().collect())),
            _ => Err("reverse requires a list or string".to_string()),
        }
    });

    // String manipulation
    define_builtin!("trim", 1, |args| {
        let arg = args
            .first()
            .ok_or_else(|| "trim requires 1 argument".to_string())?;

        match arg {
            Value::String(s) => Ok(Value::String(s.trim().to_string())),
            _ => Err("trim requires a string".to_string()),
        }
    });

    define_builtin!("replace", 3, |args| {
        let string = args
            .first()
            .ok_or_else(|| "replace requires 3 arguments".to_string())?;

        let pattern = args
            .get(1)
            .ok_or_else(|| "replace requires 3 arguments".to_string())?;

        let replacement = args
            .get(2)
            .ok_or_else(|| "replace requires 3 arguments".to_string())?;

        match (string, pattern, replacement) {
            (Value::String(s), Value::String(from), Value::String(to)) => {
                Ok(Value::String(s.replace(from, to)))
            }
            _ => Err("replace requires three strings".to_string()),
        }
    });

    // Math utilities
    define_builtin!("round", 1, |args| {
        let arg = args
            .first()
            .ok_or_else(|| "round requires 1 argument".to_string())?;

        match arg {
            Value::Number(n) => Ok(Value::Number(n.round())),
            _ => Err("round requires a number".to_string()),
        }
    });

    define_builtin!("abs", 1, |args| {
        let arg = args
            .first()
            .ok_or_else(|| "abs requires 1 argument".to_string())?;

        match arg {
            Value::Number(n) => Ok(Value::Number(n.abs())),
            _ => Err("abs requires a number".to_string()),
        }
    });

    define_builtin!("min", 2, |args| {
        let a = args
            .first()
            .ok_or_else(|| "min requires 2 arguments".to_string())?;

        let b = args
            .get(1)
            .ok_or_else(|| "min requires 2 arguments".to_string())?;

        match (a, b) {
            (Value::Number(x), Value::Number(y)) => Ok(Value::Number(x.min(*y))),
            _ => Err("min requires two numbers".to_string()),
        }
    });

    define_builtin!("max", 2, |args| {
        let a = args
            .first()
            .ok_or_else(|| "max requires 2 arguments".to_string())?;

        let b = args
            .get(1)
            .ok_or_else(|| "max requires 2 arguments".to_string())?;

        match (a, b) {
            (Value::Number(x), Value::Number(y)) => Ok(Value::Number(x.max(*y))),
            _ => Err("max requires two numbers".to_string()),
        }
    });

    define_builtin!("floor", 1, |args| {
        let num = args
            .first()
            .ok_or_else(|| "floor requires 1 argument".to_string())?;

        match num {
            Value::Number(n) => Ok(Value::Number(n.floor())),
            _ => Err("floor requires a number".to_string()),
        }
    });

    define_builtin!("ceil", 1, |args| {
        let num = args
            .first()
            .ok_or_else(|| "ceil requires 1 argument".to_string())?;

        match num {
            Value::Number(n) => Ok(Value::Number(n.ceil())),
            _ => Err("ceil requires a number".to_string()),
        }
    });

    // Type conversion
    define_builtin!("number", 1, |args| {
        let arg = args
            .first()
            .ok_or_else(|| "number requires 1 argument".to_string())?;

        match arg {
            Value::String(s) => s
                .parse::<f64>()
                .map(Value::Number)
                .map_err(|_| "Could not convert string to number".to_string()),
            Value::Number(n) => Ok(Value::Number(*n)),
            _ => Err("number requires a string or number".to_string()),
        }
    });

    define_builtin!("string", 1, |args| {
        let arg = args
            .first()
            .ok_or_else(|| "string requires 1 argument".to_string())?;

        Ok(Value::String(arg.to_string()))
    });

    // Time functions
    define_builtin!("clock", 0, |_args| {
        let start = SystemTime::now();
        let since_epoch = start
            .duration_since(UNIX_EPOCH)
            .map_err(|e| format!("Time error: {}", e))?;
        Ok(Value::Number(since_epoch.as_secs_f64()))
    });

    env
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stdlib_functions() -> Result<(), String> {
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
