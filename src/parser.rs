use crate::{
    error::{Error, Result},
    tokenizer::Token,
};

#[derive(Debug)]
pub enum Expr {
    Literal(Literal),
    Unary {
        operator: UnaryOp,
        right: Box<Expr>,
    },
    Binary {
        left: Box<Expr>,
        operator: BinaryOp,
        right: Box<Expr>,
    },
    Grouping(Box<Expr>),
    Let {
        name: String,
        initializer: Box<Expr>,
    },
    Block(Vec<Expr>),
    If {
        condition: Box<Expr>,
        then_branch: Box<Expr>,
        else_branch: Box<Expr>,
    },
    While {
        condition: Box<Expr>,
        body: Box<Expr>,
    },
    Function {
        params: Vec<String>,
        body: Box<Expr>,
    },
    Call {
        callee: Box<Expr>,
        arguments: Vec<Expr>,
    },
    Variable(String),
    Assign {
        name: String,
        value: Box<Expr>,
    },
}

#[derive(Debug)]
pub enum Literal {
    Number(f64),
    String(String),
    Boolean(bool),
    Nil,
}

#[derive(Debug)]
pub enum UnaryOp {
    Negate,
    Not,
}

#[derive(Debug)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Concat,
    And,
    Or,
}

pub fn parse(tokens: &[Token]) -> Result<Expr> {
    let mut consumed = 0;
    let mut exprs = Vec::new();

    while consumed < tokens.len() && tokens[consumed] != Token::EOF {
        let (expr, expr_consumed) = match &tokens[consumed] {
            Token::LeftBrace => parse_block(&tokens[consumed..])?,
            _ => parse_expression(&tokens[consumed..], 0)?,
        };
        consumed += expr_consumed;

        // Require semicolons after all expressions including blocks
        if consumed < tokens.len() && tokens[consumed] == Token::Semicolon {
            consumed += 1;
        } else if consumed < tokens.len() && tokens[consumed] != Token::EOF {
            return Err(Error::Parser {
                message: "Expected ';' after expression".to_string(),
            });
        }

        exprs.push(expr);
    }

    Ok(Expr::Block(exprs))
}

fn parse_expression(tokens: &[Token], precedence: u8) -> Result<(Expr, usize)> {
    if tokens.is_empty() || tokens[0] == Token::EOF {
        return Err(Error::Parser {
            message: "Expected expression".to_string(),
        });
    }

    match &tokens[0] {
        Token::Let => parse_let(tokens),
        Token::If => parse_if(tokens),
        Token::While => parse_while(tokens),
        Token::Fn => parse_function(tokens),
        Token::LeftBrace => parse_block(tokens),
        _ => parse_operator_precedence(tokens, precedence),
    }
}

fn parse_scope(tokens: &[Token]) -> Result<(Vec<Expr>, usize)> {
    let mut consumed = 0;
    let mut expressions = Vec::new();

    while consumed < tokens.len() {
        if tokens[consumed] == Token::RightBrace {
            break;
        }

        let (expr, expr_consumed) = match &tokens[consumed] {
            Token::LeftBrace => parse_block(&tokens[consumed..])?,
            _ => parse_expression(&tokens[consumed..], 0)?,
        };
        consumed += expr_consumed;

        // Handle semicolons between expressions inside blocks
        if consumed < tokens.len() && tokens[consumed] == Token::Semicolon {
            consumed += 1;
        } else if consumed < tokens.len()
            && tokens[consumed] != Token::EOF
            && tokens[consumed] != Token::RightBrace
        {
            return Err(Error::Parser {
                message: "Expected ';' after expression".to_string(),
            });
        }

        expressions.push(expr);
    }

    Ok((expressions, consumed))
}

fn parse_let(tokens: &[Token]) -> Result<(Expr, usize)> {
    let mut consumed = 1; // Skip 'let'

    if consumed >= tokens.len() {
        return Err(Error::Parser {
            message: "Expected identifier after 'let'".to_string(),
        });
    }

    // Parse the variable name
    let name = match &tokens[consumed] {
        Token::Identifier(s) => s.clone(),
        _ => {
            return Err(Error::Parser {
                message: "Expected identifier after 'let'".to_string(),
            })
        }
    };
    consumed += 1;

    // Parse the equals sign
    if consumed >= tokens.len() || tokens[consumed] != Token::Equal {
        return Err(Error::Parser {
            message: "Expected '=' after identifier in let".to_string(),
        });
    }
    consumed += 1;

    // Parse the initializer expression
    let (initializer, init_consumed) = parse_expression(&tokens[consumed..], 0)?;
    consumed += init_consumed;

    Ok((
        Expr::Let {
            name,
            initializer: Box::new(initializer),
        },
        consumed,
    ))
}

fn parse_block(tokens: &[Token]) -> Result<(Expr, usize)> {
    let mut consumed = 1; // Skip '{'

    // Handle empty blocks
    if consumed < tokens.len() && tokens[consumed] == Token::RightBrace {
        return Ok((Expr::Block(Vec::new()), consumed + 1));
    }

    let (expressions, scope_consumed) = parse_scope(&tokens[consumed..])?;
    consumed += scope_consumed;

    // Consume the closing brace
    if consumed >= tokens.len() || tokens[consumed] != Token::RightBrace {
        return Err(Error::Parser {
            message: "Unterminated block".to_string(),
        });
    }
    consumed += 1;

    // Check if we should add an implicit nil
    let mut final_expressions = expressions;

    // Add implicit nil if the last expression was terminated with a semicolon
    if !final_expressions.is_empty() {
        let last_token_pos = consumed - 2; // Position before the closing brace
        if last_token_pos < tokens.len() && tokens[last_token_pos] == Token::Semicolon {
            final_expressions.push(Expr::Literal(Literal::Nil));
        }
    }

    Ok((Expr::Block(final_expressions), consumed))
}

fn parse_if(tokens: &[Token]) -> Result<(Expr, usize)> {
    let mut consumed = 1; // Skip 'if'

    let (condition, cond_consumed) = parse_expression(&tokens[consumed..], 0)?;
    consumed += cond_consumed;

    let (then_branch, then_consumed) = parse_expression(&tokens[consumed..], 0)?;
    consumed += then_consumed;

    let (else_branch, else_consumed) = if consumed < tokens.len() && tokens[consumed] == Token::Else
    {
        consumed += 1;
        let (expr, count) = parse_expression(&tokens[consumed..], 0)?;
        (expr, count)
    } else {
        (Expr::Literal(Literal::Nil), 0)
    };

    Ok((
        Expr::If {
            condition: Box::new(condition),
            then_branch: Box::new(then_branch),
            else_branch: Box::new(else_branch),
        },
        consumed + else_consumed,
    ))
}

fn parse_while(tokens: &[Token]) -> Result<(Expr, usize)> {
    let mut consumed = 1; // Skip 'while'

    if consumed >= tokens.len() {
        return Err(Error::Parser {
            message: "Expected condition after 'while'".to_string(),
        });
    }

    let (condition, cond_consumed) = parse_expression(&tokens[consumed..], 0)?;
    consumed += cond_consumed;

    if consumed >= tokens.len() {
        return Err(Error::Parser {
            message: "Expected body after while condition".to_string(),
        });
    }

    let (body, body_consumed) = parse_expression(&tokens[consumed..], 0)?;
    consumed += body_consumed;

    Ok((
        Expr::While {
            condition: Box::new(condition),
            body: Box::new(body),
        },
        consumed,
    ))
}

fn parse_function(tokens: &[Token]) -> Result<(Expr, usize)> {
    let mut consumed = 1; // Skip 'fn'

    if consumed >= tokens.len() || tokens[consumed] != Token::LeftParen {
        return Err(Error::Parser {
            message: "Expected '(' after 'fn'".to_string(),
        });
    }
    consumed += 1;

    let mut params = Vec::new();
    while consumed < tokens.len() && tokens[consumed] != Token::RightParen {
        if !params.is_empty() {
            if tokens[consumed] != Token::Comma {
                return Err(Error::Parser {
                    message: "Expected ',' between parameters".to_string(),
                });
            }
            consumed += 1;

            if consumed >= tokens.len() {
                return Err(Error::Parser {
                    message: "Expected parameter after ','".to_string(),
                });
            }
        }

        match &tokens[consumed] {
            Token::Identifier(name) => params.push(name.clone()),
            _ => {
                return Err(Error::Parser {
                    message: "Expected parameter name".to_string(),
                })
            }
        }
        consumed += 1;
    }

    if consumed >= tokens.len() {
        return Err(Error::Parser {
            message: "Unterminated function parameters".to_string(),
        });
    }
    consumed += 1; // Skip ')'

    if consumed >= tokens.len() {
        return Err(Error::Parser {
            message: "Expected function body".to_string(),
        });
    }

    let (body, body_consumed) = parse_expression(&tokens[consumed..], 0)?;
    consumed += body_consumed;

    Ok((
        Expr::Function {
            params,
            body: Box::new(body),
        },
        consumed,
    ))
}

fn parse_operator_precedence(tokens: &[Token], precedence: u8) -> Result<(Expr, usize)> {
    let (mut expr, mut consumed) = parse_prefix(tokens)?;

    loop {
        if consumed >= tokens.len() {
            break;
        }

        // Handle function calls with high precedence
        if tokens[consumed] == Token::LeftParen {
            consumed += 1;
            let mut args = Vec::new();

            while consumed < tokens.len() && tokens[consumed] != Token::RightParen {
                if !args.is_empty() {
                    if tokens[consumed] != Token::Comma {
                        return Err(Error::Parser {
                            message: "Expected ',' between arguments".to_string(),
                        });
                    }
                    consumed += 1;
                }
                let (arg, arg_consumed) = parse_expression(&tokens[consumed..], 0)?;
                args.push(arg);
                consumed += arg_consumed;
            }

            if consumed >= tokens.len() {
                return Err(Error::Parser {
                    message: "Expected ')' after arguments".to_string(),
                });
            }
            consumed += 1; // consume the right paren

            expr = Expr::Call {
                callee: Box::new(expr),
                arguments: args,
            };
            continue;
        }

        // Handle assignment
        if tokens[consumed] == Token::Equal {
            if precedence > 1 {
                // Assignment has lowest precedence
                break;
            }

            let name = match expr {
                Expr::Variable(ref name) => name.clone(),
                _ => {
                    return Err(Error::Parser {
                        message: "Invalid assignment target".to_string(),
                    })
                }
            };

            consumed += 1;
            let (value, value_consumed) = parse_expression(&tokens[consumed..], 1)?;
            consumed += value_consumed;

            expr = Expr::Assign {
                name,
                value: Box::new(value),
            };
            continue;
        }

        let op_precedence = get_precedence(&tokens[consumed]);
        if precedence >= op_precedence {
            break;
        }

        let operator = match &tokens[consumed] {
            Token::Plus => BinaryOp::Add,
            Token::Minus => BinaryOp::Subtract,
            Token::Star => BinaryOp::Multiply,
            Token::Slash => BinaryOp::Divide,
            Token::Percent => BinaryOp::Modulo,
            Token::EqualEqual => BinaryOp::Equal,
            Token::BangEqual => BinaryOp::NotEqual,
            Token::Less => BinaryOp::Less,
            Token::LessEqual => BinaryOp::LessEqual,
            Token::Greater => BinaryOp::Greater,
            Token::GreaterEqual => BinaryOp::GreaterEqual,
            Token::Concat => BinaryOp::Concat,
            Token::And => BinaryOp::And,
            Token::Or => BinaryOp::Or,
            _ => break,
        };
        consumed += 1;

        let (right, right_consumed) = parse_expression(&tokens[consumed..], op_precedence)?;
        consumed += right_consumed;

        expr = Expr::Binary {
            left: Box::new(expr),
            operator,
            right: Box::new(right),
        };
    }

    Ok((expr, consumed))
}

fn parse_prefix(tokens: &[Token]) -> Result<(Expr, usize)> {
    match &tokens[0] {
        Token::Number(n) => Ok((Expr::Literal(Literal::Number(*n)), 1)),
        Token::String(s) => Ok((Expr::Literal(Literal::String(s.clone())), 1)),
        Token::True => Ok((Expr::Literal(Literal::Boolean(true)), 1)),
        Token::False => Ok((Expr::Literal(Literal::Boolean(false)), 1)),
        Token::Nil => Ok((Expr::Literal(Literal::Nil), 1)),
        Token::Identifier(name) => Ok((Expr::Variable(name.clone()), 1)),
        Token::LeftParen => {
            let (expr, consumed) = parse_expression(&tokens[1..], 0)?;
            if consumed + 1 >= tokens.len() || tokens[consumed + 1] != Token::RightParen {
                return Err(Error::Parser {
                    message: "Expected ')'".to_string(),
                });
            }
            Ok((Expr::Grouping(Box::new(expr)), consumed + 2))
        }
        Token::Minus => {
            let (right, consumed) = parse_expression(&tokens[1..], 8)?;
            Ok((
                Expr::Unary {
                    operator: UnaryOp::Negate,
                    right: Box::new(right),
                },
                consumed + 1,
            ))
        }
        Token::Bang => {
            let (right, consumed) = parse_expression(&tokens[1..], 8)?;
            Ok((
                Expr::Unary {
                    operator: UnaryOp::Not,
                    right: Box::new(right),
                },
                consumed + 1,
            ))
        }
        _ => Err(Error::Parser {
            message: "Expected expression".to_string(),
        }),
    }
}

fn get_precedence(token: &Token) -> u8 {
    match token {
        Token::Star | Token::Slash | Token::Percent => 7,
        Token::Plus | Token::Minus => 6,
        Token::Concat => 5,
        Token::EqualEqual
        | Token::BangEqual
        | Token::Less
        | Token::LessEqual
        | Token::Greater
        | Token::GreaterEqual => 4,
        Token::And => 3,
        Token::Or => 2,
        Token::Equal => 1,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::tokenize;

    // Helper function to tokenize and parse a string
    fn parse_str(input: &str) -> Result<Expr> {
        let tokens = tokenize(input.as_bytes())?;
        parse(&tokens)
    }

    #[test]
    fn test_literals() -> Result<()> {
        if let Expr::Block(exprs) = parse_str("42;")? {
            assert_eq!(exprs.len(), 1);
            assert!(matches!(exprs[0], Expr::Literal(Literal::Number(42.0))));
        }

        if let Expr::Block(exprs) = parse_str("\"hello\";")? {
            assert_eq!(exprs.len(), 1);
            assert!(matches!(&exprs[0], Expr::Literal(Literal::String(s)) if s == "hello"));
        }
        Ok(())
    }

    #[test]
    fn test_multiple_top_level_declarations() -> Result<()> {
        // Test multiple let declarations
        let input = r#"
            let x = 1;
            let y = 2;
            let z = 3;"#;

        if let Expr::Block(exprs) = parse_str(input)? {
            assert_eq!(exprs.len(), 3);
            for expr in exprs {
                assert!(matches!(expr, Expr::Let { .. }));
            }
        }

        // Test function definitions
        let input = r#"
            let f = fn() { 1 };
            let g = fn() { 2 };"#;

        if let Expr::Block(exprs) = parse_str(input)? {
            assert_eq!(exprs.len(), 2);
            for expr in exprs {
                if let Expr::Let { initializer, .. } = expr {
                    assert!(matches!(*initializer, Expr::Function { .. }));
                } else {
                    panic!("Expected let binding with function");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_function_definitions() -> Result<()> {
        let input = r#"
            let func = fn(x, y) {
                let z = x + y;
                z * 2
            };"#;

        if let Expr::Block(exprs) = parse_str(input)? {
            assert_eq!(exprs.len(), 1);
            if let Expr::Let { initializer, .. } = &exprs[0] {
                if let Expr::Function { params, body } = &**initializer {
                    assert_eq!(params, &vec!["x", "y"]);
                    if let Expr::Block(body_exprs) = &**body {
                        assert_eq!(body_exprs.len(), 2);
                    } else {
                        panic!("Expected block as function body");
                    }
                } else {
                    panic!("Expected function expression");
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_nested_blocks() -> Result<()> {
        let input = r#"{
            let x = 1;
            {
                let y = 2;
                x + y
            }
        }"#;

        if let Expr::Block(outer) = parse_str(input)? {
            assert_eq!(outer.len(), 1);
            if let Expr::Block(inner) = &outer[0] {
                assert_eq!(inner.len(), 2);
                assert!(matches!(inner[0], Expr::Let { .. }));
                assert!(matches!(inner[1], Expr::Block(_)));
            }
        }
        Ok(())
    }

    #[test]
    fn test_implicit_nil() -> Result<()> {
        // Test block with semicolon-terminated expression
        let input = "{ print(\"hello\"); }";
        if let Expr::Block(outer) = parse_str(input)? {
            assert_eq!(outer.len(), 1); // The outer block has one expression
            if let Expr::Block(inner) = &outer[0] {
                assert_eq!(inner.len(), 2); // print + nil
                assert!(matches!(inner[0], Expr::Call { .. }));
                assert!(matches!(inner[1], Expr::Literal(Literal::Nil)));
            }
        }

        // Test block without semicolon-terminated expression
        let input = "{ print(\"hello\") }";
        if let Expr::Block(outer) = parse_str(input)? {
            assert_eq!(outer.len(), 1);
            if let Expr::Block(inner) = &outer[0] {
                assert_eq!(inner.len(), 1);
                assert!(matches!(inner[0], Expr::Call { .. }));
            }
        }

        // Test nested blocks with semicolons
        let input = r#"{ 
        print("outer");
        {
            print("inner");
        }
    }"#;
        if let Expr::Block(outer) = parse_str(input)? {
            assert_eq!(outer.len(), 1); // The outer parse_str block
            if let Expr::Block(block1) = &outer[0] {
                assert_eq!(block1.len(), 2); // outer print and inner block. inner block has no ;
                assert!(matches!(block1[0], Expr::Call { .. }));
                if let Expr::Block(block2) = &block1[1] {
                    assert_eq!(block2.len(), 2); // inner print, nil
                    assert!(matches!(block2[0], Expr::Call { .. }));
                    assert!(matches!(block2[1], Expr::Literal(Literal::Nil)));
                }
            }
        }

        // Test the implicit nil in a terminated let expression
        let input = "let x = { print(\"hello\"); };";
        if let Expr::Block(outer) = parse_str(input)? {
            assert_eq!(outer.len(), 1);
            if let Expr::Let { initializer, .. } = &outer[0] {
                if let Expr::Block(block) = &**initializer {
                    assert_eq!(block.len(), 2);
                    assert!(matches!(block[0], Expr::Call { .. }));
                    assert!(matches!(block[1], Expr::Literal(Literal::Nil)));
                }
            }
        }

        Ok(())
    }

    #[test]
    fn test_while_loops() -> Result<()> {
        let input = r#"
            let x = 0;
            while x < 10 {
                x = x + 1;
                print(x);
            };"#;

        if let Expr::Block(exprs) = parse_str(input)? {
            assert_eq!(exprs.len(), 2);
            assert!(matches!(exprs[0], Expr::Let { .. }));
            assert!(matches!(exprs[1], Expr::While { .. }));
        }
        Ok(())
    }

    #[test]
    fn test_if_expressions() -> Result<()> {
        let input = r#"
            if x == 1 {
                print("one");
            } else {
                print("not one");
            };"#;

        if let Expr::Block(exprs) = parse_str(input)? {
            assert_eq!(exprs.len(), 1);
            if let Expr::If {
                condition,
                then_branch,
                else_branch,
            } = &exprs[0]
            {
                assert!(matches!(**condition, Expr::Binary { .. }));
                assert!(matches!(**then_branch, Expr::Block(_)));
                assert!(matches!(**else_branch, Expr::Block(_)));
            }
        }
        Ok(())
    }

    #[test]
    fn test_function_calls() -> Result<()> {
        let input = r#"
            let f = fn(x) { x * 2 };
            f(21);"#;

        if let Expr::Block(exprs) = parse_str(input)? {
            assert_eq!(exprs.len(), 2);
            assert!(matches!(exprs[0], Expr::Let { .. }));
            assert!(matches!(exprs[1], Expr::Call { .. }));
        }
        Ok(())
    }

    #[test]
    fn test_complex_program() -> Result<()> {
        // Test the magic 8-ball example structure
        let input = r#"
            let get_prediction = fn(num) {
                if num == 1 { println("Yes"); };
                if num == 2 { println("No"); };
            };
            let get_rand = fn() { rand_int_range(1, 2) };
            let main = fn() {
                get_prediction(get_rand());
            };
            main()"#;

        if let Expr::Block(exprs) = parse_str(input)? {
            assert_eq!(exprs.len(), 4);
            // First three expressions should be let bindings
            (0..3).for_each(|i| {
                assert!(matches!(exprs[i], Expr::Let { .. }));
            });
            // Last expression should be a function call
            assert!(matches!(exprs[3], Expr::Call { .. }));
        }
        Ok(())
    }

    #[test]
    fn test_error_cases() -> Result<()> {
        assert!(parse_str("let;").is_err());
        assert!(parse_str("let x;").is_err());
        assert!(parse_str("let x = ;").is_err());
        assert!(parse_str("let x = 1\nprint(x);").is_err()); // Missing semicolon
        assert!(parse_str("fn() {").is_err()); // Unterminated block
        assert!(parse_str("1 + ;").is_err()); // Missing operand
        assert!(parse_str("if { 1 }").is_err()); // Missing condition
        Ok(())
    }
}
