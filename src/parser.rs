use crate::{
    error::{parser_error, Result},
    tokenizer::{Token, TokenType},
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
    List(Vec<Expr>),
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

pub fn parse(source: &[u8], tokens: &[Token]) -> Result<Expr> {
    let mut consumed = 0;
    let mut exprs = Vec::new();

    while consumed < tokens.len() && tokens[consumed].token_type != TokenType::EOF {
        let (expr, expr_consumed) = match &tokens[consumed].token_type {
            TokenType::LeftBrace => parse_block(source, &tokens[consumed..])?,
            _ => parse_expression(source, &tokens[consumed..], 0)?,
        };
        consumed += expr_consumed;

        // Require semicolons after all expressions including blocks
        if consumed < tokens.len() && tokens[consumed].token_type == TokenType::Semicolon {
            consumed += 1;
        } else if consumed < tokens.len() && tokens[consumed].token_type != TokenType::EOF {
            return parser_error(source, "Expected ';' after expression", &tokens[consumed]);
        }

        exprs.push(expr);
    }

    Ok(Expr::Block(exprs))
}

fn parse_expression(source: &[u8], tokens: &[Token], precedence: u8) -> Result<(Expr, usize)> {
    if tokens.is_empty() || tokens[0].token_type == TokenType::EOF {
        return parser_error(source, "Expected expression got EOF", &tokens[0]);
    }

    match &tokens[0].token_type {
        TokenType::Let => parse_let(source, tokens),
        TokenType::If => parse_if(source, tokens),
        TokenType::While => parse_while(source, tokens),
        TokenType::Fn => parse_function(source, tokens),
        TokenType::LeftBrace => parse_block(source, tokens),
        _ => parse_operator_precedence(source, tokens, precedence),
    }
}

fn parse_scope(source: &[u8], tokens: &[Token]) -> Result<(Vec<Expr>, usize)> {
    let mut consumed = 0;
    let mut expressions = Vec::new();

    while consumed < tokens.len() {
        if tokens[consumed].token_type == TokenType::RightBrace {
            break;
        }

        let (expr, expr_consumed) = match &tokens[consumed].token_type {
            TokenType::LeftBrace => parse_block(source, &tokens[consumed..])?,
            _ => parse_expression(source, &tokens[consumed..], 0)?,
        };
        consumed += expr_consumed;

        // Handle semicolons between expressions inside blocks
        if consumed < tokens.len() && tokens[consumed].token_type == TokenType::Semicolon {
            consumed += 1;
        } else if consumed < tokens.len()
            && tokens[consumed].token_type != TokenType::EOF
            && tokens[consumed].token_type != TokenType::RightBrace
        {
            return parser_error(
                source,
                "Expected ';' after expression",
                &tokens[consumed - 1],
            );
        }

        expressions.push(expr);
    }

    Ok((expressions, consumed))
}

fn parse_let(source: &[u8], tokens: &[Token]) -> Result<(Expr, usize)> {
    let mut consumed = 1; // Skip 'let'

    if consumed >= tokens.len() {
        return parser_error(source, "Expected identifier after 'let'", &tokens[0]);
    }

    // Parse the variable name
    let name = match &tokens[consumed].token_type {
        TokenType::Identifier(s) => s.clone(),
        _ => {
            return parser_error(source, "Expected identifier after 'let'", &tokens[consumed]);
        }
    };
    consumed += 1;

    // Parse the equals sign
    if consumed >= tokens.len() || tokens[consumed].token_type != TokenType::Equal {
        return parser_error(
            source,
            "Expected '=' after identifier in let",
            &tokens[tokens.len() - 1],
        );
    }
    consumed += 1;

    // Parse the initializer expression
    let (initializer, init_consumed) = parse_expression(source, &tokens[consumed..], 0)?;
    consumed += init_consumed;

    Ok((
        Expr::Let {
            name,
            initializer: Box::new(initializer),
        },
        consumed,
    ))
}

fn parse_block(source: &[u8], tokens: &[Token]) -> Result<(Expr, usize)> {
    let mut consumed = 1; // Skip '{'

    // Handle empty blocks
    if consumed < tokens.len() && tokens[consumed].token_type == TokenType::RightBrace {
        return Ok((Expr::Block(Vec::new()), consumed + 1));
    }

    let (expressions, scope_consumed) = parse_scope(source, &tokens[consumed..])?;
    consumed += scope_consumed;

    // Consume the closing brace
    if consumed >= tokens.len() || tokens[consumed].token_type != TokenType::RightBrace {
        return parser_error(
            source,
            "Unterminated block",
            &tokens[consumed.saturating_sub(1).min(tokens.len() - 1)],
        );
    }
    consumed += 1;

    // Check if we should add an implicit nil
    let mut final_expressions = expressions;

    // Add implicit nil if the last expression was terminated with a semicolon
    if !final_expressions.is_empty() {
        let last_token_pos = consumed - 2; // Position before the closing brace
        if last_token_pos < tokens.len()
            && tokens[last_token_pos].token_type == TokenType::Semicolon
        {
            final_expressions.push(Expr::Literal(Literal::Nil));
        }
    }

    Ok((Expr::Block(final_expressions), consumed))
}

fn parse_if(source: &[u8], tokens: &[Token]) -> Result<(Expr, usize)> {
    let mut consumed = 1; // Skip 'if'

    let (condition, cond_consumed) = parse_expression(source, &tokens[consumed..], 0)?;
    consumed += cond_consumed;

    let (then_branch, then_consumed) = parse_expression(source, &tokens[consumed..], 0)?;
    consumed += then_consumed;

    let (else_branch, else_consumed) =
        if consumed < tokens.len() && tokens[consumed].token_type == TokenType::Else {
            consumed += 1;
            let (expr, count) = parse_expression(source, &tokens[consumed..], 0)?;
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

fn parse_while(source: &[u8], tokens: &[Token]) -> Result<(Expr, usize)> {
    let mut consumed = 1; // Skip 'while'

    if consumed >= tokens.len() {
        return parser_error(source, "Expected condition after 'while'", &tokens[0]);
    }

    let (condition, cond_consumed) = parse_expression(source, &tokens[consumed..], 0)?;
    consumed += cond_consumed;

    if consumed >= tokens.len() {
        return parser_error(
            source,
            "Expected body after while condition",
            &tokens[consumed.saturating_sub(1).min(tokens.len() - 1)],
        );
    }

    let (body, body_consumed) = parse_expression(source, &tokens[consumed..], 0)?;
    consumed += body_consumed;

    Ok((
        Expr::While {
            condition: Box::new(condition),
            body: Box::new(body),
        },
        consumed,
    ))
}

fn parse_function(source: &[u8], tokens: &[Token]) -> Result<(Expr, usize)> {
    let mut consumed = 1; // Skip 'fn'

    if consumed >= tokens.len() || tokens[consumed].token_type != TokenType::LeftParen {
        return parser_error(source, "Expected '(' after 'fn'", &tokens[tokens.len() - 1]);
    }
    consumed += 1;

    let mut params = Vec::new();
    while consumed < tokens.len() && tokens[consumed].token_type != TokenType::RightParen {
        if !params.is_empty() {
            if tokens[consumed].token_type != TokenType::Comma {
                return parser_error(source, "Expected ',' between parameters", &tokens[consumed]);
            }
            consumed += 1;

            if consumed >= tokens.len() {
                return parser_error(
                    source,
                    "Expected parameter after ','",
                    &tokens[consumed.saturating_sub(1).min(tokens.len() - 1)],
                );
            }
        }

        match &tokens[consumed].token_type {
            TokenType::Identifier(name) => params.push(name.clone()),
            _ => {
                return parser_error(source, "Expected parameter name", &tokens[consumed]);
            }
        }
        consumed += 1;
    }

    if consumed >= tokens.len() {
        return parser_error(
            source,
            "Unterminated function parameters",
            &tokens[consumed.saturating_sub(1).min(tokens.len() - 1)],
        );
    }
    consumed += 1; // Skip ')'

    if consumed >= tokens.len() {
        return parser_error(
            source,
            "Expected function body",
            &tokens[consumed.saturating_sub(1).min(tokens.len() - 1)],
        );
    }

    let (body, body_consumed) = parse_expression(source, &tokens[consumed..], 0)?;
    consumed += body_consumed;

    Ok((
        Expr::Function {
            params,
            body: Box::new(body),
        },
        consumed,
    ))
}

fn parse_list(source: &[u8], tokens: &[Token]) -> Result<(Expr, usize)> {
    let mut consumed = 1; // Skip '['
    let mut elements = Vec::new();

    // Handle empty list case
    if consumed < tokens.len() && tokens[consumed].token_type == TokenType::RightSquare {
        return Ok((Expr::List(elements), consumed + 1));
    }

    // Parse first element
    let (first_elem, first_consumed) = parse_expression(source, &tokens[consumed..], 0)?;
    elements.push(first_elem);
    consumed += first_consumed;

    // Parse remaining elements
    while consumed < tokens.len() && tokens[consumed].token_type == TokenType::Comma {
        consumed += 1; // Skip comma

        // Allow trailing comma
        if tokens[consumed].token_type == TokenType::RightSquare {
            break;
        }

        let (elem, elem_consumed) = parse_expression(source, &tokens[consumed..], 0)?;
        elements.push(elem);
        consumed += elem_consumed;
    }

    // Expect closing bracket
    if consumed >= tokens.len() || tokens[consumed].token_type != TokenType::RightSquare {
        return parser_error(
            source,
            "Expected ']' after list elements",
            &tokens[tokens.len() - 1],
        );
    }
    consumed += 1; // Skip ']'

    Ok((Expr::List(elements), consumed))
}

fn parse_operator_precedence(
    source: &[u8],
    tokens: &[Token],
    precedence: u8,
) -> Result<(Expr, usize)> {
    let (mut expr, mut consumed) = parse_prefix(source, tokens)?;

    loop {
        if consumed >= tokens.len() {
            break;
        }

        // Handle function calls with high precedence
        if tokens[consumed].token_type == TokenType::LeftParen {
            consumed += 1;
            let mut args = Vec::new();

            while consumed < tokens.len() && tokens[consumed].token_type != TokenType::RightParen {
                if !args.is_empty() {
                    if tokens[consumed].token_type != TokenType::Comma {
                        return parser_error(
                            source,
                            "Expected ',' between arguments",
                            &tokens[consumed],
                        );
                    }
                    consumed += 1;
                }
                let (arg, arg_consumed) = parse_expression(source, &tokens[consumed..], 0)?;
                args.push(arg);
                consumed += arg_consumed;
            }

            if consumed >= tokens.len() {
                return parser_error(
                    source,
                    "Expected ')' after arguments",
                    &tokens[consumed.saturating_sub(1).min(tokens.len() - 1)],
                );
            }
            consumed += 1; // consume the right paren

            expr = Expr::Call {
                callee: Box::new(expr),
                arguments: args,
            };
            continue;
        }

        // Handle assignment
        if tokens[consumed].token_type == TokenType::Equal {
            if precedence > 1 {
                // Assignment has lowest precedence
                break;
            }

            let name = match expr {
                Expr::Variable(ref name) => name.clone(),
                _ => {
                    return parser_error(
                        source,
                        "Invalid assignment target",
                        &tokens[consumed - 1],
                    );
                }
            };

            consumed += 1;
            let (value, value_consumed) = parse_expression(source, &tokens[consumed..], 1)?;
            consumed += value_consumed;

            expr = Expr::Assign {
                name,
                value: Box::new(value),
            };
            continue;
        }

        let op_precedence = get_precedence(&tokens[consumed].token_type);
        if precedence >= op_precedence {
            break;
        }

        let operator = match &tokens[consumed].token_type {
            TokenType::Plus => BinaryOp::Add,
            TokenType::Minus => BinaryOp::Subtract,
            TokenType::Star => BinaryOp::Multiply,
            TokenType::Slash => BinaryOp::Divide,
            TokenType::Percent => BinaryOp::Modulo,
            TokenType::EqualEqual => BinaryOp::Equal,
            TokenType::BangEqual => BinaryOp::NotEqual,
            TokenType::Less => BinaryOp::Less,
            TokenType::LessEqual => BinaryOp::LessEqual,
            TokenType::Greater => BinaryOp::Greater,
            TokenType::GreaterEqual => BinaryOp::GreaterEqual,
            TokenType::Concat => BinaryOp::Concat,
            TokenType::And => BinaryOp::And,
            TokenType::Or => BinaryOp::Or,
            _ => break,
        };
        consumed += 1;

        let (right, right_consumed) = parse_expression(source, &tokens[consumed..], op_precedence)?;
        consumed += right_consumed;

        expr = Expr::Binary {
            left: Box::new(expr),
            operator,
            right: Box::new(right),
        };
    }

    Ok((expr, consumed))
}

fn parse_prefix(source: &[u8], tokens: &[Token]) -> Result<(Expr, usize)> {
    match &tokens[0].token_type {
        TokenType::Number(n) => Ok((Expr::Literal(Literal::Number(*n)), 1)),
        TokenType::String(s) => Ok((Expr::Literal(Literal::String(s.clone())), 1)),
        TokenType::True => Ok((Expr::Literal(Literal::Boolean(true)), 1)),
        TokenType::False => Ok((Expr::Literal(Literal::Boolean(false)), 1)),
        TokenType::Nil => Ok((Expr::Literal(Literal::Nil), 1)),
        TokenType::Identifier(name) => Ok((Expr::Variable(name.clone()), 1)),
        TokenType::LeftSquare => parse_list(source, tokens),
        TokenType::LeftParen => {
            let (expr, consumed) = parse_expression(source, &tokens[1..], 0)?;
            if consumed + 1 >= tokens.len()
                || tokens[consumed + 1].token_type != TokenType::RightParen
            {
                return parser_error(source, "Expected ')'", &tokens[consumed + 1]);
            }
            Ok((Expr::Grouping(Box::new(expr)), consumed + 2))
        }
        TokenType::Minus => {
            let (right, consumed) = parse_expression(source, &tokens[1..], 8)?;
            Ok((
                Expr::Unary {
                    operator: UnaryOp::Negate,
                    right: Box::new(right),
                },
                consumed + 1,
            ))
        }
        TokenType::Bang => {
            let (right, consumed) = parse_expression(source, &tokens[1..], 8)?;
            Ok((
                Expr::Unary {
                    operator: UnaryOp::Not,
                    right: Box::new(right),
                },
                consumed + 1,
            ))
        }
        _ => parser_error(source, "Expected expression", &tokens[0]),
    }
}

fn get_precedence(token: &TokenType) -> u8 {
    match token {
        TokenType::Star | TokenType::Slash | TokenType::Percent => 7,
        TokenType::Plus | TokenType::Minus => 6,
        TokenType::Concat => 5,
        TokenType::EqualEqual
        | TokenType::BangEqual
        | TokenType::Less
        | TokenType::LessEqual
        | TokenType::Greater
        | TokenType::GreaterEqual => 4,
        TokenType::And => 3,
        TokenType::Or => 2,
        TokenType::Equal => 1,
        _ => 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokenizer::tokenize;

    // Helper function to tokenize and parse a string
    fn parse_str(input: &str) -> Result<Expr> {
        let source = input.as_bytes();
        let tokens = tokenize(source)?;
        parse(source, &tokens)
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
    fn test_list_literals() -> Result<()> {
        // Test empty list
        if let Expr::Block(exprs) = parse_str("[];")? {
            assert_eq!(exprs.len(), 1);
            assert!(matches!(&exprs[0], Expr::List(elements) if elements.is_empty()));
        }

        // Test simple list
        if let Expr::Block(exprs) = parse_str("[1, 2, 3];")? {
            assert_eq!(exprs.len(), 1);
            if let Expr::List(elements) = &exprs[0] {
                assert_eq!(elements.len(), 3);
                assert!(matches!(&elements[0], Expr::Literal(Literal::Number(1.0))));
                assert!(matches!(&elements[1], Expr::Literal(Literal::Number(2.0))));
                assert!(matches!(&elements[2], Expr::Literal(Literal::Number(3.0))));
            }
        }

        // Test mixed types
        if let Expr::Block(exprs) = parse_str("[1, \"hello\", true];")? {
            assert_eq!(exprs.len(), 1);
            if let Expr::List(elements) = &exprs[0] {
                assert_eq!(elements.len(), 3);
                assert!(matches!(&elements[0], Expr::Literal(Literal::Number(1.0))));
                assert!(matches!(&elements[1], Expr::Literal(Literal::String(s)) if s == "hello"));
                assert!(matches!(
                    &elements[2],
                    Expr::Literal(Literal::Boolean(true))
                ));
            }
        }

        // Test nested lists
        if let Expr::Block(exprs) = parse_str("[[1, 2], [3, 4]];")? {
            assert_eq!(exprs.len(), 1);
            if let Expr::List(elements) = &exprs[0] {
                assert_eq!(elements.len(), 2);
                for inner in elements {
                    assert!(matches!(inner, Expr::List(_)));
                }
            }
        }

        // Test trailing comma
        if let Expr::Block(exprs) = parse_str("[1, 2, 3,];")? {
            assert_eq!(exprs.len(), 1);
            if let Expr::List(elements) = &exprs[0] {
                assert_eq!(elements.len(), 3);
            }
        }

        // Test error cases
        assert!(parse_str("[").is_err()); // Unclosed list
        assert!(parse_str("[,]").is_err()); // Missing element after comma
        assert!(parse_str("[1 2]").is_err()); // Missing comma
        assert!(parse_str("[1, 2").is_err()); // Unclosed list

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
