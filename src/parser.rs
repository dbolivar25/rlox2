use crate::{
    error::{parser_error, Result},
    tokenizer::{Token, TokenType},
};
use std::ops::Range;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Expr {
    pub expr_type: ExprType,
    pub byte_span: Range<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExprType {
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
        recursive: bool,
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

#[derive(Debug, Clone)]
pub enum Literal {
    Number(f64),
    String(String),
    Boolean(bool),
    Function {
        params: Vec<String>,
        body: Box<Expr>,
    },
    List(Vec<Expr>),
    Nil,
}

impl PartialEq for Literal {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Literal::Number(a), Literal::Number(b)) => {
                const EPSILON: f64 = 1e-10;
                (a - b).abs() < EPSILON
            }
            (Literal::String(a), Literal::String(b)) => a == b,
            (Literal::Boolean(a), Literal::Boolean(b)) => a == b,
            (Literal::Function { .. }, Literal::Function { .. }) => false,
            (Literal::List(a), Literal::List(b)) => a == b,
            _ => std::mem::discriminant(self) == std::mem::discriminant(other),
        }
    }
}

impl Eq for Literal {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnaryOp {
    Negate,
    Not,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    // Assert tokens end with EOF
    assert!(
        tokens
            .last()
            .map_or(false, |t| t.token_type == TokenType::EOF),
        "Token slice must be terminated by EOF"
    );

    // Parse as a single block expression
    let (expr, consumed) = parse_scope(source, tokens)?;
    assert_eq!(consumed, tokens.len(), "Failed to consume all tokens");

    Ok(expr)
}

// Parse a sequence of expressions in a block context
fn parse_scope(source: &[u8], tokens: &[Token]) -> Result<(Expr, usize)> {
    let start = tokens.first().map_or(0, |t| t.byte_span.start);
    let mut consumed = 0;
    let mut expressions = Vec::new();

    while consumed < tokens.len() {
        // If we hit EOF, consume it and break
        if tokens[consumed].token_type == TokenType::EOF {
            consumed += 1; // Consume the EOF token
            break;
        }

        // Break if we hit right brace without consuming it
        if tokens[consumed].token_type == TokenType::RightBrace {
            break;
        }

        // Parse next expression
        let (expr, expr_consumed) = parse_expression(source, &tokens[consumed..], 0)?;
        expressions.push(expr);
        consumed += expr_consumed;

        // Handle semicolon
        match tokens[consumed].token_type {
            TokenType::Semicolon => {
                let semicolon_span = tokens[consumed].byte_span.clone();
                consumed += 1;

                // Add implicit nil if this is the last expression
                if consumed < tokens.len()
                    && (tokens[consumed].token_type == TokenType::EOF
                        || tokens[consumed].token_type == TokenType::RightBrace)
                {
                    expressions.push(Expr {
                        expr_type: ExprType::Literal(Literal::Nil),
                        byte_span: semicolon_span.end..semicolon_span.end, // Zero-width span after semicolon
                    });
                }
            }
            TokenType::EOF | TokenType::RightBrace => (),
            _ => {
                return parser_error(
                    source,
                    "Expected ';' after expression",
                    &tokens[consumed.saturating_sub(1)],
                )
            }
        }
    }

    // Calculate end position
    let end = if consumed > 0 {
        tokens[consumed - 1].byte_span.end
    } else {
        start
    };

    Ok((
        Expr {
            expr_type: ExprType::Block(expressions),
            byte_span: start..end,
        },
        consumed,
    ))
}

fn parse_expression(source: &[u8], tokens: &[Token], precedence: u8) -> Result<(Expr, usize)> {
    assert!(
        tokens
            .last()
            .map_or(false, |t| t.token_type == TokenType::EOF),
        "Token slice must be terminated by EOF"
    );

    if tokens.is_empty() || tokens[0].token_type == TokenType::EOF {
        return parser_error(source, "Expected expression", &tokens[0]);
    }

    // Parse prefix expression
    let (mut left, mut consumed) = parse_prefix(source, tokens)?;

    // Parse infix expressions while precedence allows
    while consumed < tokens.len() {
        let op_precedence = get_precedence(&tokens[consumed].token_type);
        if precedence >= op_precedence {
            break;
        }

        // Parse the operator and right side
        let (new_expr, op_consumed) =
            parse_infix(source, &tokens[consumed..], op_precedence, left)?;
        left = new_expr;
        consumed += op_consumed;
    }

    Ok((left, consumed))
}

fn parse_prefix(source: &[u8], tokens: &[Token]) -> Result<(Expr, usize)> {
    let start = tokens[0].byte_span.start;

    match &tokens[0].token_type {
        TokenType::Number(n) => Ok((
            Expr {
                expr_type: ExprType::Literal(Literal::Number(*n)),
                byte_span: tokens[0].byte_span.clone(),
            },
            1,
        )),
        TokenType::String(s) => Ok((
            Expr {
                expr_type: ExprType::Literal(Literal::String(s.clone())),
                byte_span: tokens[0].byte_span.clone(),
            },
            1,
        )),
        TokenType::True => Ok((
            Expr {
                expr_type: ExprType::Literal(Literal::Boolean(true)),
                byte_span: tokens[0].byte_span.clone(),
            },
            1,
        )),
        TokenType::False => Ok((
            Expr {
                expr_type: ExprType::Literal(Literal::Boolean(false)),
                byte_span: tokens[0].byte_span.clone(),
            },
            1,
        )),
        TokenType::Nil => Ok((
            Expr {
                expr_type: ExprType::Literal(Literal::Nil),
                byte_span: tokens[0].byte_span.clone(),
            },
            1,
        )),
        TokenType::Identifier(name) => {
            let mut consumed = 1;

            // Check for function call
            if consumed < tokens.len() && tokens[consumed].token_type == TokenType::LeftParen {
                consumed += 1;
                let mut arguments = Vec::new();

                // Parse arguments
                while consumed < tokens.len()
                    && tokens[consumed].token_type != TokenType::RightParen
                {
                    if !arguments.is_empty() {
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
                    arguments.push(arg);
                    consumed += arg_consumed;
                }

                // Expect closing parenthesis
                if consumed >= tokens.len() || tokens[consumed].token_type != TokenType::RightParen
                {
                    return parser_error(
                        source,
                        "Expected ')' after arguments",
                        &tokens[consumed.saturating_sub(1)],
                    );
                }
                consumed += 1;

                let end = tokens[consumed - 1].byte_span.end;
                Ok((
                    Expr {
                        expr_type: ExprType::Call {
                            callee: Box::new(Expr {
                                expr_type: ExprType::Variable(name.clone()),
                                byte_span: tokens[0].byte_span.clone(),
                            }),
                            arguments,
                        },
                        byte_span: start..end,
                    },
                    consumed,
                ))
            } else {
                Ok((
                    Expr {
                        expr_type: ExprType::Variable(name.clone()),
                        byte_span: tokens[0].byte_span.clone(),
                    },
                    consumed,
                ))
            }
        }
        TokenType::Let => parse_let(source, tokens),
        TokenType::If => parse_if(source, tokens),
        TokenType::While => parse_while(source, tokens),
        TokenType::Fn => parse_function(source, tokens),
        TokenType::LeftSquare => parse_list(source, tokens),
        TokenType::LeftParen => {
            let mut consumed = 1; // Skip '('

            // Parse the expression inside the parentheses
            let (expr, expr_consumed) = parse_expression(source, &tokens[consumed..], 0)?;
            consumed += expr_consumed;

            // Expect closing parenthesis
            if consumed >= tokens.len() || tokens[consumed].token_type != TokenType::RightParen {
                return parser_error(
                    source,
                    "Expected ')' after expression",
                    &tokens[consumed.saturating_sub(1)],
                );
            }
            consumed += 1; // Skip ')'

            // Check for function call
            if consumed < tokens.len() && tokens[consumed].token_type == TokenType::LeftParen {
                consumed += 1;
                let mut arguments = Vec::new();

                // Parse arguments
                while consumed < tokens.len()
                    && tokens[consumed].token_type != TokenType::RightParen
                {
                    if !arguments.is_empty() {
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
                    arguments.push(arg);
                    consumed += arg_consumed;
                }

                // Expect closing parenthesis
                if consumed >= tokens.len() || tokens[consumed].token_type != TokenType::RightParen
                {
                    return parser_error(
                        source,
                        "Expected ')' after arguments",
                        &tokens[consumed.saturating_sub(1)],
                    );
                }
                consumed += 1;

                let end = tokens[consumed - 1].byte_span.end;
                Ok((
                    Expr {
                        expr_type: ExprType::Call {
                            callee: Box::new(Expr {
                                expr_type: ExprType::Grouping(Box::new(expr)),
                                byte_span: start..end,
                            }),
                            arguments,
                        },
                        byte_span: start..end,
                    },
                    consumed,
                ))
            } else {
                let end = tokens[consumed - 1].byte_span.end;
                Ok((
                    Expr {
                        expr_type: ExprType::Grouping(Box::new(expr)),
                        byte_span: start..end,
                    },
                    consumed,
                ))
            }
        }
        TokenType::LeftBrace => {
            let mut consumed = 1; // Skip '{'
            let (block, block_consumed) = parse_scope(source, &tokens[consumed..])?;
            consumed += block_consumed;

            // Expect closing brace
            if consumed >= tokens.len() || tokens[consumed].token_type != TokenType::RightBrace {
                return parser_error(
                    source,
                    "Expected '}' after block",
                    &tokens[consumed.saturating_sub(1)],
                );
            }
            consumed += 1;

            let end = tokens[consumed - 1].byte_span.end;
            Ok((
                Expr {
                    expr_type: block.expr_type,
                    byte_span: start..end,
                },
                consumed,
            ))
        }
        TokenType::Minus | TokenType::Bang => {
            let operator = match tokens[0].token_type {
                TokenType::Minus => UnaryOp::Negate,
                TokenType::Bang => UnaryOp::Not,
                _ => unreachable!(),
            };

            let (right, right_consumed) = parse_expression(source, &tokens[1..], 8)?;
            let end = right.byte_span.end;

            Ok((
                Expr {
                    expr_type: ExprType::Unary {
                        operator,
                        right: Box::new(right),
                    },
                    byte_span: start..end,
                },
                right_consumed + 1,
            ))
        }
        _ => parser_error(source, "Expected expression", &tokens[0]),
    }
}

fn parse_infix(
    source: &[u8],
    tokens: &[Token],
    precedence: u8,
    left: Expr,
) -> Result<(Expr, usize)> {
    let start = left.byte_span.start;

    let operator = match &tokens[0].token_type {
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
        TokenType::Equal => {
            // Handle assignment
            if precedence > 1 {
                // Assignment has lowest precedence
                return parser_error(source, "Unexpected '='", &tokens[0]);
            }

            let name = match left.expr_type {
                ExprType::Variable(name) => name,
                _ => return parser_error(source, "Invalid assignment target", &tokens[0]),
            };

            let (value, consumed) = parse_expression(source, &tokens[1..], 1)?;
            let end = value.byte_span.end;

            return Ok((
                Expr {
                    expr_type: ExprType::Assign {
                        name,
                        value: Box::new(value),
                    },
                    byte_span: start..end,
                },
                consumed + 1,
            ));
        }
        _ => return parser_error(source, "Expected operator", &tokens[0]),
    };

    let consumed = 1; // Operator token
    let (right, right_consumed) = parse_expression(source, &tokens[consumed..], precedence)?;
    let end = right.byte_span.end;

    Ok((
        Expr {
            expr_type: ExprType::Binary {
                left: Box::new(left),
                operator,
                right: Box::new(right),
            },
            byte_span: start..end,
        },
        consumed + right_consumed,
    ))
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

fn parse_let(source: &[u8], tokens: &[Token]) -> Result<(Expr, usize)> {
    let start = tokens[0].byte_span.start;
    let mut consumed = 1; // Skip 'let'

    // Check for 'rec' keyword
    let recursive = if consumed < tokens.len() && tokens[consumed].token_type == TokenType::Rec {
        consumed += 1;
        true
    } else {
        false
    };

    // Parse identifier - check bounds first
    if consumed >= tokens.len() {
        return parser_error(
            source,
            "Expected identifier after 'let'",
            &tokens[consumed - 1],
        );
    }

    // Get the identifier
    let name = match &tokens[consumed].token_type {
        TokenType::Identifier(name) => name.clone(),
        _ => return parser_error(source, "Expected identifier after 'let'", &tokens[consumed]),
    };
    consumed += 1;

    // Parse equals sign
    if consumed >= tokens.len() || tokens[consumed].token_type != TokenType::Equal {
        return parser_error(
            source,
            "Expected '=' after identifier in let",
            &tokens[consumed - 1],
        );
    }
    consumed += 1;

    // Parse initializer
    let (initializer, init_consumed) = parse_expression(source, &tokens[consumed..], 0)?;
    consumed += init_consumed;

    // Get end position from last token
    let end = if consumed > 0 {
        tokens[consumed - 1].byte_span.end
    } else {
        tokens[0].byte_span.end
    };

    Ok((
        Expr {
            expr_type: ExprType::Let {
                name,
                initializer: Box::new(initializer),
                recursive,
            },
            byte_span: start..end,
        },
        consumed,
    ))
}

fn parse_if(source: &[u8], tokens: &[Token]) -> Result<(Expr, usize)> {
    let start = tokens[0].byte_span.start;
    let mut consumed = 1; // Skip 'if'

    // Parse condition
    let (condition, cond_consumed) = parse_expression(source, &tokens[consumed..], 0)?;
    consumed += cond_consumed;

    // Parse then branch
    let (then_branch, then_consumed) = parse_expression(source, &tokens[consumed..], 0)?;
    consumed += then_consumed;

    // Parse optional else branch
    let (else_branch, else_consumed) =
        if consumed < tokens.len() && tokens[consumed].token_type == TokenType::Else {
            consumed += 1;
            let (expr, count) = parse_expression(source, &tokens[consumed..], 0)?;
            (expr, count)
        } else {
            // Create Nil expression with span from then_branch end
            let nil_span = then_branch.byte_span.end..then_branch.byte_span.end;
            (
                Expr {
                    expr_type: ExprType::Literal(Literal::Nil),
                    byte_span: nil_span,
                },
                0,
            )
        };
    consumed += else_consumed;

    // Get the end position from the last token
    let end = if else_consumed > 0 {
        else_branch.byte_span.end
    } else {
        then_branch.byte_span.end
    };

    Ok((
        Expr {
            expr_type: ExprType::If {
                condition: Box::new(condition),
                then_branch: Box::new(then_branch),
                else_branch: Box::new(else_branch),
            },
            byte_span: start..end,
        },
        consumed,
    ))
}

fn parse_while(source: &[u8], tokens: &[Token]) -> Result<(Expr, usize)> {
    let start = tokens[0].byte_span.start;
    let mut consumed = 1; // Skip 'while'

    // Parse condition expression
    let (condition, cond_consumed) = parse_expression(source, &tokens[consumed..], 0)?;
    consumed += cond_consumed;

    // Parse body expression
    let (body, body_consumed) = parse_expression(source, &tokens[consumed..], 0)?;
    consumed += body_consumed;

    // Get the end position from the last token involved in the while expression
    let end = if consumed > 0 {
        tokens[consumed - 1].byte_span.end
    } else {
        tokens[0].byte_span.end
    };

    Ok((
        Expr {
            expr_type: ExprType::While {
                condition: Box::new(condition),
                body: Box::new(body),
            },
            byte_span: start..end,
        },
        consumed,
    ))
}

fn parse_function(source: &[u8], tokens: &[Token]) -> Result<(Expr, usize)> {
    let start = tokens[0].byte_span.start;
    let mut consumed = 1; // Skip 'fn'

    // Expect opening parenthesis
    if consumed >= tokens.len() || tokens[consumed].token_type != TokenType::LeftParen {
        return parser_error(source, "Expected '(' after 'fn'", &tokens[consumed - 1]);
    }
    consumed += 1;

    // Parse parameters
    let mut params = Vec::new();
    while consumed < tokens.len() && tokens[consumed].token_type != TokenType::RightParen {
        if !params.is_empty() {
            if tokens[consumed].token_type != TokenType::Comma {
                return parser_error(source, "Expected ',' between parameters", &tokens[consumed]);
            }
            consumed += 1;
        }

        match &tokens[consumed].token_type {
            TokenType::Identifier(name) => params.push(name.clone()),
            _ => return parser_error(source, "Expected parameter name", &tokens[consumed]),
        }
        consumed += 1;
    }

    // Expect closing parenthesis
    if consumed >= tokens.len() || tokens[consumed].token_type != TokenType::RightParen {
        return parser_error(
            source,
            "Expected ')' after parameters",
            &tokens[consumed - 1],
        );
    }
    consumed += 1;

    // Parse body expression
    let (body, body_consumed) = parse_expression(source, &tokens[consumed..], 0)?;
    consumed += body_consumed;

    // Get the end position from the last token involved in the function
    let end = if consumed > 0 {
        tokens[consumed - 1].byte_span.end
    } else {
        tokens[0].byte_span.end
    };

    Ok((
        Expr {
            expr_type: ExprType::Literal(Literal::Function {
                params,
                body: Box::new(body),
            }),
            byte_span: start..end,
        },
        consumed,
    ))
}

fn parse_list(source: &[u8], tokens: &[Token]) -> Result<(Expr, usize)> {
    let start = tokens[0].byte_span.start;
    let mut consumed = 1; // Skip '['
    let mut elements = Vec::new();

    // Handle empty list
    if consumed < tokens.len() && tokens[consumed].token_type == TokenType::RightSquare {
        let end = tokens[consumed].byte_span.end;
        return Ok((
            Expr {
                expr_type: ExprType::Literal(Literal::List(elements)),
                byte_span: start..end,
            },
            consumed + 1,
        ));
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
            &tokens[consumed - 1],
        );
    }

    let end = tokens[consumed].byte_span.end;
    consumed += 1;

    Ok((
        Expr {
            expr_type: ExprType::Literal(Literal::List(elements)),
            byte_span: start..end,
        },
        consumed,
    ))
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
        if let ExprType::Block(exprs) = parse_str("42")?.expr_type {
            assert_eq!(exprs.len(), 1);
            assert!(matches!(
                exprs[0].expr_type,
                ExprType::Literal(Literal::Number(42.0))
            ));
        }

        if let ExprType::Block(exprs) = parse_str("\"hello\"")?.expr_type {
            assert_eq!(exprs.len(), 1);
            assert!(
                matches!(&exprs[0].expr_type, ExprType::Literal(Literal::String(s)) if s == "hello")
            );
        }
        Ok(())
    }

    #[test]
    fn test_multiple_top_level_declarations() -> Result<()> {
        // Test multiple let declarations
        let input = r#"
            let x = 1;
            let y = 2;
            let z = 3"#;

        if let ExprType::Block(exprs) = parse_str(input)?.expr_type {
            assert_eq!(exprs.len(), 3);
            for expr in exprs {
                assert!(matches!(expr.expr_type, ExprType::Let { .. }));
            }
        }

        // Test function definitions
        let input = r#"
            let f = fn() { 1 };
            let g = fn() { 2 }"#;

        if let ExprType::Block(exprs) = parse_str(input)?.expr_type {
            assert_eq!(exprs.len(), 2);
            for expr in exprs {
                if let ExprType::Let { initializer, .. } = expr.expr_type {
                    assert!(matches!(
                        initializer.expr_type,
                        ExprType::Literal(Literal::Function { .. })
                    ));
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
            }"#;

        if let ExprType::Block(exprs) = parse_str(input)?.expr_type {
            assert_eq!(exprs.len(), 1);
            if let ExprType::Let { initializer, .. } = &exprs[0].expr_type {
                if let ExprType::Literal(Literal::Function { params, body }) =
                    &initializer.expr_type
                {
                    assert_eq!(params, &vec!["x", "y"]);
                    if let ExprType::Block(body_exprs) = &body.expr_type {
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
        if let ExprType::Block(exprs) = parse_str("[]")?.expr_type {
            assert_eq!(exprs.len(), 1);
            assert!(
                matches!(&exprs[0].expr_type, ExprType::Literal(Literal::List(elements)) if elements.is_empty())
            );
        }

        // Test simple list
        if let ExprType::Block(exprs) = parse_str("[1, 2, 3]")?.expr_type {
            assert_eq!(exprs.len(), 1);
            if let ExprType::Literal(Literal::List(elements)) = &exprs[0].expr_type {
                assert_eq!(elements.len(), 3);
                assert!(matches!(
                    &elements[0].expr_type,
                    ExprType::Literal(Literal::Number(1.0))
                ));
                assert!(matches!(
                    &elements[1].expr_type,
                    ExprType::Literal(Literal::Number(2.0))
                ));
                assert!(matches!(
                    &elements[2].expr_type,
                    ExprType::Literal(Literal::Number(3.0))
                ));
            }
        }

        // Test mixed types
        if let ExprType::Block(exprs) = parse_str("[1, \"hello\", true]")?.expr_type {
            assert_eq!(exprs.len(), 1);
            if let ExprType::Literal(Literal::List(elements)) = &exprs[0].expr_type {
                assert_eq!(elements.len(), 3);
                assert!(matches!(
                    &elements[0].expr_type,
                    ExprType::Literal(Literal::Number(1.0))
                ));
                assert!(
                    matches!(&elements[1].expr_type, ExprType::Literal(Literal::String(s)) if s == "hello")
                );
                assert!(matches!(
                    &elements[2].expr_type,
                    ExprType::Literal(Literal::Boolean(true))
                ));
            }
        }

        // Test nested lists
        if let ExprType::Literal(Literal::List(exprs)) = parse_str("[[1, 2], [3, 4]]")?.expr_type {
            assert_eq!(exprs.len(), 1);
            if let ExprType::Literal(Literal::List(elements)) = &exprs[0].expr_type {
                assert_eq!(elements.len(), 2);
                for inner in elements {
                    assert!(matches!(
                        inner.expr_type,
                        ExprType::Literal(Literal::List(_))
                    ));
                }
            }
        }

        // Test trailing comma
        if let ExprType::Block(exprs) = parse_str("[1, 2, 3,]")?.expr_type {
            assert_eq!(exprs.len(), 1);
            if let ExprType::Literal(Literal::List(elements)) = &exprs[0].expr_type {
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

        if let ExprType::Block(outer) = parse_str(input)?.expr_type {
            assert_eq!(outer.len(), 1);
            if let ExprType::Block(inner) = &outer[0].expr_type {
                assert_eq!(inner.len(), 2);
                assert!(matches!(inner[0].expr_type, ExprType::Let { .. }));
                assert!(matches!(inner[1].expr_type, ExprType::Block(_)));
            }
        }
        Ok(())
    }

    #[test]
    fn test_implicit_nil() -> Result<()> {
        // Test block with semicolon-terminated expression
        let input = "{ print(\"hello\"); }";
        if let ExprType::Block(outer) = parse_str(input)?.expr_type {
            assert_eq!(outer.len(), 1); // The outer block has one expression
            if let ExprType::Block(inner) = &outer[0].expr_type {
                assert_eq!(inner.len(), 2); // print + nil
                assert!(matches!(inner[0].expr_type, ExprType::Call { .. }));
                assert!(matches!(
                    inner[1].expr_type,
                    ExprType::Literal(Literal::Nil)
                ));
            }
        }

        // Test block without semicolon-terminated expression
        let input = "{ print(\"hello\") }";
        if let ExprType::Block(outer) = parse_str(input)?.expr_type {
            assert_eq!(outer.len(), 1);
            if let ExprType::Block(inner) = &outer[0].expr_type {
                assert_eq!(inner.len(), 1);
                assert!(matches!(inner[0].expr_type, ExprType::Call { .. }));
            }
        }

        // Test nested blocks with semicolons
        let input = r#"{
        print("outer");
        {
            print("inner");
        }
    }"#;
        if let ExprType::Block(outer) = parse_str(input)?.expr_type {
            assert_eq!(outer.len(), 1); // The outer parse_str block
            if let ExprType::Block(block1) = &outer[0].expr_type {
                assert_eq!(block1.len(), 2); // outer print and inner block. inner block has no ;
                assert!(matches!(block1[0].expr_type, ExprType::Call { .. }));
                if let ExprType::Block(block2) = &block1[1].expr_type {
                    assert_eq!(block2.len(), 2); // inner print, nil
                    assert!(matches!(block2[0].expr_type, ExprType::Call { .. }));
                    assert!(matches!(
                        block2[1].expr_type,
                        ExprType::Literal(Literal::Nil)
                    ));
                }
            }
        }

        // Test the implicit nil in a terminated let expression
        let input = "let x = { print(\"hello\"); }";
        if let ExprType::Block(outer) = parse_str(input)?.expr_type {
            assert_eq!(outer.len(), 1);
            if let ExprType::Let { initializer, .. } = &outer[0].expr_type {
                if let ExprType::Block(block) = &initializer.expr_type {
                    assert_eq!(block.len(), 2);
                    assert!(matches!(block[0].expr_type, ExprType::Call { .. }));
                    assert!(matches!(
                        block[1].expr_type,
                        ExprType::Literal(Literal::Nil)
                    ));
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
            }"#;

        if let ExprType::Block(exprs) = parse_str(input)?.expr_type {
            assert_eq!(exprs.len(), 2);
            assert!(matches!(exprs[0].expr_type, ExprType::Let { .. }));
            assert!(matches!(exprs[1].expr_type, ExprType::While { .. }));
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
            }"#;

        if let ExprType::Block(exprs) = parse_str(input)?.expr_type {
            assert_eq!(exprs.len(), 1);
            if let ExprType::If {
                condition,
                then_branch,
                else_branch,
            } = &exprs[0].expr_type
            {
                assert!(matches!(condition.expr_type, ExprType::Binary { .. }));
                assert!(matches!(then_branch.expr_type, ExprType::Block(_)));
                assert!(matches!(else_branch.expr_type, ExprType::Block(_)));
            }
        }
        Ok(())
    }

    #[test]
    fn test_function_calls() -> Result<()> {
        let input = r#"
            let f = fn(x) { x * 2 };
            f(21)"#;

        if let ExprType::Block(exprs) = parse_str(input)?.expr_type {
            assert_eq!(exprs.len(), 2);
            assert!(matches!(exprs[0].expr_type, ExprType::Let { .. }));
            assert!(matches!(exprs[1].expr_type, ExprType::Call { .. }));
        }
        Ok(())
    }

    #[test]
    fn test_iife() -> Result<()> {
        // Test expression-body IIFE
        if let ExprType::Block(exprs) = parse_str("(fn() 42)()")?.expr_type {
            assert_eq!(exprs.len(), 1);
            if let ExprType::Call { callee, arguments } = &exprs[0].expr_type {
                assert!(matches!(callee.expr_type, ExprType::Grouping(_)));
                assert!(arguments.is_empty());

                if let ExprType::Grouping(inner) = &callee.expr_type {
                    assert!(matches!(
                        inner.expr_type,
                        ExprType::Literal(Literal::Function { .. })
                    ));
                }
            } else {
                panic!("Expected Call expression");
            }
        }

        // Test block-body IIFE
        if let ExprType::Block(exprs) = parse_str("(fn() { 42 })()")?.expr_type {
            assert_eq!(exprs.len(), 1);
            if let ExprType::Call { callee, arguments } = &exprs[0].expr_type {
                assert!(matches!(callee.expr_type, ExprType::Grouping(_)));
                assert!(arguments.is_empty());

                if let ExprType::Grouping(inner) = &callee.expr_type {
                    assert!(matches!(
                        inner.expr_type,
                        ExprType::Literal(Literal::Function { .. })
                    ));
                }
            } else {
                panic!("Expected Call expression");
            }
        }

        // Test IIFE with arguments
        if let ExprType::Block(exprs) = parse_str("(fn(x) x * 2)(42)")?.expr_type {
            assert_eq!(exprs.len(), 1);
            if let ExprType::Call { callee, arguments } = &exprs[0].expr_type {
                assert!(matches!(callee.expr_type, ExprType::Grouping(_)));
                assert_eq!(arguments.len(), 1);
                assert!(matches!(
                    &arguments[0].expr_type,
                    ExprType::Literal(Literal::Number(42.0))
                ));
            }
        }

        // Test IIFE in let binding
        if let ExprType::Block(exprs) = parse_str("let f = (fn() 42)()")?.expr_type {
            assert_eq!(exprs.len(), 1);
            if let ExprType::Let { initializer, .. } = &exprs[0].expr_type {
                assert!(matches!(initializer.expr_type, ExprType::Call { .. }));
            }
        }

        // Verify that we still handle regular grouped expressions
        if let ExprType::Block(exprs) = parse_str("(42)")?.expr_type {
            assert_eq!(exprs.len(), 1);
            assert!(matches!(exprs[0].expr_type, ExprType::Grouping(_)));
        }

        Ok(())
    }

    #[test]
    fn test_iife_error_cases() -> Result<()> {
        // Missing closing paren for function
        assert!(parse_str("(fn() 42()").is_err());

        // Missing closing paren for call
        assert!(parse_str("(fn() 42)(").is_err());

        // Missing opening paren
        assert!(parse_str("fn() 42)()").is_err());

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

        if let ExprType::Block(exprs) = parse_str(input)?.expr_type {
            assert_eq!(exprs.len(), 4);
            // First three expressions should be let bindings
            (0..3).for_each(|i| {
                assert!(matches!(exprs[i].expr_type, ExprType::Let { .. }));
            });
            // Last expression should be a function call
            assert!(matches!(exprs[3].expr_type, ExprType::Call { .. }));
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
