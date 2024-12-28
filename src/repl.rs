use reedline::{
    Prompt, PromptEditMode, PromptHistorySearch, PromptHistorySearchStatus, ValidationResult,
    Validator,
};
use std::borrow::Cow;

#[derive(Clone)]
pub struct REPLPrompt;

impl Prompt for REPLPrompt {
    fn render_prompt_left(&self) -> Cow<str> {
        Cow::Borrowed("rlox")
    }

    fn render_prompt_right(&self) -> Cow<str> {
        Cow::Borrowed("")
    }

    fn render_prompt_indicator(&self, _edit_mode: PromptEditMode) -> Cow<str> {
        Cow::Borrowed("> ")
    }

    fn render_prompt_multiline_indicator(&self) -> Cow<str> {
        Cow::Borrowed("  ... ")
    }

    fn render_prompt_history_search_indicator(
        &self,
        history_search: PromptHistorySearch,
    ) -> Cow<str> {
        let prefix = match history_search.status {
            PromptHistorySearchStatus::Passing => "",
            PromptHistorySearchStatus::Failing => "failing ",
        };
        Cow::Owned(format!(
            "({}reverse-search: {}) ",
            prefix, history_search.term
        ))
    }
}

#[derive(Debug, Default)]
pub struct REPLValidator;

impl Validator for REPLValidator {
    fn validate(&self, line: &str) -> ValidationResult {
        let trimmed = line.trim_end();
        if trimmed.is_empty() {
            return ValidationResult::Complete;
        }

        if trimmed.ends_with('\\') {
            return ValidationResult::Incomplete;
        }

        let mut delimiters = Vec::new();
        let mut in_string = false;
        let mut escaped = false;

        for c in line.chars() {
            match c {
                '"' if !escaped => in_string = !in_string,
                '\\' if in_string => escaped = !escaped,
                _ if in_string => {
                    escaped = false;
                    continue;
                }

                '{' | '(' | '[' => delimiters.push(c),
                '}' => {
                    if delimiters.pop() != Some('{') {
                        return ValidationResult::Complete;
                    }
                }
                ')' => {
                    if delimiters.pop() != Some('(') {
                        return ValidationResult::Complete;
                    }
                }
                ']' => {
                    if delimiters.pop() != Some('[') {
                        return ValidationResult::Complete;
                    }
                }

                _ => escaped = false,
            }
        }

        if in_string {
            return ValidationResult::Incomplete;
        }

        if delimiters.is_empty() {
            ValidationResult::Complete
        } else {
            ValidationResult::Incomplete
        }
    }
}
