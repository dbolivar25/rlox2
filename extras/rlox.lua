-- This is the file I added to my neovim config so I didn't have
-- to stare at plain white text when writing my language. This
-- matches the syntax highlighting built into the repl interface.

local function setup_rlox_syntax()
    local colors = {
        keyword = "#43728C", -- LightBlue
        literal = "#EBC181", -- Yellow
        default = "#FFFFFF", -- White
        operator = "#8F8CA8", -- DarkGray
    }

    -- Clear any existing syntax rules
    vim.cmd([[syntax clear]])

    -- Define keywords
    vim.cmd([[
        syntax keyword rloxKeyword let rec fn if else while return and or struct nil
        syntax keyword rloxBoolean true false
    ]])

    -- Operators
    vim.cmd([[
        syntax match rloxOperator "[+\-*/<>=!{}\[\](),;%]"
        syntax match rloxOperator "=="
        syntax match rloxOperator "!="
        syntax match rloxOperator ">="
        syntax match rloxOperator "<="
        syntax match rloxOperator "<>"
    ]])

    -- Numbers and strings (literals)
    vim.cmd([[syntax match rloxNumber "\v\d+(\.\d+)?"]])
    vim.cmd([[syntax region rloxString start=/"/ skip=/\\"/ end=/"/ contains=rloxStringEscape]])
    vim.cmd([[syntax match rloxStringEscape "\\." contained]])

    -- Comments (using default color)
    vim.cmd([[syntax match rloxComment "//.*$"]])
    vim.cmd([[syntax region rloxComment start="/\*" end="\*/"]])

    -- Set up highlighting
    local function hi(group, opts)
        vim.api.nvim_set_hl(0, group, opts)
    end

    hi("rloxKeyword", { fg = colors.keyword }) -- Keywords use KEYWORD_COLOR
    hi("rloxBoolean", { fg = colors.literal }) -- Booleans are literals
    hi("rloxOperator", { fg = colors.operator }) -- Operators use OPERATOR_COLOR
    hi("rloxString", { fg = colors.literal }) -- Strings use LITERAL_COLOR
    hi("rloxNumber", { fg = colors.literal }) -- Numbers use LITERAL_COLOR
    hi("rloxComment", { fg = colors.operator }) -- Comments use OPERATOR_COLOR
    hi("rloxStringEscape", { fg = colors.literal }) -- String escapes use LITERAL_COLOR
end

return {
    setup = setup_rlox_syntax,
}
