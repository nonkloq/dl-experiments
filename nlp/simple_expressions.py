import random

import numpy as np
import torch as th
from torch.nn.utils.rnn import pad_sequence

VARIABLE_NAMES = "abcdefghijklmnopqrstuvwxyz"
OPERATIONS = "+-/*"
NUMBERS = "0123456789"

PADDING_TOKEN = "|"
START_OF_ANSWER = "["
END_OF_ANSWER = "]"
START_OF_EXPRESSION = "$"
END_OF_EXPRESSION = "!"
NULL_VALUE = "#"

EXPRESSION_SEPARATOR = ";"
VARIABLE_SEPARATOR = ","
SPACE_CHARACTER = " "
UNKNOWN_CHARECTER = "?"
ASSIGN_VALUE_CHAR = "="

expr_vocab_reverse_index = (
    # 0 for padding
    PADDING_TOKEN
    # plurals
    + VARIABLE_NAMES
    + NUMBERS
    + OPERATIONS
    # start and end indicators
    + START_OF_ANSWER
    + END_OF_ANSWER
    + START_OF_EXPRESSION
    + END_OF_EXPRESSION
    # sepical characters
    + NULL_VALUE
    + EXPRESSION_SEPARATOR
    + VARIABLE_SEPARATOR
    + SPACE_CHARACTER
    + UNKNOWN_CHARECTER
    + ASSIGN_VALUE_CHAR
    # decimal and paren (default)
    + ".()"
)
expr_vocabulary = {x: i for i, x in enumerate(expr_vocab_reverse_index)}

UNKNOWN_CHAR_ID = expr_vocabulary[UNKNOWN_CHARECTER]
PADDING_TOKEN_ID = expr_vocabulary[PADDING_TOKEN]


def encode_expression(exp: str) -> list[int]:
    return [expr_vocabulary.get(x, UNKNOWN_CHAR_ID) for x in exp]


def decode_expression(exp: list[int]) -> list[str]:
    return [expr_vocab_reverse_index[x] for x in exp]


def collate(batch):
    """
    add right padding, returns Tensor [B, T]
    """
    return pad_sequence(
        [th.tensor(x, dtype=th.long) for x in batch],
        batch_first=True,
        padding_value=PADDING_TOKEN_ID,
    )


def get_numbered_variables(
    k: int, number_range: tuple[float, float], float_decimals: int
) -> dict[str, float]:
    """return {var: num} Map, the len of the map can be 1..min(k,26)"""
    _vars = set(random.choices(VARIABLE_NAMES, k=k))
    _mi, _mx = number_range
    _delta = _mx - _mi

    numbers = np.random.random(len(_vars))
    numbers = _mi + _delta * numbers

    number_formatter = (  # noqa: E731
        lambda x: round(x, float_decimals) if float_decimals > 0 else lambda x: int(x)
    )

    return {x: number_formatter(numbers[i]) for i, x in enumerate(_vars)}  # type: ignore


def build_expression(
    var_num_map: dict[str, float],
    expression_range: tuple[int, int],
    decimal_round: int,
    parenthesis_prob: float,
) -> tuple[str, str]:
    variables = list(var_num_map.keys())

    expression_size = random.randint(
        max(len(set(variables)), expression_range[0]), expression_range[1]
    )

    operation_symbols = random.choices(OPERATIONS, k=expression_size - 1)
    if len(variables) < expression_size:
        repeated_vars = random.choices(variables, k=expression_size - len(variables))
        variables += repeated_vars
    random.shuffle(variables)

    expression_stack = []
    open_parans = 0
    for i in range(len(variables) - 1):
        op = operation_symbols[i]
        va = variables[i]

        # randomly add paran, the prob decayed by the open paren count
        if random.random() > (parenthesis_prob ** (open_parans + 1)):
            expression_stack.append("(")
            open_parans += 1

        expression_stack.append(va)
        # closing parenthesis after appending a variable
        # this may lead to : (a) or (a Op ...) and both of them are valid
        if open_parans > 0 and random.random() > parenthesis_prob:
            expression_stack.append(")")
            open_parans -= 1

        expression_stack.append(op)

    # filling remaining parenthesis
    expression_stack.append(variables[-1])
    if open_parans > 0:
        for _ in range(open_parans):
            expression_stack.append(")")

    _expr_for_eval = SPACE_CHARACTER.join(
        [x if (x in "+/-*()") else str(var_num_map[x]) for x in expression_stack]
    )

    source_expression = (
        START_OF_EXPRESSION
        + " ".join(expression_stack)
        + EXPRESSION_SEPARATOR
        + VARIABLE_SEPARATOR.join(
            [f"{v}{ASSIGN_VALUE_CHAR}{n}" for v, n in var_num_map.items()]
        )
        + END_OF_EXPRESSION
    )
    try:
        trgt = str(
            round(eval(_expr_for_eval), decimal_round)
            if decimal_round > 0
            else int(eval(_expr_for_eval))
        )
    except ZeroDivisionError:
        trgt = NULL_VALUE

    target_value = START_OF_ANSWER + trgt + END_OF_ANSWER
    return source_expression, target_value


class SimpleExpression:
    def __init__(
        self,
        n_variables: int,
        expr_range: tuple[int, int],
        number_range: tuple[float, float],
        max_decimals: int,
        parenthesis_prob: float = 0.5,
    ) -> None:
        assert expr_range[0] >= 1, "Atleast 1 variable needed in the expression"

        self.n_variables = n_variables
        self.expr_range = expr_range
        self.number_range = number_range
        self.max_decimals = max_decimals
        self.parenthesis_prob = parenthesis_prob

    def sample(self, batch_size: int):
        expressions = []
        targets = []
        for _ in range(batch_size):
            var_num_map = get_numbered_variables(
                self.n_variables, self.number_range, self.max_decimals
            )
            expression, target = build_expression(
                var_num_map, self.expr_range, self.max_decimals, self.parenthesis_prob
            )
            expressions.append(expression)
            targets.append(target)
        return expressions, targets


if __name__ == "__main__":
    assert len(expr_vocabulary) == 54
    exp_gen = SimpleExpression(3, (1, 5), (-10, 10), 2, 0.5)
    exprs, trgts = exp_gen.sample(1)
    enc = encode_expression(exprs[0])
    dec = decode_expression(enc)
    assert "".join(dec) == exprs[0]
