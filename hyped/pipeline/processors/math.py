from .base import DataProcessor, DataProcessorConfig
from datasets import Features, Sequence, Value, ClassLabel
from dataclasses import dataclass
from typing import Literal, Any
from types import SimpleNamespace
import numpy as np
import operator
import ast


def evaluate_tree(node:ast.AST, item:SimpleNamespace, index:int, rank:int):

    # supported binary operators
    bin_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    # supported unary operators
    un_ops = {
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Invert: operator.inv
    }

    if isinstance(node, ast.Expression):
        return evaluate_tree(node.body, item, index, rank)

    if isinstance(node, (ast.Num, ast.Constant)):
        return node.value

    if isinstance(node, ast.BinOp) and isinstance(node.op, tuple(bin_ops.keys())):
        return bin_ops[type(node.op)](
            evaluate_tree(node.left, item, index, rank),
            evaluate_tree(node.right, item, index, rank)
        )

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, tuple(un_ops.keys())):
        return un_ops[type(node.op)](
            evaluate_tree(node.operand, item, index, rank)
        )

    if isinstance(node, ast.Attribute):
        obj = evaluate_tree(node.value, item, index, rank)
        return getattr(obj, node.attr)

    if isinstance(node, ast.Name):
        ns = SimpleNamespace(item=item, index=index, rank=rank)
        return getattr(ns, node.id)

    raise SyntaxError(ast.dump(node, indent=4))


def check_tree(node:ast.AST, features:SimpleNamespace):

    # supported binary operators
    bin_ops = (
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.Pow
    )
    # supported unary operators
    un_ops = (
        ast.USub,
        ast.UAdd,
        ast.Invert
    )

    if isinstance(node, ast.Expression):
        return check_tree(node.body, features)

    if isinstance(node, (ast.Num, ast.Constant)):
        return Value('float32') # assumes data type

    if isinstance(node, ast.BinOp) and isinstance(node.op, bin_ops):
        left = check_tree(node.left, features)
        right = check_tree(node.right, features)

        # TODO: type-check binary operations
        if isinstance(left, (Value, ClassLabel)) and isinstance(right, (Value, ClassLabel)):
            return left

        if isinstance(left, Sequence) and isinstance(right, (ClassLabel, Value)):
            return left

        if isinstance(left, (ClassLabel, Value)) and isinstance(right, Sequence):
            return right

        if isinstance(left, Sequence) and isinstance(right, Sequence):
            # check shapes
            if (left.length != right.length):
                raise ValueError("Sequence Length mismatch, `%s` != `%s`" % (left.length, right.length))
            return left

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, un_ops):
        return check_tree(node.operand, features)

    if isinstance(node, ast.Attribute):
        obj = check_tree(node.value, features)
        return getattr(obj, node.attr)

    if isinstance(node, ast.Name):
        ns = SimpleNamespace(item=features)
        return getattr(ns, node.id)

    raise SyntaxError(ast.dump(node, indent=4))


@dataclass
class MathProcessorConfig(DataProcessorConfig):
    processor_type:Literal["math"] = "math"
    expression:str = None
    output_column:str = None

    def __post_init__(self) -> None:
        if self.expression is None:
            raise ValueError("No expresison specified in math processor")
        if self.output_column is None:
            raise ValueError("Output column not defined, please specify `output_column`")


class MathProcessor(DataProcessor):

    def __init__(self, config:MathProcessorConfig) -> None:
        super(MathProcessor, self).__init__(config=config)
        # parse expression into syntax tree
        self.tree = ast.parse(self.config.expression, mode='eval')

    @property
    def variables(self) -> set[str]:
        return {node.attr for node in ast.walk(self.tree) if isinstance(node, ast.Attribute)}

    def map_features(self, features:Features) -> Features:
        # make sure all variables in the expression are present in features
        for var in self.variables:
            if var not in features:
                raise ValueError("Variable `%s` not present in features but referenced in expression" % var)
        # check expression and infer output feature type
        return {self.config.output_column: check_tree(self.tree, SimpleNamespace(**features))}

    def process(self, example:dict[str, Any], index:int, rank:int) -> dict[str, Any]:
        # get all variables from examples and evaluate expression based on them
        item = SimpleNamespace(**{k: np.asarray(example[k]) for k in self.variables})
        out = evaluate_tree(self.tree, item, index, rank)
        # return output
        return {self.config.output_column: out.tolist()}
