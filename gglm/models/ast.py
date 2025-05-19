from __future__ import annotations
from typing import List, Union, TypeAlias, Optional
from enum import Enum
import math
import logging
import hashlib

from lark import Token

from gglm.utils import Tensor, ASTNode, ensure_args
from gglm.models.parser import ParseContext, ParseError
import gglm.wrapper as wrapper
import gglm.wrapper.marshall as marshall
from gglm.wrapper import gen


class Operation(Enum):
    VALUE = "value"
    ADD = "+"
    SUBTRACT = "-"
    MULTIPLY = "*"
    DIVIDE = "/"
    MODULO = "//"
    POWER = "^"
    FUNC_CALL = "func()"

    def apply_to_ints(self, a: int, b: int, *, function_name: Optional[str] = None) -> int:   
        match self:
            case Operation.ADD:
                return a + b
            case Operation.SUBTRACT:
                return a - b
            case Operation.MULTIPLY:
                return a * b
            case Operation.DIVIDE:
                return a // b
            case Operation.MODULO:
                return a % b
            case Operation.POWER:
                return a ** b
            case Operation.FUNC_CALL:
                if not function_name:
                    raise ValueError("Function name must be provided for FUNC_CALL")
                raise ValueError(f"Unsupported numeric function call: {function_name} for int")
            case _:
                raise ValueError(f"Unsupported operation: {self}")
            
    def apply_to_floats(self, a: float, b: Optional[float], *, function_name: Optional[str] = None) -> float:
        if self == Operation.FUNC_CALL:
            if not function_name:
                raise ValueError("Function name must be provided for FUNC_CALL")
            if function_name == "sqrt":
                return math.sqrt(a)
            else:
                raise ValueError(f"Unsupported numeric function call: {function_name} for float")

        if b is None:
            raise ValueError("Second operand must be provided for binary operations")
        match self:
            case Operation.ADD:
                return a + b
            case Operation.SUBTRACT:
                return a - b
            case Operation.MULTIPLY:
                return a * b
            case Operation.DIVIDE:
                return a // b
            case Operation.MODULO:
                return a % b
            case Operation.POWER:
                return a ** b
            case _:
                raise ValueError(f"Unsupported operation: {self}")

class Expression(ASTNode):
    ctx: ParseContext
    operation: Operation
    function_name: Optional[str]
    operands: List[Operand]

    def __init__(self, source_token: Token, ctx: ParseContext, operation: Operation, *operands: Operand, function_name: Optional[str] = None):
        super().__init__(source_token, ctx)
        self.ctx = ctx
        self.function_name = function_name
        self.operation = operation
        self.operands = list(operands)
    
    def __int__(self):
        a = None
        b = None

        match self.operands:
            case [op1]:
                a = op1
            case [op1, op2]:
                a = op1
                b = op2
            case _:
                raise ValueError(f"Unsupported number of operands: {len(self.operands)}") # shouldn't get here

        if not b:
            if isinstance(a, int):
                return a
            elif isinstance(a, str):
                return self.resolve_param(a)
            else:
                raise ValueError() # shouldn't get here
        
        a = int(a)

        if b is not None:
            b = int(b)

        return self.operation.apply_to_ints(a, b)
    
    def __float__(self):
        a = None
        b = None

        match self.operands:
            case [op1]:
                a = op1
            case [op1, op2]:
                a = op1
                b = op2
            case _:
                raise ValueError(f"Unsupported number of operands: {len(self.operands)}") # shouldn't get here

        if not b:
            if isinstance(a, int):
                return a
            elif isinstance(a, str):
                return self.resolve_param(a)
            else:
                raise ValueError() # shouldn't get here
        
        a = float(a)
        if b is not None:
            b = float(b)

        return self.operation.apply_to_floats(a, b)
    
    def __str__(self) -> str:
        name = self.operation
        if self.operation == Operation.FUNC_CALL:
            name = self.function_name
        if self.operation == Operation.VALUE:
            return str(self.operands[0])
        return f"{name}({', '.join(str(op) for op in self.operands)})"
    
    def __repr__(self) -> str:
        return f"Expression('{str(self)}')"
    
class Assignment(ASTNode):
    target: str
    expression: Expression

    def __init__(self, source_token: Token, ctx: ParseContext, target: str, expression: Expression):
        super().__init__(source_token, ctx)
        self.target = target
        self.expression = expression 

    def __str__(self) -> str:
        return f"{self.target} = {str(self.expression)}"
    
    def __repr__(self) -> str:
        return f"Assignment(target={self.target}, expression={repr(self.expression)})"

class RepeatBlock(ASTNode):
    count: Expression
    count_variable: str
    statements: List[ASTNode]

    def __init__(self, source_token: Token, ctx: ParseContext, count: Expression, count_variable: str, statements: List[ASTNode]):
        super().__init__(source_token, ctx)
        self.count = count
        self.count_variable = count_variable
        self.statements = statements

    def __str__(self) -> str:
        statements = "{\n" + '\n\t'.join(str(stmt) for stmt in self.statements) + "\n}"
        return f"repeat {self.count} {statements}"
    
    def __repr__(self) -> str:
        return f"RepeatBlock(count={repr(self.count)}, statements={[repr(s) for s in self.statements]})"
    
class FunctionCall(ASTNode):
    function_name: str
    arguments: List[Expression]

    def __init__(self, source_token: Token, ctx: ParseContext, function_name: str, arguments: List[Operand]):
        super().__init__(source_token, ctx)
        self.function_name = function_name
        self.arguments = arguments

    def __str__(self) -> str:
        return f"{self.function_name}({', '.join(str(arg) for arg in self.arguments)})"
    
    def __repr__(self) -> str:
        return f"FunctionCall(function_name={self.function_name}, arguments={[repr(a) for a in self.arguments]})"
    
Operand: TypeAlias = Union[Expression, int, float, str]

def produce_ggml_function_call_graph(ctx0: wrapper.ggml_context_p, df: wrapper.ggml_cgraph_p, node: FunctionCall | Expression) -> Optional[Tensor | int | float]:
    if isinstance(node, FunctionCall):
        func_name = node.function_name
        raw_args = node.arguments
    else:
        func_name = node.function_name
        raw_args = node.operands

    args = [produce_ggml_graph(ctx0, df, op) if isinstance(op, Expression) else op for op in raw_args]
    logging.debug(f"{func_name=} {args=} {raw_args=}")

    ggml_function = marshall.GGML_FUNCTIONS.get(str(func_name))
    if ggml_function is not None:
        ensure_args(func_name, args, ggml_function.arg_types, node.source_token)
        func_args = hashlib.md5(", ".join([arg.name if isinstance(arg, Tensor) else str(arg) for arg in args]).encode()).hexdigest()[-4:]
        # TODO: gracefully handle errors here
        result_name =  f"{func_name}_{func_args}"
        if node.ctx.repeat_index != None:
            result_name = result_name + f"_{node.ctx.repeat_index}"
        result: Tensor = ggml_function.func(ctx0, result_name, *args)
        logging.debug(f"{result=}")

        node.ctx.intermediate_tensors.append(result)
        return result
    
    # handle functions that return numbers separately for now
    if isinstance(node, Expression):
        result_num = None
        if len(args) == 1 and (isinstance(args[0], int) or isinstance(args[0], float)):
            result_num = node.operation.apply_to_floats(float(args[0]), None, function_name=func_name)
        elif len(args) == 2 and isinstance(args[0], Tensor) and isinstance(args[1], int) and func_name == "row_size":
            result_num = gen.ggml_row_size(args[0].type, args[1])
        elif len(args) == 1 and isinstance(args[0], Tensor) and func_name == "element_size":
            result_num = gen.ggml_element_size(args[0].ptr)
        
        if result_num is not None:
            logging.debug(f"{result_num=}")
            return result_num

    raise NotImplementedError(f"Unsupported function: {func_name}")

def produce_ggml_graph(ctx0: wrapper.ggml_context_p, df: wrapper.ggml_cgraph_p, node: ASTNode) -> Optional[Tensor | int | float]:
    """Performs "instruction lowering" on the AST, producing a ggml graph. This is the final step in compiling a GGML model."""

    # logging.debug(f"Building ggml graph for {node}")
    if isinstance(node, Expression):
        match node.operation:
            case Operation.VALUE:
                if isinstance(node.operands[0], str):
                    result = node.resolve_tensor(node.operands[0], raise_error=False)
                    if not result:
                        result = node.resolve_param(node.operands[0], raise_error=False)
                    if result is not None:
                        return result
                    else:
                        raise ParseError(f"Unknown name {node.operands[0]}", node.source_token)
                else:
                    raise ParseError(f"Unsupported operand type: {type(node.operands[0])}", node.source_token)
            case Operation.FUNC_CALL:
                return produce_ggml_function_call_graph(ctx0, df, node)
            case Operation.DIVIDE:
                lhs = produce_ggml_graph(ctx0, df, node.operands[0])
                rhs = produce_ggml_graph(ctx0, df, node.operands[1])

                match [lhs, rhs]:
                    case [int(a), int(b)]:
                        return a // b
                    case [float(a), float(b)] | [float(a), int(b)] | [int(a), float(b)]:
                        return float(a / b)
                    case [Tensor(), Tensor()]:
                        result_name = f"div_{id(node)}"
                        result = wrapper.div(ctx0, result_name, lhs, rhs)
                        node.ctx.intermediate_tensors.append(result)
                        return result
                    case _:
                        raise ValueError(f"Unsupported operands for division: {lhs}, {rhs}")
            case Operation.MULTIPLY:
                lhs = produce_ggml_graph(ctx0, df, node.operands[0])
                rhs = produce_ggml_graph(ctx0, df, node.operands[1])

                match [lhs, rhs]:
                    case [int(a), int(b)] | [float(a), float(b)] | [float(a), int(b)] | [int(a), float(b)]:
                        return a * b
                    case [Tensor(), Tensor()]:
                        result_name = f"mul_{id(node)}"
                        result = wrapper.mul(ctx0, result_name, lhs, rhs)
                        node.ctx.intermediate_tensors.append(result)
                        return result
                    case _:
                        raise ValueError(f"Unsupported operands for multiplication: {lhs}, {rhs}")
            case Operation.ADD:
                lhs = produce_ggml_graph(ctx0, df, node.operands[0])
                rhs = produce_ggml_graph(ctx0, df, node.operands[1])

                match [lhs, rhs]:
                    case [int(a), int(b)] | [float(a), float(b)] | [float(a), int(b)] | [int(a), float(b)]:
                        return a + b
                    case [Tensor(), Tensor()]:
                        result_name = f"add_{id(node)}"
                        result = wrapper.add(ctx0, result_name, lhs, rhs)
                        node.ctx.intermediate_tensors.append(result)
                        return result
                    case _:
                        raise ValueError(f"Unsupported operands for addition: {lhs}, {rhs}")
            case Operation.SUBTRACT:
                lhs = produce_ggml_graph(ctx0, df, node.operands[0])
                rhs = produce_ggml_graph(ctx0, df, node.operands[1])

                match [lhs, rhs]:
                    case [int(a), int(b)] | [float(a), float(b)] | [float(a), int(b)] | [int(a), float(b)]:
                        return a - b
                    case [Tensor(), Tensor()]:
                        result_name = f"sub_{id(node)}"
                        result = wrapper.sub(ctx0, result_name, lhs, rhs)
                        node.ctx.intermediate_tensors.append(result)
                        return result
                    case _:
                        raise ValueError(f"Unsupported operands for subtraction: {lhs}, {rhs}")

        raise ParseError(f"Unsupported operation: {node.operation}", node.source_token)
    elif isinstance(node, Assignment):
        expression_value = produce_ggml_graph(ctx0, df, node.expression)
        if expression_value:
            if isinstance(expression_value, Tensor):
                target_name = node.target
                node.ctx.graph[target_name] = expression_value
                
                if node.ctx.repeat_index != None:
                    target_name = target_name + f"_{node.ctx.repeat_index}"
                gen.ggml_set_name(expression_value.ptr, target_name.encode())
            else:
                raise ParseError("Can only assign values of type Tensor to graph variables", node.source_token)
        return
    elif isinstance(node, RepeatBlock):
        node.ctx.repeat_var_name = node.count_variable
        for i in range(0, int(node.count)):
            node.ctx.repeat_index = i
            for stmt in node.statements:
                produce_ggml_graph(ctx0, df, stmt)
        return
    elif isinstance(node, FunctionCall):
        function_result = produce_ggml_function_call_graph(ctx0, df, node)
        if isinstance(function_result, Tensor):
            # ensure intermediate tensors that are not assigned to any variables are properly added to the graph
            gen.ggml_build_forward_expand(df, function_result.ptr)
        return
    
    raise ParseError(f"Unsupported statement: {node}", node.source_token)