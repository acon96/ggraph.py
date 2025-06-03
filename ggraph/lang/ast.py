from __future__ import annotations
from typing import List, Union, TypeAlias, Optional, Dict, TYPE_CHECKING
from enum import Enum
import math
import logging
import hashlib
from dataclasses import dataclass, field

from lark import Token

from ggraph.utils import Tensor, ensure_args, ContextParams, ModelParams, BatchParams
from ggraph.lang.parser import ParseError
import ggraph.wrapper as wrapper
from ggraph.wrapper import gen

if TYPE_CHECKING:
    from ggraph.lang.parser import ParseContext

logger = logging.getLogger(__name__)

@dataclass(kw_only=True)
class LoweringContext:
    """Stores the context of the current ast->ggml lowering operation"""
    context_params: ContextParams
    model_params: ModelParams
    batch_params: BatchParams
    intermediate_tensors: List[Tensor] = field(default_factory=lambda: [])
    graph: Dict[str, Tensor] = field(default_factory=lambda: {})
    gguf_tensors: Dict[str, Tensor] = field(default_factory=lambda: {})
    repeat_index: Optional[int] = None
    repeat_var_name: Optional[str] = None
    
    @classmethod
    def from_parse_context(cls, parse_ctx: ParseContext, batch_params: BatchParams, gguf_tensors: Dict[str, Tensor], io_tensors: Dict[str, Tensor]) -> LoweringContext:
        return cls(
            batch_params=batch_params,
            context_params=parse_ctx.context_params,
            model_params=parse_ctx.model_params,
            intermediate_tensors=[],
            graph={**gguf_tensors, **io_tensors },
            gguf_tensors=gguf_tensors,
        )

def resolve_param(name: str, *, raise_error: bool = True, lowering_ctx: Optional[LoweringContext] = None, parse_ctx: Optional[ParseContext] = None, source_token: Optional[Token] = None) -> Optional[int | float]:

    ctx_to_check = lowering_ctx if lowering_ctx else parse_ctx
    if name.startswith("params."):
        if name.startswith("params.context."):
            return getattr(ctx_to_check.context_params, name[len("params.context."):])
        elif name.startswith("params.model."):
            return getattr(ctx_to_check.model_params, name[len("params.model."):])
        elif name.startswith("params.batch."):
            return getattr(ctx_to_check.batch_params, name[len("params.batch."):])
    
    if lowering_ctx:
        if name == lowering_ctx.repeat_var_name:
            return lowering_ctx.repeat_index
        
    try:
        return int(name)
    except ValueError:
        pass

    try:
        return float(name)
    except ValueError:
        pass

    if raise_error:
        raise ParseError(f"Unknown param {name}", source_token)
    
    return None

def resolve_tensor(lowering_ctx: LoweringContext, name: str, *, raise_error: bool = True, source_token: Optional[Token] = None) -> Optional[Tensor]:
    if "%d" in name:
        name = name % lowering_ctx.repeat_index
    
    if name in lowering_ctx.graph:
        return lowering_ctx.graph[name]
    
    if name in lowering_ctx.gguf_tensors:
        return lowering_ctx.gguf_tensors[name]

    if raise_error:
        raise ParseError(f"Unknown tensor {name}", source_token)
    
    # logger.debug(f"Attempted to resolve unknown tensor {name}; graph tensors = {self.ctx.graph.keys()}")
    
    return None

class ASTNode:
    """
    Represents a node in the abstract syntax tree (AST) of a GGML model. 
    Contains a reference to its source token and the current parsing context to allow easy reporting of syntax errors.
    """
    source_token: Token

    def __init__(self, source_token: Token):
        self.source_token = source_token

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
    operation: Operation
    function_name: Optional[str]
    operands: List[Operand]

    def __init__(self, source_token: Token, operation: Operation, *operands: Operand, function_name: Optional[str] = None):
        super().__init__(source_token)
        self.function_name = function_name
        self.operation = operation
        self.operands = list(operands)
    
    def resolve_as_int(self, *, lowering_ctx: Optional[LoweringContext] = None, parse_ctx: Optional[ParseContext] = None) -> int:
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
                result = resolve_param(a, lowering_ctx=lowering_ctx, parse_ctx=parse_ctx)
                if isinstance(result, int):
                    return result
                raise ValueError(f"Expected int, got {type(result)}")
            else:
                raise ValueError() # shouldn't get here
        
        a = a.resolve_as_int(lowering_ctx=lowering_ctx, parse_ctx=parse_ctx) if isinstance(a, Expression) else int(a)

        if b is not None:
            b = b.resolve_as_int(lowering_ctx=lowering_ctx, parse_ctx=parse_ctx) if isinstance(b, Expression) else int(b)

        return self.operation.apply_to_ints(a, b)
    
    def resolve_as_float(self, *, lowering_ctx: Optional[LoweringContext] = None, parse_ctx: Optional[ParseContext] = None) -> float:
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
                result = resolve_param(a, lowering_ctx=lowering_ctx, parse_ctx=parse_ctx)
                if isinstance(result, float):
                    return result
                raise ValueError(f"Expected float, got {type(result)}")
            else:
                raise ValueError() # shouldn't get here
        
        a = a.resolve_as_float(lowering_ctx=lowering_ctx, parse_ctx=parse_ctx) if isinstance(a, Expression) else float(a)
        if b is not None:
            b = b.resolve_as_float(lowering_ctx=lowering_ctx, parse_ctx=parse_ctx) if isinstance(b, Expression) else float(b)

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

    def __init__(self, source_token: Token, target: str, expression: Expression):
        super().__init__(source_token)
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

    def __init__(self, source_token: Token, count: Expression, count_variable: str, statements: List[ASTNode]):
        super().__init__(source_token)
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

    def __init__(self, source_token: Token, function_name: str, arguments: List[Operand]):
        super().__init__(source_token)
        self.function_name = function_name
        self.arguments = arguments

    def __str__(self) -> str:
        return f"{self.function_name}({', '.join(str(arg) for arg in self.arguments)})"
    
    def __repr__(self) -> str:
        return f"FunctionCall(function_name={self.function_name}, arguments={[repr(a) for a in self.arguments]})"
    
Operand: TypeAlias = Union[Expression, int, float, str]

def produce_ggml_function_call_graph(lowering_ctx: LoweringContext, ctx0: wrapper.ggml_context_p, df: wrapper.ggml_cgraph_p, node: FunctionCall | Expression, debug_calls: bool) -> Optional[Tensor | int | float]:
    if isinstance(node, FunctionCall):
        func_name = node.function_name
        raw_args = node.arguments
    else:
        func_name = node.function_name
        raw_args = node.operands

    args = [produce_ggml_graph(lowering_ctx, ctx0, df, op, debug_calls) if isinstance(op, Expression) else op for op in raw_args]
    if debug_calls:
        logger.debug(f"{func_name=} {args=} {raw_args=}")

    ggml_function = wrapper.GGML_FUNCTIONS.get(str(func_name))
    if ggml_function is not None:
        ensure_args(func_name, args, ggml_function.arg_types, node.source_token)
        func_args = hashlib.md5(", ".join([arg.name if isinstance(arg, Tensor) else str(arg) for arg in args]).encode()).hexdigest()[-4:]
        # TODO: gracefully handle errors here
        result_name =  f"{func_name}_{func_args}"
        if lowering_ctx.repeat_index != None:
            result_name = result_name + f"_{lowering_ctx.repeat_index}"
        result: Tensor = ggml_function.func(ctx0, result_name, *args)
        if debug_calls:
            logger.debug(f"{result=}")

        lowering_ctx.intermediate_tensors.append(result)
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
            if debug_calls:
                logger.debug(f"{result_num=}")
            return result_num

    raise NotImplementedError(f"Unsupported function: {func_name}")

def produce_ggml_graph(lowering_ctx: LoweringContext, ctx0: wrapper.ggml_context_p, df: wrapper.ggml_cgraph_p, node: ASTNode, debug_calls: bool) -> Optional[Tensor | int | float]:
    """Performs "instruction lowering" on the AST, producing a ggml graph. This is the final step in compiling a GGML model."""

    # logger.debug(f"Building ggml graph for {node}")
    if isinstance(node, Expression):
        match node.operation:
            case Operation.VALUE:
                if isinstance(node.operands[0], str):
                    result = resolve_tensor(lowering_ctx, node.operands[0], raise_error=False)
                    if not result:
                        result = resolve_param(node.operands[0], lowering_ctx=lowering_ctx, raise_error=False)
                    if result is not None:
                        return result
                    else:
                        raise ParseError(f"Unknown name {node.operands[0]}", node.source_token)
                else:
                    raise ParseError(f"Unsupported operand type: {type(node.operands[0])}", node.source_token)
            case Operation.FUNC_CALL:
                return produce_ggml_function_call_graph(lowering_ctx, ctx0, df, node, debug_calls)
            case Operation.DIVIDE:
                lhs = produce_ggml_graph(lowering_ctx, ctx0, df, node.operands[0], debug_calls)
                rhs = produce_ggml_graph(lowering_ctx, ctx0, df, node.operands[1], debug_calls)

                match [lhs, rhs]:
                    case [int(a), int(b)]:
                        return a // b
                    case [float(a), float(b)] | [float(a), int(b)] | [int(a), float(b)]:
                        return float(a / b)
                    case [Tensor(), Tensor()]:
                        result_name = f"div_{id(node)}"
                        result = wrapper.div(ctx0, result_name, lhs, rhs)
                        lowering_ctx.intermediate_tensors.append(result)
                        return result
                    case _:
                        raise ValueError(f"Unsupported operands for division: {lhs}, {rhs}")
            case Operation.MULTIPLY:
                lhs = produce_ggml_graph(lowering_ctx, ctx0, df, node.operands[0], debug_calls)
                rhs = produce_ggml_graph(lowering_ctx, ctx0, df, node.operands[1], debug_calls)

                match [lhs, rhs]:
                    case [int(a), int(b)] | [float(a), float(b)] | [float(a), int(b)] | [int(a), float(b)]:
                        return a * b
                    case [Tensor(), Tensor()]:
                        result_name = f"mul_{id(node)}"
                        result = wrapper.mul(ctx0, result_name, lhs, rhs)
                        lowering_ctx.intermediate_tensors.append(result)
                        return result
                    case _:
                        raise ValueError(f"Unsupported operands for multiplication: {lhs}, {rhs}")
            case Operation.ADD:
                lhs = produce_ggml_graph(lowering_ctx, ctx0, df, node.operands[0], debug_calls)
                rhs = produce_ggml_graph(lowering_ctx, ctx0, df, node.operands[1], debug_calls)

                match [lhs, rhs]:
                    case [int(a), int(b)] | [float(a), float(b)] | [float(a), int(b)] | [int(a), float(b)]:
                        return a + b
                    case [Tensor(), Tensor()]:
                        result_name = f"add_{id(node)}"
                        result = wrapper.add(ctx0, result_name, lhs, rhs)
                        lowering_ctx.intermediate_tensors.append(result)
                        return result
                    case _:
                        raise ValueError(f"Unsupported operands for addition: {lhs}, {rhs}")
            case Operation.SUBTRACT:
                lhs = produce_ggml_graph(lowering_ctx, ctx0, df, node.operands[0], debug_calls)
                rhs = produce_ggml_graph(lowering_ctx, ctx0, df, node.operands[1], debug_calls)

                match [lhs, rhs]:
                    case [int(a), int(b)] | [float(a), float(b)] | [float(a), int(b)] | [int(a), float(b)]:
                        return a - b
                    case [Tensor(), Tensor()]:
                        result_name = f"sub_{id(node)}"
                        result = wrapper.sub(ctx0, result_name, lhs, rhs)
                        lowering_ctx.intermediate_tensors.append(result)
                        return result
                    case _:
                        raise ValueError(f"Unsupported operands for subtraction: {lhs}, {rhs}")

        raise ParseError(f"Unsupported operation: {node.operation}", node.source_token)
    elif isinstance(node, Assignment):
        expression_value = produce_ggml_graph(lowering_ctx, ctx0, df, node.expression, debug_calls)
        if expression_value:
            if isinstance(expression_value, Tensor):
                target_name = node.target
                lowering_ctx.graph[target_name] = expression_value
                
                if lowering_ctx.repeat_index != None:
                    target_name = target_name + f"_{lowering_ctx.repeat_index}"
                gen.ggml_set_name(expression_value.ptr, target_name.encode())
            else:
                raise ParseError("Can only assign values of type Tensor to graph variables", node.source_token)
        return
    elif isinstance(node, RepeatBlock):
        lowering_ctx.repeat_var_name = node.count_variable
        for i in range(0, node.count.resolve_as_int(lowering_ctx=lowering_ctx)):
            lowering_ctx.repeat_index = i
            for stmt in node.statements:
                produce_ggml_graph(lowering_ctx, ctx0, df, stmt, debug_calls)
        return
    elif isinstance(node, FunctionCall):
        function_result = produce_ggml_function_call_graph(lowering_ctx, ctx0, df, node, debug_calls)
        if isinstance(function_result, Tensor):
            # ensure intermediate tensors that are not assigned to any variables are properly added to the graph
            gen.ggml_build_forward_expand(df, function_result.ptr)
        return
    
    raise ParseError(f"Unsupported statement: {node}", node.source_token)