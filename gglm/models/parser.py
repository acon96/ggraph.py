from __future__ import annotations
from typing import List, Dict, Final
from lark import Lark, Tree, Token
from lark.exceptions import UnexpectedToken, UnexpectedEOF, UnexpectedCharacters
import logging

from gglm.utils import ParseContext, ParseError
from gglm.models import Tensor, GGMLContextParams, ModelParams
from gglm.models.ast import Expression, Assignment, RepeatBlock, FunctionCall, Operand, Operation
from gglm.wrapper import gen

PARSER_DEFINITION = """

model: "model" ID "{" tensors graph "}"
tensors: "tensors" "{" tensor_declaration* "}"
tensor_declaration: ID ":" type "," shape ("," ATTRIBUTE)?
graph: "graph" statement_block
statement_block: "{" statement* "}"
statement: assignment | repeat_block | function_call
assignment: ID "=" expression
repeat_block: "repeat" assignment statement_block
expression: logical_or
logical_or: logical_and ("or" logical_and)*
logical_and: comparison ("and" comparison)*
comparison: arithmetic ((COMP_LT | COMP_GT | COMP_LE | COMP_GE | COMP_EQ | COMP_NE) arithmetic)*
arithmetic: term ((PLUS | MINUS) term)*
term: factor ((MULTIPLY | DIVIDE | MODULO) factor)*
factor: primary (EXPONENT primary)*
primary: ID | NUMBER | "(" expression ")" | function_call
function_call: ID "(" (expression ("," expression)*)? ")"
shape: "[" expression ("," expression)* "]"
type: ("A".."Z")+("0".."9")+

MULTIPLY: "*"
DIVIDE: "/"
MODULO: "//"
PLUS: "+"
MINUS: "-"
EXPONENT: "^"
COMP_LT: "<"
COMP_GT: ">"
COMP_LE: "<="
COMP_GE: ">="
COMP_EQ: "="
COMP_NE: "!="
ATTRIBUTE: "is_input" | "is_state"
ID: ("a".."z" | "A".."Z")("a".."z" | "A".."Z" | "0".."9" | "_" | "." | "%")*

%import common.NUMBER
%import common.WS
%import common. SH_COMMENT
%ignore WS
%ignore SH_COMMENT

"""


class GGMLParser:
    """
    Parses *.ggml files based on the "GGML language" using the Lark library.
    The "GGML language" is a custom language for representing neural network models in a string serializable format so that
    the model definition can be stored as one or more GGUF key-value pairs.
    """

    TYPE_TO_GGML_TYPE: Final[Dict[str, int]] = {
        "F32": gen.GGML_TYPE_F32,
        "F16": gen.GGML_TYPE_F16,
        "BF16": gen.GGML_TYPE_BF16,
        "I32": gen.GGML_TYPE_I32,
        "I16": gen.GGML_TYPE_I16,
        "I8": gen.GGML_TYPE_I8,
    }

    def __init__(self):
        # Initialize the Lark parser with the loaded grammar
        self.parser = Lark(PARSER_DEFINITION, start='model')

    def collapse_expr_tree(self, tree: Tree, keep_rules=None):
        """Removes unnecessary tree nodes from the parse tree."""
        if keep_rules is None:
            # Specify rules to retain; all others will be candidates for collapsing
            keep_rules = {'start', 'model', 'tensors', 'tensor_declaration', 'type', 'shape',
                        'graph', 'statement', 'assignment', 'expression', 'function_call', 'attribute'}

        if isinstance(tree, Token):
            return tree

        # Recursively collapse children
        collapsed_children = [self.collapse_expr_tree(child, keep_rules) for child in tree.children]

        # Flatten unnecessary rule levels
        if tree.data not in keep_rules and len(collapsed_children) == 1:
            return collapsed_children[0]

        return Tree(tree.data, collapsed_children)
    
    def parse_expression(self, ctx: ParseContext, expr_subtree: Tree) -> Expression:
        """Parses an expression subtree into an Expression AST object."""
        def explore_branch(tree_level: Tree | Token) -> Operand:
            if isinstance(tree_level, Token):
                return Expression(tree_level, ctx, Operation.VALUE, tree_level.value)

            # Handle function calls
            if tree_level.data == "function_call":
                func_name = tree_level.children[0].value
                args = [explore_branch(x) for x in tree_level.children[1:]]
                return Expression(tree_level.children[0], ctx, Operation.FUNC_CALL, *args, function_name=func_name)

            # Map rule names to operator tokens and Operation enums
            binary_ops = {
                "arithmetic": {
                    "+": Operation.ADD,
                    "-": Operation.SUBTRACT,
                },
                "term": {
                    "*": Operation.MULTIPLY,
                    "/": Operation.DIVIDE,
                    "//": Operation.MODULO,
                },
                "factor": {
                    "^": Operation.POWER,
                },
                # Add more for logical/comparison if needed
            }

            # Recursively handle binary operations for arithmetic, term, factor, etc.
            if tree_level.data in binary_ops:
                children = tree_level.children
                # If only one child, just recurse
                if len(children) == 1:
                    return explore_branch(children[0])
                # Otherwise, left-associative: left, op, right, [op, right, ...]
                left = explore_branch(children[0])
                i = 1
                while i < len(children):
                    op_token = children[i]
                    right = explore_branch(children[i + 1])
                    op_map = binary_ops[tree_level.data]
                    op = op_map.get(op_token.value)
                    if op is None:
                        raise ValueError(f"Unknown operator {op_token.value} in {tree_level.data}")
                    left = Expression(op_token, ctx, op, left, right)
                    i += 2
                return left

            # Handle parenthesis and single-child rules
            if len(tree_level.children) == 1:
                return explore_branch(tree_level.children[0])

            raise ValueError(f"Unexpected branch in expression tree: {tree_level}")

        rule_token: Token = expr_subtree.data
        if rule_token.type != "RULE" and rule_token.value != "expression":
            raise ParseError(f"Unexpected rule token in expression subtree: {rule_token}", rule_token)

        return explore_branch(expr_subtree)
        # """Parses an expression subtree into an Expression AST object."""
        # def explore_branch(tree_level: Tree | Token) -> Operand:
        #     if isinstance(tree_level, Token):
        #         return Expression(tree_level, ctx, Operation.VALUE, tree_level.value)
            
        #     match tree_level:
        #         case Tree(data=Token('RULE', 'function_call'), children=[Token('ID', func_name), *args]):
        #             return Expression(tree_level.children[0], ctx, Operation.FUNC_CALL, *[explore_branch(x) for x in args], function_name=func_name)
        #         case Tree(data=Token('RULE', 'term'), children=[a, raw_op, b]):
        #             if isinstance(raw_op, Token):
        #                 op = Operation.MULTIPLY if raw_op.value == "*" else Operation.DIVIDE if raw_op.value == "/" else None
        #                 return Expression(tree_level.children[0], ctx, op, explore_branch(a), explore_branch(b))
                    
        #     if len(tree_level.children) == 1:
        #         return explore_branch(tree_level.children[0])
            
        #     raise ValueError(f"Unexpected branch in expression tree: {tree_level}")

        # rule_token: Token = expr_subtree.data
        # if rule_token.type != "RULE" and rule_token.value != "expression":
        #     raise ParseError(f"Unexpected rule token in expression subtree: {rule_token}", rule_token)
        
        # return explore_branch(expr_subtree)
    
    def parse_type(self, type_subtree: Tree) -> int:
        type_tokens = type_subtree.children
        if all(isinstance(x, Token) for x in type_tokens):
            type_str = "".join([ x.value for x in type_tokens])
        else:
            raise ValueError() # should never happen
            
        result = self.TYPE_TO_GGML_TYPE.get(type_str)
        if result is not None:
            return result
        raise ParseError(f"Unknown tensor data type '{type_str}'", type_tokens[0])

    def parse_shape(self, ctx: ParseContext, shape_subtree: Tree) -> List[Expression]:
        match shape_subtree:
            case Tree(data=Token('RULE', 'shape'), children=[*dims]):
                return [self.parse_expression(ctx, dim) for dim in dims]
        raise ValueError(f"Unknown shape format: {shape_subtree}")

    def parse_tensor(self, ctx: ParseContext, tensor: Tree) -> Tensor:
        match tensor:
            case Tree(data=Token('RULE', 'tensor_declaration'), children=[name, type, shape]):
                if isinstance(name, Token) and isinstance(type, Tree) and isinstance(shape, Tree):
                    new_tensor = Tensor(name=name.value, type=self.parse_type(type), shape=self.parse_shape(ctx, shape))
                    ctx.created_tensors.append(new_tensor)
                    return new_tensor
            case Tree(data=Token('RULE', 'tensor_declaration'), children=[name, type, shape, attr]):
                if isinstance(name, Token) and isinstance(type, Tree) and isinstance(shape, Tree) and isinstance(attr, Token):
                    new_tensor = Tensor(name=name.value, type=self.parse_type(type), shape=self.parse_shape(ctx, shape), is_input=attr.value == 'is_input')
                    ctx.created_tensors.append(new_tensor)
                    return new_tensor
            
        logging.debug(f"{tensor=}")
        raise ParseError(f"Failed to parse tensor", tensor.children[0])
    
    def parse_statement(self, ctx: ParseContext, statement: Tree):
        match statement:
            case Tree(data=Token('RULE', 'assignment'), children=[name, value]):
                if isinstance(name, Token) and isinstance(value, Tree):
                    return Assignment(source_token=name, ctx=ctx, target=name.value, expression=self.parse_expression(ctx, value))
            case Tree(data=Token('RULE', 'repeat_block'), children=[Tree(data=Token('RULE', 'assignment'), children=[count_variable, count]), block]):
                if isinstance(count, Tree) and isinstance(block, Tree):
                    statements = [self.parse_statement(ctx, stmt.children[0]) for stmt in block.children]
                    return RepeatBlock(source_token=count.children[0], ctx=ctx, count=self.parse_expression(ctx, count), count_variable=count_variable, statements=statements)
            case Tree(data=Token('RULE', 'function_call'), children=[name, *args]):
                if isinstance(name, Token):
                    args = [self.parse_expression(ctx, arg) for arg in args]
                    return FunctionCall(source_token=name, ctx=ctx, function_name=name.value, arguments=args)
            case _:
                logging.debug(f"unmatched statement: {statement}")
        raise ParseError(f"Failed to parse graph statement", statement.children[0])

    def parse(self, file: str, context_params: GGMLContextParams, model_params: ModelParams):
        """Parse a file containing a graph definition. Return the parsed graph context."""
        # Read the content of the provided file
        with open(file, 'r') as f:
            text = f.read()

        # Run the parsing operation on the file's content and return the result
        try:
            parsed_content = self.parser.parse(text)
        except Exception as ex:
            if isinstance(ex, UnexpectedToken):
                raise ParseError(f"Unexpected token while parsing: {ex.token}", ex.token)
            elif isinstance(ex, UnexpectedCharacters):
                raise ParseError(f"Unexpected characters while parsing: {ex.char}", None).override_pos(ex.line, ex.column)
            elif isinstance(ex, UnexpectedEOF):
                raise ParseError(f"Unexpected end of file while parsing", None).override_pos(ex.line, ex.column)
            raise ex
            
        model_tree = self.collapse_expr_tree(parsed_content)
        model_name = model_tree.children[0].value
        
        logging.debug(f"Parsing model {model_name}")

        ctx = ParseContext(model=model_name, source=text, context_params=context_params, model_params=model_params)

        # parse tensors block
        tensors_subtree = model_tree.children[1]
        if isinstance(tensors_subtree, Tree) and tensors_subtree.data == "tensors":
            for tensor in tensors_subtree.children:
                parsed_tensor = self.parse_tensor(ctx, tensor)
                ctx.graph[parsed_tensor.name] = parsed_tensor

                logging.debug(f"Defined tensor: {parsed_tensor}")

        # parse graph block
        graph_subtree = model_tree.children[2]
        if isinstance(graph_subtree, Tree) and graph_subtree.data == "graph":
            for node in graph_subtree.children[0].children:
                ctx.ast.append(self.parse_statement(ctx, node.children[0]))

        
        repeated_tensors = [t for t in ctx.created_tensors if '%d' in t.name]
        num_repeated_tensors = len(repeated_tensors)

        repeat_blocks = [s for s in ctx.ast if isinstance(s, RepeatBlock)]
        num_repeat_blocks = len(repeat_blocks)

        if num_repeated_tensors > 0 and num_repeat_blocks > 1:
            raise ParseError(f"Multiple repeat blocks in the graph are not supported when using repeated tensors! Found {num_repeat_blocks} repeat blocks and {num_repeated_tensors} repeated tensors.", None)
        
        # exapnd repeated tensors
        if num_repeated_tensors > 0:
            if len(repeat_blocks) == 0:
                raise ParseError(f"Repeated tensors found but no repeat blocks in the graph! Found {num_repeated_tensors} repeated tensors.", None)
            num_replicas = int(repeat_blocks[0].count)
            # filter out the repeated tensor placeholders then re-add the expanded ones
            ctx.created_tensors = [t for t in ctx.created_tensors if '%d' not in t.name]
            ctx.graph = {k: v for k, v in ctx.graph.items() if '%d' not in k}
            for i in range(num_replicas):
                for tensor in repeated_tensors:
                    new_name = tensor.name % i
                    new_tensor = Tensor(
                        name=new_name, type=tensor.type, shape=tensor.shape, 
                        is_input=tensor.is_input, is_view=tensor.is_view
                    )
                    ctx.created_tensors.append(new_tensor)
                    ctx.graph[new_name] = new_tensor
        
        logging.debug(f"Finished parsing model {model_name}")
        
        return ctx