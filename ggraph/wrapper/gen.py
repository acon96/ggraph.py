# -*- coding: utf-8 -*-
#
# TARGET arch is: ['-I./include/', '-I/usr/include/clang/14']
# WORD_SIZE is: 8
# POINTER_SIZE is: 8
# LONGDOUBLE_SIZE is: 16
#
from __future__ import annotations
from typing import TYPE_CHECKING, Callable, List, Any, Dict, cast, TypeVar
import functools
import ctypes
import os

F = TypeVar("F", bound=Callable[..., Any])

c_int128 = ctypes.c_ubyte*16
c_uint128 = c_int128
void = None
if ctypes.sizeof(ctypes.c_longdouble) == 16:
    c_long_double_t = ctypes.c_longdouble
else:
    c_long_double_t = ctypes.c_ubyte*16

GGML_LIBRARY_DIR = os.environ.get('GGML_LIBRARY_DIR', '')
_libraries = {}

def ctypes_function_for_shared_library(libname: str):
    def ctypes_function(
        name: str, argtypes: List[Any], restype: Any
    ):
        enabled = libname in _libraries and hasattr(_libraries[libname], name)
        def decorator(f: F) -> F:
            if enabled:
                func = getattr(_libraries[libname], name)
                func.argtypes = argtypes
                func.restype = restype
                functools.wraps(f)(func)
                return func
            else:
                def f_(*args: Any, **kwargs: Any):
                    raise RuntimeError(
                        f"Function '{name}' is not available in the shared library '{libname}'."
                    )
                return cast(F, f_)

        return decorator

    return ctypes_function
_libraries['libggml-base.so'] = ctypes.CDLL("%slibggml-base.so" % GGML_LIBRARY_DIR)
def string_cast(char_pointer, encoding='utf-8', errors='strict'):
    value = ctypes.cast(char_pointer, ctypes.c_char_p).value
    if value is not None and encoding is not None:
        value = value.decode(encoding, errors=errors)
    return value


def char_pointer_cast(string, encoding='utf-8'):
    if encoding is not None:
        try:
            string = string.encode(encoding)
        except AttributeError:
            # In Python3, bytes has no encode attribute
            pass
    string = ctypes.c_char_p(string)
    return ctypes.cast(string, ctypes.POINTER(ctypes.c_char))



class AsDictMixin:
    @classmethod
    def as_dict(cls, self):
        result = {}
        if not isinstance(self, AsDictMixin):
            # not a structure, assume it's already a python object
            return self
        if not hasattr(cls, "_fields_"):
            return result
        # sys.version_info >= (3, 5)
        # for (field, *_) in cls._fields_:  # noqa
        for field_tuple in cls._fields_:  # noqa
            field = field_tuple[0]
            if field.startswith('PADDING_'):
                continue
            value = getattr(self, field)
            type_ = type(value)
            if hasattr(value, "_length_") and hasattr(value, "_type_"):
                # array
                type_ = type_._type_
                if hasattr(type_, 'as_dict'):
                    value = [type_.as_dict(v) for v in value]
                else:
                    value = [i for i in value]
            elif hasattr(value, "contents") and hasattr(value, "_type_"):
                # pointer
                try:
                    if not hasattr(type_, "as_dict"):
                        value = value.contents
                    else:
                        type_ = type_._type_
                        value = type_.as_dict(value.contents)
                except ValueError:
                    # nullptr
                    value = None
            elif isinstance(value, AsDictMixin):
                # other structure
                value = type_.as_dict(value)
            result[field] = value
        return result


class Structure(ctypes.Structure, AsDictMixin):

    def __init__(self, *args, **kwds):
        # We don't want to use positional arguments fill PADDING_* fields

        args = dict(zip(self.__class__._field_names_(), args))
        args.update(kwds)
        super(Structure, self).__init__(**args)

    @classmethod
    def _field_names_(cls):
        if hasattr(cls, '_fields_'):
            return (f[0] for f in cls._fields_ if not f[0].startswith('PADDING'))
        else:
            return ()

    @classmethod
    def get_type(cls, field):
        for f in cls._fields_:
            if f[0] == field:
                return f[1]
        return None

    @classmethod
    def bind(cls, bound_fields):
        fields = {}
        for name, type_ in cls._fields_:
            if hasattr(type_, "restype"):
                if name in bound_fields:
                    if bound_fields[name] is None:
                        fields[name] = type_()
                    else:
                        # use a closure to capture the callback from the loop scope
                        fields[name] = (
                            type_((lambda callback: lambda *args: callback(*args))(
                                bound_fields[name]))
                        )
                    del bound_fields[name]
                else:
                    # default callback implementation (does nothing)
                    try:
                        default_ = type_(0).restype().value
                    except TypeError:
                        default_ = None
                    fields[name] = type_((
                        lambda default_: lambda *args: default_)(default_))
            else:
                # not a callback function, use default initialization
                if name in bound_fields:
                    fields[name] = bound_fields[name]
                    del bound_fields[name]
                else:
                    fields[name] = type_()
        if len(bound_fields) != 0:
            raise ValueError(
                "Cannot bind the following unknown callback(s) {}.{}".format(
                    cls.__name__, bound_fields.keys()
            ))
        return cls(**fields)


class Union(ctypes.Union, AsDictMixin):
    pass



_libraries['libggml.so'] = ctypes.CDLL("%slibggml.so" % GGML_LIBRARY_DIR)
_libraries['libggml-cpu.so'] = ctypes.CDLL("%slibggml-cpu.so" % GGML_LIBRARY_DIR)
_libraries['libggml-cuda.so'] = ctypes.CDLL("%slibggml-cuda.so" % GGML_LIBRARY_DIR)


@ctypes_function_for_shared_library('libggml-base.so')("ggml_abort", [ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.POINTER(ctypes.c_char)], None)
def ggml_abort(file: ctypes._Pointer[ctypes.c_char], line: ctypes.c_int32, fmt: ctypes._Pointer[ctypes.c_char]) -> None:
    ...


# values for enumeration 'ggml_status'
ggml_status__enumvalues = {
    -2: 'GGML_STATUS_ALLOC_FAILED',
    -1: 'GGML_STATUS_FAILED',
    0: 'GGML_STATUS_SUCCESS',
    1: 'GGML_STATUS_ABORTED',
}
GGML_STATUS_ALLOC_FAILED = -2
GGML_STATUS_FAILED = -1
GGML_STATUS_SUCCESS = 0
GGML_STATUS_ABORTED = 1
ggml_status = ctypes.c_int32 # enum
@ctypes_function_for_shared_library('libggml-base.so')("ggml_status_to_string", [ggml_status], ctypes.POINTER(ctypes.c_char))
def ggml_status_to_string(status: ggml_status) -> ctypes._Pointer[ctypes.c_char]:
    ...

ggml_fp16_t = ctypes.c_uint16
@ctypes_function_for_shared_library('libggml-base.so')("ggml_fp16_to_fp32", [ggml_fp16_t], ctypes.c_float)
def ggml_fp16_to_fp32(p1: ggml_fp16_t) -> ctypes.c_float:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_fp32_to_fp16", [ctypes.c_float], ggml_fp16_t)
def ggml_fp32_to_fp16(p1: ctypes.c_float) -> ggml_fp16_t:
    ...

int64_t = ctypes.c_int64
@ctypes_function_for_shared_library('libggml-base.so')("ggml_fp16_to_fp32_row", [ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float), int64_t], None)
def ggml_fp16_to_fp32_row(p1: ctypes._Pointer[ctypes.c_uint16], p2: ctypes._Pointer[ctypes.c_float], p3: int64_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_fp32_to_fp16_row", [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint16), int64_t], None)
def ggml_fp32_to_fp16_row(p1: ctypes._Pointer[ctypes.c_float], p2: ctypes._Pointer[ctypes.c_uint16], p3: int64_t) -> None:
    ...

class struct_c__SA_ggml_bf16_t(Structure):
    if TYPE_CHECKING:
        bits: ctypes.c_uint16
struct_c__SA_ggml_bf16_t._pack_ = 1 # source:False
struct_c__SA_ggml_bf16_t._fields_ = [
    ('bits', ctypes.c_uint16),
]

ggml_bf16_t = struct_c__SA_ggml_bf16_t
@ctypes_function_for_shared_library('libggml-base.so')("ggml_fp32_to_bf16", [ctypes.c_float], ggml_bf16_t)
def ggml_fp32_to_bf16(p1: ctypes.c_float) -> ggml_bf16_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_bf16_to_fp32", [ggml_bf16_t], ctypes.c_float)
def ggml_bf16_to_fp32(p1: ggml_bf16_t) -> ctypes.c_float:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_bf16_to_fp32_row", [ctypes.POINTER(struct_c__SA_ggml_bf16_t), ctypes.POINTER(ctypes.c_float), int64_t], None)
def ggml_bf16_to_fp32_row(p1: ctypes._Pointer[struct_c__SA_ggml_bf16_t], p2: ctypes._Pointer[ctypes.c_float], p3: int64_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_fp32_to_bf16_row_ref", [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(struct_c__SA_ggml_bf16_t), int64_t], None)
def ggml_fp32_to_bf16_row_ref(p1: ctypes._Pointer[ctypes.c_float], p2: ctypes._Pointer[struct_c__SA_ggml_bf16_t], p3: int64_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_fp32_to_bf16_row", [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(struct_c__SA_ggml_bf16_t), int64_t], None)
def ggml_fp32_to_bf16_row(p1: ctypes._Pointer[ctypes.c_float], p2: ctypes._Pointer[struct_c__SA_ggml_bf16_t], p3: int64_t) -> None:
    ...


# values for enumeration 'ggml_type'
ggml_type__enumvalues = {
    0: 'GGML_TYPE_F32',
    1: 'GGML_TYPE_F16',
    2: 'GGML_TYPE_Q4_0',
    3: 'GGML_TYPE_Q4_1',
    6: 'GGML_TYPE_Q5_0',
    7: 'GGML_TYPE_Q5_1',
    8: 'GGML_TYPE_Q8_0',
    9: 'GGML_TYPE_Q8_1',
    10: 'GGML_TYPE_Q2_K',
    11: 'GGML_TYPE_Q3_K',
    12: 'GGML_TYPE_Q4_K',
    13: 'GGML_TYPE_Q5_K',
    14: 'GGML_TYPE_Q6_K',
    15: 'GGML_TYPE_Q8_K',
    16: 'GGML_TYPE_IQ2_XXS',
    17: 'GGML_TYPE_IQ2_XS',
    18: 'GGML_TYPE_IQ3_XXS',
    19: 'GGML_TYPE_IQ1_S',
    20: 'GGML_TYPE_IQ4_NL',
    21: 'GGML_TYPE_IQ3_S',
    22: 'GGML_TYPE_IQ2_S',
    23: 'GGML_TYPE_IQ4_XS',
    24: 'GGML_TYPE_I8',
    25: 'GGML_TYPE_I16',
    26: 'GGML_TYPE_I32',
    27: 'GGML_TYPE_I64',
    28: 'GGML_TYPE_F64',
    29: 'GGML_TYPE_IQ1_M',
    30: 'GGML_TYPE_BF16',
    34: 'GGML_TYPE_TQ1_0',
    35: 'GGML_TYPE_TQ2_0',
    39: 'GGML_TYPE_COUNT',
}
GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q4_0 = 2
GGML_TYPE_Q4_1 = 3
GGML_TYPE_Q5_0 = 6
GGML_TYPE_Q5_1 = 7
GGML_TYPE_Q8_0 = 8
GGML_TYPE_Q8_1 = 9
GGML_TYPE_Q2_K = 10
GGML_TYPE_Q3_K = 11
GGML_TYPE_Q4_K = 12
GGML_TYPE_Q5_K = 13
GGML_TYPE_Q6_K = 14
GGML_TYPE_Q8_K = 15
GGML_TYPE_IQ2_XXS = 16
GGML_TYPE_IQ2_XS = 17
GGML_TYPE_IQ3_XXS = 18
GGML_TYPE_IQ1_S = 19
GGML_TYPE_IQ4_NL = 20
GGML_TYPE_IQ3_S = 21
GGML_TYPE_IQ2_S = 22
GGML_TYPE_IQ4_XS = 23
GGML_TYPE_I8 = 24
GGML_TYPE_I16 = 25
GGML_TYPE_I32 = 26
GGML_TYPE_I64 = 27
GGML_TYPE_F64 = 28
GGML_TYPE_IQ1_M = 29
GGML_TYPE_BF16 = 30
GGML_TYPE_TQ1_0 = 34
GGML_TYPE_TQ2_0 = 35
GGML_TYPE_COUNT = 39
ggml_type = ctypes.c_uint32 # enum

# values for enumeration 'ggml_prec'
ggml_prec__enumvalues = {
    0: 'GGML_PREC_DEFAULT',
    10: 'GGML_PREC_F32',
}
GGML_PREC_DEFAULT = 0
GGML_PREC_F32 = 10
ggml_prec = ctypes.c_uint32 # enum

# values for enumeration 'ggml_ftype'
ggml_ftype__enumvalues = {
    -1: 'GGML_FTYPE_UNKNOWN',
    0: 'GGML_FTYPE_ALL_F32',
    1: 'GGML_FTYPE_MOSTLY_F16',
    2: 'GGML_FTYPE_MOSTLY_Q4_0',
    3: 'GGML_FTYPE_MOSTLY_Q4_1',
    4: 'GGML_FTYPE_MOSTLY_Q4_1_SOME_F16',
    7: 'GGML_FTYPE_MOSTLY_Q8_0',
    8: 'GGML_FTYPE_MOSTLY_Q5_0',
    9: 'GGML_FTYPE_MOSTLY_Q5_1',
    10: 'GGML_FTYPE_MOSTLY_Q2_K',
    11: 'GGML_FTYPE_MOSTLY_Q3_K',
    12: 'GGML_FTYPE_MOSTLY_Q4_K',
    13: 'GGML_FTYPE_MOSTLY_Q5_K',
    14: 'GGML_FTYPE_MOSTLY_Q6_K',
    15: 'GGML_FTYPE_MOSTLY_IQ2_XXS',
    16: 'GGML_FTYPE_MOSTLY_IQ2_XS',
    17: 'GGML_FTYPE_MOSTLY_IQ3_XXS',
    18: 'GGML_FTYPE_MOSTLY_IQ1_S',
    19: 'GGML_FTYPE_MOSTLY_IQ4_NL',
    20: 'GGML_FTYPE_MOSTLY_IQ3_S',
    21: 'GGML_FTYPE_MOSTLY_IQ2_S',
    22: 'GGML_FTYPE_MOSTLY_IQ4_XS',
    23: 'GGML_FTYPE_MOSTLY_IQ1_M',
    24: 'GGML_FTYPE_MOSTLY_BF16',
}
GGML_FTYPE_UNKNOWN = -1
GGML_FTYPE_ALL_F32 = 0
GGML_FTYPE_MOSTLY_F16 = 1
GGML_FTYPE_MOSTLY_Q4_0 = 2
GGML_FTYPE_MOSTLY_Q4_1 = 3
GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4
GGML_FTYPE_MOSTLY_Q8_0 = 7
GGML_FTYPE_MOSTLY_Q5_0 = 8
GGML_FTYPE_MOSTLY_Q5_1 = 9
GGML_FTYPE_MOSTLY_Q2_K = 10
GGML_FTYPE_MOSTLY_Q3_K = 11
GGML_FTYPE_MOSTLY_Q4_K = 12
GGML_FTYPE_MOSTLY_Q5_K = 13
GGML_FTYPE_MOSTLY_Q6_K = 14
GGML_FTYPE_MOSTLY_IQ2_XXS = 15
GGML_FTYPE_MOSTLY_IQ2_XS = 16
GGML_FTYPE_MOSTLY_IQ3_XXS = 17
GGML_FTYPE_MOSTLY_IQ1_S = 18
GGML_FTYPE_MOSTLY_IQ4_NL = 19
GGML_FTYPE_MOSTLY_IQ3_S = 20
GGML_FTYPE_MOSTLY_IQ2_S = 21
GGML_FTYPE_MOSTLY_IQ4_XS = 22
GGML_FTYPE_MOSTLY_IQ1_M = 23
GGML_FTYPE_MOSTLY_BF16 = 24
ggml_ftype = ctypes.c_int32 # enum

# values for enumeration 'ggml_op'
ggml_op__enumvalues = {
    0: 'GGML_OP_NONE',
    1: 'GGML_OP_DUP',
    2: 'GGML_OP_ADD',
    3: 'GGML_OP_ADD1',
    4: 'GGML_OP_ACC',
    5: 'GGML_OP_SUB',
    6: 'GGML_OP_MUL',
    7: 'GGML_OP_DIV',
    8: 'GGML_OP_SQR',
    9: 'GGML_OP_SQRT',
    10: 'GGML_OP_LOG',
    11: 'GGML_OP_SIN',
    12: 'GGML_OP_COS',
    13: 'GGML_OP_SUM',
    14: 'GGML_OP_SUM_ROWS',
    15: 'GGML_OP_MEAN',
    16: 'GGML_OP_ARGMAX',
    17: 'GGML_OP_COUNT_EQUAL',
    18: 'GGML_OP_REPEAT',
    19: 'GGML_OP_REPEAT_BACK',
    20: 'GGML_OP_CONCAT',
    21: 'GGML_OP_SILU_BACK',
    22: 'GGML_OP_NORM',
    23: 'GGML_OP_RMS_NORM',
    24: 'GGML_OP_RMS_NORM_BACK',
    25: 'GGML_OP_GROUP_NORM',
    26: 'GGML_OP_L2_NORM',
    27: 'GGML_OP_MUL_MAT',
    28: 'GGML_OP_MUL_MAT_ID',
    29: 'GGML_OP_OUT_PROD',
    30: 'GGML_OP_SCALE',
    31: 'GGML_OP_SET',
    32: 'GGML_OP_CPY',
    33: 'GGML_OP_CONT',
    34: 'GGML_OP_RESHAPE',
    35: 'GGML_OP_VIEW',
    36: 'GGML_OP_PERMUTE',
    37: 'GGML_OP_TRANSPOSE',
    38: 'GGML_OP_GET_ROWS',
    39: 'GGML_OP_GET_ROWS_BACK',
    40: 'GGML_OP_DIAG',
    41: 'GGML_OP_DIAG_MASK_INF',
    42: 'GGML_OP_DIAG_MASK_ZERO',
    43: 'GGML_OP_SOFT_MAX',
    44: 'GGML_OP_SOFT_MAX_BACK',
    45: 'GGML_OP_ROPE',
    46: 'GGML_OP_ROPE_BACK',
    47: 'GGML_OP_CLAMP',
    48: 'GGML_OP_CONV_TRANSPOSE_1D',
    49: 'GGML_OP_IM2COL',
    50: 'GGML_OP_IM2COL_BACK',
    51: 'GGML_OP_CONV_2D_DW',
    52: 'GGML_OP_CONV_TRANSPOSE_2D',
    53: 'GGML_OP_POOL_1D',
    54: 'GGML_OP_POOL_2D',
    55: 'GGML_OP_POOL_2D_BACK',
    56: 'GGML_OP_UPSCALE',
    57: 'GGML_OP_PAD',
    58: 'GGML_OP_PAD_REFLECT_1D',
    59: 'GGML_OP_ARANGE',
    60: 'GGML_OP_TIMESTEP_EMBEDDING',
    61: 'GGML_OP_ARGSORT',
    62: 'GGML_OP_LEAKY_RELU',
    63: 'GGML_OP_FLASH_ATTN_EXT',
    64: 'GGML_OP_FLASH_ATTN_BACK',
    65: 'GGML_OP_SSM_CONV',
    66: 'GGML_OP_SSM_SCAN',
    67: 'GGML_OP_WIN_PART',
    68: 'GGML_OP_WIN_UNPART',
    69: 'GGML_OP_GET_REL_POS',
    70: 'GGML_OP_ADD_REL_POS',
    71: 'GGML_OP_RWKV_WKV6',
    72: 'GGML_OP_GATED_LINEAR_ATTN',
    73: 'GGML_OP_RWKV_WKV7',
    74: 'GGML_OP_UNARY',
    75: 'GGML_OP_MAP_CUSTOM1',
    76: 'GGML_OP_MAP_CUSTOM2',
    77: 'GGML_OP_MAP_CUSTOM3',
    78: 'GGML_OP_CUSTOM',
    79: 'GGML_OP_CROSS_ENTROPY_LOSS',
    80: 'GGML_OP_CROSS_ENTROPY_LOSS_BACK',
    81: 'GGML_OP_OPT_STEP_ADAMW',
    82: 'GGML_OP_COUNT',
}
GGML_OP_NONE = 0
GGML_OP_DUP = 1
GGML_OP_ADD = 2
GGML_OP_ADD1 = 3
GGML_OP_ACC = 4
GGML_OP_SUB = 5
GGML_OP_MUL = 6
GGML_OP_DIV = 7
GGML_OP_SQR = 8
GGML_OP_SQRT = 9
GGML_OP_LOG = 10
GGML_OP_SIN = 11
GGML_OP_COS = 12
GGML_OP_SUM = 13
GGML_OP_SUM_ROWS = 14
GGML_OP_MEAN = 15
GGML_OP_ARGMAX = 16
GGML_OP_COUNT_EQUAL = 17
GGML_OP_REPEAT = 18
GGML_OP_REPEAT_BACK = 19
GGML_OP_CONCAT = 20
GGML_OP_SILU_BACK = 21
GGML_OP_NORM = 22
GGML_OP_RMS_NORM = 23
GGML_OP_RMS_NORM_BACK = 24
GGML_OP_GROUP_NORM = 25
GGML_OP_L2_NORM = 26
GGML_OP_MUL_MAT = 27
GGML_OP_MUL_MAT_ID = 28
GGML_OP_OUT_PROD = 29
GGML_OP_SCALE = 30
GGML_OP_SET = 31
GGML_OP_CPY = 32
GGML_OP_CONT = 33
GGML_OP_RESHAPE = 34
GGML_OP_VIEW = 35
GGML_OP_PERMUTE = 36
GGML_OP_TRANSPOSE = 37
GGML_OP_GET_ROWS = 38
GGML_OP_GET_ROWS_BACK = 39
GGML_OP_DIAG = 40
GGML_OP_DIAG_MASK_INF = 41
GGML_OP_DIAG_MASK_ZERO = 42
GGML_OP_SOFT_MAX = 43
GGML_OP_SOFT_MAX_BACK = 44
GGML_OP_ROPE = 45
GGML_OP_ROPE_BACK = 46
GGML_OP_CLAMP = 47
GGML_OP_CONV_TRANSPOSE_1D = 48
GGML_OP_IM2COL = 49
GGML_OP_IM2COL_BACK = 50
GGML_OP_CONV_2D_DW = 51
GGML_OP_CONV_TRANSPOSE_2D = 52
GGML_OP_POOL_1D = 53
GGML_OP_POOL_2D = 54
GGML_OP_POOL_2D_BACK = 55
GGML_OP_UPSCALE = 56
GGML_OP_PAD = 57
GGML_OP_PAD_REFLECT_1D = 58
GGML_OP_ARANGE = 59
GGML_OP_TIMESTEP_EMBEDDING = 60
GGML_OP_ARGSORT = 61
GGML_OP_LEAKY_RELU = 62
GGML_OP_FLASH_ATTN_EXT = 63
GGML_OP_FLASH_ATTN_BACK = 64
GGML_OP_SSM_CONV = 65
GGML_OP_SSM_SCAN = 66
GGML_OP_WIN_PART = 67
GGML_OP_WIN_UNPART = 68
GGML_OP_GET_REL_POS = 69
GGML_OP_ADD_REL_POS = 70
GGML_OP_RWKV_WKV6 = 71
GGML_OP_GATED_LINEAR_ATTN = 72
GGML_OP_RWKV_WKV7 = 73
GGML_OP_UNARY = 74
GGML_OP_MAP_CUSTOM1 = 75
GGML_OP_MAP_CUSTOM2 = 76
GGML_OP_MAP_CUSTOM3 = 77
GGML_OP_CUSTOM = 78
GGML_OP_CROSS_ENTROPY_LOSS = 79
GGML_OP_CROSS_ENTROPY_LOSS_BACK = 80
GGML_OP_OPT_STEP_ADAMW = 81
GGML_OP_COUNT = 82
ggml_op = ctypes.c_uint32 # enum

# values for enumeration 'ggml_unary_op'
ggml_unary_op__enumvalues = {
    0: 'GGML_UNARY_OP_ABS',
    1: 'GGML_UNARY_OP_SGN',
    2: 'GGML_UNARY_OP_NEG',
    3: 'GGML_UNARY_OP_STEP',
    4: 'GGML_UNARY_OP_TANH',
    5: 'GGML_UNARY_OP_ELU',
    6: 'GGML_UNARY_OP_RELU',
    7: 'GGML_UNARY_OP_SIGMOID',
    8: 'GGML_UNARY_OP_GELU',
    9: 'GGML_UNARY_OP_GELU_QUICK',
    10: 'GGML_UNARY_OP_SILU',
    11: 'GGML_UNARY_OP_HARDSWISH',
    12: 'GGML_UNARY_OP_HARDSIGMOID',
    13: 'GGML_UNARY_OP_EXP',
    14: 'GGML_UNARY_OP_GELU_ERF',
    15: 'GGML_UNARY_OP_COUNT',
}
GGML_UNARY_OP_ABS = 0
GGML_UNARY_OP_SGN = 1
GGML_UNARY_OP_NEG = 2
GGML_UNARY_OP_STEP = 3
GGML_UNARY_OP_TANH = 4
GGML_UNARY_OP_ELU = 5
GGML_UNARY_OP_RELU = 6
GGML_UNARY_OP_SIGMOID = 7
GGML_UNARY_OP_GELU = 8
GGML_UNARY_OP_GELU_QUICK = 9
GGML_UNARY_OP_SILU = 10
GGML_UNARY_OP_HARDSWISH = 11
GGML_UNARY_OP_HARDSIGMOID = 12
GGML_UNARY_OP_EXP = 13
GGML_UNARY_OP_GELU_ERF = 14
GGML_UNARY_OP_COUNT = 15
ggml_unary_op = ctypes.c_uint32 # enum

# values for enumeration 'ggml_object_type'
ggml_object_type__enumvalues = {
    0: 'GGML_OBJECT_TYPE_TENSOR',
    1: 'GGML_OBJECT_TYPE_GRAPH',
    2: 'GGML_OBJECT_TYPE_WORK_BUFFER',
}
GGML_OBJECT_TYPE_TENSOR = 0
GGML_OBJECT_TYPE_GRAPH = 1
GGML_OBJECT_TYPE_WORK_BUFFER = 2
ggml_object_type = ctypes.c_uint32 # enum

# values for enumeration 'ggml_log_level'
ggml_log_level__enumvalues = {
    0: 'GGML_LOG_LEVEL_NONE',
    1: 'GGML_LOG_LEVEL_DEBUG',
    2: 'GGML_LOG_LEVEL_INFO',
    3: 'GGML_LOG_LEVEL_WARN',
    4: 'GGML_LOG_LEVEL_ERROR',
    5: 'GGML_LOG_LEVEL_CONT',
}
GGML_LOG_LEVEL_NONE = 0
GGML_LOG_LEVEL_DEBUG = 1
GGML_LOG_LEVEL_INFO = 2
GGML_LOG_LEVEL_WARN = 3
GGML_LOG_LEVEL_ERROR = 4
GGML_LOG_LEVEL_CONT = 5
ggml_log_level = ctypes.c_uint32 # enum

# values for enumeration 'ggml_tensor_flag'
ggml_tensor_flag__enumvalues = {
    1: 'GGML_TENSOR_FLAG_INPUT',
    2: 'GGML_TENSOR_FLAG_OUTPUT',
    4: 'GGML_TENSOR_FLAG_PARAM',
    8: 'GGML_TENSOR_FLAG_LOSS',
}
GGML_TENSOR_FLAG_INPUT = 1
GGML_TENSOR_FLAG_OUTPUT = 2
GGML_TENSOR_FLAG_PARAM = 4
GGML_TENSOR_FLAG_LOSS = 8
ggml_tensor_flag = ctypes.c_uint32 # enum
class struct_ggml_init_params(Structure):
    if TYPE_CHECKING:
        mem_size: ctypes.c_uint64
        mem_buffer: ctypes.c_void_p
        no_alloc: ctypes.c_bool
struct_ggml_init_params._pack_ = 1 # source:False
struct_ggml_init_params._fields_ = [
    ('mem_size', ctypes.c_uint64),
    ('mem_buffer', ctypes.POINTER(None)),
    ('no_alloc', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 7),
]

class struct_ggml_tensor(Structure):
    if TYPE_CHECKING:
        type: ggml_type
        buffer: ctypes._Pointer[struct_ggml_backend_buffer]
        ne: ctypes.Array[ctypes.c_int64]
        nb: ctypes.Array[ctypes.c_uint64]
        op: ggml_op
        op_params: ctypes.Array[ctypes.c_int32]
        flags: ctypes.c_int32
        src: ctypes.Array[ctypes._Pointer[struct_ggml_tensor]]
        view_src: ctypes._Pointer[struct_ggml_tensor]
        view_offs: ctypes.c_uint64
        data: ctypes.c_void_p
        name: ctypes.Array[ctypes.c_char]
        extra: ctypes.c_void_p
        padding: ctypes.Array[ctypes.c_char]
class struct_ggml_backend_buffer(Structure):
    if TYPE_CHECKING:
        iface: struct_ggml_backend_buffer_i
        buft: ctypes._Pointer[struct_ggml_backend_buffer_type]
        context: ctypes.c_void_p
        size: ctypes.c_uint64
        usage: ggml_backend_buffer_usage
struct_ggml_tensor._pack_ = 1 # source:False
struct_ggml_tensor._fields_ = [
    ('type', ggml_type),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('buffer', ctypes.POINTER(struct_ggml_backend_buffer)),
    ('ne', ctypes.c_int64 * 4),
    ('nb', ctypes.c_uint64 * 4),
    ('op', ggml_op),
    ('op_params', ctypes.c_int32 * 16),
    ('flags', ctypes.c_int32),
    ('src', ctypes.POINTER(struct_ggml_tensor) * 10),
    ('view_src', ctypes.POINTER(struct_ggml_tensor)),
    ('view_offs', ctypes.c_uint64),
    ('data', ctypes.POINTER(None)),
    ('name', ctypes.c_char * 64),
    ('extra', ctypes.POINTER(None)),
    ('padding', ctypes.c_char * 8),
]

class struct_ggml_backend_buffer_type(Structure):
    if TYPE_CHECKING:
        iface: struct_ggml_backend_buffer_type_i
        device: ctypes._Pointer[struct_ggml_backend_device]
        context: ctypes.c_void_p
class struct_ggml_backend_buffer_i(Structure):
    if TYPE_CHECKING:
        free_buffer: Callable[[ctypes._Pointer[struct_ggml_backend_buffer]], None]
        get_base: Callable[[ctypes._Pointer[struct_ggml_backend_buffer]], ctypes.c_void_p]
        init_tensor: Callable[[ctypes._Pointer[struct_ggml_backend_buffer], ctypes._Pointer[struct_ggml_tensor]], ggml_status]
        memset_tensor: Callable[[ctypes._Pointer[struct_ggml_backend_buffer], ctypes._Pointer[struct_ggml_tensor], ctypes.c_ubyte, ctypes.c_uint64, ctypes.c_uint64], None]
        set_tensor: Callable[[ctypes._Pointer[struct_ggml_backend_buffer], ctypes._Pointer[struct_ggml_tensor], ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64], None]
        get_tensor: Callable[[ctypes._Pointer[struct_ggml_backend_buffer], ctypes._Pointer[struct_ggml_tensor], ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64], None]
        cpy_tensor: Callable[[ctypes._Pointer[struct_ggml_backend_buffer], ctypes._Pointer[struct_ggml_tensor], ctypes._Pointer[struct_ggml_tensor]], ctypes.c_bool]
        clear: Callable[[ctypes._Pointer[struct_ggml_backend_buffer], ctypes.c_ubyte], None]
        reset: Callable[[ctypes._Pointer[struct_ggml_backend_buffer]], None]
struct_ggml_backend_buffer_i._pack_ = 1 # source:False
struct_ggml_backend_buffer_i._fields_ = [
    ('free_buffer', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_backend_buffer))),
    ('get_base', ctypes.CFUNCTYPE(ctypes.POINTER(None), ctypes.POINTER(struct_ggml_backend_buffer))),
    ('init_tensor', ctypes.CFUNCTYPE(ggml_status, ctypes.POINTER(struct_ggml_backend_buffer), ctypes.POINTER(struct_ggml_tensor))),
    ('memset_tensor', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_backend_buffer), ctypes.POINTER(struct_ggml_tensor), ctypes.c_ubyte, ctypes.c_uint64, ctypes.c_uint64)),
    ('set_tensor', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_backend_buffer), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(None), ctypes.c_uint64, ctypes.c_uint64)),
    ('get_tensor', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_backend_buffer), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(None), ctypes.c_uint64, ctypes.c_uint64)),
    ('cpy_tensor', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_ggml_backend_buffer), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor))),
    ('clear', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_backend_buffer), ctypes.c_ubyte)),
    ('reset', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_backend_buffer))),
]


# values for enumeration 'ggml_backend_buffer_usage'
ggml_backend_buffer_usage__enumvalues = {
    0: 'GGML_BACKEND_BUFFER_USAGE_ANY',
    1: 'GGML_BACKEND_BUFFER_USAGE_WEIGHTS',
    2: 'GGML_BACKEND_BUFFER_USAGE_COMPUTE',
}
GGML_BACKEND_BUFFER_USAGE_ANY = 0
GGML_BACKEND_BUFFER_USAGE_WEIGHTS = 1
GGML_BACKEND_BUFFER_USAGE_COMPUTE = 2
ggml_backend_buffer_usage = ctypes.c_uint32 # enum
struct_ggml_backend_buffer._pack_ = 1 # source:False
struct_ggml_backend_buffer._fields_ = [
    ('iface', struct_ggml_backend_buffer_i),
    ('buft', ctypes.POINTER(struct_ggml_backend_buffer_type)),
    ('context', ctypes.POINTER(None)),
    ('size', ctypes.c_uint64),
    ('usage', ggml_backend_buffer_usage),
]

ggml_abort_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(None))
ggml_guid = ctypes.c_ubyte * 16
ggml_guid_t = ctypes.POINTER(ctypes.c_ubyte * 16)
@ctypes_function_for_shared_library('libggml-base.so')("ggml_guid_matches", [ggml_guid_t, ggml_guid_t], ctypes.c_bool)
def ggml_guid_matches(guid_a: ggml_guid_t, guid_b: ggml_guid_t) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_time_init", [], None)
def ggml_time_init() -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_time_ms", [], int64_t)
def ggml_time_ms() -> int64_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_time_us", [], int64_t)
def ggml_time_us() -> int64_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_cycles", [], int64_t)
def ggml_cycles() -> int64_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_cycles_per_ms", [], int64_t)
def ggml_cycles_per_ms() -> int64_t:
    ...

class struct__IO_FILE(Structure):
    if TYPE_CHECKING:
        _flags: ctypes.c_int32
        _IO_read_ptr: ctypes._Pointer[ctypes.c_char]
        _IO_read_end: ctypes._Pointer[ctypes.c_char]
        _IO_read_base: ctypes._Pointer[ctypes.c_char]
        _IO_write_base: ctypes._Pointer[ctypes.c_char]
        _IO_write_ptr: ctypes._Pointer[ctypes.c_char]
        _IO_write_end: ctypes._Pointer[ctypes.c_char]
        _IO_buf_base: ctypes._Pointer[ctypes.c_char]
        _IO_buf_end: ctypes._Pointer[ctypes.c_char]
        _IO_save_base: ctypes._Pointer[ctypes.c_char]
        _IO_backup_base: ctypes._Pointer[ctypes.c_char]
        _IO_save_end: ctypes._Pointer[ctypes.c_char]
        _markers: ctypes._Pointer[struct__IO_marker]
        _chain: ctypes._Pointer[struct__IO_FILE]
        _fileno: ctypes.c_int32
        _flags2: ctypes.c_int32
        _old_offset: ctypes.c_int64
        _cur_column: ctypes.c_uint16
        _vtable_offset: ctypes.c_byte
        _shortbuf: ctypes.Array[ctypes.c_char]
        _lock: ctypes.c_void_p
        _offset: ctypes.c_int64
        _codecvt: ctypes._Pointer[struct__IO_codecvt]
        _wide_data: ctypes._Pointer[struct__IO_wide_data]
        _freeres_list: ctypes._Pointer[struct__IO_FILE]
        _freeres_buf: ctypes.c_void_p
        __pad5: ctypes.c_uint64
        _mode: ctypes.c_int32
        _unused2: ctypes.Array[ctypes.c_char]
class struct__IO_marker(Structure):
    pass

class struct__IO_codecvt(Structure):
    pass

class struct__IO_wide_data(Structure):
    pass

struct__IO_FILE._pack_ = 1 # source:False
struct__IO_FILE._fields_ = [
    ('_flags', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('_IO_read_ptr', ctypes.POINTER(ctypes.c_char)),
    ('_IO_read_end', ctypes.POINTER(ctypes.c_char)),
    ('_IO_read_base', ctypes.POINTER(ctypes.c_char)),
    ('_IO_write_base', ctypes.POINTER(ctypes.c_char)),
    ('_IO_write_ptr', ctypes.POINTER(ctypes.c_char)),
    ('_IO_write_end', ctypes.POINTER(ctypes.c_char)),
    ('_IO_buf_base', ctypes.POINTER(ctypes.c_char)),
    ('_IO_buf_end', ctypes.POINTER(ctypes.c_char)),
    ('_IO_save_base', ctypes.POINTER(ctypes.c_char)),
    ('_IO_backup_base', ctypes.POINTER(ctypes.c_char)),
    ('_IO_save_end', ctypes.POINTER(ctypes.c_char)),
    ('_markers', ctypes.POINTER(struct__IO_marker)),
    ('_chain', ctypes.POINTER(struct__IO_FILE)),
    ('_fileno', ctypes.c_int32),
    ('_flags2', ctypes.c_int32),
    ('_old_offset', ctypes.c_int64),
    ('_cur_column', ctypes.c_uint16),
    ('_vtable_offset', ctypes.c_byte),
    ('_shortbuf', ctypes.c_char * 1),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('_lock', ctypes.POINTER(None)),
    ('_offset', ctypes.c_int64),
    ('_codecvt', ctypes.POINTER(struct__IO_codecvt)),
    ('_wide_data', ctypes.POINTER(struct__IO_wide_data)),
    ('_freeres_list', ctypes.POINTER(struct__IO_FILE)),
    ('_freeres_buf', ctypes.POINTER(None)),
    ('__pad5', ctypes.c_uint64),
    ('_mode', ctypes.c_int32),
    ('_unused2', ctypes.c_char * 20),
]

@ctypes_function_for_shared_library('libggml-base.so')("ggml_fopen", [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)], ctypes.POINTER(struct__IO_FILE))
def ggml_fopen(fname: ctypes._Pointer[ctypes.c_char], mode: ctypes._Pointer[ctypes.c_char]) -> ctypes._Pointer[struct__IO_FILE]:
    ...

class struct_ggml_object(Structure):
    if TYPE_CHECKING:
        offs: ctypes.c_uint64
        size: ctypes.c_uint64
        next: ctypes._Pointer[struct_ggml_object]
        type: ggml_object_type
        padding: ctypes.Array[ctypes.c_char]
@ctypes_function_for_shared_library('libggml-base.so')("ggml_print_object", [ctypes.POINTER(struct_ggml_object)], None)
def ggml_print_object(obj: ctypes._Pointer[struct_ggml_object]) -> None:
    ...

class struct_ggml_context(Structure):
    if TYPE_CHECKING:
        mem_size: ctypes.c_uint64
        mem_buffer: ctypes.c_void_p
        mem_buffer_owned: ctypes.c_bool
        no_alloc: ctypes.c_bool
        n_objects: ctypes.c_int32
        objects_begin: ctypes._Pointer[struct_ggml_object]
        objects_end: ctypes._Pointer[struct_ggml_object]
@ctypes_function_for_shared_library('libggml-base.so')("ggml_print_objects", [ctypes.POINTER(struct_ggml_context)], None)
def ggml_print_objects(ctx: ctypes._Pointer[struct_ggml_context]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_nelements", [ctypes.POINTER(struct_ggml_tensor)], int64_t)
def ggml_nelements(tensor: ctypes._Pointer[struct_ggml_tensor]) -> int64_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_nrows", [ctypes.POINTER(struct_ggml_tensor)], int64_t)
def ggml_nrows(tensor: ctypes._Pointer[struct_ggml_tensor]) -> int64_t:
    ...

size_t = ctypes.c_uint64
@ctypes_function_for_shared_library('libggml-base.so')("ggml_nbytes", [ctypes.POINTER(struct_ggml_tensor)], size_t)
def ggml_nbytes(tensor: ctypes._Pointer[struct_ggml_tensor]) -> size_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_nbytes_pad", [ctypes.POINTER(struct_ggml_tensor)], size_t)
def ggml_nbytes_pad(tensor: ctypes._Pointer[struct_ggml_tensor]) -> size_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_blck_size", [ggml_type], int64_t)
def ggml_blck_size(type: ggml_type) -> int64_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_type_size", [ggml_type], size_t)
def ggml_type_size(type: ggml_type) -> size_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_row_size", [ggml_type, int64_t], size_t)
def ggml_row_size(type: ggml_type, ne: int64_t) -> size_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_type_sizef", [ggml_type], ctypes.c_double)
def ggml_type_sizef(type: ggml_type) -> ctypes.c_double:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_type_name", [ggml_type], ctypes.POINTER(ctypes.c_char))
def ggml_type_name(type: ggml_type) -> ctypes._Pointer[ctypes.c_char]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_op_name", [ggml_op], ctypes.POINTER(ctypes.c_char))
def ggml_op_name(op: ggml_op) -> ctypes._Pointer[ctypes.c_char]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_op_symbol", [ggml_op], ctypes.POINTER(ctypes.c_char))
def ggml_op_symbol(op: ggml_op) -> ctypes._Pointer[ctypes.c_char]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_unary_op_name", [ggml_unary_op], ctypes.POINTER(ctypes.c_char))
def ggml_unary_op_name(op: ggml_unary_op) -> ctypes._Pointer[ctypes.c_char]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_op_desc", [ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(ctypes.c_char))
def ggml_op_desc(t: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[ctypes.c_char]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_element_size", [ctypes.POINTER(struct_ggml_tensor)], size_t)
def ggml_element_size(tensor: ctypes._Pointer[struct_ggml_tensor]) -> size_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_is_quantized", [ggml_type], ctypes.c_bool)
def ggml_is_quantized(type: ggml_type) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_ftype_to_ggml_type", [ggml_ftype], ggml_type)
def ggml_ftype_to_ggml_type(ftype: ggml_ftype) -> ggml_type:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_is_transposed", [ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_is_transposed(tensor: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_is_permuted", [ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_is_permuted(tensor: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_is_empty", [ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_is_empty(tensor: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_is_scalar", [ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_is_scalar(tensor: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_is_vector", [ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_is_vector(tensor: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_is_matrix", [ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_is_matrix(tensor: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_is_3d", [ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_is_3d(tensor: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_n_dims", [ctypes.POINTER(struct_ggml_tensor)], ctypes.c_int32)
def ggml_n_dims(tensor: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_is_contiguous", [ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_is_contiguous(tensor: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_is_contiguous_0", [ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_is_contiguous_0(tensor: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_is_contiguous_1", [ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_is_contiguous_1(tensor: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_is_contiguous_2", [ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_is_contiguous_2(tensor: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_is_contiguously_allocated", [ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_is_contiguously_allocated(tensor: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_is_contiguous_channels", [ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_is_contiguous_channels(tensor: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_are_same_shape", [ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_are_same_shape(t0: ctypes._Pointer[struct_ggml_tensor], t1: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_are_same_stride", [ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_are_same_stride(t0: ctypes._Pointer[struct_ggml_tensor], t1: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_can_repeat", [ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_can_repeat(t0: ctypes._Pointer[struct_ggml_tensor], t1: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_tensor_overhead", [], size_t)
def ggml_tensor_overhead() -> size_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_validate_row_data", [ggml_type, ctypes.POINTER(None), size_t], ctypes.c_bool)
def ggml_validate_row_data(type: ggml_type, data: ctypes.c_void_p, nbytes: size_t) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_init", [struct_ggml_init_params], ctypes.POINTER(struct_ggml_context))
def ggml_init(params: struct_ggml_init_params) -> ctypes._Pointer[struct_ggml_context]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_reset", [ctypes.POINTER(struct_ggml_context)], None)
def ggml_reset(ctx: ctypes._Pointer[struct_ggml_context]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_free", [ctypes.POINTER(struct_ggml_context)], None)
def ggml_free(ctx: ctypes._Pointer[struct_ggml_context]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_used_mem", [ctypes.POINTER(struct_ggml_context)], size_t)
def ggml_used_mem(ctx: ctypes._Pointer[struct_ggml_context]) -> size_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_get_no_alloc", [ctypes.POINTER(struct_ggml_context)], ctypes.c_bool)
def ggml_get_no_alloc(ctx: ctypes._Pointer[struct_ggml_context]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_set_no_alloc", [ctypes.POINTER(struct_ggml_context), ctypes.c_bool], None)
def ggml_set_no_alloc(ctx: ctypes._Pointer[struct_ggml_context], no_alloc: ctypes.c_bool) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_get_mem_buffer", [ctypes.POINTER(struct_ggml_context)], ctypes.POINTER(None))
def ggml_get_mem_buffer(ctx: ctypes._Pointer[struct_ggml_context]) -> ctypes.c_void_p:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_get_mem_size", [ctypes.POINTER(struct_ggml_context)], size_t)
def ggml_get_mem_size(ctx: ctypes._Pointer[struct_ggml_context]) -> size_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_get_max_tensor_size", [ctypes.POINTER(struct_ggml_context)], size_t)
def ggml_get_max_tensor_size(ctx: ctypes._Pointer[struct_ggml_context]) -> size_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_new_tensor", [ctypes.POINTER(struct_ggml_context), ggml_type, ctypes.c_int32, ctypes.POINTER(ctypes.c_int64)], ctypes.POINTER(struct_ggml_tensor))
def ggml_new_tensor(ctx: ctypes._Pointer[struct_ggml_context], type: ggml_type, n_dims: ctypes.c_int32, ne: ctypes._Pointer[ctypes.c_int64]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_new_tensor_1d", [ctypes.POINTER(struct_ggml_context), ggml_type, int64_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_new_tensor_1d(ctx: ctypes._Pointer[struct_ggml_context], type: ggml_type, ne0: int64_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_new_tensor_2d", [ctypes.POINTER(struct_ggml_context), ggml_type, int64_t, int64_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_new_tensor_2d(ctx: ctypes._Pointer[struct_ggml_context], type: ggml_type, ne0: int64_t, ne1: int64_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_new_tensor_3d", [ctypes.POINTER(struct_ggml_context), ggml_type, int64_t, int64_t, int64_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_new_tensor_3d(ctx: ctypes._Pointer[struct_ggml_context], type: ggml_type, ne0: int64_t, ne1: int64_t, ne2: int64_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_new_tensor_4d", [ctypes.POINTER(struct_ggml_context), ggml_type, int64_t, int64_t, int64_t, int64_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_new_tensor_4d(ctx: ctypes._Pointer[struct_ggml_context], type: ggml_type, ne0: int64_t, ne1: int64_t, ne2: int64_t, ne3: int64_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_new_buffer", [ctypes.POINTER(struct_ggml_context), size_t], ctypes.POINTER(None))
def ggml_new_buffer(ctx: ctypes._Pointer[struct_ggml_context], nbytes: size_t) -> ctypes.c_void_p:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_dup_tensor", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_dup_tensor(ctx: ctypes._Pointer[struct_ggml_context], src: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_view_tensor", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_view_tensor(ctx: ctypes._Pointer[struct_ggml_context], src: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_get_first_tensor", [ctypes.POINTER(struct_ggml_context)], ctypes.POINTER(struct_ggml_tensor))
def ggml_get_first_tensor(ctx: ctypes._Pointer[struct_ggml_context]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_get_next_tensor", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_get_next_tensor(ctx: ctypes._Pointer[struct_ggml_context], tensor: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_get_tensor", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(ctypes.c_char)], ctypes.POINTER(struct_ggml_tensor))
def ggml_get_tensor(ctx: ctypes._Pointer[struct_ggml_context], name: ctypes._Pointer[ctypes.c_char]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_unravel_index", [ctypes.POINTER(struct_ggml_tensor), int64_t, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int64)], None)
def ggml_unravel_index(tensor: ctypes._Pointer[struct_ggml_tensor], i: int64_t, i0: ctypes._Pointer[ctypes.c_int64], i1: ctypes._Pointer[ctypes.c_int64], i2: ctypes._Pointer[ctypes.c_int64], i3: ctypes._Pointer[ctypes.c_int64]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_get_unary_op", [ctypes.POINTER(struct_ggml_tensor)], ggml_unary_op)
def ggml_get_unary_op(tensor: ctypes._Pointer[struct_ggml_tensor]) -> ggml_unary_op:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_get_data", [ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(None))
def ggml_get_data(tensor: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_void_p:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_get_data_f32", [ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(ctypes.c_float))
def ggml_get_data_f32(tensor: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[ctypes.c_float]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_get_name", [ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(ctypes.c_char))
def ggml_get_name(tensor: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[ctypes.c_char]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_set_name", [ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(ctypes.c_char)], ctypes.POINTER(struct_ggml_tensor))
def ggml_set_name(tensor: ctypes._Pointer[struct_ggml_tensor], name: ctypes._Pointer[ctypes.c_char]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_format_name", [ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(ctypes.c_char)], ctypes.POINTER(struct_ggml_tensor))
def ggml_format_name(tensor: ctypes._Pointer[struct_ggml_tensor], fmt: ctypes._Pointer[ctypes.c_char]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_set_input", [ctypes.POINTER(struct_ggml_tensor)], None)
def ggml_set_input(tensor: ctypes._Pointer[struct_ggml_tensor]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_set_output", [ctypes.POINTER(struct_ggml_tensor)], None)
def ggml_set_output(tensor: ctypes._Pointer[struct_ggml_tensor]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_set_param", [ctypes.POINTER(struct_ggml_tensor)], None)
def ggml_set_param(tensor: ctypes._Pointer[struct_ggml_tensor]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_set_loss", [ctypes.POINTER(struct_ggml_tensor)], None)
def ggml_set_loss(tensor: ctypes._Pointer[struct_ggml_tensor]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_dup", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_dup(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_dup_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_dup_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_add", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_add(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_add_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_add_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_add_cast", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ggml_type], ctypes.POINTER(struct_ggml_tensor))
def ggml_add_cast(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], type: ggml_type) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_add1", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_add1(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_add1_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_add1_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_acc", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), size_t, size_t, size_t, size_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_acc(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], nb1: size_t, nb2: size_t, nb3: size_t, offset: size_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_acc_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), size_t, size_t, size_t, size_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_acc_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], nb1: size_t, nb2: size_t, nb3: size_t, offset: size_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_sub", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_sub(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_sub_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_sub_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_mul", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_mul(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_mul_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_mul_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_div", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_div(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_div_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_div_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_sqr", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_sqr(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_sqr_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_sqr_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_sqrt", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_sqrt(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_sqrt_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_sqrt_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_log", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_log(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_log_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_log_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_sin", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_sin(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_sin_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_sin_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_cos", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_cos(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_cos_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_cos_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_sum", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_sum(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_sum_rows", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_sum_rows(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_mean", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_mean(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_argmax", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_argmax(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_count_equal", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_count_equal(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_repeat", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_repeat(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_repeat_4d", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), int64_t, int64_t, int64_t, int64_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_repeat_4d(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], ne0: int64_t, ne1: int64_t, ne2: int64_t, ne3: int64_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_repeat_back", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_repeat_back(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_concat", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_concat(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], dim: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_abs", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_abs(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_abs_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_abs_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_sgn", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_sgn(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_sgn_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_sgn_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_neg", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_neg(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_neg_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_neg_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_step", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_step(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_step_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_step_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_tanh", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_tanh(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_tanh_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_tanh_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_elu", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_elu(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_elu_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_elu_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_relu", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_relu(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_leaky_relu", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_float, ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_leaky_relu(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], negative_slope: ctypes.c_float, inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_relu_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_relu_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_sigmoid", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_sigmoid(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_sigmoid_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_sigmoid_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_gelu", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_gelu(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_gelu_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_gelu_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_gelu_erf", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_gelu_erf(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_gelu_erf_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_gelu_erf_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_gelu_quick", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_gelu_quick(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_gelu_quick_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_gelu_quick_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_silu", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_silu(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_silu_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_silu_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_silu_back", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_silu_back(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_hardswish", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_hardswish(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_hardsigmoid", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_hardsigmoid(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_exp", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_exp(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_exp_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_exp_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_norm", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_norm(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], eps: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_norm_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_norm_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], eps: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_rms_norm", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_rms_norm(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], eps: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_rms_norm_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_rms_norm_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], eps: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_group_norm", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_group_norm(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], n_groups: ctypes.c_int32, eps: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_group_norm_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_group_norm_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], n_groups: ctypes.c_int32, eps: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_l2_norm", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_l2_norm(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], eps: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_l2_norm_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_l2_norm_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], eps: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_rms_norm_back", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_rms_norm_back(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], eps: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_mul_mat", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_mul_mat(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_mul_mat_set_prec", [ctypes.POINTER(struct_ggml_tensor), ggml_prec], None)
def ggml_mul_mat_set_prec(a: ctypes._Pointer[struct_ggml_tensor], prec: ggml_prec) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_mul_mat_id", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_mul_mat_id(ctx: ctypes._Pointer[struct_ggml_context], _as: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], ids: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_out_prod", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_out_prod(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_scale", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_scale(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], s: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_scale_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_scale_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], s: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_set", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), size_t, size_t, size_t, size_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_set(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], nb1: size_t, nb2: size_t, nb3: size_t, offset: size_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_set_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), size_t, size_t, size_t, size_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_set_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], nb1: size_t, nb2: size_t, nb3: size_t, offset: size_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_set_1d", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), size_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_set_1d(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], offset: size_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_set_1d_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), size_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_set_1d_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], offset: size_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_set_2d", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), size_t, size_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_set_2d(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], nb1: size_t, offset: size_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_set_2d_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), size_t, size_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_set_2d_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], nb1: size_t, offset: size_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_cpy", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_cpy(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_cast", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ggml_type], ctypes.POINTER(struct_ggml_tensor))
def ggml_cast(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], type: ggml_type) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_cont", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_cont(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_cont_1d", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), int64_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_cont_1d(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], ne0: int64_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_cont_2d", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), int64_t, int64_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_cont_2d(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], ne0: int64_t, ne1: int64_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_cont_3d", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), int64_t, int64_t, int64_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_cont_3d(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], ne0: int64_t, ne1: int64_t, ne2: int64_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_cont_4d", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), int64_t, int64_t, int64_t, int64_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_cont_4d(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], ne0: int64_t, ne1: int64_t, ne2: int64_t, ne3: int64_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_reshape", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_reshape(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_reshape_1d", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), int64_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_reshape_1d(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], ne0: int64_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_reshape_2d", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), int64_t, int64_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_reshape_2d(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], ne0: int64_t, ne1: int64_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_reshape_3d", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), int64_t, int64_t, int64_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_reshape_3d(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], ne0: int64_t, ne1: int64_t, ne2: int64_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_reshape_4d", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), int64_t, int64_t, int64_t, int64_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_reshape_4d(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], ne0: int64_t, ne1: int64_t, ne2: int64_t, ne3: int64_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_view_1d", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), int64_t, size_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_view_1d(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], ne0: int64_t, offset: size_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_view_2d", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), int64_t, int64_t, size_t, size_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_view_2d(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], ne0: int64_t, ne1: int64_t, nb1: size_t, offset: size_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_view_3d", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), int64_t, int64_t, int64_t, size_t, size_t, size_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_view_3d(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], ne0: int64_t, ne1: int64_t, ne2: int64_t, nb1: size_t, nb2: size_t, offset: size_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_view_4d", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), int64_t, int64_t, int64_t, int64_t, size_t, size_t, size_t, size_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_view_4d(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], ne0: int64_t, ne1: int64_t, ne2: int64_t, ne3: int64_t, nb1: size_t, nb2: size_t, nb3: size_t, offset: size_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_permute", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_permute(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], axis0: ctypes.c_int32, axis1: ctypes.c_int32, axis2: ctypes.c_int32, axis3: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_transpose", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_transpose(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_get_rows", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_get_rows(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_get_rows_back", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_get_rows_back(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], c: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_diag", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_diag(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_diag_mask_inf", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_diag_mask_inf(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], n_past: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_diag_mask_inf_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_diag_mask_inf_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], n_past: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_diag_mask_zero", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_diag_mask_zero(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], n_past: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_diag_mask_zero_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_diag_mask_zero_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], n_past: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_soft_max", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_soft_max(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_soft_max_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_soft_max_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_soft_max_ext", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_float, ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_soft_max_ext(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], mask: ctypes._Pointer[struct_ggml_tensor], scale: ctypes.c_float, max_bias: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_soft_max_ext_back", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_float, ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_soft_max_ext_back(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], scale: ctypes.c_float, max_bias: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_soft_max_ext_back_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_float, ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_soft_max_ext_back_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], scale: ctypes.c_float, max_bias: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_rope", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_rope(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], n_dims: ctypes.c_int32, mode: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_rope_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_rope_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], n_dims: ctypes.c_int32, mode: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_rope_ext", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_rope_ext(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], c: ctypes._Pointer[struct_ggml_tensor], n_dims: ctypes.c_int32, mode: ctypes.c_int32, n_ctx_orig: ctypes.c_int32, freq_base: ctypes.c_float, freq_scale: ctypes.c_float, ext_factor: ctypes.c_float, attn_factor: ctypes.c_float, beta_fast: ctypes.c_float, beta_slow: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_rope_multi", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32 * 4, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_rope_multi(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], c: ctypes._Pointer[struct_ggml_tensor], n_dims: ctypes.c_int32, sections: ctypes.Array[ctypes.c_int32], mode: ctypes.c_int32, n_ctx_orig: ctypes.c_int32, freq_base: ctypes.c_float, freq_scale: ctypes.c_float, ext_factor: ctypes.c_float, attn_factor: ctypes.c_float, beta_fast: ctypes.c_float, beta_slow: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_rope_ext_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_rope_ext_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], c: ctypes._Pointer[struct_ggml_tensor], n_dims: ctypes.c_int32, mode: ctypes.c_int32, n_ctx_orig: ctypes.c_int32, freq_base: ctypes.c_float, freq_scale: ctypes.c_float, ext_factor: ctypes.c_float, attn_factor: ctypes.c_float, beta_fast: ctypes.c_float, beta_slow: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_rope_custom", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_rope_custom(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], n_dims: ctypes.c_int32, mode: ctypes.c_int32, n_ctx_orig: ctypes.c_int32, freq_base: ctypes.c_float, freq_scale: ctypes.c_float, ext_factor: ctypes.c_float, attn_factor: ctypes.c_float, beta_fast: ctypes.c_float, beta_slow: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_rope_custom_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_rope_custom_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], n_dims: ctypes.c_int32, mode: ctypes.c_int32, n_ctx_orig: ctypes.c_int32, freq_base: ctypes.c_float, freq_scale: ctypes.c_float, ext_factor: ctypes.c_float, attn_factor: ctypes.c_float, beta_fast: ctypes.c_float, beta_slow: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_rope_yarn_corr_dims", [ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float * 2], None)
def ggml_rope_yarn_corr_dims(n_dims: ctypes.c_int32, n_ctx_orig: ctypes.c_int32, freq_base: ctypes.c_float, beta_fast: ctypes.c_float, beta_slow: ctypes.c_float, dims: ctypes.Array[ctypes.c_float]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_rope_ext_back", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_rope_ext_back(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], c: ctypes._Pointer[struct_ggml_tensor], n_dims: ctypes.c_int32, mode: ctypes.c_int32, n_ctx_orig: ctypes.c_int32, freq_base: ctypes.c_float, freq_scale: ctypes.c_float, ext_factor: ctypes.c_float, attn_factor: ctypes.c_float, beta_fast: ctypes.c_float, beta_slow: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_rope_multi_back", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32 * 4, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_rope_multi_back(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], c: ctypes._Pointer[struct_ggml_tensor], n_dims: ctypes.c_int32, sections: ctypes.Array[ctypes.c_int32], mode: ctypes.c_int32, n_ctx_orig: ctypes.c_int32, freq_base: ctypes.c_float, freq_scale: ctypes.c_float, ext_factor: ctypes.c_float, attn_factor: ctypes.c_float, beta_fast: ctypes.c_float, beta_slow: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_clamp", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_float, ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_clamp(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], min: ctypes.c_float, max: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_im2col", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_bool, ggml_type], ctypes.POINTER(struct_ggml_tensor))
def ggml_im2col(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], s0: ctypes.c_int32, s1: ctypes.c_int32, p0: ctypes.c_int32, p1: ctypes.c_int32, d0: ctypes.c_int32, d1: ctypes.c_int32, is_2D: ctypes.c_bool, dst_type: ggml_type) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_im2col_back", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(ctypes.c_int64), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_im2col_back(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], ne: ctypes._Pointer[ctypes.c_int64], s0: ctypes.c_int32, s1: ctypes.c_int32, p0: ctypes.c_int32, p1: ctypes.c_int32, d0: ctypes.c_int32, d1: ctypes.c_int32, is_2D: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_conv_1d", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_conv_1d(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], s0: ctypes.c_int32, p0: ctypes.c_int32, d0: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_conv_1d_ph", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_conv_1d_ph(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], s: ctypes.c_int32, d: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_conv_1d_dw", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_conv_1d_dw(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], s0: ctypes.c_int32, p0: ctypes.c_int32, d0: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_conv_1d_dw_ph", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_conv_1d_dw_ph(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], s0: ctypes.c_int32, d0: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_conv_transpose_1d", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_conv_transpose_1d(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], s0: ctypes.c_int32, p0: ctypes.c_int32, d0: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_conv_2d", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_conv_2d(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], s0: ctypes.c_int32, s1: ctypes.c_int32, p0: ctypes.c_int32, p1: ctypes.c_int32, d0: ctypes.c_int32, d1: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_conv_2d_sk_p0", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_conv_2d_sk_p0(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_conv_2d_s1_ph", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_conv_2d_s1_ph(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_conv_2d_dw", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_conv_2d_dw(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], s0: ctypes.c_int32, s1: ctypes.c_int32, p0: ctypes.c_int32, p1: ctypes.c_int32, d0: ctypes.c_int32, d1: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_conv_2d_dw_direct", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_conv_2d_dw_direct(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], stride0: ctypes.c_int32, stride1: ctypes.c_int32, pad0: ctypes.c_int32, pad1: ctypes.c_int32, dilation0: ctypes.c_int32, dilation1: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_conv_transpose_2d_p0", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_conv_transpose_2d_p0(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], stride: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...


# values for enumeration 'ggml_op_pool'
ggml_op_pool__enumvalues = {
    0: 'GGML_OP_POOL_MAX',
    1: 'GGML_OP_POOL_AVG',
    2: 'GGML_OP_POOL_COUNT',
}
GGML_OP_POOL_MAX = 0
GGML_OP_POOL_AVG = 1
GGML_OP_POOL_COUNT = 2
ggml_op_pool = ctypes.c_uint32 # enum
@ctypes_function_for_shared_library('libggml-base.so')("ggml_pool_1d", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ggml_op_pool, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_pool_1d(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], op: ggml_op_pool, k0: ctypes.c_int32, s0: ctypes.c_int32, p0: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_pool_2d", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ggml_op_pool, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_pool_2d(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], op: ggml_op_pool, k0: ctypes.c_int32, k1: ctypes.c_int32, s0: ctypes.c_int32, s1: ctypes.c_int32, p0: ctypes.c_float, p1: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_pool_2d_back", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ggml_op_pool, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_pool_2d_back(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], af: ctypes._Pointer[struct_ggml_tensor], op: ggml_op_pool, k0: ctypes.c_int32, k1: ctypes.c_int32, s0: ctypes.c_int32, s1: ctypes.c_int32, p0: ctypes.c_float, p1: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...


# values for enumeration 'ggml_scale_mode'
ggml_scale_mode__enumvalues = {
    0: 'GGML_SCALE_MODE_NEAREST',
    1: 'GGML_SCALE_MODE_BILINEAR',
}
GGML_SCALE_MODE_NEAREST = 0
GGML_SCALE_MODE_BILINEAR = 1
ggml_scale_mode = ctypes.c_uint32 # enum
@ctypes_function_for_shared_library('libggml-base.so')("ggml_upscale", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ggml_scale_mode], ctypes.POINTER(struct_ggml_tensor))
def ggml_upscale(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], scale_factor: ctypes.c_int32, mode: ggml_scale_mode) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_upscale_ext", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ggml_scale_mode], ctypes.POINTER(struct_ggml_tensor))
def ggml_upscale_ext(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], ne0: ctypes.c_int32, ne1: ctypes.c_int32, ne2: ctypes.c_int32, ne3: ctypes.c_int32, mode: ggml_scale_mode) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_pad", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_pad(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], p0: ctypes.c_int32, p1: ctypes.c_int32, p2: ctypes.c_int32, p3: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_pad_reflect_1d", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_pad_reflect_1d(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], p0: ctypes.c_int32, p1: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_timestep_embedding", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_timestep_embedding(ctx: ctypes._Pointer[struct_ggml_context], timesteps: ctypes._Pointer[struct_ggml_tensor], dim: ctypes.c_int32, max_period: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...


# values for enumeration 'ggml_sort_order'
ggml_sort_order__enumvalues = {
    0: 'GGML_SORT_ORDER_ASC',
    1: 'GGML_SORT_ORDER_DESC',
}
GGML_SORT_ORDER_ASC = 0
GGML_SORT_ORDER_DESC = 1
ggml_sort_order = ctypes.c_uint32 # enum
@ctypes_function_for_shared_library('libggml-base.so')("ggml_argsort", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ggml_sort_order], ctypes.POINTER(struct_ggml_tensor))
def ggml_argsort(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], order: ggml_sort_order) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_arange", [ctypes.POINTER(struct_ggml_context), ctypes.c_float, ctypes.c_float, ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_arange(ctx: ctypes._Pointer[struct_ggml_context], start: ctypes.c_float, stop: ctypes.c_float, step: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_top_k", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_top_k(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], k: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_flash_attn_ext", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_float, ctypes.c_float, ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_flash_attn_ext(ctx: ctypes._Pointer[struct_ggml_context], q: ctypes._Pointer[struct_ggml_tensor], k: ctypes._Pointer[struct_ggml_tensor], v: ctypes._Pointer[struct_ggml_tensor], mask: ctypes._Pointer[struct_ggml_tensor], scale: ctypes.c_float, max_bias: ctypes.c_float, logit_softcap: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_flash_attn_ext_set_prec", [ctypes.POINTER(struct_ggml_tensor), ggml_prec], None)
def ggml_flash_attn_ext_set_prec(a: ctypes._Pointer[struct_ggml_tensor], prec: ggml_prec) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_flash_attn_ext_get_prec", [ctypes.POINTER(struct_ggml_tensor)], ggml_prec)
def ggml_flash_attn_ext_get_prec(a: ctypes._Pointer[struct_ggml_tensor]) -> ggml_prec:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_flash_attn_back", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_flash_attn_back(ctx: ctypes._Pointer[struct_ggml_context], q: ctypes._Pointer[struct_ggml_tensor], k: ctypes._Pointer[struct_ggml_tensor], v: ctypes._Pointer[struct_ggml_tensor], d: ctypes._Pointer[struct_ggml_tensor], masked: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_ssm_conv", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_ssm_conv(ctx: ctypes._Pointer[struct_ggml_context], sx: ctypes._Pointer[struct_ggml_tensor], c: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_ssm_scan", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_ssm_scan(ctx: ctypes._Pointer[struct_ggml_context], s: ctypes._Pointer[struct_ggml_tensor], x: ctypes._Pointer[struct_ggml_tensor], dt: ctypes._Pointer[struct_ggml_tensor], A: ctypes._Pointer[struct_ggml_tensor], B: ctypes._Pointer[struct_ggml_tensor], C: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_win_part", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_win_part(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], w: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_win_unpart", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_win_unpart(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], w0: ctypes.c_int32, h0: ctypes.c_int32, w: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_unary", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ggml_unary_op], ctypes.POINTER(struct_ggml_tensor))
def ggml_unary(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], op: ggml_unary_op) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_unary_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ggml_unary_op], ctypes.POINTER(struct_ggml_tensor))
def ggml_unary_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], op: ggml_unary_op) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_get_rel_pos", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_get_rel_pos(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], qh: ctypes.c_int32, kh: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_add_rel_pos", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_add_rel_pos(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], pw: ctypes._Pointer[struct_ggml_tensor], ph: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_add_rel_pos_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_add_rel_pos_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], pw: ctypes._Pointer[struct_ggml_tensor], ph: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_rwkv_wkv6", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_rwkv_wkv6(ctx: ctypes._Pointer[struct_ggml_context], k: ctypes._Pointer[struct_ggml_tensor], v: ctypes._Pointer[struct_ggml_tensor], r: ctypes._Pointer[struct_ggml_tensor], tf: ctypes._Pointer[struct_ggml_tensor], td: ctypes._Pointer[struct_ggml_tensor], state: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_gated_linear_attn", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_gated_linear_attn(ctx: ctypes._Pointer[struct_ggml_context], k: ctypes._Pointer[struct_ggml_tensor], v: ctypes._Pointer[struct_ggml_tensor], q: ctypes._Pointer[struct_ggml_tensor], g: ctypes._Pointer[struct_ggml_tensor], state: ctypes._Pointer[struct_ggml_tensor], scale: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_rwkv_wkv7", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_rwkv_wkv7(ctx: ctypes._Pointer[struct_ggml_context], r: ctypes._Pointer[struct_ggml_tensor], w: ctypes._Pointer[struct_ggml_tensor], k: ctypes._Pointer[struct_ggml_tensor], v: ctypes._Pointer[struct_ggml_tensor], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], state: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

ggml_custom1_op_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(None))
ggml_custom2_op_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(None))
ggml_custom3_op_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(None))
@ctypes_function_for_shared_library('libggml-base.so')("ggml_map_custom1", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ggml_custom1_op_t, ctypes.c_int32, ctypes.POINTER(None)], ctypes.POINTER(struct_ggml_tensor))
def ggml_map_custom1(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], fun: ggml_custom1_op_t, n_tasks: ctypes.c_int32, userdata: ctypes.c_void_p) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_map_custom1_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ggml_custom1_op_t, ctypes.c_int32, ctypes.POINTER(None)], ctypes.POINTER(struct_ggml_tensor))
def ggml_map_custom1_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], fun: ggml_custom1_op_t, n_tasks: ctypes.c_int32, userdata: ctypes.c_void_p) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_map_custom2", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ggml_custom2_op_t, ctypes.c_int32, ctypes.POINTER(None)], ctypes.POINTER(struct_ggml_tensor))
def ggml_map_custom2(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], fun: ggml_custom2_op_t, n_tasks: ctypes.c_int32, userdata: ctypes.c_void_p) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_map_custom2_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ggml_custom2_op_t, ctypes.c_int32, ctypes.POINTER(None)], ctypes.POINTER(struct_ggml_tensor))
def ggml_map_custom2_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], fun: ggml_custom2_op_t, n_tasks: ctypes.c_int32, userdata: ctypes.c_void_p) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_map_custom3", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ggml_custom3_op_t, ctypes.c_int32, ctypes.POINTER(None)], ctypes.POINTER(struct_ggml_tensor))
def ggml_map_custom3(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], c: ctypes._Pointer[struct_ggml_tensor], fun: ggml_custom3_op_t, n_tasks: ctypes.c_int32, userdata: ctypes.c_void_p) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_map_custom3_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ggml_custom3_op_t, ctypes.c_int32, ctypes.POINTER(None)], ctypes.POINTER(struct_ggml_tensor))
def ggml_map_custom3_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], c: ctypes._Pointer[struct_ggml_tensor], fun: ggml_custom3_op_t, n_tasks: ctypes.c_int32, userdata: ctypes.c_void_p) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

ggml_custom_op_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.POINTER(None))
@ctypes_function_for_shared_library('libggml-base.so')("ggml_custom_4d", [ctypes.POINTER(struct_ggml_context), ggml_type, int64_t, int64_t, int64_t, int64_t, ctypes.POINTER(ctypes.POINTER(struct_ggml_tensor)), ctypes.c_int32, ggml_custom_op_t, ctypes.c_int32, ctypes.POINTER(None)], ctypes.POINTER(struct_ggml_tensor))
def ggml_custom_4d(ctx: ctypes._Pointer[struct_ggml_context], type: ggml_type, ne0: int64_t, ne1: int64_t, ne2: int64_t, ne3: int64_t, args: ctypes._Pointer[ctypes._Pointer[struct_ggml_tensor]], n_args: ctypes.c_int32, fun: ggml_custom_op_t, n_tasks: ctypes.c_int32, userdata: ctypes.c_void_p) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_custom_inplace", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(ctypes.POINTER(struct_ggml_tensor)), ctypes.c_int32, ggml_custom_op_t, ctypes.c_int32, ctypes.POINTER(None)], ctypes.POINTER(struct_ggml_tensor))
def ggml_custom_inplace(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], args: ctypes._Pointer[ctypes._Pointer[struct_ggml_tensor]], n_args: ctypes.c_int32, fun: ggml_custom_op_t, n_tasks: ctypes.c_int32, userdata: ctypes.c_void_p) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_cross_entropy_loss", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_cross_entropy_loss(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_cross_entropy_loss_back", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_cross_entropy_loss_back(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], c: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_opt_step_adamw", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_opt_step_adamw(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], grad: ctypes._Pointer[struct_ggml_tensor], m: ctypes._Pointer[struct_ggml_tensor], v: ctypes._Pointer[struct_ggml_tensor], adamw_params: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

class struct_ggml_cgraph(Structure):
    if TYPE_CHECKING:
        size: ctypes.c_int32
        n_nodes: ctypes.c_int32
        n_leafs: ctypes.c_int32
        nodes: ctypes._Pointer[ctypes._Pointer[struct_ggml_tensor]]
        grads: ctypes._Pointer[ctypes._Pointer[struct_ggml_tensor]]
        grad_accs: ctypes._Pointer[ctypes._Pointer[struct_ggml_tensor]]
        leafs: ctypes._Pointer[ctypes._Pointer[struct_ggml_tensor]]
        visited_hash_set: struct_ggml_hash_set
        order: ggml_cgraph_eval_order
@ctypes_function_for_shared_library('libggml-base.so')("ggml_build_forward_expand", [ctypes.POINTER(struct_ggml_cgraph), ctypes.POINTER(struct_ggml_tensor)], None)
def ggml_build_forward_expand(cgraph: ctypes._Pointer[struct_ggml_cgraph], tensor: ctypes._Pointer[struct_ggml_tensor]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_build_backward_expand", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_cgraph), ctypes.POINTER(ctypes.POINTER(struct_ggml_tensor))], None)
def ggml_build_backward_expand(ctx: ctypes._Pointer[struct_ggml_context], cgraph: ctypes._Pointer[struct_ggml_cgraph], grad_accs: ctypes._Pointer[ctypes._Pointer[struct_ggml_tensor]]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_new_graph", [ctypes.POINTER(struct_ggml_context)], ctypes.POINTER(struct_ggml_cgraph))
def ggml_new_graph(ctx: ctypes._Pointer[struct_ggml_context]) -> ctypes._Pointer[struct_ggml_cgraph]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_new_graph_custom", [ctypes.POINTER(struct_ggml_context), size_t, ctypes.c_bool], ctypes.POINTER(struct_ggml_cgraph))
def ggml_new_graph_custom(ctx: ctypes._Pointer[struct_ggml_context], size: size_t, grads: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_cgraph]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_graph_dup", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_cgraph), ctypes.c_bool], ctypes.POINTER(struct_ggml_cgraph))
def ggml_graph_dup(ctx: ctypes._Pointer[struct_ggml_context], cgraph: ctypes._Pointer[struct_ggml_cgraph], force_grads: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_cgraph]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_graph_cpy", [ctypes.POINTER(struct_ggml_cgraph), ctypes.POINTER(struct_ggml_cgraph)], None)
def ggml_graph_cpy(src: ctypes._Pointer[struct_ggml_cgraph], dst: ctypes._Pointer[struct_ggml_cgraph]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_graph_reset", [ctypes.POINTER(struct_ggml_cgraph)], None)
def ggml_graph_reset(cgraph: ctypes._Pointer[struct_ggml_cgraph]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_graph_clear", [ctypes.POINTER(struct_ggml_cgraph)], None)
def ggml_graph_clear(cgraph: ctypes._Pointer[struct_ggml_cgraph]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_graph_size", [ctypes.POINTER(struct_ggml_cgraph)], ctypes.c_int32)
def ggml_graph_size(cgraph: ctypes._Pointer[struct_ggml_cgraph]) -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_graph_node", [ctypes.POINTER(struct_ggml_cgraph), ctypes.c_int32], ctypes.POINTER(struct_ggml_tensor))
def ggml_graph_node(cgraph: ctypes._Pointer[struct_ggml_cgraph], i: ctypes.c_int32) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_graph_nodes", [ctypes.POINTER(struct_ggml_cgraph)], ctypes.POINTER(ctypes.POINTER(struct_ggml_tensor)))
def ggml_graph_nodes(cgraph: ctypes._Pointer[struct_ggml_cgraph]) -> ctypes._Pointer[ctypes._Pointer[struct_ggml_tensor]]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_graph_n_nodes", [ctypes.POINTER(struct_ggml_cgraph)], ctypes.c_int32)
def ggml_graph_n_nodes(cgraph: ctypes._Pointer[struct_ggml_cgraph]) -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_graph_add_node", [ctypes.POINTER(struct_ggml_cgraph), ctypes.POINTER(struct_ggml_tensor)], None)
def ggml_graph_add_node(cgraph: ctypes._Pointer[struct_ggml_cgraph], tensor: ctypes._Pointer[struct_ggml_tensor]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_graph_overhead", [], size_t)
def ggml_graph_overhead() -> size_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_graph_overhead_custom", [size_t, ctypes.c_bool], size_t)
def ggml_graph_overhead_custom(size: size_t, grads: ctypes.c_bool) -> size_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_graph_get_tensor", [ctypes.POINTER(struct_ggml_cgraph), ctypes.POINTER(ctypes.c_char)], ctypes.POINTER(struct_ggml_tensor))
def ggml_graph_get_tensor(cgraph: ctypes._Pointer[struct_ggml_cgraph], name: ctypes._Pointer[ctypes.c_char]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_graph_get_grad", [ctypes.POINTER(struct_ggml_cgraph), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_graph_get_grad(cgraph: ctypes._Pointer[struct_ggml_cgraph], node: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_graph_get_grad_acc", [ctypes.POINTER(struct_ggml_cgraph), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_graph_get_grad_acc(cgraph: ctypes._Pointer[struct_ggml_cgraph], node: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_graph_export", [ctypes.POINTER(struct_ggml_cgraph), ctypes.POINTER(ctypes.c_char)], None)
def ggml_graph_export(cgraph: ctypes._Pointer[struct_ggml_cgraph], fname: ctypes._Pointer[ctypes.c_char]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_graph_import", [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(struct_ggml_context)), ctypes.POINTER(ctypes.POINTER(struct_ggml_context))], ctypes.POINTER(struct_ggml_cgraph))
def ggml_graph_import(fname: ctypes._Pointer[ctypes.c_char], ctx_data: ctypes._Pointer[ctypes._Pointer[struct_ggml_context]], ctx_eval: ctypes._Pointer[ctypes._Pointer[struct_ggml_context]]) -> ctypes._Pointer[struct_ggml_cgraph]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_graph_print", [ctypes.POINTER(struct_ggml_cgraph)], None)
def ggml_graph_print(cgraph: ctypes._Pointer[struct_ggml_cgraph]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_graph_dump_dot", [ctypes.POINTER(struct_ggml_cgraph), ctypes.POINTER(struct_ggml_cgraph), ctypes.POINTER(ctypes.c_char)], None)
def ggml_graph_dump_dot(gb: ctypes._Pointer[struct_ggml_cgraph], gf: ctypes._Pointer[struct_ggml_cgraph], filename: ctypes._Pointer[ctypes.c_char]) -> None:
    ...

ggml_log_callback = ctypes.CFUNCTYPE(None, ggml_log_level, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None))
@ctypes_function_for_shared_library('libggml-base.so')("ggml_log_set", [ggml_log_callback, ctypes.POINTER(None)], None)
def ggml_log_set(log_callback: ggml_log_callback, user_data: ctypes.c_void_p) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_set_zero", [ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_set_zero(tensor: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_quantize_init", [ggml_type], None)
def ggml_quantize_init(type: ggml_type) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_quantize_free", [], None)
def ggml_quantize_free() -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_quantize_requires_imatrix", [ggml_type], ctypes.c_bool)
def ggml_quantize_requires_imatrix(type: ggml_type) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_quantize_chunk", [ggml_type, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(None), int64_t, int64_t, int64_t, ctypes.POINTER(ctypes.c_float)], size_t)
def ggml_quantize_chunk(type: ggml_type, src: ctypes._Pointer[ctypes.c_float], dst: ctypes.c_void_p, start: int64_t, nrows: int64_t, n_per_row: int64_t, imatrix: ctypes._Pointer[ctypes.c_float]) -> size_t:
    ...

ggml_to_float_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_float), ctypes.c_int64)
ggml_from_float_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(None), ctypes.c_int64)
class struct_ggml_type_traits(Structure):
    if TYPE_CHECKING:
        type_name: ctypes._Pointer[ctypes.c_char]
        blck_size: ctypes.c_int64
        blck_size_interleave: ctypes.c_int64
        type_size: ctypes.c_uint64
        is_quantized: ctypes.c_bool
        to_float: Callable[[ctypes.c_void_p, ctypes._Pointer[ctypes.c_float], ctypes.c_int64], None]
        from_float_ref: Callable[[ctypes._Pointer[ctypes.c_float], ctypes.c_void_p, ctypes.c_int64], None]
struct_ggml_type_traits._pack_ = 1 # source:False
struct_ggml_type_traits._fields_ = [
    ('type_name', ctypes.POINTER(ctypes.c_char)),
    ('blck_size', ctypes.c_int64),
    ('blck_size_interleave', ctypes.c_int64),
    ('type_size', ctypes.c_uint64),
    ('is_quantized', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 7),
    ('to_float', ctypes.CFUNCTYPE(None, ctypes.POINTER(None), ctypes.POINTER(ctypes.c_float), ctypes.c_int64)),
    ('from_float_ref', ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(None), ctypes.c_int64)),
]

@ctypes_function_for_shared_library('libggml-base.so')("ggml_get_type_traits", [ggml_type], ctypes.POINTER(struct_ggml_type_traits))
def ggml_get_type_traits(type: ggml_type) -> ctypes._Pointer[struct_ggml_type_traits]:
    ...


# values for enumeration 'ggml_sched_priority'
ggml_sched_priority__enumvalues = {
    0: 'GGML_SCHED_PRIO_NORMAL',
    1: 'GGML_SCHED_PRIO_MEDIUM',
    2: 'GGML_SCHED_PRIO_HIGH',
    3: 'GGML_SCHED_PRIO_REALTIME',
}
GGML_SCHED_PRIO_NORMAL = 0
GGML_SCHED_PRIO_MEDIUM = 1
GGML_SCHED_PRIO_HIGH = 2
GGML_SCHED_PRIO_REALTIME = 3
ggml_sched_priority = ctypes.c_uint32 # enum
class struct_ggml_threadpool_params(Structure):
    if TYPE_CHECKING:
        cpumask: ctypes.Array[ctypes.c_bool]
        n_threads: ctypes.c_int32
        prio: ggml_sched_priority
        poll: ctypes.c_uint32
        strict_cpu: ctypes.c_bool
        paused: ctypes.c_bool
struct_ggml_threadpool_params._pack_ = 1 # source:False
struct_ggml_threadpool_params._fields_ = [
    ('cpumask', ctypes.c_bool * 512),
    ('n_threads', ctypes.c_int32),
    ('prio', ggml_sched_priority),
    ('poll', ctypes.c_uint32),
    ('strict_cpu', ctypes.c_bool),
    ('paused', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 2),
]

class struct_ggml_threadpool(Structure):
    pass

ggml_threadpool_t = ctypes.POINTER(struct_ggml_threadpool)
@ctypes_function_for_shared_library('libggml-base.so')("ggml_threadpool_params_default", [ctypes.c_int32], struct_ggml_threadpool_params)
def ggml_threadpool_params_default(n_threads: ctypes.c_int32) -> struct_ggml_threadpool_params:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_threadpool_params_init", [ctypes.POINTER(struct_ggml_threadpool_params), ctypes.c_int32], None)
def ggml_threadpool_params_init(p: ctypes._Pointer[struct_ggml_threadpool_params], n_threads: ctypes.c_int32) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_threadpool_params_match", [ctypes.POINTER(struct_ggml_threadpool_params), ctypes.POINTER(struct_ggml_threadpool_params)], ctypes.c_bool)
def ggml_threadpool_params_match(p0: ctypes._Pointer[struct_ggml_threadpool_params], p1: ctypes._Pointer[struct_ggml_threadpool_params]) -> ctypes.c_bool:
    ...

class struct_ggml_backend_device(Structure):
    if TYPE_CHECKING:
        iface: struct_ggml_backend_device_i
        reg: ctypes._Pointer[struct_ggml_backend_reg]
        context: ctypes.c_void_p
class struct_ggml_backend_buffer_type_i(Structure):
    if TYPE_CHECKING:
        get_name: Callable[[ctypes._Pointer[struct_ggml_backend_buffer_type]], ctypes._Pointer[ctypes.c_char]]
        alloc_buffer: Callable[[ctypes._Pointer[struct_ggml_backend_buffer_type], ctypes.c_uint64], ctypes._Pointer[struct_ggml_backend_buffer]]
        get_alignment: Callable[[ctypes._Pointer[struct_ggml_backend_buffer_type]], ctypes.c_uint64]
        get_max_size: Callable[[ctypes._Pointer[struct_ggml_backend_buffer_type]], ctypes.c_uint64]
        get_alloc_size: Callable[[ctypes._Pointer[struct_ggml_backend_buffer_type], ctypes._Pointer[struct_ggml_tensor]], ctypes.c_uint64]
        is_host: Callable[[ctypes._Pointer[struct_ggml_backend_buffer_type]], ctypes.c_bool]
struct_ggml_backend_buffer_type_i._pack_ = 1 # source:False
struct_ggml_backend_buffer_type_i._fields_ = [
    ('get_name', ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_ggml_backend_buffer_type))),
    ('alloc_buffer', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ggml_backend_buffer), ctypes.POINTER(struct_ggml_backend_buffer_type), ctypes.c_uint64)),
    ('get_alignment', ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.POINTER(struct_ggml_backend_buffer_type))),
    ('get_max_size', ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.POINTER(struct_ggml_backend_buffer_type))),
    ('get_alloc_size', ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.POINTER(struct_ggml_backend_buffer_type), ctypes.POINTER(struct_ggml_tensor))),
    ('is_host', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_ggml_backend_buffer_type))),
]

struct_ggml_backend_buffer_type._pack_ = 1 # source:False
struct_ggml_backend_buffer_type._fields_ = [
    ('iface', struct_ggml_backend_buffer_type_i),
    ('device', ctypes.POINTER(struct_ggml_backend_device)),
    ('context', ctypes.POINTER(None)),
]

ggml_backend_buffer_type_t = ctypes.POINTER(struct_ggml_backend_buffer_type)
ggml_backend_buffer_t = ctypes.POINTER(struct_ggml_backend_buffer)
class struct_ggml_backend(Structure):
    if TYPE_CHECKING:
        guid: ctypes._Pointer[ctypes.Array[ctypes.c_ubyte]]
        iface: struct_ggml_backend_i
        device: ctypes._Pointer[struct_ggml_backend_device]
        context: ctypes.c_void_p
class struct_ggml_backend_i(Structure):
    if TYPE_CHECKING:
        get_name: Callable[[ctypes._Pointer[struct_ggml_backend]], ctypes._Pointer[ctypes.c_char]]
        free: Callable[[ctypes._Pointer[struct_ggml_backend]], None]
        set_tensor_async: Callable[[ctypes._Pointer[struct_ggml_backend], ctypes._Pointer[struct_ggml_tensor], ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64], None]
        get_tensor_async: Callable[[ctypes._Pointer[struct_ggml_backend], ctypes._Pointer[struct_ggml_tensor], ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64], None]
        cpy_tensor_async: Callable[[ctypes._Pointer[struct_ggml_backend], ctypes._Pointer[struct_ggml_backend], ctypes._Pointer[struct_ggml_tensor], ctypes._Pointer[struct_ggml_tensor]], ctypes.c_bool]
        synchronize: Callable[[ctypes._Pointer[struct_ggml_backend]], None]
        graph_plan_create: Callable[[ctypes._Pointer[struct_ggml_backend], ctypes._Pointer[struct_ggml_cgraph]], ctypes.c_void_p]
        graph_plan_free: Callable[[ctypes._Pointer[struct_ggml_backend], ctypes.c_void_p], None]
        graph_plan_update: Callable[[ctypes._Pointer[struct_ggml_backend], ctypes.c_void_p, ctypes._Pointer[struct_ggml_cgraph]], None]
        graph_plan_compute: Callable[[ctypes._Pointer[struct_ggml_backend], ctypes.c_void_p], ggml_status]
        graph_compute: Callable[[ctypes._Pointer[struct_ggml_backend], ctypes._Pointer[struct_ggml_cgraph]], ggml_status]
        event_record: Callable[[ctypes._Pointer[struct_ggml_backend], ctypes._Pointer[struct_ggml_backend_event]], None]
        event_wait: Callable[[ctypes._Pointer[struct_ggml_backend], ctypes._Pointer[struct_ggml_backend_event]], None]
class struct_ggml_backend_event(Structure):
    if TYPE_CHECKING:
        device: ctypes._Pointer[struct_ggml_backend_device]
        context: ctypes.c_void_p
struct_ggml_backend_i._pack_ = 1 # source:False
struct_ggml_backend_i._fields_ = [
    ('get_name', ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_ggml_backend))),
    ('free', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_backend))),
    ('set_tensor_async', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_backend), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(None), ctypes.c_uint64, ctypes.c_uint64)),
    ('get_tensor_async', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_backend), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(None), ctypes.c_uint64, ctypes.c_uint64)),
    ('cpy_tensor_async', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_ggml_backend), ctypes.POINTER(struct_ggml_backend), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor))),
    ('synchronize', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_backend))),
    ('graph_plan_create', ctypes.CFUNCTYPE(ctypes.POINTER(None), ctypes.POINTER(struct_ggml_backend), ctypes.POINTER(struct_ggml_cgraph))),
    ('graph_plan_free', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_backend), ctypes.POINTER(None))),
    ('graph_plan_update', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_backend), ctypes.POINTER(None), ctypes.POINTER(struct_ggml_cgraph))),
    ('graph_plan_compute', ctypes.CFUNCTYPE(ggml_status, ctypes.POINTER(struct_ggml_backend), ctypes.POINTER(None))),
    ('graph_compute', ctypes.CFUNCTYPE(ggml_status, ctypes.POINTER(struct_ggml_backend), ctypes.POINTER(struct_ggml_cgraph))),
    ('event_record', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_backend), ctypes.POINTER(struct_ggml_backend_event))),
    ('event_wait', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_backend), ctypes.POINTER(struct_ggml_backend_event))),
]

struct_ggml_backend._pack_ = 1 # source:False
struct_ggml_backend._fields_ = [
    ('guid', ctypes.POINTER(ctypes.c_ubyte * 16)),
    ('iface', struct_ggml_backend_i),
    ('device', ctypes.POINTER(struct_ggml_backend_device)),
    ('context', ctypes.POINTER(None)),
]

ggml_backend_t = ctypes.POINTER(struct_ggml_backend)
class struct_ggml_tallocr(Structure):
    if TYPE_CHECKING:
        buffer: ctypes._Pointer[struct_ggml_backend_buffer]
        base: ctypes.c_void_p
        alignment: ctypes.c_uint64
        offset: ctypes.c_uint64
struct_ggml_tallocr._pack_ = 1 # source:False
struct_ggml_tallocr._fields_ = [
    ('buffer', ctypes.POINTER(struct_ggml_backend_buffer)),
    ('base', ctypes.POINTER(None)),
    ('alignment', ctypes.c_uint64),
    ('offset', ctypes.c_uint64),
]

@ctypes_function_for_shared_library('libggml-base.so')("ggml_tallocr_new", [ggml_backend_buffer_t], struct_ggml_tallocr)
def ggml_tallocr_new(buffer: ggml_backend_buffer_t) -> struct_ggml_tallocr:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_tallocr_alloc", [ctypes.POINTER(struct_ggml_tallocr), ctypes.POINTER(struct_ggml_tensor)], ggml_status)
def ggml_tallocr_alloc(talloc: ctypes._Pointer[struct_ggml_tallocr], tensor: ctypes._Pointer[struct_ggml_tensor]) -> ggml_status:
    ...

class struct_ggml_gallocr(Structure):
    pass

ggml_gallocr_t = ctypes.POINTER(struct_ggml_gallocr)
@ctypes_function_for_shared_library('libggml-base.so')("ggml_gallocr_new", [ggml_backend_buffer_type_t], ggml_gallocr_t)
def ggml_gallocr_new(buft: ggml_backend_buffer_type_t) -> ggml_gallocr_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_gallocr_new_n", [ctypes.POINTER(ctypes.POINTER(struct_ggml_backend_buffer_type)), ctypes.c_int32], ggml_gallocr_t)
def ggml_gallocr_new_n(bufts: ctypes._Pointer[ctypes._Pointer[struct_ggml_backend_buffer_type]], n_bufs: ctypes.c_int32) -> ggml_gallocr_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_gallocr_free", [ggml_gallocr_t], None)
def ggml_gallocr_free(galloc: ggml_gallocr_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_gallocr_reserve", [ggml_gallocr_t, ctypes.POINTER(struct_ggml_cgraph)], ctypes.c_bool)
def ggml_gallocr_reserve(galloc: ggml_gallocr_t, graph: ctypes._Pointer[struct_ggml_cgraph]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_gallocr_reserve_n", [ggml_gallocr_t, ctypes.POINTER(struct_ggml_cgraph), ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32)], ctypes.c_bool)
def ggml_gallocr_reserve_n(galloc: ggml_gallocr_t, graph: ctypes._Pointer[struct_ggml_cgraph], node_buffer_ids: ctypes._Pointer[ctypes.c_int32], leaf_buffer_ids: ctypes._Pointer[ctypes.c_int32]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_gallocr_alloc_graph", [ggml_gallocr_t, ctypes.POINTER(struct_ggml_cgraph)], ctypes.c_bool)
def ggml_gallocr_alloc_graph(galloc: ggml_gallocr_t, graph: ctypes._Pointer[struct_ggml_cgraph]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_gallocr_get_buffer_size", [ggml_gallocr_t, ctypes.c_int32], size_t)
def ggml_gallocr_get_buffer_size(galloc: ggml_gallocr_t, buffer_id: ctypes.c_int32) -> size_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_alloc_ctx_tensors_from_buft", [ctypes.POINTER(struct_ggml_context), ggml_backend_buffer_type_t], ctypes.POINTER(struct_ggml_backend_buffer))
def ggml_backend_alloc_ctx_tensors_from_buft(ctx: ctypes._Pointer[struct_ggml_context], buft: ggml_backend_buffer_type_t) -> ctypes._Pointer[struct_ggml_backend_buffer]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_alloc_ctx_tensors", [ctypes.POINTER(struct_ggml_context), ggml_backend_t], ctypes.POINTER(struct_ggml_backend_buffer))
def ggml_backend_alloc_ctx_tensors(ctx: ctypes._Pointer[struct_ggml_context], backend: ggml_backend_t) -> ctypes._Pointer[struct_ggml_backend_buffer]:
    ...

struct_ggml_backend_event._pack_ = 1 # source:False
struct_ggml_backend_event._fields_ = [
    ('device', ctypes.POINTER(struct_ggml_backend_device)),
    ('context', ctypes.POINTER(None)),
]

ggml_backend_event_t = ctypes.POINTER(struct_ggml_backend_event)
ggml_backend_graph_plan_t = ctypes.POINTER(None)
class struct_ggml_backend_reg(Structure):
    if TYPE_CHECKING:
        api_version: ctypes.c_int32
        iface: struct_ggml_backend_reg_i
        context: ctypes.c_void_p
class struct_ggml_backend_reg_i(Structure):
    if TYPE_CHECKING:
        get_name: Callable[[ctypes._Pointer[struct_ggml_backend_reg]], ctypes._Pointer[ctypes.c_char]]
        get_device_count: Callable[[ctypes._Pointer[struct_ggml_backend_reg]], ctypes.c_uint64]
        get_device: Callable[[ctypes._Pointer[struct_ggml_backend_reg], ctypes.c_uint64], ctypes._Pointer[struct_ggml_backend_device]]
        get_proc_address: Callable[[ctypes._Pointer[struct_ggml_backend_reg], ctypes._Pointer[ctypes.c_char]], ctypes.c_void_p]
struct_ggml_backend_reg_i._pack_ = 1 # source:False
struct_ggml_backend_reg_i._fields_ = [
    ('get_name', ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_ggml_backend_reg))),
    ('get_device_count', ctypes.CFUNCTYPE(ctypes.c_uint64, ctypes.POINTER(struct_ggml_backend_reg))),
    ('get_device', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ggml_backend_device), ctypes.POINTER(struct_ggml_backend_reg), ctypes.c_uint64)),
    ('get_proc_address', ctypes.CFUNCTYPE(ctypes.POINTER(None), ctypes.POINTER(struct_ggml_backend_reg), ctypes.POINTER(ctypes.c_char))),
]

struct_ggml_backend_reg._pack_ = 1 # source:False
struct_ggml_backend_reg._fields_ = [
    ('api_version', ctypes.c_int32),
    ('iface', struct_ggml_backend_reg_i),
    ('context', ctypes.POINTER(None)),
]

ggml_backend_reg_t = ctypes.POINTER(struct_ggml_backend_reg)
class struct_ggml_backend_device_i(Structure):
    if TYPE_CHECKING:
        get_name: Callable[[ctypes._Pointer[struct_ggml_backend_device]], ctypes._Pointer[ctypes.c_char]]
        get_description: Callable[[ctypes._Pointer[struct_ggml_backend_device]], ctypes._Pointer[ctypes.c_char]]
        get_memory: Callable[[ctypes._Pointer[struct_ggml_backend_device], ctypes._Pointer[ctypes.c_uint64], ctypes._Pointer[ctypes.c_uint64]], None]
        get_type: Callable[[ctypes._Pointer[struct_ggml_backend_device]], ggml_backend_dev_type]
        get_props: Callable[[ctypes._Pointer[struct_ggml_backend_device], ctypes._Pointer[struct_ggml_backend_dev_props]], None]
        init_backend: Callable[[ctypes._Pointer[struct_ggml_backend_device], ctypes._Pointer[ctypes.c_char]], ctypes._Pointer[struct_ggml_backend]]
        get_buffer_type: Callable[[ctypes._Pointer[struct_ggml_backend_device]], ctypes._Pointer[struct_ggml_backend_buffer_type]]
        get_host_buffer_type: Callable[[ctypes._Pointer[struct_ggml_backend_device]], ctypes._Pointer[struct_ggml_backend_buffer_type]]
        buffer_from_host_ptr: Callable[[ctypes._Pointer[struct_ggml_backend_device], ctypes.c_void_p, ctypes.c_uint64, ctypes.c_uint64], ctypes._Pointer[struct_ggml_backend_buffer]]
        supports_op: Callable[[ctypes._Pointer[struct_ggml_backend_device], ctypes._Pointer[struct_ggml_tensor]], ctypes.c_bool]
        supports_buft: Callable[[ctypes._Pointer[struct_ggml_backend_device], ctypes._Pointer[struct_ggml_backend_buffer_type]], ctypes.c_bool]
        offload_op: Callable[[ctypes._Pointer[struct_ggml_backend_device], ctypes._Pointer[struct_ggml_tensor]], ctypes.c_bool]
        event_new: Callable[[ctypes._Pointer[struct_ggml_backend_device]], ctypes._Pointer[struct_ggml_backend_event]]
        event_free: Callable[[ctypes._Pointer[struct_ggml_backend_device], ctypes._Pointer[struct_ggml_backend_event]], None]
        event_synchronize: Callable[[ctypes._Pointer[struct_ggml_backend_device], ctypes._Pointer[struct_ggml_backend_event]], None]

# values for enumeration 'ggml_backend_dev_type'
ggml_backend_dev_type__enumvalues = {
    0: 'GGML_BACKEND_DEVICE_TYPE_CPU',
    1: 'GGML_BACKEND_DEVICE_TYPE_GPU',
    2: 'GGML_BACKEND_DEVICE_TYPE_ACCEL',
}
GGML_BACKEND_DEVICE_TYPE_CPU = 0
GGML_BACKEND_DEVICE_TYPE_GPU = 1
GGML_BACKEND_DEVICE_TYPE_ACCEL = 2
ggml_backend_dev_type = ctypes.c_uint32 # enum
class struct_ggml_backend_dev_props(Structure):
    if TYPE_CHECKING:
        name: ctypes._Pointer[ctypes.c_char]
        description: ctypes._Pointer[ctypes.c_char]
        memory_free: ctypes.c_uint64
        memory_total: ctypes.c_uint64
        type: ggml_backend_dev_type
        caps: struct_ggml_backend_dev_caps
struct_ggml_backend_device_i._pack_ = 1 # source:False
struct_ggml_backend_device_i._fields_ = [
    ('get_name', ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_ggml_backend_device))),
    ('get_description', ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_ggml_backend_device))),
    ('get_memory', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_backend_device), ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64))),
    ('get_type', ctypes.CFUNCTYPE(ggml_backend_dev_type, ctypes.POINTER(struct_ggml_backend_device))),
    ('get_props', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_backend_device), ctypes.POINTER(struct_ggml_backend_dev_props))),
    ('init_backend', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ggml_backend), ctypes.POINTER(struct_ggml_backend_device), ctypes.POINTER(ctypes.c_char))),
    ('get_buffer_type', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ggml_backend_buffer_type), ctypes.POINTER(struct_ggml_backend_device))),
    ('get_host_buffer_type', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ggml_backend_buffer_type), ctypes.POINTER(struct_ggml_backend_device))),
    ('buffer_from_host_ptr', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ggml_backend_buffer), ctypes.POINTER(struct_ggml_backend_device), ctypes.POINTER(None), ctypes.c_uint64, ctypes.c_uint64)),
    ('supports_op', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_ggml_backend_device), ctypes.POINTER(struct_ggml_tensor))),
    ('supports_buft', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_ggml_backend_device), ctypes.POINTER(struct_ggml_backend_buffer_type))),
    ('offload_op', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_ggml_backend_device), ctypes.POINTER(struct_ggml_tensor))),
    ('event_new', ctypes.CFUNCTYPE(ctypes.POINTER(struct_ggml_backend_event), ctypes.POINTER(struct_ggml_backend_device))),
    ('event_free', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_backend_device), ctypes.POINTER(struct_ggml_backend_event))),
    ('event_synchronize', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_backend_device), ctypes.POINTER(struct_ggml_backend_event))),
]

struct_ggml_backend_device._pack_ = 1 # source:False
struct_ggml_backend_device._fields_ = [
    ('iface', struct_ggml_backend_device_i),
    ('reg', ctypes.POINTER(struct_ggml_backend_reg)),
    ('context', ctypes.POINTER(None)),
]

ggml_backend_dev_t = ctypes.POINTER(struct_ggml_backend_device)
@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buft_name", [ggml_backend_buffer_type_t], ctypes.POINTER(ctypes.c_char))
def ggml_backend_buft_name(buft: ggml_backend_buffer_type_t) -> ctypes._Pointer[ctypes.c_char]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buft_alloc_buffer", [ggml_backend_buffer_type_t, size_t], ggml_backend_buffer_t)
def ggml_backend_buft_alloc_buffer(buft: ggml_backend_buffer_type_t, size: size_t) -> ggml_backend_buffer_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buft_get_alignment", [ggml_backend_buffer_type_t], size_t)
def ggml_backend_buft_get_alignment(buft: ggml_backend_buffer_type_t) -> size_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buft_get_max_size", [ggml_backend_buffer_type_t], size_t)
def ggml_backend_buft_get_max_size(buft: ggml_backend_buffer_type_t) -> size_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buft_get_alloc_size", [ggml_backend_buffer_type_t, ctypes.POINTER(struct_ggml_tensor)], size_t)
def ggml_backend_buft_get_alloc_size(buft: ggml_backend_buffer_type_t, tensor: ctypes._Pointer[struct_ggml_tensor]) -> size_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buft_is_host", [ggml_backend_buffer_type_t], ctypes.c_bool)
def ggml_backend_buft_is_host(buft: ggml_backend_buffer_type_t) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buft_get_device", [ggml_backend_buffer_type_t], ggml_backend_dev_t)
def ggml_backend_buft_get_device(buft: ggml_backend_buffer_type_t) -> ggml_backend_dev_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buffer_name", [ggml_backend_buffer_t], ctypes.POINTER(ctypes.c_char))
def ggml_backend_buffer_name(buffer: ggml_backend_buffer_t) -> ctypes._Pointer[ctypes.c_char]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buffer_free", [ggml_backend_buffer_t], None)
def ggml_backend_buffer_free(buffer: ggml_backend_buffer_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buffer_get_base", [ggml_backend_buffer_t], ctypes.POINTER(None))
def ggml_backend_buffer_get_base(buffer: ggml_backend_buffer_t) -> ctypes.c_void_p:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buffer_get_size", [ggml_backend_buffer_t], size_t)
def ggml_backend_buffer_get_size(buffer: ggml_backend_buffer_t) -> size_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buffer_init_tensor", [ggml_backend_buffer_t, ctypes.POINTER(struct_ggml_tensor)], ggml_status)
def ggml_backend_buffer_init_tensor(buffer: ggml_backend_buffer_t, tensor: ctypes._Pointer[struct_ggml_tensor]) -> ggml_status:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buffer_get_alignment", [ggml_backend_buffer_t], size_t)
def ggml_backend_buffer_get_alignment(buffer: ggml_backend_buffer_t) -> size_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buffer_get_max_size", [ggml_backend_buffer_t], size_t)
def ggml_backend_buffer_get_max_size(buffer: ggml_backend_buffer_t) -> size_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buffer_get_alloc_size", [ggml_backend_buffer_t, ctypes.POINTER(struct_ggml_tensor)], size_t)
def ggml_backend_buffer_get_alloc_size(buffer: ggml_backend_buffer_t, tensor: ctypes._Pointer[struct_ggml_tensor]) -> size_t:
    ...

uint8_t = ctypes.c_uint8
@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buffer_clear", [ggml_backend_buffer_t, uint8_t], None)
def ggml_backend_buffer_clear(buffer: ggml_backend_buffer_t, value: uint8_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buffer_is_host", [ggml_backend_buffer_t], ctypes.c_bool)
def ggml_backend_buffer_is_host(buffer: ggml_backend_buffer_t) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buffer_set_usage", [ggml_backend_buffer_t, ggml_backend_buffer_usage], None)
def ggml_backend_buffer_set_usage(buffer: ggml_backend_buffer_t, usage: ggml_backend_buffer_usage) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buffer_get_usage", [ggml_backend_buffer_t], ggml_backend_buffer_usage)
def ggml_backend_buffer_get_usage(buffer: ggml_backend_buffer_t) -> ggml_backend_buffer_usage:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buffer_get_type", [ggml_backend_buffer_t], ggml_backend_buffer_type_t)
def ggml_backend_buffer_get_type(buffer: ggml_backend_buffer_t) -> ggml_backend_buffer_type_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buffer_reset", [ggml_backend_buffer_t], None)
def ggml_backend_buffer_reset(buffer: ggml_backend_buffer_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_tensor_copy", [ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], None)
def ggml_backend_tensor_copy(src: ctypes._Pointer[struct_ggml_tensor], dst: ctypes._Pointer[struct_ggml_tensor]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_guid", [ggml_backend_t], ggml_guid_t)
def ggml_backend_guid(backend: ggml_backend_t) -> ggml_guid_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_name", [ggml_backend_t], ctypes.POINTER(ctypes.c_char))
def ggml_backend_name(backend: ggml_backend_t) -> ctypes._Pointer[ctypes.c_char]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_free", [ggml_backend_t], None)
def ggml_backend_free(backend: ggml_backend_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_get_default_buffer_type", [ggml_backend_t], ggml_backend_buffer_type_t)
def ggml_backend_get_default_buffer_type(backend: ggml_backend_t) -> ggml_backend_buffer_type_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_alloc_buffer", [ggml_backend_t, size_t], ggml_backend_buffer_t)
def ggml_backend_alloc_buffer(backend: ggml_backend_t, size: size_t) -> ggml_backend_buffer_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_get_alignment", [ggml_backend_t], size_t)
def ggml_backend_get_alignment(backend: ggml_backend_t) -> size_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_get_max_size", [ggml_backend_t], size_t)
def ggml_backend_get_max_size(backend: ggml_backend_t) -> size_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_tensor_set_async", [ggml_backend_t, ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(None), size_t, size_t], None)
def ggml_backend_tensor_set_async(backend: ggml_backend_t, tensor: ctypes._Pointer[struct_ggml_tensor], data: ctypes.c_void_p, offset: size_t, size: size_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_tensor_get_async", [ggml_backend_t, ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(None), size_t, size_t], None)
def ggml_backend_tensor_get_async(backend: ggml_backend_t, tensor: ctypes._Pointer[struct_ggml_tensor], data: ctypes.c_void_p, offset: size_t, size: size_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_tensor_set", [ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(None), size_t, size_t], None)
def ggml_backend_tensor_set(tensor: ctypes._Pointer[struct_ggml_tensor], data: ctypes.c_void_p, offset: size_t, size: size_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_tensor_get", [ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(None), size_t, size_t], None)
def ggml_backend_tensor_get(tensor: ctypes._Pointer[struct_ggml_tensor], data: ctypes.c_void_p, offset: size_t, size: size_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_tensor_memset", [ctypes.POINTER(struct_ggml_tensor), uint8_t, size_t, size_t], None)
def ggml_backend_tensor_memset(tensor: ctypes._Pointer[struct_ggml_tensor], value: uint8_t, offset: size_t, size: size_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_synchronize", [ggml_backend_t], None)
def ggml_backend_synchronize(backend: ggml_backend_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_graph_plan_create", [ggml_backend_t, ctypes.POINTER(struct_ggml_cgraph)], ggml_backend_graph_plan_t)
def ggml_backend_graph_plan_create(backend: ggml_backend_t, cgraph: ctypes._Pointer[struct_ggml_cgraph]) -> ggml_backend_graph_plan_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_graph_plan_free", [ggml_backend_t, ggml_backend_graph_plan_t], None)
def ggml_backend_graph_plan_free(backend: ggml_backend_t, plan: ggml_backend_graph_plan_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_graph_plan_compute", [ggml_backend_t, ggml_backend_graph_plan_t], ggml_status)
def ggml_backend_graph_plan_compute(backend: ggml_backend_t, plan: ggml_backend_graph_plan_t) -> ggml_status:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_graph_compute", [ggml_backend_t, ctypes.POINTER(struct_ggml_cgraph)], ggml_status)
def ggml_backend_graph_compute(backend: ggml_backend_t, cgraph: ctypes._Pointer[struct_ggml_cgraph]) -> ggml_status:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_graph_compute_async", [ggml_backend_t, ctypes.POINTER(struct_ggml_cgraph)], ggml_status)
def ggml_backend_graph_compute_async(backend: ggml_backend_t, cgraph: ctypes._Pointer[struct_ggml_cgraph]) -> ggml_status:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_supports_op", [ggml_backend_t, ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_backend_supports_op(backend: ggml_backend_t, op: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_supports_buft", [ggml_backend_t, ggml_backend_buffer_type_t], ctypes.c_bool)
def ggml_backend_supports_buft(backend: ggml_backend_t, buft: ggml_backend_buffer_type_t) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_offload_op", [ggml_backend_t, ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_backend_offload_op(backend: ggml_backend_t, op: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_tensor_copy_async", [ggml_backend_t, ggml_backend_t, ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], None)
def ggml_backend_tensor_copy_async(backend_src: ggml_backend_t, backend_dst: ggml_backend_t, src: ctypes._Pointer[struct_ggml_tensor], dst: ctypes._Pointer[struct_ggml_tensor]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_get_device", [ggml_backend_t], ggml_backend_dev_t)
def ggml_backend_get_device(backend: ggml_backend_t) -> ggml_backend_dev_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_event_new", [ggml_backend_dev_t], ggml_backend_event_t)
def ggml_backend_event_new(device: ggml_backend_dev_t) -> ggml_backend_event_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_event_free", [ggml_backend_event_t], None)
def ggml_backend_event_free(event: ggml_backend_event_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_event_record", [ggml_backend_event_t, ggml_backend_t], None)
def ggml_backend_event_record(event: ggml_backend_event_t, backend: ggml_backend_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_event_synchronize", [ggml_backend_event_t], None)
def ggml_backend_event_synchronize(event: ggml_backend_event_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_event_wait", [ggml_backend_t, ggml_backend_event_t], None)
def ggml_backend_event_wait(backend: ggml_backend_t, event: ggml_backend_event_t) -> None:
    ...

class struct_ggml_backend_dev_caps(Structure):
    if TYPE_CHECKING:
        _async: ctypes.c_bool
        host_buffer: ctypes.c_bool
        buffer_from_host_ptr: ctypes.c_bool
        events: ctypes.c_bool
struct_ggml_backend_dev_caps._pack_ = 1 # source:False
struct_ggml_backend_dev_caps._fields_ = [
    ('_async', ctypes.c_bool),
    ('host_buffer', ctypes.c_bool),
    ('buffer_from_host_ptr', ctypes.c_bool),
    ('events', ctypes.c_bool),
]

struct_ggml_backend_dev_props._pack_ = 1 # source:False
struct_ggml_backend_dev_props._fields_ = [
    ('name', ctypes.POINTER(ctypes.c_char)),
    ('description', ctypes.POINTER(ctypes.c_char)),
    ('memory_free', ctypes.c_uint64),
    ('memory_total', ctypes.c_uint64),
    ('type', ggml_backend_dev_type),
    ('caps', struct_ggml_backend_dev_caps),
]

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_dev_name", [ggml_backend_dev_t], ctypes.POINTER(ctypes.c_char))
def ggml_backend_dev_name(device: ggml_backend_dev_t) -> ctypes._Pointer[ctypes.c_char]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_dev_description", [ggml_backend_dev_t], ctypes.POINTER(ctypes.c_char))
def ggml_backend_dev_description(device: ggml_backend_dev_t) -> ctypes._Pointer[ctypes.c_char]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_dev_memory", [ggml_backend_dev_t, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)], None)
def ggml_backend_dev_memory(device: ggml_backend_dev_t, free: ctypes._Pointer[ctypes.c_uint64], total: ctypes._Pointer[ctypes.c_uint64]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_dev_get_props", [ggml_backend_dev_t, ctypes.POINTER(struct_ggml_backend_dev_props)], None)
def ggml_backend_dev_get_props(device: ggml_backend_dev_t, props: ctypes._Pointer[struct_ggml_backend_dev_props]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_dev_backend_reg", [ggml_backend_dev_t], ggml_backend_reg_t)
def ggml_backend_dev_backend_reg(device: ggml_backend_dev_t) -> ggml_backend_reg_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_dev_init", [ggml_backend_dev_t, ctypes.POINTER(ctypes.c_char)], ggml_backend_t)
def ggml_backend_dev_init(device: ggml_backend_dev_t, params: ctypes._Pointer[ctypes.c_char]) -> ggml_backend_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_dev_buffer_type", [ggml_backend_dev_t], ggml_backend_buffer_type_t)
def ggml_backend_dev_buffer_type(device: ggml_backend_dev_t) -> ggml_backend_buffer_type_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_dev_host_buffer_type", [ggml_backend_dev_t], ggml_backend_buffer_type_t)
def ggml_backend_dev_host_buffer_type(device: ggml_backend_dev_t) -> ggml_backend_buffer_type_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_dev_buffer_from_host_ptr", [ggml_backend_dev_t, ctypes.POINTER(None), size_t, size_t], ggml_backend_buffer_t)
def ggml_backend_dev_buffer_from_host_ptr(device: ggml_backend_dev_t, ptr: ctypes.c_void_p, size: size_t, max_tensor_size: size_t) -> ggml_backend_buffer_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_dev_supports_op", [ggml_backend_dev_t, ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_backend_dev_supports_op(device: ggml_backend_dev_t, op: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_dev_supports_buft", [ggml_backend_dev_t, ggml_backend_buffer_type_t], ctypes.c_bool)
def ggml_backend_dev_supports_buft(device: ggml_backend_dev_t, buft: ggml_backend_buffer_type_t) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_dev_offload_op", [ggml_backend_dev_t, ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_backend_dev_offload_op(device: ggml_backend_dev_t, op: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_reg_name", [ggml_backend_reg_t], ctypes.POINTER(ctypes.c_char))
def ggml_backend_reg_name(reg: ggml_backend_reg_t) -> ctypes._Pointer[ctypes.c_char]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_reg_dev_count", [ggml_backend_reg_t], size_t)
def ggml_backend_reg_dev_count(reg: ggml_backend_reg_t) -> size_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_reg_dev_get", [ggml_backend_reg_t, size_t], ggml_backend_dev_t)
def ggml_backend_reg_dev_get(reg: ggml_backend_reg_t, index: size_t) -> ggml_backend_dev_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_reg_get_proc_address", [ggml_backend_reg_t, ctypes.POINTER(ctypes.c_char)], ctypes.POINTER(None))
def ggml_backend_reg_get_proc_address(reg: ggml_backend_reg_t, name: ctypes._Pointer[ctypes.c_char]) -> ctypes.c_void_p:
    ...

ggml_backend_split_buffer_type_t = ctypes.CFUNCTYPE(ctypes.POINTER(struct_ggml_backend_buffer_type), ctypes.c_int32, ctypes.POINTER(ctypes.c_float))
ggml_backend_set_n_threads_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_backend), ctypes.c_int32)
ggml_backend_dev_get_extra_bufts_t = ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.POINTER(struct_ggml_backend_buffer_type)), ctypes.POINTER(struct_ggml_backend_device))
ggml_backend_set_abort_callback_t = ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_ggml_backend), ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(None)), ctypes.POINTER(None))
class struct_ggml_backend_feature(Structure):
    if TYPE_CHECKING:
        name: ctypes._Pointer[ctypes.c_char]
        value: ctypes._Pointer[ctypes.c_char]
struct_ggml_backend_feature._pack_ = 1 # source:False
struct_ggml_backend_feature._fields_ = [
    ('name', ctypes.POINTER(ctypes.c_char)),
    ('value', ctypes.POINTER(ctypes.c_char)),
]

ggml_backend_get_features_t = ctypes.CFUNCTYPE(ctypes.POINTER(struct_ggml_backend_feature), ctypes.POINTER(struct_ggml_backend_reg))
@ctypes_function_for_shared_library('libggml.so')("ggml_backend_device_register", [ggml_backend_dev_t], None)
def ggml_backend_device_register(device: ggml_backend_dev_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml.so')("ggml_backend_reg_count", [], size_t)
def ggml_backend_reg_count() -> size_t:
    ...

@ctypes_function_for_shared_library('libggml.so')("ggml_backend_reg_get", [size_t], ggml_backend_reg_t)
def ggml_backend_reg_get(index: size_t) -> ggml_backend_reg_t:
    ...

@ctypes_function_for_shared_library('libggml.so')("ggml_backend_reg_by_name", [ctypes.POINTER(ctypes.c_char)], ggml_backend_reg_t)
def ggml_backend_reg_by_name(name: ctypes._Pointer[ctypes.c_char]) -> ggml_backend_reg_t:
    ...

@ctypes_function_for_shared_library('libggml.so')("ggml_backend_dev_count", [], size_t)
def ggml_backend_dev_count() -> size_t:
    ...

@ctypes_function_for_shared_library('libggml.so')("ggml_backend_dev_get", [size_t], ggml_backend_dev_t)
def ggml_backend_dev_get(index: size_t) -> ggml_backend_dev_t:
    ...

@ctypes_function_for_shared_library('libggml.so')("ggml_backend_dev_by_name", [ctypes.POINTER(ctypes.c_char)], ggml_backend_dev_t)
def ggml_backend_dev_by_name(name: ctypes._Pointer[ctypes.c_char]) -> ggml_backend_dev_t:
    ...

@ctypes_function_for_shared_library('libggml.so')("ggml_backend_dev_by_type", [ggml_backend_dev_type], ggml_backend_dev_t)
def ggml_backend_dev_by_type(type: ggml_backend_dev_type) -> ggml_backend_dev_t:
    ...

@ctypes_function_for_shared_library('libggml.so')("ggml_backend_init_by_name", [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)], ggml_backend_t)
def ggml_backend_init_by_name(name: ctypes._Pointer[ctypes.c_char], params: ctypes._Pointer[ctypes.c_char]) -> ggml_backend_t:
    ...

@ctypes_function_for_shared_library('libggml.so')("ggml_backend_init_by_type", [ggml_backend_dev_type, ctypes.POINTER(ctypes.c_char)], ggml_backend_t)
def ggml_backend_init_by_type(type: ggml_backend_dev_type, params: ctypes._Pointer[ctypes.c_char]) -> ggml_backend_t:
    ...

@ctypes_function_for_shared_library('libggml.so')("ggml_backend_init_best", [], ggml_backend_t)
def ggml_backend_init_best() -> ggml_backend_t:
    ...

@ctypes_function_for_shared_library('libggml.so')("ggml_backend_load", [ctypes.POINTER(ctypes.c_char)], ggml_backend_reg_t)
def ggml_backend_load(path: ctypes._Pointer[ctypes.c_char]) -> ggml_backend_reg_t:
    ...

@ctypes_function_for_shared_library('libggml.so')("ggml_backend_unload", [ggml_backend_reg_t], None)
def ggml_backend_unload(reg: ggml_backend_reg_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml.so')("ggml_backend_load_all", [], None)
def ggml_backend_load_all() -> None:
    ...

@ctypes_function_for_shared_library('libggml.so')("ggml_backend_load_all_from_path", [ctypes.POINTER(ctypes.c_char)], None)
def ggml_backend_load_all_from_path(dir_path: ctypes._Pointer[ctypes.c_char]) -> None:
    ...

class struct_ggml_backend_sched(Structure):
    if TYPE_CHECKING:
        is_reset: ctypes.c_bool
        is_alloc: ctypes.c_bool
        n_backends: ctypes.c_int32
        backends: ctypes.Array[ctypes._Pointer[struct_ggml_backend]]
        bufts: ctypes.Array[ctypes._Pointer[struct_ggml_backend_buffer_type]]
        galloc: ctypes._Pointer[struct_ggml_gallocr]
        hash_set: struct_ggml_hash_set
        hv_tensor_backend_ids: ctypes._Pointer[ctypes.c_int32]
        hv_tensor_copies: ctypes._Pointer[ctypes._Pointer[struct_ggml_tensor]]
        node_backend_ids: ctypes._Pointer[ctypes.c_int32]
        leaf_backend_ids: ctypes._Pointer[ctypes.c_int32]
        prev_node_backend_ids: ctypes._Pointer[ctypes.c_int32]
        prev_leaf_backend_ids: ctypes._Pointer[ctypes.c_int32]
        graph: struct_ggml_cgraph
        splits: ctypes._Pointer[struct_ggml_backend_sched_split]
        n_splits: ctypes.c_int32
        splits_capacity: ctypes.c_int32
        n_copies: ctypes.c_int32
        cur_copy: ctypes.c_int32
        events: ctypes.Array[ctypes.Array[ctypes._Pointer[struct_ggml_backend_event]]]
        graph_inputs: ctypes.Array[ctypes._Pointer[struct_ggml_tensor]]
        n_graph_inputs: ctypes.c_int32
        ctx: ctypes._Pointer[struct_ggml_context]
        callback_eval: Callable[[ctypes._Pointer[struct_ggml_tensor], ctypes.c_bool, ctypes.c_void_p], ctypes.c_bool]
        callback_eval_user_data: ctypes.c_void_p
        context_buffer: ctypes._Pointer[ctypes.c_char]
        context_buffer_size: ctypes.c_uint64
        op_offload: ctypes.c_bool
        debug: ctypes.c_int32
class struct_ggml_backend_sched_split(Structure):
    if TYPE_CHECKING:
        backend_id: ctypes.c_int32
        i_start: ctypes.c_int32
        i_end: ctypes.c_int32
        inputs: ctypes.Array[ctypes._Pointer[struct_ggml_tensor]]
        n_inputs: ctypes.c_int32
        graph: struct_ggml_cgraph
class struct_ggml_hash_set(Structure):
    if TYPE_CHECKING:
        size: ctypes.c_uint64
        used: ctypes._Pointer[ctypes.c_uint32]
        keys: ctypes._Pointer[ctypes._Pointer[struct_ggml_tensor]]
struct_ggml_hash_set._pack_ = 1 # source:False
struct_ggml_hash_set._fields_ = [
    ('size', ctypes.c_uint64),
    ('used', ctypes.POINTER(ctypes.c_uint32)),
    ('keys', ctypes.POINTER(ctypes.POINTER(struct_ggml_tensor))),
]


# values for enumeration 'ggml_cgraph_eval_order'
ggml_cgraph_eval_order__enumvalues = {
    0: 'GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT',
    1: 'GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT',
    2: 'GGML_CGRAPH_EVAL_ORDER_COUNT',
}
GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0
GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT = 1
GGML_CGRAPH_EVAL_ORDER_COUNT = 2
ggml_cgraph_eval_order = ctypes.c_uint32 # enum
struct_ggml_cgraph._pack_ = 1 # source:False
struct_ggml_cgraph._fields_ = [
    ('size', ctypes.c_int32),
    ('n_nodes', ctypes.c_int32),
    ('n_leafs', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('nodes', ctypes.POINTER(ctypes.POINTER(struct_ggml_tensor))),
    ('grads', ctypes.POINTER(ctypes.POINTER(struct_ggml_tensor))),
    ('grad_accs', ctypes.POINTER(ctypes.POINTER(struct_ggml_tensor))),
    ('leafs', ctypes.POINTER(ctypes.POINTER(struct_ggml_tensor))),
    ('visited_hash_set', struct_ggml_hash_set),
    ('order', ggml_cgraph_eval_order),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

struct_ggml_backend_sched._pack_ = 1 # source:False
struct_ggml_backend_sched._fields_ = [
    ('is_reset', ctypes.c_bool),
    ('is_alloc', ctypes.c_bool),
    ('n_backends', ctypes.c_int32),
    ('backends', ctypes.POINTER(struct_ggml_backend) * 16),
    ('bufts', ctypes.POINTER(struct_ggml_backend_buffer_type) * 16),
    ('galloc', ctypes.POINTER(struct_ggml_gallocr)),
    ('hash_set', struct_ggml_hash_set),
    ('hv_tensor_backend_ids', ctypes.POINTER(ctypes.c_int32)),
    ('hv_tensor_copies', ctypes.POINTER(ctypes.POINTER(struct_ggml_tensor))),
    ('node_backend_ids', ctypes.POINTER(ctypes.c_int32)),
    ('leaf_backend_ids', ctypes.POINTER(ctypes.c_int32)),
    ('prev_node_backend_ids', ctypes.POINTER(ctypes.c_int32)),
    ('prev_leaf_backend_ids', ctypes.POINTER(ctypes.c_int32)),
    ('graph', struct_ggml_cgraph),
    ('splits', ctypes.POINTER(struct_ggml_backend_sched_split)),
    ('n_splits', ctypes.c_int32),
    ('splits_capacity', ctypes.c_int32),
    ('n_copies', ctypes.c_int32),
    ('cur_copy', ctypes.c_int32),
    ('events', ctypes.POINTER(struct_ggml_backend_event) * 4 * 16),
    ('graph_inputs', ctypes.POINTER(struct_ggml_tensor) * 10),
    ('n_graph_inputs', ctypes.c_int32),
    ('ctx', ctypes.POINTER(struct_ggml_context)),
    ('callback_eval', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_ggml_tensor), ctypes.c_bool, ctypes.POINTER(None))),
    ('callback_eval_user_data', ctypes.POINTER(None)),
    ('context_buffer', ctypes.POINTER(ctypes.c_char)),
    ('context_buffer_size', ctypes.c_uint64),
    ('op_offload', ctypes.c_bool),
    ('debug', ctypes.c_int32),
]

ggml_backend_sched_t = ctypes.POINTER(struct_ggml_backend_sched)
ggml_backend_sched_eval_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_ggml_tensor), ctypes.c_bool, ctypes.POINTER(None))
@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_sched_new", [ctypes.POINTER(ctypes.POINTER(struct_ggml_backend)), ctypes.POINTER(ctypes.POINTER(struct_ggml_backend_buffer_type)), ctypes.c_int32, size_t, ctypes.c_bool, ctypes.c_bool], ggml_backend_sched_t)
def ggml_backend_sched_new(backends: ctypes._Pointer[ctypes._Pointer[struct_ggml_backend]], bufts: ctypes._Pointer[ctypes._Pointer[struct_ggml_backend_buffer_type]], n_backends: ctypes.c_int32, graph_size: size_t, parallel: ctypes.c_bool, op_offload: ctypes.c_bool) -> ggml_backend_sched_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_sched_free", [ggml_backend_sched_t], None)
def ggml_backend_sched_free(sched: ggml_backend_sched_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_sched_reserve", [ggml_backend_sched_t, ctypes.POINTER(struct_ggml_cgraph)], ctypes.c_bool)
def ggml_backend_sched_reserve(sched: ggml_backend_sched_t, measure_graph: ctypes._Pointer[struct_ggml_cgraph]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_sched_get_n_backends", [ggml_backend_sched_t], ctypes.c_int32)
def ggml_backend_sched_get_n_backends(sched: ggml_backend_sched_t) -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_sched_get_backend", [ggml_backend_sched_t, ctypes.c_int32], ggml_backend_t)
def ggml_backend_sched_get_backend(sched: ggml_backend_sched_t, i: ctypes.c_int32) -> ggml_backend_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_sched_get_n_splits", [ggml_backend_sched_t], ctypes.c_int32)
def ggml_backend_sched_get_n_splits(sched: ggml_backend_sched_t) -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_sched_get_n_copies", [ggml_backend_sched_t], ctypes.c_int32)
def ggml_backend_sched_get_n_copies(sched: ggml_backend_sched_t) -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_sched_get_buffer_size", [ggml_backend_sched_t, ggml_backend_t], size_t)
def ggml_backend_sched_get_buffer_size(sched: ggml_backend_sched_t, backend: ggml_backend_t) -> size_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_sched_set_tensor_backend", [ggml_backend_sched_t, ctypes.POINTER(struct_ggml_tensor), ggml_backend_t], None)
def ggml_backend_sched_set_tensor_backend(sched: ggml_backend_sched_t, node: ctypes._Pointer[struct_ggml_tensor], backend: ggml_backend_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_sched_get_tensor_backend", [ggml_backend_sched_t, ctypes.POINTER(struct_ggml_tensor)], ggml_backend_t)
def ggml_backend_sched_get_tensor_backend(sched: ggml_backend_sched_t, node: ctypes._Pointer[struct_ggml_tensor]) -> ggml_backend_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_sched_alloc_graph", [ggml_backend_sched_t, ctypes.POINTER(struct_ggml_cgraph)], ctypes.c_bool)
def ggml_backend_sched_alloc_graph(sched: ggml_backend_sched_t, graph: ctypes._Pointer[struct_ggml_cgraph]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_sched_graph_compute", [ggml_backend_sched_t, ctypes.POINTER(struct_ggml_cgraph)], ggml_status)
def ggml_backend_sched_graph_compute(sched: ggml_backend_sched_t, graph: ctypes._Pointer[struct_ggml_cgraph]) -> ggml_status:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_sched_graph_compute_async", [ggml_backend_sched_t, ctypes.POINTER(struct_ggml_cgraph)], ggml_status)
def ggml_backend_sched_graph_compute_async(sched: ggml_backend_sched_t, graph: ctypes._Pointer[struct_ggml_cgraph]) -> ggml_status:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_sched_synchronize", [ggml_backend_sched_t], None)
def ggml_backend_sched_synchronize(sched: ggml_backend_sched_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_sched_reset", [ggml_backend_sched_t], None)
def ggml_backend_sched_reset(sched: ggml_backend_sched_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_sched_set_eval_callback", [ggml_backend_sched_t, ggml_backend_sched_eval_callback, ctypes.POINTER(None)], None)
def ggml_backend_sched_set_eval_callback(sched: ggml_backend_sched_t, callback: ggml_backend_sched_eval_callback, user_data: ctypes.c_void_p) -> None:
    ...

class struct_ggml_backend_graph_copy(Structure):
    if TYPE_CHECKING:
        buffer: ctypes._Pointer[struct_ggml_backend_buffer]
        ctx_allocated: ctypes._Pointer[struct_ggml_context]
        ctx_unallocated: ctypes._Pointer[struct_ggml_context]
        graph: ctypes._Pointer[struct_ggml_cgraph]
struct_ggml_backend_graph_copy._pack_ = 1 # source:False
struct_ggml_backend_graph_copy._fields_ = [
    ('buffer', ctypes.POINTER(struct_ggml_backend_buffer)),
    ('ctx_allocated', ctypes.POINTER(struct_ggml_context)),
    ('ctx_unallocated', ctypes.POINTER(struct_ggml_context)),
    ('graph', ctypes.POINTER(struct_ggml_cgraph)),
]

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_graph_copy", [ggml_backend_t, ctypes.POINTER(struct_ggml_cgraph)], struct_ggml_backend_graph_copy)
def ggml_backend_graph_copy(backend: ggml_backend_t, graph: ctypes._Pointer[struct_ggml_cgraph]) -> struct_ggml_backend_graph_copy:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_graph_copy_free", [struct_ggml_backend_graph_copy], None)
def ggml_backend_graph_copy_free(copy: struct_ggml_backend_graph_copy) -> None:
    ...

ggml_backend_eval_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_int32, ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(None))
@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_compare_graph_backend", [ggml_backend_t, ggml_backend_t, ctypes.POINTER(struct_ggml_cgraph), ggml_backend_eval_callback, ctypes.POINTER(None)], ctypes.c_bool)
def ggml_backend_compare_graph_backend(backend1: ggml_backend_t, backend2: ggml_backend_t, graph: ctypes._Pointer[struct_ggml_cgraph], callback: ggml_backend_eval_callback, user_data: ctypes.c_void_p) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_tensor_alloc", [ggml_backend_buffer_t, ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(None)], ggml_status)
def ggml_backend_tensor_alloc(buffer: ggml_backend_buffer_t, tensor: ctypes._Pointer[struct_ggml_tensor], addr: ctypes.c_void_p) -> ggml_status:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_view_init", [ctypes.POINTER(struct_ggml_tensor)], ggml_status)
def ggml_backend_view_init(tensor: ctypes._Pointer[struct_ggml_tensor]) -> ggml_status:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_cpu_buffer_from_ptr", [ctypes.POINTER(None), size_t], ggml_backend_buffer_t)
def ggml_backend_cpu_buffer_from_ptr(ptr: ctypes.c_void_p, size: size_t) -> ggml_backend_buffer_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_cpu_buffer_type", [], ggml_backend_buffer_type_t)
def ggml_backend_cpu_buffer_type() -> ggml_backend_buffer_type_t:
    ...

class struct_ggml_cplan(Structure):
    if TYPE_CHECKING:
        work_size: ctypes.c_uint64
        work_data: ctypes._Pointer[ctypes.c_ubyte]
        n_threads: ctypes.c_int32
        threadpool: ctypes._Pointer[struct_ggml_threadpool]
        abort_callback: Callable[[ctypes.c_void_p], ctypes.c_bool]
        abort_callback_data: ctypes.c_void_p
struct_ggml_cplan._pack_ = 1 # source:False
struct_ggml_cplan._fields_ = [
    ('work_size', ctypes.c_uint64),
    ('work_data', ctypes.POINTER(ctypes.c_ubyte)),
    ('n_threads', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('threadpool', ctypes.POINTER(struct_ggml_threadpool)),
    ('abort_callback', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(None))),
    ('abort_callback_data', ctypes.POINTER(None)),
]


# values for enumeration 'ggml_numa_strategy'
ggml_numa_strategy__enumvalues = {
    0: 'GGML_NUMA_STRATEGY_DISABLED',
    1: 'GGML_NUMA_STRATEGY_DISTRIBUTE',
    2: 'GGML_NUMA_STRATEGY_ISOLATE',
    3: 'GGML_NUMA_STRATEGY_NUMACTL',
    4: 'GGML_NUMA_STRATEGY_MIRROR',
    5: 'GGML_NUMA_STRATEGY_COUNT',
}
GGML_NUMA_STRATEGY_DISABLED = 0
GGML_NUMA_STRATEGY_DISTRIBUTE = 1
GGML_NUMA_STRATEGY_ISOLATE = 2
GGML_NUMA_STRATEGY_NUMACTL = 3
GGML_NUMA_STRATEGY_MIRROR = 4
GGML_NUMA_STRATEGY_COUNT = 5
ggml_numa_strategy = ctypes.c_uint32 # enum
@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_numa_init", [ggml_numa_strategy], None)
def ggml_numa_init(numa: ggml_numa_strategy) -> None:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_is_numa", [], ctypes.c_bool)
def ggml_is_numa() -> ctypes.c_bool:
    ...

int32_t = ctypes.c_int32
@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_new_i32", [ctypes.POINTER(struct_ggml_context), int32_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_new_i32(ctx: ctypes._Pointer[struct_ggml_context], value: int32_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_new_f32", [ctypes.POINTER(struct_ggml_context), ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_new_f32(ctx: ctypes._Pointer[struct_ggml_context], value: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_set_i32", [ctypes.POINTER(struct_ggml_tensor), int32_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_set_i32(tensor: ctypes._Pointer[struct_ggml_tensor], value: int32_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_set_f32", [ctypes.POINTER(struct_ggml_tensor), ctypes.c_float], ctypes.POINTER(struct_ggml_tensor))
def ggml_set_f32(tensor: ctypes._Pointer[struct_ggml_tensor], value: ctypes.c_float) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_get_i32_1d", [ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32], int32_t)
def ggml_get_i32_1d(tensor: ctypes._Pointer[struct_ggml_tensor], i: ctypes.c_int32) -> int32_t:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_set_i32_1d", [ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, int32_t], None)
def ggml_set_i32_1d(tensor: ctypes._Pointer[struct_ggml_tensor], i: ctypes.c_int32, value: int32_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_get_i32_nd", [ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32], int32_t)
def ggml_get_i32_nd(tensor: ctypes._Pointer[struct_ggml_tensor], i0: ctypes.c_int32, i1: ctypes.c_int32, i2: ctypes.c_int32, i3: ctypes.c_int32) -> int32_t:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_set_i32_nd", [ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, int32_t], None)
def ggml_set_i32_nd(tensor: ctypes._Pointer[struct_ggml_tensor], i0: ctypes.c_int32, i1: ctypes.c_int32, i2: ctypes.c_int32, i3: ctypes.c_int32, value: int32_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_get_f32_1d", [ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32], ctypes.c_float)
def ggml_get_f32_1d(tensor: ctypes._Pointer[struct_ggml_tensor], i: ctypes.c_int32) -> ctypes.c_float:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_set_f32_1d", [ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_float], None)
def ggml_set_f32_1d(tensor: ctypes._Pointer[struct_ggml_tensor], i: ctypes.c_int32, value: ctypes.c_float) -> None:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_get_f32_nd", [ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32], ctypes.c_float)
def ggml_get_f32_nd(tensor: ctypes._Pointer[struct_ggml_tensor], i0: ctypes.c_int32, i1: ctypes.c_int32, i2: ctypes.c_int32, i3: ctypes.c_int32) -> ctypes.c_float:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_set_f32_nd", [ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float], None)
def ggml_set_f32_nd(tensor: ctypes._Pointer[struct_ggml_tensor], i0: ctypes.c_int32, i1: ctypes.c_int32, i2: ctypes.c_int32, i3: ctypes.c_int32, value: ctypes.c_float) -> None:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_threadpool_new", [ctypes.POINTER(struct_ggml_threadpool_params)], ctypes.POINTER(struct_ggml_threadpool))
def ggml_threadpool_new(params: ctypes._Pointer[struct_ggml_threadpool_params]) -> ctypes._Pointer[struct_ggml_threadpool]:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_threadpool_free", [ctypes.POINTER(struct_ggml_threadpool)], None)
def ggml_threadpool_free(threadpool: ctypes._Pointer[struct_ggml_threadpool]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_threadpool_get_n_threads", [ctypes.POINTER(struct_ggml_threadpool)], ctypes.c_int32)
def ggml_threadpool_get_n_threads(threadpool: ctypes._Pointer[struct_ggml_threadpool]) -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_threadpool_pause", [ctypes.POINTER(struct_ggml_threadpool)], None)
def ggml_threadpool_pause(threadpool: ctypes._Pointer[struct_ggml_threadpool]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_threadpool_resume", [ctypes.POINTER(struct_ggml_threadpool)], None)
def ggml_threadpool_resume(threadpool: ctypes._Pointer[struct_ggml_threadpool]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_graph_plan", [ctypes.POINTER(struct_ggml_cgraph), ctypes.c_int32, ctypes.POINTER(struct_ggml_threadpool)], struct_ggml_cplan)
def ggml_graph_plan(cgraph: ctypes._Pointer[struct_ggml_cgraph], n_threads: ctypes.c_int32, threadpool: ctypes._Pointer[struct_ggml_threadpool]) -> struct_ggml_cplan:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_graph_compute", [ctypes.POINTER(struct_ggml_cgraph), ctypes.POINTER(struct_ggml_cplan)], ggml_status)
def ggml_graph_compute(cgraph: ctypes._Pointer[struct_ggml_cgraph], cplan: ctypes._Pointer[struct_ggml_cplan]) -> ggml_status:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_graph_compute_with_ctx", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_cgraph), ctypes.c_int32], ggml_status)
def ggml_graph_compute_with_ctx(ctx: ctypes._Pointer[struct_ggml_context], cgraph: ctypes._Pointer[struct_ggml_cgraph], n_threads: ctypes.c_int32) -> ggml_status:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_sse3", [], ctypes.c_int32)
def ggml_cpu_has_sse3() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_ssse3", [], ctypes.c_int32)
def ggml_cpu_has_ssse3() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_avx", [], ctypes.c_int32)
def ggml_cpu_has_avx() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_avx_vnni", [], ctypes.c_int32)
def ggml_cpu_has_avx_vnni() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_avx2", [], ctypes.c_int32)
def ggml_cpu_has_avx2() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_bmi2", [], ctypes.c_int32)
def ggml_cpu_has_bmi2() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_f16c", [], ctypes.c_int32)
def ggml_cpu_has_f16c() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_fma", [], ctypes.c_int32)
def ggml_cpu_has_fma() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_avx512", [], ctypes.c_int32)
def ggml_cpu_has_avx512() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_avx512_vbmi", [], ctypes.c_int32)
def ggml_cpu_has_avx512_vbmi() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_avx512_vnni", [], ctypes.c_int32)
def ggml_cpu_has_avx512_vnni() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_avx512_bf16", [], ctypes.c_int32)
def ggml_cpu_has_avx512_bf16() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_amx_int8", [], ctypes.c_int32)
def ggml_cpu_has_amx_int8() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_neon", [], ctypes.c_int32)
def ggml_cpu_has_neon() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_arm_fma", [], ctypes.c_int32)
def ggml_cpu_has_arm_fma() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_fp16_va", [], ctypes.c_int32)
def ggml_cpu_has_fp16_va() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_dotprod", [], ctypes.c_int32)
def ggml_cpu_has_dotprod() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_matmul_int8", [], ctypes.c_int32)
def ggml_cpu_has_matmul_int8() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_sve", [], ctypes.c_int32)
def ggml_cpu_has_sve() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_get_sve_cnt", [], ctypes.c_int32)
def ggml_cpu_get_sve_cnt() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_sme", [], ctypes.c_int32)
def ggml_cpu_has_sme() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_riscv_v", [], ctypes.c_int32)
def ggml_cpu_has_riscv_v() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_vsx", [], ctypes.c_int32)
def ggml_cpu_has_vsx() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_vxe", [], ctypes.c_int32)
def ggml_cpu_has_vxe() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_wasm_simd", [], ctypes.c_int32)
def ggml_cpu_has_wasm_simd() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_has_llamafile", [], ctypes.c_int32)
def ggml_cpu_has_llamafile() -> ctypes.c_int32:
    ...

ggml_vec_dot_t = ctypes.CFUNCTYPE(None, ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_uint64, ctypes.POINTER(None), ctypes.c_uint64, ctypes.POINTER(None), ctypes.c_uint64, ctypes.c_int32)
class struct_ggml_type_traits_cpu(Structure):
    if TYPE_CHECKING:
        from_float: Callable[[ctypes._Pointer[ctypes.c_float], ctypes.c_void_p, ctypes.c_int64], None]
        vec_dot: Callable[[ctypes.c_int32, ctypes._Pointer[ctypes.c_float], ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int32], None]
        vec_dot_type: ggml_type
        nrows: ctypes.c_int64
struct_ggml_type_traits_cpu._pack_ = 1 # source:False
struct_ggml_type_traits_cpu._fields_ = [
    ('from_float', ctypes.CFUNCTYPE(None, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(None), ctypes.c_int64)),
    ('vec_dot', ctypes.CFUNCTYPE(None, ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_uint64, ctypes.POINTER(None), ctypes.c_uint64, ctypes.POINTER(None), ctypes.c_uint64, ctypes.c_int32)),
    ('vec_dot_type', ggml_type),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('nrows', ctypes.c_int64),
]

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_get_type_traits_cpu", [ggml_type], ctypes.POINTER(struct_ggml_type_traits_cpu))
def ggml_get_type_traits_cpu(type: ggml_type) -> ctypes._Pointer[struct_ggml_type_traits_cpu]:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_init", [], None)
def ggml_cpu_init() -> None:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_backend_cpu_init", [], ggml_backend_t)
def ggml_backend_cpu_init() -> ggml_backend_t:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_backend_is_cpu", [ggml_backend_t], ctypes.c_bool)
def ggml_backend_is_cpu(backend: ggml_backend_t) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_backend_cpu_set_n_threads", [ggml_backend_t, ctypes.c_int32], None)
def ggml_backend_cpu_set_n_threads(backend_cpu: ggml_backend_t, n_threads: ctypes.c_int32) -> None:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_backend_cpu_set_threadpool", [ggml_backend_t, ggml_threadpool_t], None)
def ggml_backend_cpu_set_threadpool(backend_cpu: ggml_backend_t, threadpool: ggml_threadpool_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_backend_cpu_set_abort_callback", [ggml_backend_t, ggml_abort_callback, ctypes.POINTER(None)], None)
def ggml_backend_cpu_set_abort_callback(backend_cpu: ggml_backend_t, abort_callback: ggml_abort_callback, abort_callback_data: ctypes.c_void_p) -> None:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_backend_cpu_reg", [], ggml_backend_reg_t)
def ggml_backend_cpu_reg() -> ggml_backend_reg_t:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_fp32_to_fp16", [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_uint16), int64_t], None)
def ggml_cpu_fp32_to_fp16(p1: ctypes._Pointer[ctypes.c_float], p2: ctypes._Pointer[ctypes.c_uint16], p3: int64_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_fp16_to_fp32", [ctypes.POINTER(ctypes.c_uint16), ctypes.POINTER(ctypes.c_float), int64_t], None)
def ggml_cpu_fp16_to_fp32(p1: ctypes._Pointer[ctypes.c_uint16], p2: ctypes._Pointer[ctypes.c_float], p3: int64_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_fp32_to_bf16", [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(struct_c__SA_ggml_bf16_t), int64_t], None)
def ggml_cpu_fp32_to_bf16(p1: ctypes._Pointer[ctypes.c_float], p2: ctypes._Pointer[struct_c__SA_ggml_bf16_t], p3: int64_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_cpu_bf16_to_fp32", [ctypes.POINTER(struct_c__SA_ggml_bf16_t), ctypes.POINTER(ctypes.c_float), int64_t], None)
def ggml_cpu_bf16_to_fp32(p1: ctypes._Pointer[struct_c__SA_ggml_bf16_t], p2: ctypes._Pointer[ctypes.c_float], p3: int64_t) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_print_backtrace_symbols", [], None)
def ggml_print_backtrace_symbols() -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_print_backtrace", [], None)
def ggml_print_backtrace() -> None:
    ...

class struct_ggml_logger_state(Structure):
    if TYPE_CHECKING:
        log_callback: Callable[[ggml_log_level, ctypes._Pointer[ctypes.c_char], ctypes.c_void_p], None]
        log_callback_user_data: ctypes.c_void_p
struct_ggml_logger_state._pack_ = 1 # source:False
struct_ggml_logger_state._fields_ = [
    ('log_callback', ctypes.CFUNCTYPE(None, ggml_log_level, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None))),
    ('log_callback_user_data', ctypes.POINTER(None)),
]

class struct___va_list_tag(Structure):
    if TYPE_CHECKING:
        gp_offset: ctypes.c_uint32
        fp_offset: ctypes.c_uint32
        overflow_arg_area: ctypes.c_void_p
        reg_save_area: ctypes.c_void_p
struct___va_list_tag._pack_ = 1 # source:False
struct___va_list_tag._fields_ = [
    ('gp_offset', ctypes.c_uint32),
    ('fp_offset', ctypes.c_uint32),
    ('overflow_arg_area', ctypes.POINTER(None)),
    ('reg_save_area', ctypes.POINTER(None)),
]

va_list = struct___va_list_tag * 1
@ctypes_function_for_shared_library('FIXME_STUB')("ggml_log_internal_v", [ggml_log_level, ctypes.POINTER(ctypes.c_char), va_list], None)
def ggml_log_internal_v(level: ggml_log_level, format: ctypes._Pointer[ctypes.c_char], args: va_list) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_log_internal", [ggml_log_level, ctypes.POINTER(ctypes.c_char)], None)
def ggml_log_internal(level: ggml_log_level, format: ctypes._Pointer[ctypes.c_char]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_log_callback_default", [ggml_log_level, ctypes.POINTER(ctypes.c_char), ctypes.POINTER(None)], None)
def ggml_log_callback_default(level: ggml_log_level, text: ctypes._Pointer[ctypes.c_char], user_data: ctypes.c_void_p) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_aligned_malloc", [size_t], ctypes.POINTER(None))
def ggml_aligned_malloc(size: size_t) -> ctypes.c_void_p:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_aligned_free", [ctypes.POINTER(None), size_t], None)
def ggml_aligned_free(ptr: ctypes.c_void_p, size: size_t) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_malloc", [size_t], ctypes.POINTER(None))
def ggml_malloc(size: size_t) -> ctypes.c_void_p:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_calloc", [size_t, size_t], ctypes.POINTER(None))
def ggml_calloc(num: size_t, size: size_t) -> ctypes.c_void_p:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_vec_dot_f32", [ctypes.c_int32, ctypes.POINTER(ctypes.c_float), size_t, ctypes.POINTER(ctypes.c_float), size_t, ctypes.POINTER(ctypes.c_float), size_t, ctypes.c_int32], None)
def ggml_vec_dot_f32(n: ctypes.c_int32, s: ctypes._Pointer[ctypes.c_float], bs: size_t, x: ctypes._Pointer[ctypes.c_float], bx: size_t, y: ctypes._Pointer[ctypes.c_float], by: size_t, nrc: ctypes.c_int32) -> None:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_vec_dot_f16", [ctypes.c_int32, ctypes.POINTER(ctypes.c_float), size_t, ctypes.POINTER(ctypes.c_uint16), size_t, ctypes.POINTER(ctypes.c_uint16), size_t, ctypes.c_int32], None)
def ggml_vec_dot_f16(n: ctypes.c_int32, s: ctypes._Pointer[ctypes.c_float], bs: size_t, x: ctypes._Pointer[ctypes.c_uint16], bx: size_t, y: ctypes._Pointer[ctypes.c_uint16], by: size_t, nrc: ctypes.c_int32) -> None:
    ...

@ctypes_function_for_shared_library('libggml-cpu.so')("ggml_vec_dot_bf16", [ctypes.c_int32, ctypes.POINTER(ctypes.c_float), size_t, ctypes.POINTER(struct_c__SA_ggml_bf16_t), size_t, ctypes.POINTER(struct_c__SA_ggml_bf16_t), size_t, ctypes.c_int32], None)
def ggml_vec_dot_bf16(n: ctypes.c_int32, s: ctypes._Pointer[ctypes.c_float], bs: size_t, x: ctypes._Pointer[struct_c__SA_ggml_bf16_t], bx: size_t, y: ctypes._Pointer[struct_c__SA_ggml_bf16_t], by: size_t, nrc: ctypes.c_int32) -> None:
    ...

class struct_ggml_context_container(Structure):
    if TYPE_CHECKING:
        used: ctypes.c_bool
        context: struct_ggml_context
struct_ggml_context._pack_ = 1 # source:False
struct_ggml_context._fields_ = [
    ('mem_size', ctypes.c_uint64),
    ('mem_buffer', ctypes.POINTER(None)),
    ('mem_buffer_owned', ctypes.c_bool),
    ('no_alloc', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 2),
    ('n_objects', ctypes.c_int32),
    ('objects_begin', ctypes.POINTER(struct_ggml_object)),
    ('objects_end', ctypes.POINTER(struct_ggml_object)),
]

struct_ggml_context_container._pack_ = 1 # source:False
struct_ggml_context_container._fields_ = [
    ('used', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 7),
    ('context', struct_ggml_context),
]

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_is_contiguous_n", [ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32], ctypes.c_bool)
def ggml_is_contiguous_n(tensor: ctypes._Pointer[struct_ggml_tensor], n: ctypes.c_int32) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_is_padded_1d", [ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_is_padded_1d(tensor: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_can_repeat_rows", [ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_can_repeat_rows(t0: ctypes._Pointer[struct_ggml_tensor], t1: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_new_object", [ctypes.POINTER(struct_ggml_context), ggml_object_type, size_t], ctypes.POINTER(struct_ggml_object))
def ggml_new_object(ctx: ctypes._Pointer[struct_ggml_context], type: ggml_object_type, size: size_t) -> ctypes._Pointer[struct_ggml_object]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_new_tensor_impl", [ctypes.POINTER(struct_ggml_context), ggml_type, ctypes.c_int32, ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(struct_ggml_tensor), size_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_new_tensor_impl(ctx: ctypes._Pointer[struct_ggml_context], type: ggml_type, n_dims: ctypes.c_int32, ne: ctypes._Pointer[ctypes.c_int64], view_src: ctypes._Pointer[struct_ggml_tensor], view_offs: size_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_dup_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_dup_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_add_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_add_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_add_cast_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ggml_type], ctypes.POINTER(struct_ggml_tensor))
def ggml_add_cast_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], type: ggml_type) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_add1_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_add1_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_acc_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), size_t, size_t, size_t, size_t, ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_acc_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], nb1: size_t, nb2: size_t, nb3: size_t, offset: size_t, inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_sub_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_sub_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_mul_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_mul_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_div_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_div_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_sqr_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_sqr_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_sqrt_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_sqrt_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_log_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_log_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_sin_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_sin_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_cos_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_cos_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_norm_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_float, ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_norm_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], eps: ctypes.c_float, inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_rms_norm_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_float, ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_rms_norm_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], eps: ctypes.c_float, inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_group_norm_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_float, ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_group_norm_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], n_groups: ctypes.c_int32, eps: ctypes.c_float, inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_l2_norm_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_float, ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_l2_norm_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], eps: ctypes.c_float, inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_can_mul_mat", [ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_can_mul_mat(t0: ctypes._Pointer[struct_ggml_tensor], t1: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_can_out_prod", [ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_can_out_prod(t0: ctypes._Pointer[struct_ggml_tensor], t1: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_scale_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_float, ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_scale_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], s: ctypes.c_float, inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_set_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), size_t, size_t, size_t, size_t, ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_set_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], nb1: size_t, nb2: size_t, nb3: size_t, offset: size_t, inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_cpy_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_cpy_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_cont_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_cont_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_view_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.POINTER(ctypes.c_int64), size_t], ctypes.POINTER(struct_ggml_tensor))
def ggml_view_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], n_dims: ctypes.c_int32, ne: ctypes._Pointer[ctypes.c_int64], offset: size_t) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_diag_mask_inf_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_diag_mask_inf_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], n_past: ctypes.c_int32, inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_diag_mask_zero_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_diag_mask_zero_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], n_past: ctypes.c_int32, inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_soft_max_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_float, ctypes.c_float, ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_soft_max_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], mask: ctypes._Pointer[struct_ggml_tensor], scale: ctypes.c_float, max_bias: ctypes.c_float, inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_soft_max_ext_back_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_float, ctypes.c_float, ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_soft_max_ext_back_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], scale: ctypes.c_float, max_bias: ctypes.c_float, inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_rope_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_rope_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], c: ctypes._Pointer[struct_ggml_tensor], n_dims: ctypes.c_int32, mode: ctypes.c_int32, n_ctx_orig: ctypes.c_int32, freq_base: ctypes.c_float, freq_scale: ctypes.c_float, ext_factor: ctypes.c_float, attn_factor: ctypes.c_float, beta_fast: ctypes.c_float, beta_slow: ctypes.c_float, inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_rope_yarn_corr_dim", [ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_float], ctypes.c_float)
def ggml_rope_yarn_corr_dim(n_dims: ctypes.c_int32, n_ctx_orig: ctypes.c_int32, n_rot: ctypes.c_float, base: ctypes.c_float) -> ctypes.c_float:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_calc_conv_output_size", [int64_t, int64_t, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32], int64_t)
def ggml_calc_conv_output_size(ins: int64_t, ks: int64_t, s: ctypes.c_int32, p: ctypes.c_int32, d: ctypes.c_int32) -> int64_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_calc_conv_transpose_1d_output_size", [int64_t, int64_t, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32], int64_t)
def ggml_calc_conv_transpose_1d_output_size(ins: int64_t, ks: int64_t, s: ctypes.c_int32, p: ctypes.c_int32, d: ctypes.c_int32) -> int64_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_calc_conv_transpose_output_size", [int64_t, int64_t, ctypes.c_int32, ctypes.c_int32], int64_t)
def ggml_calc_conv_transpose_output_size(ins: int64_t, ks: int64_t, s: ctypes.c_int32, p: ctypes.c_int32) -> int64_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_calc_pool_output_size", [int64_t, ctypes.c_int32, ctypes.c_int32, ctypes.c_float], int64_t)
def ggml_calc_pool_output_size(ins: int64_t, ks: ctypes.c_int32, s: ctypes.c_int32, p: ctypes.c_float) -> int64_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_upscale_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, ggml_scale_mode], ctypes.POINTER(struct_ggml_tensor))
def ggml_upscale_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], ne0: ctypes.c_int32, ne1: ctypes.c_int32, ne2: ctypes.c_int32, ne3: ctypes.c_int32, mode: ggml_scale_mode) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_add_rel_pos_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_add_rel_pos_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], pw: ctypes._Pointer[struct_ggml_tensor], ph: ctypes._Pointer[struct_ggml_tensor], inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_unary_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ggml_unary_op, ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_unary_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], op: ggml_unary_op, inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_map_custom1_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ggml_custom1_op_t, ctypes.c_int32, ctypes.POINTER(None), ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_map_custom1_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], fun: ggml_custom1_op_t, n_tasks: ctypes.c_int32, userdata: ctypes.c_void_p, inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_map_custom2_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ggml_custom2_op_t, ctypes.c_int32, ctypes.POINTER(None), ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_map_custom2_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], fun: ggml_custom2_op_t, n_tasks: ctypes.c_int32, userdata: ctypes.c_void_p, inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_map_custom3_impl", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ggml_custom3_op_t, ctypes.c_int32, ctypes.POINTER(None), ctypes.c_bool], ctypes.POINTER(struct_ggml_tensor))
def ggml_map_custom3_impl(ctx: ctypes._Pointer[struct_ggml_context], a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor], c: ctypes._Pointer[struct_ggml_tensor], fun: ggml_custom3_op_t, n_tasks: ctypes.c_int32, userdata: ctypes.c_void_p, inplace: ctypes.c_bool) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_hash_set_new", [size_t], struct_ggml_hash_set)
def ggml_hash_set_new(size: size_t) -> struct_ggml_hash_set:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_hash_set_reset", [ctypes.POINTER(struct_ggml_hash_set)], None)
def ggml_hash_set_reset(hash_set: ctypes._Pointer[struct_ggml_hash_set]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_hash_set_free", [ctypes.POINTER(struct_ggml_hash_set)], None)
def ggml_hash_set_free(hash_set: ctypes._Pointer[struct_ggml_hash_set]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_hash_size", [size_t], size_t)
def ggml_hash_size(min_sz: size_t) -> size_t:
    ...

class struct_hash_map(Structure):
    if TYPE_CHECKING:
        set: struct_ggml_hash_set
        vals: ctypes._Pointer[ctypes._Pointer[struct_ggml_tensor]]
struct_hash_map._pack_ = 1 # source:False
struct_hash_map._fields_ = [
    ('set', struct_ggml_hash_set),
    ('vals', ctypes.POINTER(ctypes.POINTER(struct_ggml_tensor))),
]

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_new_hash_map", [size_t], ctypes.POINTER(struct_hash_map))
def ggml_new_hash_map(size: size_t) -> ctypes._Pointer[struct_hash_map]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_hash_map_free", [ctypes.POINTER(struct_hash_map)], None)
def ggml_hash_map_free(map: ctypes._Pointer[struct_hash_map]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_add_or_set", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_cgraph), size_t, ctypes.POINTER(struct_ggml_tensor)], None)
def ggml_add_or_set(ctx: ctypes._Pointer[struct_ggml_context], cgraph: ctypes._Pointer[struct_ggml_cgraph], isrc: size_t, tensor: ctypes._Pointer[struct_ggml_tensor]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_acc_or_set", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_cgraph), size_t, ctypes.POINTER(struct_ggml_tensor), size_t, size_t, size_t, size_t], None)
def ggml_acc_or_set(ctx: ctypes._Pointer[struct_ggml_context], cgraph: ctypes._Pointer[struct_ggml_cgraph], isrc: size_t, tensor: ctypes._Pointer[struct_ggml_tensor], nb1: size_t, nb2: size_t, nb3: size_t, offset: size_t) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_add1_or_set", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_cgraph), size_t, ctypes.POINTER(struct_ggml_tensor)], None)
def ggml_add1_or_set(ctx: ctypes._Pointer[struct_ggml_context], cgraph: ctypes._Pointer[struct_ggml_cgraph], isrc: size_t, tensor: ctypes._Pointer[struct_ggml_tensor]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_sub_or_set", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_cgraph), size_t, ctypes.POINTER(struct_ggml_tensor)], None)
def ggml_sub_or_set(ctx: ctypes._Pointer[struct_ggml_context], cgraph: ctypes._Pointer[struct_ggml_cgraph], isrc: size_t, tensor: ctypes._Pointer[struct_ggml_tensor]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_compute_backward", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_cgraph), ctypes.c_int32, ctypes.POINTER(ctypes.c_bool)], None)
def ggml_compute_backward(ctx: ctypes._Pointer[struct_ggml_context], cgraph: ctypes._Pointer[struct_ggml_cgraph], i: ctypes.c_int32, grads_needed: ctypes._Pointer[ctypes.c_bool]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_visit_parents", [ctypes.POINTER(struct_ggml_cgraph), ctypes.POINTER(struct_ggml_tensor)], None)
def ggml_visit_parents(cgraph: ctypes._Pointer[struct_ggml_cgraph], node: ctypes._Pointer[struct_ggml_tensor]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_build_forward_impl", [ctypes.POINTER(struct_ggml_cgraph), ctypes.POINTER(struct_ggml_tensor), ctypes.c_bool], None)
def ggml_build_forward_impl(cgraph: ctypes._Pointer[struct_ggml_cgraph], tensor: ctypes._Pointer[struct_ggml_tensor], expand: ctypes.c_bool) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("incr_ptr_aligned", [ctypes.POINTER(ctypes.POINTER(None)), size_t, size_t], ctypes.POINTER(None))
def incr_ptr_aligned(p: ctypes._Pointer[ctypes.c_void_p], size: size_t, align: size_t) -> ctypes.c_void_p:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_graph_nbytes", [size_t, ctypes.c_bool], size_t)
def ggml_graph_nbytes(size: size_t, grads: ctypes.c_bool) -> size_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_graph_view", [ctypes.POINTER(struct_ggml_cgraph), ctypes.c_int32, ctypes.c_int32], struct_ggml_cgraph)
def ggml_graph_view(cgraph0: ctypes._Pointer[struct_ggml_cgraph], i0: ctypes.c_int32, i1: ctypes.c_int32) -> struct_ggml_cgraph:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_graph_find", [ctypes.POINTER(struct_ggml_cgraph), ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_graph_find(cgraph: ctypes._Pointer[struct_ggml_cgraph], node: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_graph_get_parent", [ctypes.POINTER(struct_ggml_cgraph), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_graph_get_parent(cgraph: ctypes._Pointer[struct_ggml_cgraph], node: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_graph_dump_dot_node_edge", [ctypes.POINTER(struct__IO_FILE), ctypes.POINTER(struct_ggml_cgraph), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(ctypes.c_char)], None)
def ggml_graph_dump_dot_node_edge(fp: ctypes._Pointer[struct__IO_FILE], gb: ctypes._Pointer[struct_ggml_cgraph], node: ctypes._Pointer[struct_ggml_tensor], parent: ctypes._Pointer[struct_ggml_tensor], label: ctypes._Pointer[ctypes.c_char]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_graph_dump_dot_leaf_edge", [ctypes.POINTER(struct__IO_FILE), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(ctypes.c_char)], None)
def ggml_graph_dump_dot_leaf_edge(fp: ctypes._Pointer[struct__IO_FILE], node: ctypes._Pointer[struct_ggml_tensor], parent: ctypes._Pointer[struct_ggml_tensor], label: ctypes._Pointer[ctypes.c_char]) -> None:
    ...

class struct__0(Structure):
    if TYPE_CHECKING:
        bits: ctypes.c_uint16
struct__0._pack_ = 1 # source:False
struct__0._fields_ = [
    ('bits', ctypes.c_uint16),
]

struct_ggml_backend_sched_split._pack_ = 1 # source:False
struct_ggml_backend_sched_split._fields_ = [
    ('backend_id', ctypes.c_int32),
    ('i_start', ctypes.c_int32),
    ('i_end', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('inputs', ctypes.POINTER(struct_ggml_tensor) * 10),
    ('n_inputs', ctypes.c_int32),
    ('PADDING_1', ctypes.c_ubyte * 4),
    ('graph', struct_ggml_cgraph),
]

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buffer_init", [ggml_backend_buffer_type_t, struct_ggml_backend_buffer_i, ctypes.POINTER(None), size_t], ggml_backend_buffer_t)
def ggml_backend_buffer_init(buft: ggml_backend_buffer_type_t, iface: struct_ggml_backend_buffer_i, context: ctypes.c_void_p, size: size_t) -> ggml_backend_buffer_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buffer_copy_tensor", [ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_backend_buffer_copy_tensor(src: ctypes._Pointer[struct_ggml_tensor], dst: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_multi_buffer_alloc_buffer", [ctypes.POINTER(ctypes.POINTER(struct_ggml_backend_buffer)), size_t], ggml_backend_buffer_t)
def ggml_backend_multi_buffer_alloc_buffer(buffers: ctypes._Pointer[ctypes._Pointer[struct_ggml_backend_buffer]], n_buffers: size_t) -> ggml_backend_buffer_t:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_buffer_is_multi_buffer", [ggml_backend_buffer_t], ctypes.c_bool)
def ggml_backend_buffer_is_multi_buffer(buffer: ggml_backend_buffer_t) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-base.so')("ggml_backend_multi_buffer_set_usage", [ggml_backend_buffer_t, ggml_backend_buffer_usage], None)
def ggml_backend_multi_buffer_set_usage(buffer: ggml_backend_buffer_t, usage: ggml_backend_buffer_usage) -> None:
    ...

@ctypes_function_for_shared_library('libggml.so')("ggml_backend_register", [ggml_backend_reg_t], None)
def ggml_backend_register(reg: ggml_backend_reg_t) -> None:
    ...

ggml_backend_init_t = ctypes.CFUNCTYPE(ctypes.POINTER(struct_ggml_backend_reg))
ggml_backend_score_t = ctypes.CFUNCTYPE(ctypes.c_int32)
@ctypes_function_for_shared_library('FIXME_STUB')("ggml_are_same_layout", [ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_are_same_layout(a: ctypes._Pointer[struct_ggml_tensor], b: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

class struct_ggml_backend_multi_buffer_context(Structure):
    if TYPE_CHECKING:
        buffers: ctypes._Pointer[ctypes._Pointer[struct_ggml_backend_buffer]]
        n_buffers: ctypes.c_uint64
struct_ggml_backend_multi_buffer_context._pack_ = 1 # source:False
struct_ggml_backend_multi_buffer_context._fields_ = [
    ('buffers', ctypes.POINTER(ctypes.POINTER(struct_ggml_backend_buffer))),
    ('n_buffers', ctypes.c_uint64),
]

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_multi_buffer_free_buffer", [ggml_backend_buffer_t], None)
def ggml_backend_multi_buffer_free_buffer(buffer: ggml_backend_buffer_t) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_multi_buffer_clear", [ggml_backend_buffer_t, uint8_t], None)
def ggml_backend_multi_buffer_clear(buffer: ggml_backend_buffer_t, value: uint8_t) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_dup_tensor_layout", [ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def ggml_dup_tensor_layout(ctx: ctypes._Pointer[struct_ggml_context], tensor: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_is_view_op", [ggml_op], ctypes.c_bool)
def ggml_is_view_op(op: ggml_op) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_sched_backend_id", [ggml_backend_sched_t, ggml_backend_t], ctypes.c_int32)
def ggml_backend_sched_backend_id(sched: ggml_backend_sched_t, backend: ggml_backend_t) -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_sched_backend_from_buffer", [ggml_backend_sched_t, ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.c_int32)
def ggml_backend_sched_backend_from_buffer(sched: ggml_backend_sched_t, tensor: ctypes._Pointer[struct_ggml_tensor], op: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_sched_backend_id_from_cur", [ggml_backend_sched_t, ctypes.POINTER(struct_ggml_tensor)], ctypes.c_int32)
def ggml_backend_sched_backend_id_from_cur(sched: ggml_backend_sched_t, tensor: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("fmt_size", [size_t], ctypes.POINTER(ctypes.c_char))
def fmt_size(size: size_t) -> ctypes._Pointer[ctypes.c_char]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_sched_print_assignments", [ggml_backend_sched_t, ctypes.POINTER(struct_ggml_cgraph)], None)
def ggml_backend_sched_print_assignments(sched: ggml_backend_sched_t, graph: ctypes._Pointer[struct_ggml_cgraph]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_sched_buffer_supported", [ggml_backend_sched_t, ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32], ctypes.c_bool)
def ggml_backend_sched_buffer_supported(sched: ggml_backend_sched_t, t: ctypes._Pointer[struct_ggml_tensor], backend_id: ctypes.c_int32) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_sched_set_if_supported", [ggml_backend_sched_t, ctypes.POINTER(struct_ggml_tensor), ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)], None)
def ggml_backend_sched_set_if_supported(sched: ggml_backend_sched_t, node: ctypes._Pointer[struct_ggml_tensor], cur_backend_id: ctypes.c_int32, node_backend_id: ctypes._Pointer[ctypes.c_int32]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_sched_split_graph", [ggml_backend_sched_t, ctypes.POINTER(struct_ggml_cgraph)], None)
def ggml_backend_sched_split_graph(sched: ggml_backend_sched_t, graph: ctypes._Pointer[struct_ggml_cgraph]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_sched_alloc_splits", [ggml_backend_sched_t], ctypes.c_bool)
def ggml_backend_sched_alloc_splits(sched: ggml_backend_sched_t) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_sched_compute_splits", [ggml_backend_sched_t], ggml_status)
def ggml_backend_sched_compute_splits(sched: ggml_backend_sched_t) -> ggml_status:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("graph_copy_dup_tensor", [struct_ggml_hash_set, ctypes.POINTER(ctypes.POINTER(struct_ggml_tensor)), ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_context), ctypes.POINTER(struct_ggml_tensor)], ctypes.POINTER(struct_ggml_tensor))
def graph_copy_dup_tensor(hash_set: struct_ggml_hash_set, node_copies: ctypes._Pointer[ctypes._Pointer[struct_ggml_tensor]], ctx_allocated: ctypes._Pointer[struct_ggml_context], ctx_unallocated: ctypes._Pointer[struct_ggml_context], src: ctypes._Pointer[struct_ggml_tensor]) -> ctypes._Pointer[struct_ggml_tensor]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("graph_copy_init_tensor", [ctypes.POINTER(struct_ggml_hash_set), ctypes.POINTER(ctypes.POINTER(struct_ggml_tensor)), ctypes.POINTER(ctypes.c_bool), ctypes.POINTER(struct_ggml_tensor)], None)
def graph_copy_init_tensor(hash_set: ctypes._Pointer[struct_ggml_hash_set], node_copies: ctypes._Pointer[ctypes._Pointer[struct_ggml_tensor]], node_init: ctypes._Pointer[ctypes.c_bool], src: ctypes._Pointer[struct_ggml_tensor]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_cpu_buffer_get_base", [ggml_backend_buffer_t], ctypes.POINTER(None))
def ggml_backend_cpu_buffer_get_base(buffer: ggml_backend_buffer_t) -> ctypes.c_void_p:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_cpu_buffer_free_buffer", [ggml_backend_buffer_t], None)
def ggml_backend_cpu_buffer_free_buffer(buffer: ggml_backend_buffer_t) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_cpu_buffer_memset_tensor", [ggml_backend_buffer_t, ctypes.POINTER(struct_ggml_tensor), uint8_t, size_t, size_t], None)
def ggml_backend_cpu_buffer_memset_tensor(buffer: ggml_backend_buffer_t, tensor: ctypes._Pointer[struct_ggml_tensor], value: uint8_t, offset: size_t, size: size_t) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_cpu_buffer_set_tensor", [ggml_backend_buffer_t, ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(None), size_t, size_t], None)
def ggml_backend_cpu_buffer_set_tensor(buffer: ggml_backend_buffer_t, tensor: ctypes._Pointer[struct_ggml_tensor], data: ctypes.c_void_p, offset: size_t, size: size_t) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_cpu_buffer_get_tensor", [ggml_backend_buffer_t, ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(None), size_t, size_t], None)
def ggml_backend_cpu_buffer_get_tensor(buffer: ggml_backend_buffer_t, tensor: ctypes._Pointer[struct_ggml_tensor], data: ctypes.c_void_p, offset: size_t, size: size_t) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_cpu_buffer_cpy_tensor", [ggml_backend_buffer_t, ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(struct_ggml_tensor)], ctypes.c_bool)
def ggml_backend_cpu_buffer_cpy_tensor(buffer: ggml_backend_buffer_t, src: ctypes._Pointer[struct_ggml_tensor], dst: ctypes._Pointer[struct_ggml_tensor]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_cpu_buffer_clear", [ggml_backend_buffer_t, uint8_t], None)
def ggml_backend_cpu_buffer_clear(buffer: ggml_backend_buffer_t, value: uint8_t) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_cpu_buffer_type_get_name", [ggml_backend_buffer_type_t], ctypes.POINTER(ctypes.c_char))
def ggml_backend_cpu_buffer_type_get_name(buft: ggml_backend_buffer_type_t) -> ctypes._Pointer[ctypes.c_char]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_cpu_buffer_type_alloc_buffer", [ggml_backend_buffer_type_t, size_t], ggml_backend_buffer_t)
def ggml_backend_cpu_buffer_type_alloc_buffer(buft: ggml_backend_buffer_type_t, size: size_t) -> ggml_backend_buffer_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_cpu_buffer_type_get_alignment", [ggml_backend_buffer_type_t], size_t)
def ggml_backend_cpu_buffer_type_get_alignment(buft: ggml_backend_buffer_type_t) -> size_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_cpu_buffer_type_is_host", [ggml_backend_buffer_type_t], ctypes.c_bool)
def ggml_backend_cpu_buffer_type_is_host(buft: ggml_backend_buffer_type_t) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_cpu_buffer_from_ptr_type_get_name", [ggml_backend_buffer_type_t], ctypes.POINTER(ctypes.c_char))
def ggml_backend_cpu_buffer_from_ptr_type_get_name(buft: ggml_backend_buffer_type_t) -> ctypes._Pointer[ctypes.c_char]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("ggml_backend_cpu_buffer_from_ptr_type", [], ggml_backend_buffer_type_t)
def ggml_backend_cpu_buffer_from_ptr_type() -> ggml_backend_buffer_type_t:
    ...

@ctypes_function_for_shared_library('libggml-cuda.so')("ggml_backend_cuda_init", [ctypes.c_int32], ggml_backend_t)
def ggml_backend_cuda_init(device: ctypes.c_int32) -> ggml_backend_t:
    ...

@ctypes_function_for_shared_library('libggml-cuda.so')("ggml_backend_is_cuda", [ggml_backend_t], ctypes.c_bool)
def ggml_backend_is_cuda(backend: ggml_backend_t) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-cuda.so')("ggml_backend_cuda_buffer_type", [ctypes.c_int32], ggml_backend_buffer_type_t)
def ggml_backend_cuda_buffer_type(device: ctypes.c_int32) -> ggml_backend_buffer_type_t:
    ...

@ctypes_function_for_shared_library('libggml-cuda.so')("ggml_backend_cuda_split_buffer_type", [ctypes.c_int32, ctypes.POINTER(ctypes.c_float)], ggml_backend_buffer_type_t)
def ggml_backend_cuda_split_buffer_type(main_device: ctypes.c_int32, tensor_split: ctypes._Pointer[ctypes.c_float]) -> ggml_backend_buffer_type_t:
    ...

@ctypes_function_for_shared_library('libggml-cuda.so')("ggml_backend_cuda_host_buffer_type", [], ggml_backend_buffer_type_t)
def ggml_backend_cuda_host_buffer_type() -> ggml_backend_buffer_type_t:
    ...

@ctypes_function_for_shared_library('libggml-cuda.so')("ggml_backend_cuda_get_device_count", [], ctypes.c_int32)
def ggml_backend_cuda_get_device_count() -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('libggml-cuda.so')("ggml_backend_cuda_get_device_description", [ctypes.c_int32, ctypes.POINTER(ctypes.c_char), size_t], None)
def ggml_backend_cuda_get_device_description(device: ctypes.c_int32, description: ctypes._Pointer[ctypes.c_char], description_size: size_t) -> None:
    ...

@ctypes_function_for_shared_library('libggml-cuda.so')("ggml_backend_cuda_get_device_memory", [ctypes.c_int32, ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)], None)
def ggml_backend_cuda_get_device_memory(device: ctypes.c_int32, free: ctypes._Pointer[ctypes.c_uint64], total: ctypes._Pointer[ctypes.c_uint64]) -> None:
    ...

@ctypes_function_for_shared_library('libggml-cuda.so')("ggml_backend_cuda_register_host_buffer", [ctypes.POINTER(None), size_t], ctypes.c_bool)
def ggml_backend_cuda_register_host_buffer(buffer: ctypes.c_void_p, size: size_t) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('libggml-cuda.so')("ggml_backend_cuda_unregister_host_buffer", [ctypes.POINTER(None)], None)
def ggml_backend_cuda_unregister_host_buffer(buffer: ctypes.c_void_p) -> None:
    ...

@ctypes_function_for_shared_library('libggml-cuda.so')("ggml_backend_cuda_reg", [], ggml_backend_reg_t)
def ggml_backend_cuda_reg() -> ggml_backend_reg_t:
    ...

class struct_llama_vocab(Structure):
    pass

class struct_llama_model(Structure):
    pass

class struct_llama_context(Structure):
    pass

class struct_llama_kv_cache(Structure):
    pass

llama_pos = ctypes.c_int32
llama_token = ctypes.c_int32
llama_seq_id = ctypes.c_int32

# values for enumeration 'llama_vocab_type'
llama_vocab_type__enumvalues = {
    0: 'LLAMA_VOCAB_TYPE_NONE',
    1: 'LLAMA_VOCAB_TYPE_SPM',
    2: 'LLAMA_VOCAB_TYPE_BPE',
    3: 'LLAMA_VOCAB_TYPE_WPM',
    4: 'LLAMA_VOCAB_TYPE_UGM',
    5: 'LLAMA_VOCAB_TYPE_RWKV',
}
LLAMA_VOCAB_TYPE_NONE = 0
LLAMA_VOCAB_TYPE_SPM = 1
LLAMA_VOCAB_TYPE_BPE = 2
LLAMA_VOCAB_TYPE_WPM = 3
LLAMA_VOCAB_TYPE_UGM = 4
LLAMA_VOCAB_TYPE_RWKV = 5
llama_vocab_type = ctypes.c_uint32 # enum

# values for enumeration 'llama_vocab_pre_type'
llama_vocab_pre_type__enumvalues = {
    0: 'LLAMA_VOCAB_PRE_TYPE_DEFAULT',
    1: 'LLAMA_VOCAB_PRE_TYPE_LLAMA3',
    2: 'LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM',
    3: 'LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER',
    4: 'LLAMA_VOCAB_PRE_TYPE_FALCON',
    5: 'LLAMA_VOCAB_PRE_TYPE_MPT',
    6: 'LLAMA_VOCAB_PRE_TYPE_STARCODER',
    7: 'LLAMA_VOCAB_PRE_TYPE_GPT2',
    8: 'LLAMA_VOCAB_PRE_TYPE_REFACT',
    9: 'LLAMA_VOCAB_PRE_TYPE_COMMAND_R',
    10: 'LLAMA_VOCAB_PRE_TYPE_STABLELM2',
    11: 'LLAMA_VOCAB_PRE_TYPE_QWEN2',
    12: 'LLAMA_VOCAB_PRE_TYPE_OLMO',
    13: 'LLAMA_VOCAB_PRE_TYPE_DBRX',
    14: 'LLAMA_VOCAB_PRE_TYPE_SMAUG',
    15: 'LLAMA_VOCAB_PRE_TYPE_PORO',
    16: 'LLAMA_VOCAB_PRE_TYPE_CHATGLM3',
    17: 'LLAMA_VOCAB_PRE_TYPE_CHATGLM4',
    18: 'LLAMA_VOCAB_PRE_TYPE_VIKING',
    19: 'LLAMA_VOCAB_PRE_TYPE_JAIS',
    20: 'LLAMA_VOCAB_PRE_TYPE_TEKKEN',
    21: 'LLAMA_VOCAB_PRE_TYPE_SMOLLM',
    22: 'LLAMA_VOCAB_PRE_TYPE_CODESHELL',
    23: 'LLAMA_VOCAB_PRE_TYPE_BLOOM',
    24: 'LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH',
    25: 'LLAMA_VOCAB_PRE_TYPE_EXAONE',
    26: 'LLAMA_VOCAB_PRE_TYPE_CHAMELEON',
    27: 'LLAMA_VOCAB_PRE_TYPE_MINERVA',
    28: 'LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM',
    29: 'LLAMA_VOCAB_PRE_TYPE_GPT4O',
    30: 'LLAMA_VOCAB_PRE_TYPE_SUPERBPE',
    31: 'LLAMA_VOCAB_PRE_TYPE_TRILLION',
    32: 'LLAMA_VOCAB_PRE_TYPE_BAILINGMOE',
    33: 'LLAMA_VOCAB_PRE_TYPE_LLAMA4',
    34: 'LLAMA_VOCAB_PRE_TYPE_PIXTRAL',
    35: 'LLAMA_VOCAB_PRE_TYPE_SEED_CODER',
}
LLAMA_VOCAB_PRE_TYPE_DEFAULT = 0
LLAMA_VOCAB_PRE_TYPE_LLAMA3 = 1
LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM = 2
LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER = 3
LLAMA_VOCAB_PRE_TYPE_FALCON = 4
LLAMA_VOCAB_PRE_TYPE_MPT = 5
LLAMA_VOCAB_PRE_TYPE_STARCODER = 6
LLAMA_VOCAB_PRE_TYPE_GPT2 = 7
LLAMA_VOCAB_PRE_TYPE_REFACT = 8
LLAMA_VOCAB_PRE_TYPE_COMMAND_R = 9
LLAMA_VOCAB_PRE_TYPE_STABLELM2 = 10
LLAMA_VOCAB_PRE_TYPE_QWEN2 = 11
LLAMA_VOCAB_PRE_TYPE_OLMO = 12
LLAMA_VOCAB_PRE_TYPE_DBRX = 13
LLAMA_VOCAB_PRE_TYPE_SMAUG = 14
LLAMA_VOCAB_PRE_TYPE_PORO = 15
LLAMA_VOCAB_PRE_TYPE_CHATGLM3 = 16
LLAMA_VOCAB_PRE_TYPE_CHATGLM4 = 17
LLAMA_VOCAB_PRE_TYPE_VIKING = 18
LLAMA_VOCAB_PRE_TYPE_JAIS = 19
LLAMA_VOCAB_PRE_TYPE_TEKKEN = 20
LLAMA_VOCAB_PRE_TYPE_SMOLLM = 21
LLAMA_VOCAB_PRE_TYPE_CODESHELL = 22
LLAMA_VOCAB_PRE_TYPE_BLOOM = 23
LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH = 24
LLAMA_VOCAB_PRE_TYPE_EXAONE = 25
LLAMA_VOCAB_PRE_TYPE_CHAMELEON = 26
LLAMA_VOCAB_PRE_TYPE_MINERVA = 27
LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM = 28
LLAMA_VOCAB_PRE_TYPE_GPT4O = 29
LLAMA_VOCAB_PRE_TYPE_SUPERBPE = 30
LLAMA_VOCAB_PRE_TYPE_TRILLION = 31
LLAMA_VOCAB_PRE_TYPE_BAILINGMOE = 32
LLAMA_VOCAB_PRE_TYPE_LLAMA4 = 33
LLAMA_VOCAB_PRE_TYPE_PIXTRAL = 34
LLAMA_VOCAB_PRE_TYPE_SEED_CODER = 35
llama_vocab_pre_type = ctypes.c_uint32 # enum

# values for enumeration 'llama_rope_type'
llama_rope_type__enumvalues = {
    -1: 'LLAMA_ROPE_TYPE_NONE',
    0: 'LLAMA_ROPE_TYPE_NORM',
    2: 'LLAMA_ROPE_TYPE_NEOX',
    8: 'LLAMA_ROPE_TYPE_MROPE',
    24: 'LLAMA_ROPE_TYPE_VISION',
}
LLAMA_ROPE_TYPE_NONE = -1
LLAMA_ROPE_TYPE_NORM = 0
LLAMA_ROPE_TYPE_NEOX = 2
LLAMA_ROPE_TYPE_MROPE = 8
LLAMA_ROPE_TYPE_VISION = 24
llama_rope_type = ctypes.c_int32 # enum

# values for enumeration 'llama_token_type'
llama_token_type__enumvalues = {
    0: 'LLAMA_TOKEN_TYPE_UNDEFINED',
    1: 'LLAMA_TOKEN_TYPE_NORMAL',
    2: 'LLAMA_TOKEN_TYPE_UNKNOWN',
    3: 'LLAMA_TOKEN_TYPE_CONTROL',
    4: 'LLAMA_TOKEN_TYPE_USER_DEFINED',
    5: 'LLAMA_TOKEN_TYPE_UNUSED',
    6: 'LLAMA_TOKEN_TYPE_BYTE',
}
LLAMA_TOKEN_TYPE_UNDEFINED = 0
LLAMA_TOKEN_TYPE_NORMAL = 1
LLAMA_TOKEN_TYPE_UNKNOWN = 2
LLAMA_TOKEN_TYPE_CONTROL = 3
LLAMA_TOKEN_TYPE_USER_DEFINED = 4
LLAMA_TOKEN_TYPE_UNUSED = 5
LLAMA_TOKEN_TYPE_BYTE = 6
llama_token_type = ctypes.c_uint32 # enum

# values for enumeration 'llama_token_attr'
llama_token_attr__enumvalues = {
    0: 'LLAMA_TOKEN_ATTR_UNDEFINED',
    1: 'LLAMA_TOKEN_ATTR_UNKNOWN',
    2: 'LLAMA_TOKEN_ATTR_UNUSED',
    4: 'LLAMA_TOKEN_ATTR_NORMAL',
    8: 'LLAMA_TOKEN_ATTR_CONTROL',
    16: 'LLAMA_TOKEN_ATTR_USER_DEFINED',
    32: 'LLAMA_TOKEN_ATTR_BYTE',
    64: 'LLAMA_TOKEN_ATTR_NORMALIZED',
    128: 'LLAMA_TOKEN_ATTR_LSTRIP',
    256: 'LLAMA_TOKEN_ATTR_RSTRIP',
    512: 'LLAMA_TOKEN_ATTR_SINGLE_WORD',
}
LLAMA_TOKEN_ATTR_UNDEFINED = 0
LLAMA_TOKEN_ATTR_UNKNOWN = 1
LLAMA_TOKEN_ATTR_UNUSED = 2
LLAMA_TOKEN_ATTR_NORMAL = 4
LLAMA_TOKEN_ATTR_CONTROL = 8
LLAMA_TOKEN_ATTR_USER_DEFINED = 16
LLAMA_TOKEN_ATTR_BYTE = 32
LLAMA_TOKEN_ATTR_NORMALIZED = 64
LLAMA_TOKEN_ATTR_LSTRIP = 128
LLAMA_TOKEN_ATTR_RSTRIP = 256
LLAMA_TOKEN_ATTR_SINGLE_WORD = 512
llama_token_attr = ctypes.c_uint32 # enum

# values for enumeration 'llama_ftype'
llama_ftype__enumvalues = {
    0: 'LLAMA_FTYPE_ALL_F32',
    1: 'LLAMA_FTYPE_MOSTLY_F16',
    2: 'LLAMA_FTYPE_MOSTLY_Q4_0',
    3: 'LLAMA_FTYPE_MOSTLY_Q4_1',
    7: 'LLAMA_FTYPE_MOSTLY_Q8_0',
    8: 'LLAMA_FTYPE_MOSTLY_Q5_0',
    9: 'LLAMA_FTYPE_MOSTLY_Q5_1',
    10: 'LLAMA_FTYPE_MOSTLY_Q2_K',
    11: 'LLAMA_FTYPE_MOSTLY_Q3_K_S',
    12: 'LLAMA_FTYPE_MOSTLY_Q3_K_M',
    13: 'LLAMA_FTYPE_MOSTLY_Q3_K_L',
    14: 'LLAMA_FTYPE_MOSTLY_Q4_K_S',
    15: 'LLAMA_FTYPE_MOSTLY_Q4_K_M',
    16: 'LLAMA_FTYPE_MOSTLY_Q5_K_S',
    17: 'LLAMA_FTYPE_MOSTLY_Q5_K_M',
    18: 'LLAMA_FTYPE_MOSTLY_Q6_K',
    19: 'LLAMA_FTYPE_MOSTLY_IQ2_XXS',
    20: 'LLAMA_FTYPE_MOSTLY_IQ2_XS',
    21: 'LLAMA_FTYPE_MOSTLY_Q2_K_S',
    22: 'LLAMA_FTYPE_MOSTLY_IQ3_XS',
    23: 'LLAMA_FTYPE_MOSTLY_IQ3_XXS',
    24: 'LLAMA_FTYPE_MOSTLY_IQ1_S',
    25: 'LLAMA_FTYPE_MOSTLY_IQ4_NL',
    26: 'LLAMA_FTYPE_MOSTLY_IQ3_S',
    27: 'LLAMA_FTYPE_MOSTLY_IQ3_M',
    28: 'LLAMA_FTYPE_MOSTLY_IQ2_S',
    29: 'LLAMA_FTYPE_MOSTLY_IQ2_M',
    30: 'LLAMA_FTYPE_MOSTLY_IQ4_XS',
    31: 'LLAMA_FTYPE_MOSTLY_IQ1_M',
    32: 'LLAMA_FTYPE_MOSTLY_BF16',
    36: 'LLAMA_FTYPE_MOSTLY_TQ1_0',
    37: 'LLAMA_FTYPE_MOSTLY_TQ2_0',
    1024: 'LLAMA_FTYPE_GUESSED',
}
LLAMA_FTYPE_ALL_F32 = 0
LLAMA_FTYPE_MOSTLY_F16 = 1
LLAMA_FTYPE_MOSTLY_Q4_0 = 2
LLAMA_FTYPE_MOSTLY_Q4_1 = 3
LLAMA_FTYPE_MOSTLY_Q8_0 = 7
LLAMA_FTYPE_MOSTLY_Q5_0 = 8
LLAMA_FTYPE_MOSTLY_Q5_1 = 9
LLAMA_FTYPE_MOSTLY_Q2_K = 10
LLAMA_FTYPE_MOSTLY_Q3_K_S = 11
LLAMA_FTYPE_MOSTLY_Q3_K_M = 12
LLAMA_FTYPE_MOSTLY_Q3_K_L = 13
LLAMA_FTYPE_MOSTLY_Q4_K_S = 14
LLAMA_FTYPE_MOSTLY_Q4_K_M = 15
LLAMA_FTYPE_MOSTLY_Q5_K_S = 16
LLAMA_FTYPE_MOSTLY_Q5_K_M = 17
LLAMA_FTYPE_MOSTLY_Q6_K = 18
LLAMA_FTYPE_MOSTLY_IQ2_XXS = 19
LLAMA_FTYPE_MOSTLY_IQ2_XS = 20
LLAMA_FTYPE_MOSTLY_Q2_K_S = 21
LLAMA_FTYPE_MOSTLY_IQ3_XS = 22
LLAMA_FTYPE_MOSTLY_IQ3_XXS = 23
LLAMA_FTYPE_MOSTLY_IQ1_S = 24
LLAMA_FTYPE_MOSTLY_IQ4_NL = 25
LLAMA_FTYPE_MOSTLY_IQ3_S = 26
LLAMA_FTYPE_MOSTLY_IQ3_M = 27
LLAMA_FTYPE_MOSTLY_IQ2_S = 28
LLAMA_FTYPE_MOSTLY_IQ2_M = 29
LLAMA_FTYPE_MOSTLY_IQ4_XS = 30
LLAMA_FTYPE_MOSTLY_IQ1_M = 31
LLAMA_FTYPE_MOSTLY_BF16 = 32
LLAMA_FTYPE_MOSTLY_TQ1_0 = 36
LLAMA_FTYPE_MOSTLY_TQ2_0 = 37
LLAMA_FTYPE_GUESSED = 1024
llama_ftype = ctypes.c_uint32 # enum

# values for enumeration 'llama_rope_scaling_type'
llama_rope_scaling_type__enumvalues = {
    -1: 'LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED',
    0: 'LLAMA_ROPE_SCALING_TYPE_NONE',
    1: 'LLAMA_ROPE_SCALING_TYPE_LINEAR',
    2: 'LLAMA_ROPE_SCALING_TYPE_YARN',
    3: 'LLAMA_ROPE_SCALING_TYPE_LONGROPE',
    3: 'LLAMA_ROPE_SCALING_TYPE_MAX_VALUE',
}
LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED = -1
LLAMA_ROPE_SCALING_TYPE_NONE = 0
LLAMA_ROPE_SCALING_TYPE_LINEAR = 1
LLAMA_ROPE_SCALING_TYPE_YARN = 2
LLAMA_ROPE_SCALING_TYPE_LONGROPE = 3
LLAMA_ROPE_SCALING_TYPE_MAX_VALUE = 3
llama_rope_scaling_type = ctypes.c_int32 # enum

# values for enumeration 'llama_pooling_type'
llama_pooling_type__enumvalues = {
    -1: 'LLAMA_POOLING_TYPE_UNSPECIFIED',
    0: 'LLAMA_POOLING_TYPE_NONE',
    1: 'LLAMA_POOLING_TYPE_MEAN',
    2: 'LLAMA_POOLING_TYPE_CLS',
    3: 'LLAMA_POOLING_TYPE_LAST',
    4: 'LLAMA_POOLING_TYPE_RANK',
}
LLAMA_POOLING_TYPE_UNSPECIFIED = -1
LLAMA_POOLING_TYPE_NONE = 0
LLAMA_POOLING_TYPE_MEAN = 1
LLAMA_POOLING_TYPE_CLS = 2
LLAMA_POOLING_TYPE_LAST = 3
LLAMA_POOLING_TYPE_RANK = 4
llama_pooling_type = ctypes.c_int32 # enum

# values for enumeration 'llama_attention_type'
llama_attention_type__enumvalues = {
    -1: 'LLAMA_ATTENTION_TYPE_UNSPECIFIED',
    0: 'LLAMA_ATTENTION_TYPE_CAUSAL',
    1: 'LLAMA_ATTENTION_TYPE_NON_CAUSAL',
}
LLAMA_ATTENTION_TYPE_UNSPECIFIED = -1
LLAMA_ATTENTION_TYPE_CAUSAL = 0
LLAMA_ATTENTION_TYPE_NON_CAUSAL = 1
llama_attention_type = ctypes.c_int32 # enum

# values for enumeration 'llama_split_mode'
llama_split_mode__enumvalues = {
    0: 'LLAMA_SPLIT_MODE_NONE',
    1: 'LLAMA_SPLIT_MODE_LAYER',
    2: 'LLAMA_SPLIT_MODE_ROW',
}
LLAMA_SPLIT_MODE_NONE = 0
LLAMA_SPLIT_MODE_LAYER = 1
LLAMA_SPLIT_MODE_ROW = 2
llama_split_mode = ctypes.c_uint32 # enum
class struct_llama_token_data(Structure):
    if TYPE_CHECKING:
        id: ctypes.c_int32
        logit: ctypes.c_float
        p: ctypes.c_float
struct_llama_token_data._pack_ = 1 # source:False
struct_llama_token_data._fields_ = [
    ('id', ctypes.c_int32),
    ('logit', ctypes.c_float),
    ('p', ctypes.c_float),
]

llama_token_data = struct_llama_token_data
class struct_llama_token_data_array(Structure):
    if TYPE_CHECKING:
        data: ctypes._Pointer[struct_llama_token_data]
        size: ctypes.c_uint64
        selected: ctypes.c_int64
        sorted: ctypes.c_bool
struct_llama_token_data_array._pack_ = 1 # source:False
struct_llama_token_data_array._fields_ = [
    ('data', ctypes.POINTER(struct_llama_token_data)),
    ('size', ctypes.c_uint64),
    ('selected', ctypes.c_int64),
    ('sorted', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 7),
]

llama_token_data_array = struct_llama_token_data_array
llama_progress_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_float, ctypes.POINTER(None))
class struct_llama_batch(Structure):
    if TYPE_CHECKING:
        n_tokens: ctypes.c_int32
        token: ctypes._Pointer[ctypes.c_int32]
        embd: ctypes._Pointer[ctypes.c_float]
        pos: ctypes._Pointer[ctypes.c_int32]
        n_seq_id: ctypes._Pointer[ctypes.c_int32]
        seq_id: ctypes._Pointer[ctypes._Pointer[ctypes.c_int32]]
        logits: ctypes._Pointer[ctypes.c_byte]
struct_llama_batch._pack_ = 1 # source:False
struct_llama_batch._fields_ = [
    ('n_tokens', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('token', ctypes.POINTER(ctypes.c_int32)),
    ('embd', ctypes.POINTER(ctypes.c_float)),
    ('pos', ctypes.POINTER(ctypes.c_int32)),
    ('n_seq_id', ctypes.POINTER(ctypes.c_int32)),
    ('seq_id', ctypes.POINTER(ctypes.POINTER(ctypes.c_int32))),
    ('logits', ctypes.POINTER(ctypes.c_byte)),
]

llama_batch = struct_llama_batch

# values for enumeration 'llama_model_kv_override_type'
llama_model_kv_override_type__enumvalues = {
    0: 'LLAMA_KV_OVERRIDE_TYPE_INT',
    1: 'LLAMA_KV_OVERRIDE_TYPE_FLOAT',
    2: 'LLAMA_KV_OVERRIDE_TYPE_BOOL',
    3: 'LLAMA_KV_OVERRIDE_TYPE_STR',
}
LLAMA_KV_OVERRIDE_TYPE_INT = 0
LLAMA_KV_OVERRIDE_TYPE_FLOAT = 1
LLAMA_KV_OVERRIDE_TYPE_BOOL = 2
LLAMA_KV_OVERRIDE_TYPE_STR = 3
llama_model_kv_override_type = ctypes.c_uint32 # enum
class struct_llama_model_kv_override(Structure):
    if TYPE_CHECKING:
        tag: llama_model_kv_override_type
        key: ctypes.Array[ctypes.c_char]
        anonymous_union: union_llama_model_kv_override_0
class union_llama_model_kv_override_0(Union):
    if TYPE_CHECKING:
        val_i64: ctypes.c_int64
        val_f64: ctypes.c_double
        val_bool: ctypes.c_bool
        val_str: ctypes.Array[ctypes.c_char]
union_llama_model_kv_override_0._pack_ = 1 # source:False
union_llama_model_kv_override_0._fields_ = [
    ('val_i64', ctypes.c_int64),
    ('val_f64', ctypes.c_double),
    ('val_bool', ctypes.c_bool),
    ('val_str', ctypes.c_char * 128),
]

struct_llama_model_kv_override._pack_ = 1 # source:False
struct_llama_model_kv_override._anonymous_ = ('_0',)
struct_llama_model_kv_override._fields_ = [
    ('tag', llama_model_kv_override_type),
    ('key', ctypes.c_char * 128),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('_0', union_llama_model_kv_override_0),
]

class struct_llama_model_tensor_buft_override(Structure):
    if TYPE_CHECKING:
        pattern: ctypes._Pointer[ctypes.c_char]
        buft: ctypes._Pointer[struct_ggml_backend_buffer_type]
struct_llama_model_tensor_buft_override._pack_ = 1 # source:False
struct_llama_model_tensor_buft_override._fields_ = [
    ('pattern', ctypes.POINTER(ctypes.c_char)),
    ('buft', ctypes.POINTER(struct_ggml_backend_buffer_type)),
]

class struct_llama_model_params(Structure):
    if TYPE_CHECKING:
        devices: ctypes._Pointer[ctypes._Pointer[struct_ggml_backend_device]]
        tensor_buft_overrides: ctypes._Pointer[struct_llama_model_tensor_buft_override]
        n_gpu_layers: ctypes.c_int32
        split_mode: llama_split_mode
        main_gpu: ctypes.c_int32
        tensor_split: ctypes._Pointer[ctypes.c_float]
        progress_callback: Callable[[ctypes.c_float, ctypes.c_void_p], ctypes.c_bool]
        progress_callback_user_data: ctypes.c_void_p
        kv_overrides: ctypes._Pointer[struct_llama_model_kv_override]
        vocab_only: ctypes.c_bool
        use_mmap: ctypes.c_bool
        use_mlock: ctypes.c_bool
        check_tensors: ctypes.c_bool
struct_llama_model_params._pack_ = 1 # source:False
struct_llama_model_params._fields_ = [
    ('devices', ctypes.POINTER(ctypes.POINTER(struct_ggml_backend_device))),
    ('tensor_buft_overrides', ctypes.POINTER(struct_llama_model_tensor_buft_override)),
    ('n_gpu_layers', ctypes.c_int32),
    ('split_mode', llama_split_mode),
    ('main_gpu', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('tensor_split', ctypes.POINTER(ctypes.c_float)),
    ('progress_callback', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_float, ctypes.POINTER(None))),
    ('progress_callback_user_data', ctypes.POINTER(None)),
    ('kv_overrides', ctypes.POINTER(struct_llama_model_kv_override)),
    ('vocab_only', ctypes.c_bool),
    ('use_mmap', ctypes.c_bool),
    ('use_mlock', ctypes.c_bool),
    ('check_tensors', ctypes.c_bool),
    ('PADDING_1', ctypes.c_ubyte * 4),
]

class struct_llama_context_params(Structure):
    if TYPE_CHECKING:
        n_ctx: ctypes.c_uint32
        n_batch: ctypes.c_uint32
        n_ubatch: ctypes.c_uint32
        n_seq_max: ctypes.c_uint32
        n_threads: ctypes.c_int32
        n_threads_batch: ctypes.c_int32
        rope_scaling_type: llama_rope_scaling_type
        pooling_type: llama_pooling_type
        attention_type: llama_attention_type
        rope_freq_base: ctypes.c_float
        rope_freq_scale: ctypes.c_float
        yarn_ext_factor: ctypes.c_float
        yarn_attn_factor: ctypes.c_float
        yarn_beta_fast: ctypes.c_float
        yarn_beta_slow: ctypes.c_float
        yarn_orig_ctx: ctypes.c_uint32
        defrag_thold: ctypes.c_float
        cb_eval: Callable[[ctypes._Pointer[struct_ggml_tensor], ctypes.c_bool, ctypes.c_void_p], ctypes.c_bool]
        cb_eval_user_data: ctypes.c_void_p
        type_k: ggml_type
        type_v: ggml_type
        abort_callback: Callable[[ctypes.c_void_p], ctypes.c_bool]
        abort_callback_data: ctypes.c_void_p
        embeddings: ctypes.c_bool
        offload_kqv: ctypes.c_bool
        flash_attn: ctypes.c_bool
        no_perf: ctypes.c_bool
        op_offload: ctypes.c_bool
        swa_full: ctypes.c_bool
struct_llama_context_params._pack_ = 1 # source:False
struct_llama_context_params._fields_ = [
    ('n_ctx', ctypes.c_uint32),
    ('n_batch', ctypes.c_uint32),
    ('n_ubatch', ctypes.c_uint32),
    ('n_seq_max', ctypes.c_uint32),
    ('n_threads', ctypes.c_int32),
    ('n_threads_batch', ctypes.c_int32),
    ('rope_scaling_type', llama_rope_scaling_type),
    ('pooling_type', llama_pooling_type),
    ('attention_type', llama_attention_type),
    ('rope_freq_base', ctypes.c_float),
    ('rope_freq_scale', ctypes.c_float),
    ('yarn_ext_factor', ctypes.c_float),
    ('yarn_attn_factor', ctypes.c_float),
    ('yarn_beta_fast', ctypes.c_float),
    ('yarn_beta_slow', ctypes.c_float),
    ('yarn_orig_ctx', ctypes.c_uint32),
    ('defrag_thold', ctypes.c_float),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('cb_eval', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_ggml_tensor), ctypes.c_bool, ctypes.POINTER(None))),
    ('cb_eval_user_data', ctypes.POINTER(None)),
    ('type_k', ggml_type),
    ('type_v', ggml_type),
    ('abort_callback', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(None))),
    ('abort_callback_data', ctypes.POINTER(None)),
    ('embeddings', ctypes.c_bool),
    ('offload_kqv', ctypes.c_bool),
    ('flash_attn', ctypes.c_bool),
    ('no_perf', ctypes.c_bool),
    ('op_offload', ctypes.c_bool),
    ('swa_full', ctypes.c_bool),
    ('PADDING_1', ctypes.c_ubyte * 2),
]

class struct_llama_model_quantize_params(Structure):
    if TYPE_CHECKING:
        nthread: ctypes.c_int32
        ftype: llama_ftype
        output_tensor_type: ggml_type
        token_embedding_type: ggml_type
        allow_requantize: ctypes.c_bool
        quantize_output_tensor: ctypes.c_bool
        only_copy: ctypes.c_bool
        pure: ctypes.c_bool
        keep_split: ctypes.c_bool
        imatrix: ctypes.c_void_p
        kv_overrides: ctypes.c_void_p
        tensor_types: ctypes.c_void_p
struct_llama_model_quantize_params._pack_ = 1 # source:False
struct_llama_model_quantize_params._fields_ = [
    ('nthread', ctypes.c_int32),
    ('ftype', llama_ftype),
    ('output_tensor_type', ggml_type),
    ('token_embedding_type', ggml_type),
    ('allow_requantize', ctypes.c_bool),
    ('quantize_output_tensor', ctypes.c_bool),
    ('only_copy', ctypes.c_bool),
    ('pure', ctypes.c_bool),
    ('keep_split', ctypes.c_bool),
    ('PADDING_0', ctypes.c_ubyte * 3),
    ('imatrix', ctypes.POINTER(None)),
    ('kv_overrides', ctypes.POINTER(None)),
    ('tensor_types', ctypes.POINTER(None)),
]

llama_model_quantize_params = struct_llama_model_quantize_params
class struct_llama_logit_bias(Structure):
    if TYPE_CHECKING:
        token: ctypes.c_int32
        bias: ctypes.c_float
struct_llama_logit_bias._pack_ = 1 # source:False
struct_llama_logit_bias._fields_ = [
    ('token', ctypes.c_int32),
    ('bias', ctypes.c_float),
]

llama_logit_bias = struct_llama_logit_bias
class struct_llama_sampler_chain_params(Structure):
    if TYPE_CHECKING:
        no_perf: ctypes.c_bool
struct_llama_sampler_chain_params._pack_ = 1 # source:False
struct_llama_sampler_chain_params._fields_ = [
    ('no_perf', ctypes.c_bool),
]

llama_sampler_chain_params = struct_llama_sampler_chain_params
class struct_llama_chat_message(Structure):
    if TYPE_CHECKING:
        role: ctypes._Pointer[ctypes.c_char]
        content: ctypes._Pointer[ctypes.c_char]
struct_llama_chat_message._pack_ = 1 # source:False
struct_llama_chat_message._fields_ = [
    ('role', ctypes.POINTER(ctypes.c_char)),
    ('content', ctypes.POINTER(ctypes.c_char)),
]

llama_chat_message = struct_llama_chat_message
class struct_llama_adapter_lora(Structure):
    pass

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_default_params", [], struct_llama_model_params)
def llama_model_default_params() -> struct_llama_model_params:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_context_default_params", [], struct_llama_context_params)
def llama_context_default_params() -> struct_llama_context_params:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_chain_default_params", [], struct_llama_sampler_chain_params)
def llama_sampler_chain_default_params() -> struct_llama_sampler_chain_params:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_quantize_default_params", [], struct_llama_model_quantize_params)
def llama_model_quantize_default_params() -> struct_llama_model_quantize_params:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_backend_init", [], None)
def llama_backend_init() -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_backend_free", [], None)
def llama_backend_free() -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_numa_init", [ggml_numa_strategy], None)
def llama_numa_init(numa: ggml_numa_strategy) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_attach_threadpool", [ctypes.POINTER(struct_llama_context), ggml_threadpool_t, ggml_threadpool_t], None)
def llama_attach_threadpool(ctx: ctypes._Pointer[struct_llama_context], threadpool: ggml_threadpool_t, threadpool_batch: ggml_threadpool_t) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_detach_threadpool", [ctypes.POINTER(struct_llama_context)], None)
def llama_detach_threadpool(ctx: ctypes._Pointer[struct_llama_context]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_load_model_from_file", [ctypes.POINTER(ctypes.c_char), struct_llama_model_params], ctypes.POINTER(struct_llama_model))
def llama_load_model_from_file(path_model: ctypes._Pointer[ctypes.c_char], params: struct_llama_model_params) -> ctypes._Pointer[struct_llama_model]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_load_from_file", [ctypes.POINTER(ctypes.c_char), struct_llama_model_params], ctypes.POINTER(struct_llama_model))
def llama_model_load_from_file(path_model: ctypes._Pointer[ctypes.c_char], params: struct_llama_model_params) -> ctypes._Pointer[struct_llama_model]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_load_from_splits", [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), size_t, struct_llama_model_params], ctypes.POINTER(struct_llama_model))
def llama_model_load_from_splits(paths: ctypes._Pointer[ctypes._Pointer[ctypes.c_char]], n_paths: size_t, params: struct_llama_model_params) -> ctypes._Pointer[struct_llama_model]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_save_to_file", [ctypes.POINTER(struct_llama_model), ctypes.POINTER(ctypes.c_char)], None)
def llama_model_save_to_file(model: ctypes._Pointer[struct_llama_model], path_model: ctypes._Pointer[ctypes.c_char]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_free_model", [ctypes.POINTER(struct_llama_model)], None)
def llama_free_model(model: ctypes._Pointer[struct_llama_model]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_free", [ctypes.POINTER(struct_llama_model)], None)
def llama_model_free(model: ctypes._Pointer[struct_llama_model]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_init_from_model", [ctypes.POINTER(struct_llama_model), struct_llama_context_params], ctypes.POINTER(struct_llama_context))
def llama_init_from_model(model: ctypes._Pointer[struct_llama_model], params: struct_llama_context_params) -> ctypes._Pointer[struct_llama_context]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_new_context_with_model", [ctypes.POINTER(struct_llama_model), struct_llama_context_params], ctypes.POINTER(struct_llama_context))
def llama_new_context_with_model(model: ctypes._Pointer[struct_llama_model], params: struct_llama_context_params) -> ctypes._Pointer[struct_llama_context]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_free", [ctypes.POINTER(struct_llama_context)], None)
def llama_free(ctx: ctypes._Pointer[struct_llama_context]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_time_us", [], int64_t)
def llama_time_us() -> int64_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_max_devices", [], size_t)
def llama_max_devices() -> size_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_max_parallel_sequences", [], size_t)
def llama_max_parallel_sequences() -> size_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_supports_mmap", [], ctypes.c_bool)
def llama_supports_mmap() -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_supports_mlock", [], ctypes.c_bool)
def llama_supports_mlock() -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_supports_gpu_offload", [], ctypes.c_bool)
def llama_supports_gpu_offload() -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_supports_rpc", [], ctypes.c_bool)
def llama_supports_rpc() -> ctypes.c_bool:
    ...

uint32_t = ctypes.c_uint32
@ctypes_function_for_shared_library('FIXME_STUB')("llama_n_ctx", [ctypes.POINTER(struct_llama_context)], uint32_t)
def llama_n_ctx(ctx: ctypes._Pointer[struct_llama_context]) -> uint32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_n_batch", [ctypes.POINTER(struct_llama_context)], uint32_t)
def llama_n_batch(ctx: ctypes._Pointer[struct_llama_context]) -> uint32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_n_ubatch", [ctypes.POINTER(struct_llama_context)], uint32_t)
def llama_n_ubatch(ctx: ctypes._Pointer[struct_llama_context]) -> uint32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_n_seq_max", [ctypes.POINTER(struct_llama_context)], uint32_t)
def llama_n_seq_max(ctx: ctypes._Pointer[struct_llama_context]) -> uint32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_n_ctx_train", [ctypes.POINTER(struct_llama_model)], int32_t)
def llama_n_ctx_train(model: ctypes._Pointer[struct_llama_model]) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_n_embd", [ctypes.POINTER(struct_llama_model)], int32_t)
def llama_n_embd(model: ctypes._Pointer[struct_llama_model]) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_n_layer", [ctypes.POINTER(struct_llama_model)], int32_t)
def llama_n_layer(model: ctypes._Pointer[struct_llama_model]) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_n_head", [ctypes.POINTER(struct_llama_model)], int32_t)
def llama_n_head(model: ctypes._Pointer[struct_llama_model]) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_n_vocab", [ctypes.POINTER(struct_llama_vocab)], int32_t)
def llama_n_vocab(vocab: ctypes._Pointer[struct_llama_vocab]) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_get_model", [ctypes.POINTER(struct_llama_context)], ctypes.POINTER(struct_llama_model))
def llama_get_model(ctx: ctypes._Pointer[struct_llama_context]) -> ctypes._Pointer[struct_llama_model]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_get_kv_self", [ctypes.POINTER(struct_llama_context)], ctypes.POINTER(struct_llama_kv_cache))
def llama_get_kv_self(ctx: ctypes._Pointer[struct_llama_context]) -> ctypes._Pointer[struct_llama_kv_cache]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_get_vocab", [ctypes.POINTER(struct_llama_model)], ctypes.POINTER(struct_llama_vocab))
def llama_model_get_vocab(model: ctypes._Pointer[struct_llama_model]) -> ctypes._Pointer[struct_llama_vocab]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_rope_type", [ctypes.POINTER(struct_llama_model)], llama_rope_type)
def llama_model_rope_type(model: ctypes._Pointer[struct_llama_model]) -> llama_rope_type:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_n_ctx_train", [ctypes.POINTER(struct_llama_model)], int32_t)
def llama_model_n_ctx_train(model: ctypes._Pointer[struct_llama_model]) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_n_embd", [ctypes.POINTER(struct_llama_model)], int32_t)
def llama_model_n_embd(model: ctypes._Pointer[struct_llama_model]) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_n_layer", [ctypes.POINTER(struct_llama_model)], int32_t)
def llama_model_n_layer(model: ctypes._Pointer[struct_llama_model]) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_n_head", [ctypes.POINTER(struct_llama_model)], int32_t)
def llama_model_n_head(model: ctypes._Pointer[struct_llama_model]) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_n_head_kv", [ctypes.POINTER(struct_llama_model)], int32_t)
def llama_model_n_head_kv(model: ctypes._Pointer[struct_llama_model]) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_rope_freq_scale_train", [ctypes.POINTER(struct_llama_model)], ctypes.c_float)
def llama_model_rope_freq_scale_train(model: ctypes._Pointer[struct_llama_model]) -> ctypes.c_float:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_vocab_n_tokens", [ctypes.POINTER(struct_llama_vocab)], int32_t)
def llama_vocab_n_tokens(vocab: ctypes._Pointer[struct_llama_vocab]) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_meta_val_str", [ctypes.POINTER(struct_llama_model), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), size_t], int32_t)
def llama_model_meta_val_str(model: ctypes._Pointer[struct_llama_model], key: ctypes._Pointer[ctypes.c_char], buf: ctypes._Pointer[ctypes.c_char], buf_size: size_t) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_meta_count", [ctypes.POINTER(struct_llama_model)], int32_t)
def llama_model_meta_count(model: ctypes._Pointer[struct_llama_model]) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_meta_key_by_index", [ctypes.POINTER(struct_llama_model), int32_t, ctypes.POINTER(ctypes.c_char), size_t], int32_t)
def llama_model_meta_key_by_index(model: ctypes._Pointer[struct_llama_model], i: int32_t, buf: ctypes._Pointer[ctypes.c_char], buf_size: size_t) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_meta_val_str_by_index", [ctypes.POINTER(struct_llama_model), int32_t, ctypes.POINTER(ctypes.c_char), size_t], int32_t)
def llama_model_meta_val_str_by_index(model: ctypes._Pointer[struct_llama_model], i: int32_t, buf: ctypes._Pointer[ctypes.c_char], buf_size: size_t) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_desc", [ctypes.POINTER(struct_llama_model), ctypes.POINTER(ctypes.c_char), size_t], int32_t)
def llama_model_desc(model: ctypes._Pointer[struct_llama_model], buf: ctypes._Pointer[ctypes.c_char], buf_size: size_t) -> int32_t:
    ...

uint64_t = ctypes.c_uint64
@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_size", [ctypes.POINTER(struct_llama_model)], uint64_t)
def llama_model_size(model: ctypes._Pointer[struct_llama_model]) -> uint64_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_chat_template", [ctypes.POINTER(struct_llama_model), ctypes.POINTER(ctypes.c_char)], ctypes.POINTER(ctypes.c_char))
def llama_model_chat_template(model: ctypes._Pointer[struct_llama_model], name: ctypes._Pointer[ctypes.c_char]) -> ctypes._Pointer[ctypes.c_char]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_n_params", [ctypes.POINTER(struct_llama_model)], uint64_t)
def llama_model_n_params(model: ctypes._Pointer[struct_llama_model]) -> uint64_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_has_encoder", [ctypes.POINTER(struct_llama_model)], ctypes.c_bool)
def llama_model_has_encoder(model: ctypes._Pointer[struct_llama_model]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_has_decoder", [ctypes.POINTER(struct_llama_model)], ctypes.c_bool)
def llama_model_has_decoder(model: ctypes._Pointer[struct_llama_model]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_decoder_start_token", [ctypes.POINTER(struct_llama_model)], llama_token)
def llama_model_decoder_start_token(model: ctypes._Pointer[struct_llama_model]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_is_recurrent", [ctypes.POINTER(struct_llama_model)], ctypes.c_bool)
def llama_model_is_recurrent(model: ctypes._Pointer[struct_llama_model]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_model_quantize", [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_llama_model_quantize_params)], uint32_t)
def llama_model_quantize(fname_inp: ctypes._Pointer[ctypes.c_char], fname_out: ctypes._Pointer[ctypes.c_char], params: ctypes._Pointer[struct_llama_model_quantize_params]) -> uint32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_adapter_lora_init", [ctypes.POINTER(struct_llama_model), ctypes.POINTER(ctypes.c_char)], ctypes.POINTER(struct_llama_adapter_lora))
def llama_adapter_lora_init(model: ctypes._Pointer[struct_llama_model], path_lora: ctypes._Pointer[ctypes.c_char]) -> ctypes._Pointer[struct_llama_adapter_lora]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_adapter_lora_free", [ctypes.POINTER(struct_llama_adapter_lora)], None)
def llama_adapter_lora_free(adapter: ctypes._Pointer[struct_llama_adapter_lora]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_set_adapter_lora", [ctypes.POINTER(struct_llama_context), ctypes.POINTER(struct_llama_adapter_lora), ctypes.c_float], int32_t)
def llama_set_adapter_lora(ctx: ctypes._Pointer[struct_llama_context], adapter: ctypes._Pointer[struct_llama_adapter_lora], scale: ctypes.c_float) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_rm_adapter_lora", [ctypes.POINTER(struct_llama_context), ctypes.POINTER(struct_llama_adapter_lora)], int32_t)
def llama_rm_adapter_lora(ctx: ctypes._Pointer[struct_llama_context], adapter: ctypes._Pointer[struct_llama_adapter_lora]) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_clear_adapter_lora", [ctypes.POINTER(struct_llama_context)], None)
def llama_clear_adapter_lora(ctx: ctypes._Pointer[struct_llama_context]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_apply_adapter_cvec", [ctypes.POINTER(struct_llama_context), ctypes.POINTER(ctypes.c_float), size_t, int32_t, int32_t, int32_t], int32_t)
def llama_apply_adapter_cvec(ctx: ctypes._Pointer[struct_llama_context], data: ctypes._Pointer[ctypes.c_float], len: size_t, n_embd: int32_t, il_start: int32_t, il_end: int32_t) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_kv_self_n_tokens", [ctypes.POINTER(struct_llama_context)], int32_t)
def llama_kv_self_n_tokens(ctx: ctypes._Pointer[struct_llama_context]) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_kv_self_used_cells", [ctypes.POINTER(struct_llama_context)], int32_t)
def llama_kv_self_used_cells(ctx: ctypes._Pointer[struct_llama_context]) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_kv_self_clear", [ctypes.POINTER(struct_llama_context)], None)
def llama_kv_self_clear(ctx: ctypes._Pointer[struct_llama_context]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_kv_self_seq_rm", [ctypes.POINTER(struct_llama_context), llama_seq_id, llama_pos, llama_pos], ctypes.c_bool)
def llama_kv_self_seq_rm(ctx: ctypes._Pointer[struct_llama_context], seq_id: llama_seq_id, p0: llama_pos, p1: llama_pos) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_kv_self_seq_cp", [ctypes.POINTER(struct_llama_context), llama_seq_id, llama_seq_id, llama_pos, llama_pos], None)
def llama_kv_self_seq_cp(ctx: ctypes._Pointer[struct_llama_context], seq_id_src: llama_seq_id, seq_id_dst: llama_seq_id, p0: llama_pos, p1: llama_pos) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_kv_self_seq_keep", [ctypes.POINTER(struct_llama_context), llama_seq_id], None)
def llama_kv_self_seq_keep(ctx: ctypes._Pointer[struct_llama_context], seq_id: llama_seq_id) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_kv_self_seq_add", [ctypes.POINTER(struct_llama_context), llama_seq_id, llama_pos, llama_pos, llama_pos], None)
def llama_kv_self_seq_add(ctx: ctypes._Pointer[struct_llama_context], seq_id: llama_seq_id, p0: llama_pos, p1: llama_pos, delta: llama_pos) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_kv_self_seq_div", [ctypes.POINTER(struct_llama_context), llama_seq_id, llama_pos, llama_pos, ctypes.c_int32], None)
def llama_kv_self_seq_div(ctx: ctypes._Pointer[struct_llama_context], seq_id: llama_seq_id, p0: llama_pos, p1: llama_pos, d: ctypes.c_int32) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_kv_self_seq_pos_min", [ctypes.POINTER(struct_llama_context), llama_seq_id], llama_pos)
def llama_kv_self_seq_pos_min(ctx: ctypes._Pointer[struct_llama_context], seq_id: llama_seq_id) -> llama_pos:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_kv_self_seq_pos_max", [ctypes.POINTER(struct_llama_context), llama_seq_id], llama_pos)
def llama_kv_self_seq_pos_max(ctx: ctypes._Pointer[struct_llama_context], seq_id: llama_seq_id) -> llama_pos:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_kv_self_defrag", [ctypes.POINTER(struct_llama_context)], None)
def llama_kv_self_defrag(ctx: ctypes._Pointer[struct_llama_context]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_kv_self_can_shift", [ctypes.POINTER(struct_llama_context)], ctypes.c_bool)
def llama_kv_self_can_shift(ctx: ctypes._Pointer[struct_llama_context]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_kv_self_update", [ctypes.POINTER(struct_llama_context)], None)
def llama_kv_self_update(ctx: ctypes._Pointer[struct_llama_context]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_state_get_size", [ctypes.POINTER(struct_llama_context)], size_t)
def llama_state_get_size(ctx: ctypes._Pointer[struct_llama_context]) -> size_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_get_state_size", [ctypes.POINTER(struct_llama_context)], size_t)
def llama_get_state_size(ctx: ctypes._Pointer[struct_llama_context]) -> size_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_state_get_data", [ctypes.POINTER(struct_llama_context), ctypes.POINTER(ctypes.c_ubyte), size_t], size_t)
def llama_state_get_data(ctx: ctypes._Pointer[struct_llama_context], dst: ctypes._Pointer[ctypes.c_ubyte], size: size_t) -> size_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_copy_state_data", [ctypes.POINTER(struct_llama_context), ctypes.POINTER(ctypes.c_ubyte)], size_t)
def llama_copy_state_data(ctx: ctypes._Pointer[struct_llama_context], dst: ctypes._Pointer[ctypes.c_ubyte]) -> size_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_state_set_data", [ctypes.POINTER(struct_llama_context), ctypes.POINTER(ctypes.c_ubyte), size_t], size_t)
def llama_state_set_data(ctx: ctypes._Pointer[struct_llama_context], src: ctypes._Pointer[ctypes.c_ubyte], size: size_t) -> size_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_set_state_data", [ctypes.POINTER(struct_llama_context), ctypes.POINTER(ctypes.c_ubyte)], size_t)
def llama_set_state_data(ctx: ctypes._Pointer[struct_llama_context], src: ctypes._Pointer[ctypes.c_ubyte]) -> size_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_state_load_file", [ctypes.POINTER(struct_llama_context), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_int32), size_t, ctypes.POINTER(ctypes.c_uint64)], ctypes.c_bool)
def llama_state_load_file(ctx: ctypes._Pointer[struct_llama_context], path_session: ctypes._Pointer[ctypes.c_char], tokens_out: ctypes._Pointer[ctypes.c_int32], n_token_capacity: size_t, n_token_count_out: ctypes._Pointer[ctypes.c_uint64]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_load_session_file", [ctypes.POINTER(struct_llama_context), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_int32), size_t, ctypes.POINTER(ctypes.c_uint64)], ctypes.c_bool)
def llama_load_session_file(ctx: ctypes._Pointer[struct_llama_context], path_session: ctypes._Pointer[ctypes.c_char], tokens_out: ctypes._Pointer[ctypes.c_int32], n_token_capacity: size_t, n_token_count_out: ctypes._Pointer[ctypes.c_uint64]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_state_save_file", [ctypes.POINTER(struct_llama_context), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_int32), size_t], ctypes.c_bool)
def llama_state_save_file(ctx: ctypes._Pointer[struct_llama_context], path_session: ctypes._Pointer[ctypes.c_char], tokens: ctypes._Pointer[ctypes.c_int32], n_token_count: size_t) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_save_session_file", [ctypes.POINTER(struct_llama_context), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_int32), size_t], ctypes.c_bool)
def llama_save_session_file(ctx: ctypes._Pointer[struct_llama_context], path_session: ctypes._Pointer[ctypes.c_char], tokens: ctypes._Pointer[ctypes.c_int32], n_token_count: size_t) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_state_seq_get_size", [ctypes.POINTER(struct_llama_context), llama_seq_id], size_t)
def llama_state_seq_get_size(ctx: ctypes._Pointer[struct_llama_context], seq_id: llama_seq_id) -> size_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_state_seq_get_data", [ctypes.POINTER(struct_llama_context), ctypes.POINTER(ctypes.c_ubyte), size_t, llama_seq_id], size_t)
def llama_state_seq_get_data(ctx: ctypes._Pointer[struct_llama_context], dst: ctypes._Pointer[ctypes.c_ubyte], size: size_t, seq_id: llama_seq_id) -> size_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_state_seq_set_data", [ctypes.POINTER(struct_llama_context), ctypes.POINTER(ctypes.c_ubyte), size_t, llama_seq_id], size_t)
def llama_state_seq_set_data(ctx: ctypes._Pointer[struct_llama_context], src: ctypes._Pointer[ctypes.c_ubyte], size: size_t, dest_seq_id: llama_seq_id) -> size_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_state_seq_save_file", [ctypes.POINTER(struct_llama_context), ctypes.POINTER(ctypes.c_char), llama_seq_id, ctypes.POINTER(ctypes.c_int32), size_t], size_t)
def llama_state_seq_save_file(ctx: ctypes._Pointer[struct_llama_context], filepath: ctypes._Pointer[ctypes.c_char], seq_id: llama_seq_id, tokens: ctypes._Pointer[ctypes.c_int32], n_token_count: size_t) -> size_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_state_seq_load_file", [ctypes.POINTER(struct_llama_context), ctypes.POINTER(ctypes.c_char), llama_seq_id, ctypes.POINTER(ctypes.c_int32), size_t, ctypes.POINTER(ctypes.c_uint64)], size_t)
def llama_state_seq_load_file(ctx: ctypes._Pointer[struct_llama_context], filepath: ctypes._Pointer[ctypes.c_char], dest_seq_id: llama_seq_id, tokens_out: ctypes._Pointer[ctypes.c_int32], n_token_capacity: size_t, n_token_count_out: ctypes._Pointer[ctypes.c_uint64]) -> size_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_batch_get_one", [ctypes.POINTER(ctypes.c_int32), int32_t], struct_llama_batch)
def llama_batch_get_one(tokens: ctypes._Pointer[ctypes.c_int32], n_tokens: int32_t) -> struct_llama_batch:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_batch_init", [int32_t, int32_t, int32_t], struct_llama_batch)
def llama_batch_init(n_tokens: int32_t, embd: int32_t, n_seq_max: int32_t) -> struct_llama_batch:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_batch_free", [struct_llama_batch], None)
def llama_batch_free(batch: struct_llama_batch) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_encode", [ctypes.POINTER(struct_llama_context), struct_llama_batch], int32_t)
def llama_encode(ctx: ctypes._Pointer[struct_llama_context], batch: struct_llama_batch) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_decode", [ctypes.POINTER(struct_llama_context), struct_llama_batch], int32_t)
def llama_decode(ctx: ctypes._Pointer[struct_llama_context], batch: struct_llama_batch) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_set_n_threads", [ctypes.POINTER(struct_llama_context), int32_t, int32_t], None)
def llama_set_n_threads(ctx: ctypes._Pointer[struct_llama_context], n_threads: int32_t, n_threads_batch: int32_t) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_n_threads", [ctypes.POINTER(struct_llama_context)], int32_t)
def llama_n_threads(ctx: ctypes._Pointer[struct_llama_context]) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_n_threads_batch", [ctypes.POINTER(struct_llama_context)], int32_t)
def llama_n_threads_batch(ctx: ctypes._Pointer[struct_llama_context]) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_set_embeddings", [ctypes.POINTER(struct_llama_context), ctypes.c_bool], None)
def llama_set_embeddings(ctx: ctypes._Pointer[struct_llama_context], embeddings: ctypes.c_bool) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_set_causal_attn", [ctypes.POINTER(struct_llama_context), ctypes.c_bool], None)
def llama_set_causal_attn(ctx: ctypes._Pointer[struct_llama_context], causal_attn: ctypes.c_bool) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_set_warmup", [ctypes.POINTER(struct_llama_context), ctypes.c_bool], None)
def llama_set_warmup(ctx: ctypes._Pointer[struct_llama_context], warmup: ctypes.c_bool) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_set_abort_callback", [ctypes.POINTER(struct_llama_context), ggml_abort_callback, ctypes.POINTER(None)], None)
def llama_set_abort_callback(ctx: ctypes._Pointer[struct_llama_context], abort_callback: ggml_abort_callback, abort_callback_data: ctypes.c_void_p) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_synchronize", [ctypes.POINTER(struct_llama_context)], None)
def llama_synchronize(ctx: ctypes._Pointer[struct_llama_context]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_get_logits", [ctypes.POINTER(struct_llama_context)], ctypes.POINTER(ctypes.c_float))
def llama_get_logits(ctx: ctypes._Pointer[struct_llama_context]) -> ctypes._Pointer[ctypes.c_float]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_get_logits_ith", [ctypes.POINTER(struct_llama_context), int32_t], ctypes.POINTER(ctypes.c_float))
def llama_get_logits_ith(ctx: ctypes._Pointer[struct_llama_context], i: int32_t) -> ctypes._Pointer[ctypes.c_float]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_get_embeddings", [ctypes.POINTER(struct_llama_context)], ctypes.POINTER(ctypes.c_float))
def llama_get_embeddings(ctx: ctypes._Pointer[struct_llama_context]) -> ctypes._Pointer[ctypes.c_float]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_get_embeddings_ith", [ctypes.POINTER(struct_llama_context), int32_t], ctypes.POINTER(ctypes.c_float))
def llama_get_embeddings_ith(ctx: ctypes._Pointer[struct_llama_context], i: int32_t) -> ctypes._Pointer[ctypes.c_float]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_get_embeddings_seq", [ctypes.POINTER(struct_llama_context), llama_seq_id], ctypes.POINTER(ctypes.c_float))
def llama_get_embeddings_seq(ctx: ctypes._Pointer[struct_llama_context], seq_id: llama_seq_id) -> ctypes._Pointer[ctypes.c_float]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_vocab_get_text", [ctypes.POINTER(struct_llama_vocab), llama_token], ctypes.POINTER(ctypes.c_char))
def llama_vocab_get_text(vocab: ctypes._Pointer[struct_llama_vocab], token: llama_token) -> ctypes._Pointer[ctypes.c_char]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_vocab_get_score", [ctypes.POINTER(struct_llama_vocab), llama_token], ctypes.c_float)
def llama_vocab_get_score(vocab: ctypes._Pointer[struct_llama_vocab], token: llama_token) -> ctypes.c_float:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_vocab_get_attr", [ctypes.POINTER(struct_llama_vocab), llama_token], llama_token_attr)
def llama_vocab_get_attr(vocab: ctypes._Pointer[struct_llama_vocab], token: llama_token) -> llama_token_attr:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_vocab_is_eog", [ctypes.POINTER(struct_llama_vocab), llama_token], ctypes.c_bool)
def llama_vocab_is_eog(vocab: ctypes._Pointer[struct_llama_vocab], token: llama_token) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_vocab_is_control", [ctypes.POINTER(struct_llama_vocab), llama_token], ctypes.c_bool)
def llama_vocab_is_control(vocab: ctypes._Pointer[struct_llama_vocab], token: llama_token) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_vocab_bos", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_vocab_bos(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_vocab_eos", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_vocab_eos(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_vocab_eot", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_vocab_eot(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_vocab_sep", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_vocab_sep(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_vocab_nl", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_vocab_nl(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_vocab_pad", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_vocab_pad(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_vocab_get_add_bos", [ctypes.POINTER(struct_llama_vocab)], ctypes.c_bool)
def llama_vocab_get_add_bos(vocab: ctypes._Pointer[struct_llama_vocab]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_vocab_get_add_eos", [ctypes.POINTER(struct_llama_vocab)], ctypes.c_bool)
def llama_vocab_get_add_eos(vocab: ctypes._Pointer[struct_llama_vocab]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_vocab_fim_pre", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_vocab_fim_pre(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_vocab_fim_suf", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_vocab_fim_suf(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_vocab_fim_mid", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_vocab_fim_mid(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_vocab_fim_pad", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_vocab_fim_pad(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_vocab_fim_rep", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_vocab_fim_rep(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_vocab_fim_sep", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_vocab_fim_sep(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_token_get_text", [ctypes.POINTER(struct_llama_vocab), llama_token], ctypes.POINTER(ctypes.c_char))
def llama_token_get_text(vocab: ctypes._Pointer[struct_llama_vocab], token: llama_token) -> ctypes._Pointer[ctypes.c_char]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_token_get_score", [ctypes.POINTER(struct_llama_vocab), llama_token], ctypes.c_float)
def llama_token_get_score(vocab: ctypes._Pointer[struct_llama_vocab], token: llama_token) -> ctypes.c_float:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_token_get_attr", [ctypes.POINTER(struct_llama_vocab), llama_token], llama_token_attr)
def llama_token_get_attr(vocab: ctypes._Pointer[struct_llama_vocab], token: llama_token) -> llama_token_attr:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_token_is_eog", [ctypes.POINTER(struct_llama_vocab), llama_token], ctypes.c_bool)
def llama_token_is_eog(vocab: ctypes._Pointer[struct_llama_vocab], token: llama_token) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_token_is_control", [ctypes.POINTER(struct_llama_vocab), llama_token], ctypes.c_bool)
def llama_token_is_control(vocab: ctypes._Pointer[struct_llama_vocab], token: llama_token) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_token_bos", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_token_bos(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_token_eos", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_token_eos(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_token_eot", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_token_eot(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_token_cls", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_token_cls(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_token_sep", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_token_sep(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_token_nl", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_token_nl(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_token_pad", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_token_pad(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_add_bos_token", [ctypes.POINTER(struct_llama_vocab)], ctypes.c_bool)
def llama_add_bos_token(vocab: ctypes._Pointer[struct_llama_vocab]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_add_eos_token", [ctypes.POINTER(struct_llama_vocab)], ctypes.c_bool)
def llama_add_eos_token(vocab: ctypes._Pointer[struct_llama_vocab]) -> ctypes.c_bool:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_token_fim_pre", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_token_fim_pre(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_token_fim_suf", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_token_fim_suf(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_token_fim_mid", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_token_fim_mid(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_token_fim_pad", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_token_fim_pad(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_token_fim_rep", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_token_fim_rep(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_token_fim_sep", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_token_fim_sep(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_vocab_cls", [ctypes.POINTER(struct_llama_vocab)], llama_token)
def llama_vocab_cls(vocab: ctypes._Pointer[struct_llama_vocab]) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_tokenize", [ctypes.POINTER(struct_llama_vocab), ctypes.POINTER(ctypes.c_char), int32_t, ctypes.POINTER(ctypes.c_int32), int32_t, ctypes.c_bool, ctypes.c_bool], int32_t)
def llama_tokenize(vocab: ctypes._Pointer[struct_llama_vocab], text: ctypes._Pointer[ctypes.c_char], text_len: int32_t, tokens: ctypes._Pointer[ctypes.c_int32], n_tokens_max: int32_t, add_special: ctypes.c_bool, parse_special: ctypes.c_bool) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_token_to_piece", [ctypes.POINTER(struct_llama_vocab), llama_token, ctypes.POINTER(ctypes.c_char), int32_t, int32_t, ctypes.c_bool], int32_t)
def llama_token_to_piece(vocab: ctypes._Pointer[struct_llama_vocab], token: llama_token, buf: ctypes._Pointer[ctypes.c_char], length: int32_t, lstrip: int32_t, special: ctypes.c_bool) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_detokenize", [ctypes.POINTER(struct_llama_vocab), ctypes.POINTER(ctypes.c_int32), int32_t, ctypes.POINTER(ctypes.c_char), int32_t, ctypes.c_bool, ctypes.c_bool], int32_t)
def llama_detokenize(vocab: ctypes._Pointer[struct_llama_vocab], tokens: ctypes._Pointer[ctypes.c_int32], n_tokens: int32_t, text: ctypes._Pointer[ctypes.c_char], text_len_max: int32_t, remove_special: ctypes.c_bool, unparse_special: ctypes.c_bool) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_chat_apply_template", [ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_llama_chat_message), size_t, ctypes.c_bool, ctypes.POINTER(ctypes.c_char), int32_t], int32_t)
def llama_chat_apply_template(tmpl: ctypes._Pointer[ctypes.c_char], chat: ctypes._Pointer[struct_llama_chat_message], n_msg: size_t, add_ass: ctypes.c_bool, buf: ctypes._Pointer[ctypes.c_char], length: int32_t) -> int32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_chat_builtin_templates", [ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), size_t], int32_t)
def llama_chat_builtin_templates(output: ctypes._Pointer[ctypes._Pointer[ctypes.c_char]], len: size_t) -> int32_t:
    ...

llama_sampler_context_t = ctypes.POINTER(None)
class struct_llama_sampler_i(Structure):
    if TYPE_CHECKING:
        name: Callable[[ctypes._Pointer[struct_llama_sampler]], ctypes._Pointer[ctypes.c_char]]
        accept: Callable[[ctypes._Pointer[struct_llama_sampler], ctypes.c_int32], None]
        apply: Callable[[ctypes._Pointer[struct_llama_sampler], ctypes._Pointer[struct_llama_token_data_array]], None]
        reset: Callable[[ctypes._Pointer[struct_llama_sampler]], None]
        clone: Callable[[ctypes._Pointer[struct_llama_sampler]], ctypes._Pointer[struct_llama_sampler]]
        free: Callable[[ctypes._Pointer[struct_llama_sampler]], None]
class struct_llama_sampler(Structure):
    if TYPE_CHECKING:
        iface: ctypes._Pointer[struct_llama_sampler_i]
        ctx: ctypes.c_void_p
struct_llama_sampler_i._pack_ = 1 # source:False
struct_llama_sampler_i._fields_ = [
    ('name', ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.c_char), ctypes.POINTER(struct_llama_sampler))),
    ('accept', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_llama_sampler), ctypes.c_int32)),
    ('apply', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_llama_sampler), ctypes.POINTER(struct_llama_token_data_array))),
    ('reset', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_llama_sampler))),
    ('clone', ctypes.CFUNCTYPE(ctypes.POINTER(struct_llama_sampler), ctypes.POINTER(struct_llama_sampler))),
    ('free', ctypes.CFUNCTYPE(None, ctypes.POINTER(struct_llama_sampler))),
]

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_init", [ctypes.POINTER(struct_llama_sampler_i), llama_sampler_context_t], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_init(iface: ctypes._Pointer[struct_llama_sampler_i], ctx: llama_sampler_context_t) -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_name", [ctypes.POINTER(struct_llama_sampler)], ctypes.POINTER(ctypes.c_char))
def llama_sampler_name(smpl: ctypes._Pointer[struct_llama_sampler]) -> ctypes._Pointer[ctypes.c_char]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_accept", [ctypes.POINTER(struct_llama_sampler), llama_token], None)
def llama_sampler_accept(smpl: ctypes._Pointer[struct_llama_sampler], token: llama_token) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_apply", [ctypes.POINTER(struct_llama_sampler), ctypes.POINTER(struct_llama_token_data_array)], None)
def llama_sampler_apply(smpl: ctypes._Pointer[struct_llama_sampler], cur_p: ctypes._Pointer[struct_llama_token_data_array]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_reset", [ctypes.POINTER(struct_llama_sampler)], None)
def llama_sampler_reset(smpl: ctypes._Pointer[struct_llama_sampler]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_clone", [ctypes.POINTER(struct_llama_sampler)], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_clone(smpl: ctypes._Pointer[struct_llama_sampler]) -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_free", [ctypes.POINTER(struct_llama_sampler)], None)
def llama_sampler_free(smpl: ctypes._Pointer[struct_llama_sampler]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_chain_init", [struct_llama_sampler_chain_params], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_chain_init(params: struct_llama_sampler_chain_params) -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_chain_add", [ctypes.POINTER(struct_llama_sampler), ctypes.POINTER(struct_llama_sampler)], None)
def llama_sampler_chain_add(chain: ctypes._Pointer[struct_llama_sampler], smpl: ctypes._Pointer[struct_llama_sampler]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_chain_get", [ctypes.POINTER(struct_llama_sampler), int32_t], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_chain_get(chain: ctypes._Pointer[struct_llama_sampler], i: int32_t) -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_chain_n", [ctypes.POINTER(struct_llama_sampler)], ctypes.c_int32)
def llama_sampler_chain_n(chain: ctypes._Pointer[struct_llama_sampler]) -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_chain_remove", [ctypes.POINTER(struct_llama_sampler), int32_t], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_chain_remove(chain: ctypes._Pointer[struct_llama_sampler], i: int32_t) -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_init_greedy", [], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_init_greedy() -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_init_dist", [uint32_t], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_init_dist(seed: uint32_t) -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_init_softmax", [], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_init_softmax() -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_init_top_k", [int32_t], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_init_top_k(k: int32_t) -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_init_top_p", [ctypes.c_float, size_t], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_init_top_p(p: ctypes.c_float, min_keep: size_t) -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_init_min_p", [ctypes.c_float, size_t], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_init_min_p(p: ctypes.c_float, min_keep: size_t) -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_init_typical", [ctypes.c_float, size_t], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_init_typical(p: ctypes.c_float, min_keep: size_t) -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_init_temp", [ctypes.c_float], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_init_temp(t: ctypes.c_float) -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_init_temp_ext", [ctypes.c_float, ctypes.c_float, ctypes.c_float], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_init_temp_ext(t: ctypes.c_float, delta: ctypes.c_float, exponent: ctypes.c_float) -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_init_xtc", [ctypes.c_float, ctypes.c_float, size_t, uint32_t], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_init_xtc(p: ctypes.c_float, t: ctypes.c_float, min_keep: size_t, seed: uint32_t) -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_init_top_n_sigma", [ctypes.c_float], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_init_top_n_sigma(n: ctypes.c_float) -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_init_mirostat", [int32_t, uint32_t, ctypes.c_float, ctypes.c_float, int32_t], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_init_mirostat(n_vocab: int32_t, seed: uint32_t, tau: ctypes.c_float, eta: ctypes.c_float, m: int32_t) -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_init_mirostat_v2", [uint32_t, ctypes.c_float, ctypes.c_float], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_init_mirostat_v2(seed: uint32_t, tau: ctypes.c_float, eta: ctypes.c_float) -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_init_grammar", [ctypes.POINTER(struct_llama_vocab), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char)], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_init_grammar(vocab: ctypes._Pointer[struct_llama_vocab], grammar_str: ctypes._Pointer[ctypes.c_char], grammar_root: ctypes._Pointer[ctypes.c_char]) -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_init_grammar_lazy", [ctypes.POINTER(struct_llama_vocab), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), size_t, ctypes.POINTER(ctypes.c_int32), size_t], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_init_grammar_lazy(vocab: ctypes._Pointer[struct_llama_vocab], grammar_str: ctypes._Pointer[ctypes.c_char], grammar_root: ctypes._Pointer[ctypes.c_char], trigger_words: ctypes._Pointer[ctypes._Pointer[ctypes.c_char]], num_trigger_words: size_t, trigger_tokens: ctypes._Pointer[ctypes.c_int32], num_trigger_tokens: size_t) -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_init_grammar_lazy_patterns", [ctypes.POINTER(struct_llama_vocab), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.c_char), ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), size_t, ctypes.POINTER(ctypes.c_int32), size_t], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_init_grammar_lazy_patterns(vocab: ctypes._Pointer[struct_llama_vocab], grammar_str: ctypes._Pointer[ctypes.c_char], grammar_root: ctypes._Pointer[ctypes.c_char], trigger_patterns: ctypes._Pointer[ctypes._Pointer[ctypes.c_char]], num_trigger_patterns: size_t, trigger_tokens: ctypes._Pointer[ctypes.c_int32], num_trigger_tokens: size_t) -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_init_penalties", [int32_t, ctypes.c_float, ctypes.c_float, ctypes.c_float], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_init_penalties(penalty_last_n: int32_t, penalty_repeat: ctypes.c_float, penalty_freq: ctypes.c_float, penalty_present: ctypes.c_float) -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_init_dry", [ctypes.POINTER(struct_llama_vocab), int32_t, ctypes.c_float, ctypes.c_float, int32_t, int32_t, ctypes.POINTER(ctypes.POINTER(ctypes.c_char)), size_t], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_init_dry(vocab: ctypes._Pointer[struct_llama_vocab], n_ctx_train: int32_t, dry_multiplier: ctypes.c_float, dry_base: ctypes.c_float, dry_allowed_length: int32_t, dry_penalty_last_n: int32_t, seq_breakers: ctypes._Pointer[ctypes._Pointer[ctypes.c_char]], num_breakers: size_t) -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_init_logit_bias", [int32_t, int32_t, ctypes.POINTER(struct_llama_logit_bias)], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_init_logit_bias(n_vocab: int32_t, n_logit_bias: int32_t, logit_bias: ctypes._Pointer[struct_llama_logit_bias]) -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_init_infill", [ctypes.POINTER(struct_llama_vocab)], ctypes.POINTER(struct_llama_sampler))
def llama_sampler_init_infill(vocab: ctypes._Pointer[struct_llama_vocab]) -> ctypes._Pointer[struct_llama_sampler]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_get_seed", [ctypes.POINTER(struct_llama_sampler)], uint32_t)
def llama_sampler_get_seed(smpl: ctypes._Pointer[struct_llama_sampler]) -> uint32_t:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_sampler_sample", [ctypes.POINTER(struct_llama_sampler), ctypes.POINTER(struct_llama_context), int32_t], llama_token)
def llama_sampler_sample(smpl: ctypes._Pointer[struct_llama_sampler], ctx: ctypes._Pointer[struct_llama_context], idx: int32_t) -> llama_token:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_split_path", [ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.c_int32], ctypes.c_int32)
def llama_split_path(split_path: ctypes._Pointer[ctypes.c_char], maxlen: size_t, path_prefix: ctypes._Pointer[ctypes.c_char], split_no: ctypes.c_int32, split_count: ctypes.c_int32) -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_split_prefix", [ctypes.POINTER(ctypes.c_char), size_t, ctypes.POINTER(ctypes.c_char), ctypes.c_int32, ctypes.c_int32], ctypes.c_int32)
def llama_split_prefix(split_prefix: ctypes._Pointer[ctypes.c_char], maxlen: size_t, split_path: ctypes._Pointer[ctypes.c_char], split_no: ctypes.c_int32, split_count: ctypes.c_int32) -> ctypes.c_int32:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_print_system_info", [], ctypes.POINTER(ctypes.c_char))
def llama_print_system_info() -> ctypes._Pointer[ctypes.c_char]:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_log_set", [ggml_log_callback, ctypes.POINTER(None)], None)
def llama_log_set(log_callback: ggml_log_callback, user_data: ctypes.c_void_p) -> None:
    ...

class struct_llama_perf_context_data(Structure):
    if TYPE_CHECKING:
        t_start_ms: ctypes.c_double
        t_load_ms: ctypes.c_double
        t_p_eval_ms: ctypes.c_double
        t_eval_ms: ctypes.c_double
        n_p_eval: ctypes.c_int32
        n_eval: ctypes.c_int32
struct_llama_perf_context_data._pack_ = 1 # source:False
struct_llama_perf_context_data._fields_ = [
    ('t_start_ms', ctypes.c_double),
    ('t_load_ms', ctypes.c_double),
    ('t_p_eval_ms', ctypes.c_double),
    ('t_eval_ms', ctypes.c_double),
    ('n_p_eval', ctypes.c_int32),
    ('n_eval', ctypes.c_int32),
]

class struct_llama_perf_sampler_data(Structure):
    if TYPE_CHECKING:
        t_sample_ms: ctypes.c_double
        n_sample: ctypes.c_int32
struct_llama_perf_sampler_data._pack_ = 1 # source:False
struct_llama_perf_sampler_data._fields_ = [
    ('t_sample_ms', ctypes.c_double),
    ('n_sample', ctypes.c_int32),
    ('PADDING_0', ctypes.c_ubyte * 4),
]

@ctypes_function_for_shared_library('FIXME_STUB')("llama_perf_context", [ctypes.POINTER(struct_llama_context)], struct_llama_perf_context_data)
def llama_perf_context(ctx: ctypes._Pointer[struct_llama_context]) -> struct_llama_perf_context_data:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_perf_context_print", [ctypes.POINTER(struct_llama_context)], None)
def llama_perf_context_print(ctx: ctypes._Pointer[struct_llama_context]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_perf_context_reset", [ctypes.POINTER(struct_llama_context)], None)
def llama_perf_context_reset(ctx: ctypes._Pointer[struct_llama_context]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_perf_sampler", [ctypes.POINTER(struct_llama_sampler)], struct_llama_perf_sampler_data)
def llama_perf_sampler(chain: ctypes._Pointer[struct_llama_sampler]) -> struct_llama_perf_sampler_data:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_perf_sampler_print", [ctypes.POINTER(struct_llama_sampler)], None)
def llama_perf_sampler_print(chain: ctypes._Pointer[struct_llama_sampler]) -> None:
    ...

@ctypes_function_for_shared_library('FIXME_STUB')("llama_perf_sampler_reset", [ctypes.POINTER(struct_llama_sampler)], None)
def llama_perf_sampler_reset(chain: ctypes._Pointer[struct_llama_sampler]) -> None:
    ...

llama_opt_param_filter = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(None))
@ctypes_function_for_shared_library('FIXME_STUB')("llama_opt_param_filter_all", [ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(None)], ctypes.c_bool)
def llama_opt_param_filter_all(tensor: ctypes._Pointer[struct_ggml_tensor], userdata: ctypes.c_void_p) -> ctypes.c_bool:
    ...

class struct_llama_opt_params(Structure):
    if TYPE_CHECKING:
        n_ctx_train: ctypes.c_uint32
        param_filter: Callable[[ctypes._Pointer[struct_ggml_tensor], ctypes.c_void_p], ctypes.c_bool]
        param_filter_ud: ctypes.c_void_p
        get_opt_pars: Callable[[ctypes.c_void_p], struct_ggml_opt_optimizer_params]
        get_opt_pars_ud: ctypes.c_void_p
class struct_ggml_opt_optimizer_params(Structure):
    if TYPE_CHECKING:
        adamw: struct_ggml_opt_optimizer_params_adamw
class struct_ggml_opt_optimizer_params_adamw(Structure):
    if TYPE_CHECKING:
        alpha: ctypes.c_float
        beta1: ctypes.c_float
        beta2: ctypes.c_float
        eps: ctypes.c_float
        wd: ctypes.c_float
struct_ggml_opt_optimizer_params_adamw._pack_ = 1 # source:False
struct_ggml_opt_optimizer_params_adamw._fields_ = [
    ('alpha', ctypes.c_float),
    ('beta1', ctypes.c_float),
    ('beta2', ctypes.c_float),
    ('eps', ctypes.c_float),
    ('wd', ctypes.c_float),
]

struct_ggml_opt_optimizer_params._pack_ = 1 # source:False
struct_ggml_opt_optimizer_params._fields_ = [
    ('adamw', struct_ggml_opt_optimizer_params_adamw),
]

struct_llama_opt_params._pack_ = 1 # source:False
struct_llama_opt_params._fields_ = [
    ('n_ctx_train', ctypes.c_uint32),
    ('PADDING_0', ctypes.c_ubyte * 4),
    ('param_filter', ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.POINTER(struct_ggml_tensor), ctypes.POINTER(None))),
    ('param_filter_ud', ctypes.POINTER(None)),
    ('get_opt_pars', ctypes.CFUNCTYPE(struct_ggml_opt_optimizer_params, ctypes.POINTER(None))),
    ('get_opt_pars_ud', ctypes.POINTER(None)),
]

@ctypes_function_for_shared_library('FIXME_STUB')("llama_opt_init", [ctypes.POINTER(struct_llama_context), ctypes.POINTER(struct_llama_model), struct_llama_opt_params], None)
def llama_opt_init(lctx: ctypes._Pointer[struct_llama_context], model: ctypes._Pointer[struct_llama_model], lopt_params: struct_llama_opt_params) -> None:
    ...

class struct_ggml_opt_dataset(Structure):
    pass

ggml_opt_dataset_t = ctypes.POINTER(struct_ggml_opt_dataset)
class struct_ggml_opt_result(Structure):
    pass

ggml_opt_result_t = ctypes.POINTER(struct_ggml_opt_result)
class struct_ggml_opt_context(Structure):
    pass

ggml_opt_epoch_callback = ctypes.CFUNCTYPE(None, ctypes.c_bool, ctypes.POINTER(struct_ggml_opt_context), ctypes.POINTER(struct_ggml_opt_dataset), ctypes.POINTER(struct_ggml_opt_result), ctypes.c_int64, ctypes.c_int64, ctypes.c_int64)
@ctypes_function_for_shared_library('FIXME_STUB')("llama_opt_epoch", [ctypes.POINTER(struct_llama_context), ggml_opt_dataset_t, ggml_opt_result_t, ggml_opt_result_t, int64_t, ggml_opt_epoch_callback, ggml_opt_epoch_callback], None)
def llama_opt_epoch(lctx: ctypes._Pointer[struct_llama_context], dataset: ggml_opt_dataset_t, result_train: ggml_opt_result_t, result_eval: ggml_opt_result_t, idata_split: int64_t, callback_train: ggml_opt_epoch_callback, callback_eval: ggml_opt_epoch_callback) -> None:
    ...

struct_ggml_object._pack_ = 1 # source:False
struct_ggml_object._fields_ = [
    ('offs', ctypes.c_uint64),
    ('size', ctypes.c_uint64),
    ('next', ctypes.POINTER(struct_ggml_object)),
    ('type', ggml_object_type),
    ('padding', ctypes.c_char * 4),
]

struct_llama_sampler._pack_ = 1 # source:False
struct_llama_sampler._fields_ = [
    ('iface', ctypes.POINTER(struct_llama_sampler_i)),
    ('ctx', ctypes.POINTER(None)),
]

ggml_init_params = struct_ggml_init_params
ggml_tensor = struct_ggml_tensor
ggml_backend_buffer = struct_ggml_backend_buffer
ggml_backend_buffer_type = struct_ggml_backend_buffer_type
ggml_backend_buffer_i = struct_ggml_backend_buffer_i
ggml_object = struct_ggml_object
ggml_context = struct_ggml_context
ggml_cgraph = struct_ggml_cgraph
ggml_type_traits = struct_ggml_type_traits
ggml_threadpool_params = struct_ggml_threadpool_params
ggml_threadpool = struct_ggml_threadpool
ggml_backend_device = struct_ggml_backend_device
ggml_backend_buffer_type_i = struct_ggml_backend_buffer_type_i
ggml_backend = struct_ggml_backend
ggml_backend_i = struct_ggml_backend_i
ggml_backend_event = struct_ggml_backend_event
ggml_tallocr = struct_ggml_tallocr
ggml_gallocr = struct_ggml_gallocr
ggml_backend_reg = struct_ggml_backend_reg
ggml_backend_reg_i = struct_ggml_backend_reg_i
ggml_backend_device_i = struct_ggml_backend_device_i
ggml_backend_dev_props = struct_ggml_backend_dev_props
ggml_backend_dev_caps = struct_ggml_backend_dev_caps
ggml_backend_feature = struct_ggml_backend_feature
ggml_backend_sched = struct_ggml_backend_sched
ggml_backend_sched_split = struct_ggml_backend_sched_split
ggml_hash_set = struct_ggml_hash_set
ggml_backend_graph_copy = struct_ggml_backend_graph_copy
ggml_cplan = struct_ggml_cplan
ggml_type_traits_cpu = struct_ggml_type_traits_cpu
ggml_logger_state = struct_ggml_logger_state
ggml_context_container = struct_ggml_context_container
ggml_backend_multi_buffer_context = struct_ggml_backend_multi_buffer_context
ggml_opt_optimizer_params = struct_ggml_opt_optimizer_params
ggml_opt_optimizer_params_adamw = struct_ggml_opt_optimizer_params_adamw
ggml_opt_dataset = struct_ggml_opt_dataset
ggml_opt_result = struct_ggml_opt_result
ggml_opt_context = struct_ggml_opt_context
if TYPE_CHECKING:
    ggml_init_params_p = ctypes._Pointer[struct_ggml_init_params]
    ggml_tensor_p = ctypes._Pointer[struct_ggml_tensor]
    ggml_backend_buffer_p = ctypes._Pointer[struct_ggml_backend_buffer]
    ggml_backend_buffer_type_p = ctypes._Pointer[struct_ggml_backend_buffer_type]
    ggml_backend_buffer_i_p = ctypes._Pointer[struct_ggml_backend_buffer_i]
    ggml_object_p = ctypes._Pointer[struct_ggml_object]
    ggml_context_p = ctypes._Pointer[struct_ggml_context]
    ggml_cgraph_p = ctypes._Pointer[struct_ggml_cgraph]
    ggml_type_traits_p = ctypes._Pointer[struct_ggml_type_traits]
    ggml_threadpool_params_p = ctypes._Pointer[struct_ggml_threadpool_params]
    ggml_threadpool_p = ctypes._Pointer[struct_ggml_threadpool]
    ggml_backend_device_p = ctypes._Pointer[struct_ggml_backend_device]
    ggml_backend_buffer_type_i_p = ctypes._Pointer[struct_ggml_backend_buffer_type_i]
    ggml_backend_p = ctypes._Pointer[struct_ggml_backend]
    ggml_backend_i_p = ctypes._Pointer[struct_ggml_backend_i]
    ggml_backend_event_p = ctypes._Pointer[struct_ggml_backend_event]
    ggml_tallocr_p = ctypes._Pointer[struct_ggml_tallocr]
    ggml_gallocr_p = ctypes._Pointer[struct_ggml_gallocr]
    ggml_backend_reg_p = ctypes._Pointer[struct_ggml_backend_reg]
    ggml_backend_reg_i_p = ctypes._Pointer[struct_ggml_backend_reg_i]
    ggml_backend_device_i_p = ctypes._Pointer[struct_ggml_backend_device_i]
    ggml_backend_dev_props_p = ctypes._Pointer[struct_ggml_backend_dev_props]
    ggml_backend_dev_caps_p = ctypes._Pointer[struct_ggml_backend_dev_caps]
    ggml_backend_feature_p = ctypes._Pointer[struct_ggml_backend_feature]
    ggml_backend_sched_p = ctypes._Pointer[struct_ggml_backend_sched]
    ggml_backend_sched_split_p = ctypes._Pointer[struct_ggml_backend_sched_split]
    ggml_hash_set_p = ctypes._Pointer[struct_ggml_hash_set]
    ggml_backend_graph_copy_p = ctypes._Pointer[struct_ggml_backend_graph_copy]
    ggml_cplan_p = ctypes._Pointer[struct_ggml_cplan]
    ggml_type_traits_cpu_p = ctypes._Pointer[struct_ggml_type_traits_cpu]
    ggml_logger_state_p = ctypes._Pointer[struct_ggml_logger_state]
    ggml_context_container_p = ctypes._Pointer[struct_ggml_context_container]
    ggml_backend_multi_buffer_context_p = ctypes._Pointer[struct_ggml_backend_multi_buffer_context]
    ggml_opt_optimizer_params_p = ctypes._Pointer[struct_ggml_opt_optimizer_params]
    ggml_opt_optimizer_params_adamw_p = ctypes._Pointer[struct_ggml_opt_optimizer_params_adamw]
    ggml_opt_dataset_p = ctypes._Pointer[struct_ggml_opt_dataset]
    ggml_opt_result_p = ctypes._Pointer[struct_ggml_opt_result]
    ggml_opt_context_p = ctypes._Pointer[struct_ggml_opt_context]
else:
    ggml_init_params_p = ctypes.POINTER(struct_ggml_init_params)
    ggml_tensor_p = ctypes.POINTER(struct_ggml_tensor)
    ggml_backend_buffer_p = ctypes.POINTER(struct_ggml_backend_buffer)
    ggml_backend_buffer_type_p = ctypes.POINTER(struct_ggml_backend_buffer_type)
    ggml_backend_buffer_i_p = ctypes.POINTER(struct_ggml_backend_buffer_i)
    ggml_object_p = ctypes.POINTER(struct_ggml_object)
    ggml_context_p = ctypes.POINTER(struct_ggml_context)
    ggml_cgraph_p = ctypes.POINTER(struct_ggml_cgraph)
    ggml_type_traits_p = ctypes.POINTER(struct_ggml_type_traits)
    ggml_threadpool_params_p = ctypes.POINTER(struct_ggml_threadpool_params)
    ggml_threadpool_p = ctypes.POINTER(struct_ggml_threadpool)
    ggml_backend_device_p = ctypes.POINTER(struct_ggml_backend_device)
    ggml_backend_buffer_type_i_p = ctypes.POINTER(struct_ggml_backend_buffer_type_i)
    ggml_backend_p = ctypes.POINTER(struct_ggml_backend)
    ggml_backend_i_p = ctypes.POINTER(struct_ggml_backend_i)
    ggml_backend_event_p = ctypes.POINTER(struct_ggml_backend_event)
    ggml_tallocr_p = ctypes.POINTER(struct_ggml_tallocr)
    ggml_gallocr_p = ctypes.POINTER(struct_ggml_gallocr)
    ggml_backend_reg_p = ctypes.POINTER(struct_ggml_backend_reg)
    ggml_backend_reg_i_p = ctypes.POINTER(struct_ggml_backend_reg_i)
    ggml_backend_device_i_p = ctypes.POINTER(struct_ggml_backend_device_i)
    ggml_backend_dev_props_p = ctypes.POINTER(struct_ggml_backend_dev_props)
    ggml_backend_dev_caps_p = ctypes.POINTER(struct_ggml_backend_dev_caps)
    ggml_backend_feature_p = ctypes.POINTER(struct_ggml_backend_feature)
    ggml_backend_sched_p = ctypes.POINTER(struct_ggml_backend_sched)
    ggml_backend_sched_split_p = ctypes.POINTER(struct_ggml_backend_sched_split)
    ggml_hash_set_p = ctypes.POINTER(struct_ggml_hash_set)
    ggml_backend_graph_copy_p = ctypes.POINTER(struct_ggml_backend_graph_copy)
    ggml_cplan_p = ctypes.POINTER(struct_ggml_cplan)
    ggml_type_traits_cpu_p = ctypes.POINTER(struct_ggml_type_traits_cpu)
    ggml_logger_state_p = ctypes.POINTER(struct_ggml_logger_state)
    ggml_context_container_p = ctypes.POINTER(struct_ggml_context_container)
    ggml_backend_multi_buffer_context_p = ctypes.POINTER(struct_ggml_backend_multi_buffer_context)
    ggml_opt_optimizer_params_p = ctypes.POINTER(struct_ggml_opt_optimizer_params)
    ggml_opt_optimizer_params_adamw_p = ctypes.POINTER(struct_ggml_opt_optimizer_params_adamw)
    ggml_opt_dataset_p = ctypes.POINTER(struct_ggml_opt_dataset)
    ggml_opt_result_p = ctypes.POINTER(struct_ggml_opt_result)
    ggml_opt_context_p = ctypes.POINTER(struct_ggml_opt_context)
__all__ = \
    ['GGML_BACKEND_BUFFER_USAGE_ANY',
    'GGML_BACKEND_BUFFER_USAGE_COMPUTE',
    'GGML_BACKEND_BUFFER_USAGE_WEIGHTS',
    'GGML_BACKEND_DEVICE_TYPE_ACCEL', 'GGML_BACKEND_DEVICE_TYPE_CPU',
    'GGML_BACKEND_DEVICE_TYPE_GPU', 'GGML_CGRAPH_EVAL_ORDER_COUNT',
    'GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT',
    'GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT', 'GGML_FTYPE_ALL_F32',
    'GGML_FTYPE_MOSTLY_BF16', 'GGML_FTYPE_MOSTLY_F16',
    'GGML_FTYPE_MOSTLY_IQ1_M', 'GGML_FTYPE_MOSTLY_IQ1_S',
    'GGML_FTYPE_MOSTLY_IQ2_S', 'GGML_FTYPE_MOSTLY_IQ2_XS',
    'GGML_FTYPE_MOSTLY_IQ2_XXS', 'GGML_FTYPE_MOSTLY_IQ3_S',
    'GGML_FTYPE_MOSTLY_IQ3_XXS', 'GGML_FTYPE_MOSTLY_IQ4_NL',
    'GGML_FTYPE_MOSTLY_IQ4_XS', 'GGML_FTYPE_MOSTLY_Q2_K',
    'GGML_FTYPE_MOSTLY_Q3_K', 'GGML_FTYPE_MOSTLY_Q4_0',
    'GGML_FTYPE_MOSTLY_Q4_1', 'GGML_FTYPE_MOSTLY_Q4_1_SOME_F16',
    'GGML_FTYPE_MOSTLY_Q4_K', 'GGML_FTYPE_MOSTLY_Q5_0',
    'GGML_FTYPE_MOSTLY_Q5_1', 'GGML_FTYPE_MOSTLY_Q5_K',
    'GGML_FTYPE_MOSTLY_Q6_K', 'GGML_FTYPE_MOSTLY_Q8_0',
    'GGML_FTYPE_UNKNOWN', 'GGML_LOG_LEVEL_CONT',
    'GGML_LOG_LEVEL_DEBUG', 'GGML_LOG_LEVEL_ERROR',
    'GGML_LOG_LEVEL_INFO', 'GGML_LOG_LEVEL_NONE',
    'GGML_LOG_LEVEL_WARN', 'GGML_NUMA_STRATEGY_COUNT',
    'GGML_NUMA_STRATEGY_DISABLED', 'GGML_NUMA_STRATEGY_DISTRIBUTE',
    'GGML_NUMA_STRATEGY_ISOLATE', 'GGML_NUMA_STRATEGY_MIRROR',
    'GGML_NUMA_STRATEGY_NUMACTL', 'GGML_OBJECT_TYPE_GRAPH',
    'GGML_OBJECT_TYPE_TENSOR', 'GGML_OBJECT_TYPE_WORK_BUFFER',
    'GGML_OP_ACC', 'GGML_OP_ADD', 'GGML_OP_ADD1',
    'GGML_OP_ADD_REL_POS', 'GGML_OP_ARANGE', 'GGML_OP_ARGMAX',
    'GGML_OP_ARGSORT', 'GGML_OP_CLAMP', 'GGML_OP_CONCAT',
    'GGML_OP_CONT', 'GGML_OP_CONV_2D_DW', 'GGML_OP_CONV_TRANSPOSE_1D',
    'GGML_OP_CONV_TRANSPOSE_2D', 'GGML_OP_COS', 'GGML_OP_COUNT',
    'GGML_OP_COUNT_EQUAL', 'GGML_OP_CPY',
    'GGML_OP_CROSS_ENTROPY_LOSS', 'GGML_OP_CROSS_ENTROPY_LOSS_BACK',
    'GGML_OP_CUSTOM', 'GGML_OP_DIAG', 'GGML_OP_DIAG_MASK_INF',
    'GGML_OP_DIAG_MASK_ZERO', 'GGML_OP_DIV', 'GGML_OP_DUP',
    'GGML_OP_FLASH_ATTN_BACK', 'GGML_OP_FLASH_ATTN_EXT',
    'GGML_OP_GATED_LINEAR_ATTN', 'GGML_OP_GET_REL_POS',
    'GGML_OP_GET_ROWS', 'GGML_OP_GET_ROWS_BACK', 'GGML_OP_GROUP_NORM',
    'GGML_OP_IM2COL', 'GGML_OP_IM2COL_BACK', 'GGML_OP_L2_NORM',
    'GGML_OP_LEAKY_RELU', 'GGML_OP_LOG', 'GGML_OP_MAP_CUSTOM1',
    'GGML_OP_MAP_CUSTOM2', 'GGML_OP_MAP_CUSTOM3', 'GGML_OP_MEAN',
    'GGML_OP_MUL', 'GGML_OP_MUL_MAT', 'GGML_OP_MUL_MAT_ID',
    'GGML_OP_NONE', 'GGML_OP_NORM', 'GGML_OP_OPT_STEP_ADAMW',
    'GGML_OP_OUT_PROD', 'GGML_OP_PAD', 'GGML_OP_PAD_REFLECT_1D',
    'GGML_OP_PERMUTE', 'GGML_OP_POOL_1D', 'GGML_OP_POOL_2D',
    'GGML_OP_POOL_2D_BACK', 'GGML_OP_POOL_AVG', 'GGML_OP_POOL_COUNT',
    'GGML_OP_POOL_MAX', 'GGML_OP_REPEAT', 'GGML_OP_REPEAT_BACK',
    'GGML_OP_RESHAPE', 'GGML_OP_RMS_NORM', 'GGML_OP_RMS_NORM_BACK',
    'GGML_OP_ROPE', 'GGML_OP_ROPE_BACK', 'GGML_OP_RWKV_WKV6',
    'GGML_OP_RWKV_WKV7', 'GGML_OP_SCALE', 'GGML_OP_SET',
    'GGML_OP_SILU_BACK', 'GGML_OP_SIN', 'GGML_OP_SOFT_MAX',
    'GGML_OP_SOFT_MAX_BACK', 'GGML_OP_SQR', 'GGML_OP_SQRT',
    'GGML_OP_SSM_CONV', 'GGML_OP_SSM_SCAN', 'GGML_OP_SUB',
    'GGML_OP_SUM', 'GGML_OP_SUM_ROWS', 'GGML_OP_TIMESTEP_EMBEDDING',
    'GGML_OP_TRANSPOSE', 'GGML_OP_UNARY', 'GGML_OP_UPSCALE',
    'GGML_OP_VIEW', 'GGML_OP_WIN_PART', 'GGML_OP_WIN_UNPART',
    'GGML_PREC_DEFAULT', 'GGML_PREC_F32', 'GGML_SCALE_MODE_BILINEAR',
    'GGML_SCALE_MODE_NEAREST', 'GGML_SCHED_PRIO_HIGH',
    'GGML_SCHED_PRIO_MEDIUM', 'GGML_SCHED_PRIO_NORMAL',
    'GGML_SCHED_PRIO_REALTIME', 'GGML_SORT_ORDER_ASC',
    'GGML_SORT_ORDER_DESC', 'GGML_STATUS_ABORTED',
    'GGML_STATUS_ALLOC_FAILED', 'GGML_STATUS_FAILED',
    'GGML_STATUS_SUCCESS', 'GGML_TENSOR_FLAG_INPUT',
    'GGML_TENSOR_FLAG_LOSS', 'GGML_TENSOR_FLAG_OUTPUT',
    'GGML_TENSOR_FLAG_PARAM', 'GGML_TYPE_BF16', 'GGML_TYPE_COUNT',
    'GGML_TYPE_F16', 'GGML_TYPE_F32', 'GGML_TYPE_F64',
    'GGML_TYPE_I16', 'GGML_TYPE_I32', 'GGML_TYPE_I64', 'GGML_TYPE_I8',
    'GGML_TYPE_IQ1_M', 'GGML_TYPE_IQ1_S', 'GGML_TYPE_IQ2_S',
    'GGML_TYPE_IQ2_XS', 'GGML_TYPE_IQ2_XXS', 'GGML_TYPE_IQ3_S',
    'GGML_TYPE_IQ3_XXS', 'GGML_TYPE_IQ4_NL', 'GGML_TYPE_IQ4_XS',
    'GGML_TYPE_Q2_K', 'GGML_TYPE_Q3_K', 'GGML_TYPE_Q4_0',
    'GGML_TYPE_Q4_1', 'GGML_TYPE_Q4_K', 'GGML_TYPE_Q5_0',
    'GGML_TYPE_Q5_1', 'GGML_TYPE_Q5_K', 'GGML_TYPE_Q6_K',
    'GGML_TYPE_Q8_0', 'GGML_TYPE_Q8_1', 'GGML_TYPE_Q8_K',
    'GGML_TYPE_TQ1_0', 'GGML_TYPE_TQ2_0', 'GGML_UNARY_OP_ABS',
    'GGML_UNARY_OP_COUNT', 'GGML_UNARY_OP_ELU', 'GGML_UNARY_OP_EXP',
    'GGML_UNARY_OP_GELU', 'GGML_UNARY_OP_GELU_ERF',
    'GGML_UNARY_OP_GELU_QUICK', 'GGML_UNARY_OP_HARDSIGMOID',
    'GGML_UNARY_OP_HARDSWISH', 'GGML_UNARY_OP_NEG',
    'GGML_UNARY_OP_RELU', 'GGML_UNARY_OP_SGN',
    'GGML_UNARY_OP_SIGMOID', 'GGML_UNARY_OP_SILU',
    'GGML_UNARY_OP_STEP', 'GGML_UNARY_OP_TANH',
    'LLAMA_ATTENTION_TYPE_CAUSAL', 'LLAMA_ATTENTION_TYPE_NON_CAUSAL',
    'LLAMA_ATTENTION_TYPE_UNSPECIFIED', 'LLAMA_FTYPE_ALL_F32',
    'LLAMA_FTYPE_GUESSED', 'LLAMA_FTYPE_MOSTLY_BF16',
    'LLAMA_FTYPE_MOSTLY_F16', 'LLAMA_FTYPE_MOSTLY_IQ1_M',
    'LLAMA_FTYPE_MOSTLY_IQ1_S', 'LLAMA_FTYPE_MOSTLY_IQ2_M',
    'LLAMA_FTYPE_MOSTLY_IQ2_S', 'LLAMA_FTYPE_MOSTLY_IQ2_XS',
    'LLAMA_FTYPE_MOSTLY_IQ2_XXS', 'LLAMA_FTYPE_MOSTLY_IQ3_M',
    'LLAMA_FTYPE_MOSTLY_IQ3_S', 'LLAMA_FTYPE_MOSTLY_IQ3_XS',
    'LLAMA_FTYPE_MOSTLY_IQ3_XXS', 'LLAMA_FTYPE_MOSTLY_IQ4_NL',
    'LLAMA_FTYPE_MOSTLY_IQ4_XS', 'LLAMA_FTYPE_MOSTLY_Q2_K',
    'LLAMA_FTYPE_MOSTLY_Q2_K_S', 'LLAMA_FTYPE_MOSTLY_Q3_K_L',
    'LLAMA_FTYPE_MOSTLY_Q3_K_M', 'LLAMA_FTYPE_MOSTLY_Q3_K_S',
    'LLAMA_FTYPE_MOSTLY_Q4_0', 'LLAMA_FTYPE_MOSTLY_Q4_1',
    'LLAMA_FTYPE_MOSTLY_Q4_K_M', 'LLAMA_FTYPE_MOSTLY_Q4_K_S',
    'LLAMA_FTYPE_MOSTLY_Q5_0', 'LLAMA_FTYPE_MOSTLY_Q5_1',
    'LLAMA_FTYPE_MOSTLY_Q5_K_M', 'LLAMA_FTYPE_MOSTLY_Q5_K_S',
    'LLAMA_FTYPE_MOSTLY_Q6_K', 'LLAMA_FTYPE_MOSTLY_Q8_0',
    'LLAMA_FTYPE_MOSTLY_TQ1_0', 'LLAMA_FTYPE_MOSTLY_TQ2_0',
    'LLAMA_KV_OVERRIDE_TYPE_BOOL', 'LLAMA_KV_OVERRIDE_TYPE_FLOAT',
    'LLAMA_KV_OVERRIDE_TYPE_INT', 'LLAMA_KV_OVERRIDE_TYPE_STR',
    'LLAMA_POOLING_TYPE_CLS', 'LLAMA_POOLING_TYPE_LAST',
    'LLAMA_POOLING_TYPE_MEAN', 'LLAMA_POOLING_TYPE_NONE',
    'LLAMA_POOLING_TYPE_RANK', 'LLAMA_POOLING_TYPE_UNSPECIFIED',
    'LLAMA_ROPE_SCALING_TYPE_LINEAR',
    'LLAMA_ROPE_SCALING_TYPE_LONGROPE',
    'LLAMA_ROPE_SCALING_TYPE_MAX_VALUE',
    'LLAMA_ROPE_SCALING_TYPE_NONE',
    'LLAMA_ROPE_SCALING_TYPE_UNSPECIFIED',
    'LLAMA_ROPE_SCALING_TYPE_YARN', 'LLAMA_ROPE_TYPE_MROPE',
    'LLAMA_ROPE_TYPE_NEOX', 'LLAMA_ROPE_TYPE_NONE',
    'LLAMA_ROPE_TYPE_NORM', 'LLAMA_ROPE_TYPE_VISION',
    'LLAMA_SPLIT_MODE_LAYER', 'LLAMA_SPLIT_MODE_NONE',
    'LLAMA_SPLIT_MODE_ROW', 'LLAMA_TOKEN_ATTR_BYTE',
    'LLAMA_TOKEN_ATTR_CONTROL', 'LLAMA_TOKEN_ATTR_LSTRIP',
    'LLAMA_TOKEN_ATTR_NORMAL', 'LLAMA_TOKEN_ATTR_NORMALIZED',
    'LLAMA_TOKEN_ATTR_RSTRIP', 'LLAMA_TOKEN_ATTR_SINGLE_WORD',
    'LLAMA_TOKEN_ATTR_UNDEFINED', 'LLAMA_TOKEN_ATTR_UNKNOWN',
    'LLAMA_TOKEN_ATTR_UNUSED', 'LLAMA_TOKEN_ATTR_USER_DEFINED',
    'LLAMA_TOKEN_TYPE_BYTE', 'LLAMA_TOKEN_TYPE_CONTROL',
    'LLAMA_TOKEN_TYPE_NORMAL', 'LLAMA_TOKEN_TYPE_UNDEFINED',
    'LLAMA_TOKEN_TYPE_UNKNOWN', 'LLAMA_TOKEN_TYPE_UNUSED',
    'LLAMA_TOKEN_TYPE_USER_DEFINED',
    'LLAMA_VOCAB_PRE_TYPE_BAILINGMOE', 'LLAMA_VOCAB_PRE_TYPE_BLOOM',
    'LLAMA_VOCAB_PRE_TYPE_CHAMELEON', 'LLAMA_VOCAB_PRE_TYPE_CHATGLM3',
    'LLAMA_VOCAB_PRE_TYPE_CHATGLM4', 'LLAMA_VOCAB_PRE_TYPE_CODESHELL',
    'LLAMA_VOCAB_PRE_TYPE_COMMAND_R', 'LLAMA_VOCAB_PRE_TYPE_DBRX',
    'LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM',
    'LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_CODER',
    'LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM',
    'LLAMA_VOCAB_PRE_TYPE_DEFAULT', 'LLAMA_VOCAB_PRE_TYPE_EXAONE',
    'LLAMA_VOCAB_PRE_TYPE_FALCON', 'LLAMA_VOCAB_PRE_TYPE_GPT2',
    'LLAMA_VOCAB_PRE_TYPE_GPT3_FINNISH', 'LLAMA_VOCAB_PRE_TYPE_GPT4O',
    'LLAMA_VOCAB_PRE_TYPE_JAIS', 'LLAMA_VOCAB_PRE_TYPE_LLAMA3',
    'LLAMA_VOCAB_PRE_TYPE_LLAMA4', 'LLAMA_VOCAB_PRE_TYPE_MINERVA',
    'LLAMA_VOCAB_PRE_TYPE_MPT', 'LLAMA_VOCAB_PRE_TYPE_OLMO',
    'LLAMA_VOCAB_PRE_TYPE_PIXTRAL', 'LLAMA_VOCAB_PRE_TYPE_PORO',
    'LLAMA_VOCAB_PRE_TYPE_QWEN2', 'LLAMA_VOCAB_PRE_TYPE_REFACT',
    'LLAMA_VOCAB_PRE_TYPE_SEED_CODER', 'LLAMA_VOCAB_PRE_TYPE_SMAUG',
    'LLAMA_VOCAB_PRE_TYPE_SMOLLM', 'LLAMA_VOCAB_PRE_TYPE_STABLELM2',
    'LLAMA_VOCAB_PRE_TYPE_STARCODER', 'LLAMA_VOCAB_PRE_TYPE_SUPERBPE',
    'LLAMA_VOCAB_PRE_TYPE_TEKKEN', 'LLAMA_VOCAB_PRE_TYPE_TRILLION',
    'LLAMA_VOCAB_PRE_TYPE_VIKING', 'LLAMA_VOCAB_TYPE_BPE',
    'LLAMA_VOCAB_TYPE_NONE', 'LLAMA_VOCAB_TYPE_RWKV',
    'LLAMA_VOCAB_TYPE_SPM', 'LLAMA_VOCAB_TYPE_UGM',
    'LLAMA_VOCAB_TYPE_WPM', 'fmt_size', 'ggml_abort',
    'ggml_abort_callback', 'ggml_abs', 'ggml_abs_inplace', 'ggml_acc',
    'ggml_acc_impl', 'ggml_acc_inplace', 'ggml_acc_or_set',
    'ggml_add', 'ggml_add1', 'ggml_add1_impl', 'ggml_add1_inplace',
    'ggml_add1_or_set', 'ggml_add_cast', 'ggml_add_cast_impl',
    'ggml_add_impl', 'ggml_add_inplace', 'ggml_add_or_set',
    'ggml_add_rel_pos', 'ggml_add_rel_pos_impl',
    'ggml_add_rel_pos_inplace', 'ggml_aligned_free',
    'ggml_aligned_malloc', 'ggml_arange', 'ggml_are_same_layout',
    'ggml_are_same_shape', 'ggml_are_same_stride', 'ggml_argmax',
    'ggml_argsort', 'ggml_backend', 'ggml_backend_alloc_buffer',
    'ggml_backend_alloc_ctx_tensors',
    'ggml_backend_alloc_ctx_tensors_from_buft', 'ggml_backend_buffer',
    'ggml_backend_buffer_clear', 'ggml_backend_buffer_copy_tensor',
    'ggml_backend_buffer_free', 'ggml_backend_buffer_get_alignment',
    'ggml_backend_buffer_get_alloc_size',
    'ggml_backend_buffer_get_base',
    'ggml_backend_buffer_get_max_size',
    'ggml_backend_buffer_get_size', 'ggml_backend_buffer_get_type',
    'ggml_backend_buffer_get_usage', 'ggml_backend_buffer_i',
    'ggml_backend_buffer_i_p', 'ggml_backend_buffer_init',
    'ggml_backend_buffer_init_tensor', 'ggml_backend_buffer_is_host',
    'ggml_backend_buffer_is_multi_buffer', 'ggml_backend_buffer_name',
    'ggml_backend_buffer_p', 'ggml_backend_buffer_reset',
    'ggml_backend_buffer_set_usage', 'ggml_backend_buffer_t',
    'ggml_backend_buffer_type', 'ggml_backend_buffer_type_i',
    'ggml_backend_buffer_type_i_p', 'ggml_backend_buffer_type_p',
    'ggml_backend_buffer_type_t', 'ggml_backend_buffer_usage',
    'ggml_backend_buft_alloc_buffer',
    'ggml_backend_buft_get_alignment',
    'ggml_backend_buft_get_alloc_size',
    'ggml_backend_buft_get_device', 'ggml_backend_buft_get_max_size',
    'ggml_backend_buft_is_host', 'ggml_backend_buft_name',
    'ggml_backend_compare_graph_backend',
    'ggml_backend_cpu_buffer_clear',
    'ggml_backend_cpu_buffer_cpy_tensor',
    'ggml_backend_cpu_buffer_free_buffer',
    'ggml_backend_cpu_buffer_from_ptr',
    'ggml_backend_cpu_buffer_from_ptr_type',
    'ggml_backend_cpu_buffer_from_ptr_type_get_name',
    'ggml_backend_cpu_buffer_get_base',
    'ggml_backend_cpu_buffer_get_tensor',
    'ggml_backend_cpu_buffer_memset_tensor',
    'ggml_backend_cpu_buffer_set_tensor',
    'ggml_backend_cpu_buffer_type',
    'ggml_backend_cpu_buffer_type_alloc_buffer',
    'ggml_backend_cpu_buffer_type_get_alignment',
    'ggml_backend_cpu_buffer_type_get_name',
    'ggml_backend_cpu_buffer_type_is_host', 'ggml_backend_cpu_init',
    'ggml_backend_cpu_reg', 'ggml_backend_cpu_set_abort_callback',
    'ggml_backend_cpu_set_n_threads',
    'ggml_backend_cpu_set_threadpool',
    'ggml_backend_cuda_buffer_type',
    'ggml_backend_cuda_get_device_count',
    'ggml_backend_cuda_get_device_description',
    'ggml_backend_cuda_get_device_memory',
    'ggml_backend_cuda_host_buffer_type', 'ggml_backend_cuda_init',
    'ggml_backend_cuda_reg', 'ggml_backend_cuda_register_host_buffer',
    'ggml_backend_cuda_split_buffer_type',
    'ggml_backend_cuda_unregister_host_buffer',
    'ggml_backend_dev_backend_reg',
    'ggml_backend_dev_buffer_from_host_ptr',
    'ggml_backend_dev_buffer_type', 'ggml_backend_dev_by_name',
    'ggml_backend_dev_by_type', 'ggml_backend_dev_caps',
    'ggml_backend_dev_caps_p', 'ggml_backend_dev_count',
    'ggml_backend_dev_description', 'ggml_backend_dev_get',
    'ggml_backend_dev_get_extra_bufts_t',
    'ggml_backend_dev_get_props', 'ggml_backend_dev_host_buffer_type',
    'ggml_backend_dev_init', 'ggml_backend_dev_memory',
    'ggml_backend_dev_name', 'ggml_backend_dev_offload_op',
    'ggml_backend_dev_props', 'ggml_backend_dev_props_p',
    'ggml_backend_dev_supports_buft', 'ggml_backend_dev_supports_op',
    'ggml_backend_dev_t', 'ggml_backend_dev_type',
    'ggml_backend_device', 'ggml_backend_device_i',
    'ggml_backend_device_i_p', 'ggml_backend_device_p',
    'ggml_backend_device_register', 'ggml_backend_eval_callback',
    'ggml_backend_event', 'ggml_backend_event_free',
    'ggml_backend_event_new', 'ggml_backend_event_p',
    'ggml_backend_event_record', 'ggml_backend_event_synchronize',
    'ggml_backend_event_t', 'ggml_backend_event_wait',
    'ggml_backend_feature', 'ggml_backend_feature_p',
    'ggml_backend_free', 'ggml_backend_get_alignment',
    'ggml_backend_get_default_buffer_type', 'ggml_backend_get_device',
    'ggml_backend_get_features_t', 'ggml_backend_get_max_size',
    'ggml_backend_graph_compute', 'ggml_backend_graph_compute_async',
    'ggml_backend_graph_copy', 'ggml_backend_graph_copy',
    'ggml_backend_graph_copy_free', 'ggml_backend_graph_copy_p',
    'ggml_backend_graph_plan_compute',
    'ggml_backend_graph_plan_create', 'ggml_backend_graph_plan_free',
    'ggml_backend_graph_plan_t', 'ggml_backend_guid',
    'ggml_backend_i', 'ggml_backend_i_p', 'ggml_backend_init_best',
    'ggml_backend_init_by_name', 'ggml_backend_init_by_type',
    'ggml_backend_init_t', 'ggml_backend_is_cpu',
    'ggml_backend_is_cuda', 'ggml_backend_load',
    'ggml_backend_load_all', 'ggml_backend_load_all_from_path',
    'ggml_backend_multi_buffer_alloc_buffer',
    'ggml_backend_multi_buffer_clear',
    'ggml_backend_multi_buffer_context',
    'ggml_backend_multi_buffer_context_p',
    'ggml_backend_multi_buffer_free_buffer',
    'ggml_backend_multi_buffer_set_usage', 'ggml_backend_name',
    'ggml_backend_offload_op', 'ggml_backend_p', 'ggml_backend_reg',
    'ggml_backend_reg_by_name', 'ggml_backend_reg_count',
    'ggml_backend_reg_dev_count', 'ggml_backend_reg_dev_get',
    'ggml_backend_reg_get', 'ggml_backend_reg_get_proc_address',
    'ggml_backend_reg_i', 'ggml_backend_reg_i_p',
    'ggml_backend_reg_name', 'ggml_backend_reg_p',
    'ggml_backend_reg_t', 'ggml_backend_register',
    'ggml_backend_sched', 'ggml_backend_sched_alloc_graph',
    'ggml_backend_sched_alloc_splits',
    'ggml_backend_sched_backend_from_buffer',
    'ggml_backend_sched_backend_id',
    'ggml_backend_sched_backend_id_from_cur',
    'ggml_backend_sched_buffer_supported',
    'ggml_backend_sched_compute_splits',
    'ggml_backend_sched_eval_callback', 'ggml_backend_sched_free',
    'ggml_backend_sched_get_backend',
    'ggml_backend_sched_get_buffer_size',
    'ggml_backend_sched_get_n_backends',
    'ggml_backend_sched_get_n_copies',
    'ggml_backend_sched_get_n_splits',
    'ggml_backend_sched_get_tensor_backend',
    'ggml_backend_sched_graph_compute',
    'ggml_backend_sched_graph_compute_async',
    'ggml_backend_sched_new', 'ggml_backend_sched_p',
    'ggml_backend_sched_print_assignments',
    'ggml_backend_sched_reserve', 'ggml_backend_sched_reset',
    'ggml_backend_sched_set_eval_callback',
    'ggml_backend_sched_set_if_supported',
    'ggml_backend_sched_set_tensor_backend',
    'ggml_backend_sched_split', 'ggml_backend_sched_split_graph',
    'ggml_backend_sched_split_p', 'ggml_backend_sched_synchronize',
    'ggml_backend_sched_t', 'ggml_backend_score_t',
    'ggml_backend_set_abort_callback_t',
    'ggml_backend_set_n_threads_t',
    'ggml_backend_split_buffer_type_t', 'ggml_backend_supports_buft',
    'ggml_backend_supports_op', 'ggml_backend_synchronize',
    'ggml_backend_t', 'ggml_backend_tensor_alloc',
    'ggml_backend_tensor_copy', 'ggml_backend_tensor_copy_async',
    'ggml_backend_tensor_get', 'ggml_backend_tensor_get_async',
    'ggml_backend_tensor_memset', 'ggml_backend_tensor_set',
    'ggml_backend_tensor_set_async', 'ggml_backend_unload',
    'ggml_backend_view_init', 'ggml_bf16_t', 'ggml_bf16_to_fp32',
    'ggml_bf16_to_fp32_row', 'ggml_blck_size',
    'ggml_build_backward_expand', 'ggml_build_forward_expand',
    'ggml_build_forward_impl', 'ggml_calc_conv_output_size',
    'ggml_calc_conv_transpose_1d_output_size',
    'ggml_calc_conv_transpose_output_size',
    'ggml_calc_pool_output_size', 'ggml_calloc', 'ggml_can_mul_mat',
    'ggml_can_out_prod', 'ggml_can_repeat', 'ggml_can_repeat_rows',
    'ggml_cast', 'ggml_cgraph', 'ggml_cgraph_eval_order',
    'ggml_cgraph_p', 'ggml_clamp', 'ggml_compute_backward',
    'ggml_concat', 'ggml_cont', 'ggml_cont_1d', 'ggml_cont_2d',
    'ggml_cont_3d', 'ggml_cont_4d', 'ggml_cont_impl', 'ggml_context',
    'ggml_context_container', 'ggml_context_container_p',
    'ggml_context_p', 'ggml_conv_1d', 'ggml_conv_1d_dw',
    'ggml_conv_1d_dw_ph', 'ggml_conv_1d_ph', 'ggml_conv_2d',
    'ggml_conv_2d_dw', 'ggml_conv_2d_dw_direct', 'ggml_conv_2d_s1_ph',
    'ggml_conv_2d_sk_p0', 'ggml_conv_transpose_1d',
    'ggml_conv_transpose_2d_p0', 'ggml_cos', 'ggml_cos_impl',
    'ggml_cos_inplace', 'ggml_count_equal', 'ggml_cplan',
    'ggml_cplan_p', 'ggml_cpu_bf16_to_fp32', 'ggml_cpu_fp16_to_fp32',
    'ggml_cpu_fp32_to_bf16', 'ggml_cpu_fp32_to_fp16',
    'ggml_cpu_get_sve_cnt', 'ggml_cpu_has_amx_int8',
    'ggml_cpu_has_arm_fma', 'ggml_cpu_has_avx', 'ggml_cpu_has_avx2',
    'ggml_cpu_has_avx512', 'ggml_cpu_has_avx512_bf16',
    'ggml_cpu_has_avx512_vbmi', 'ggml_cpu_has_avx512_vnni',
    'ggml_cpu_has_avx_vnni', 'ggml_cpu_has_bmi2',
    'ggml_cpu_has_dotprod', 'ggml_cpu_has_f16c', 'ggml_cpu_has_fma',
    'ggml_cpu_has_fp16_va', 'ggml_cpu_has_llamafile',
    'ggml_cpu_has_matmul_int8', 'ggml_cpu_has_neon',
    'ggml_cpu_has_riscv_v', 'ggml_cpu_has_sme', 'ggml_cpu_has_sse3',
    'ggml_cpu_has_ssse3', 'ggml_cpu_has_sve', 'ggml_cpu_has_vsx',
    'ggml_cpu_has_vxe', 'ggml_cpu_has_wasm_simd', 'ggml_cpu_init',
    'ggml_cpy', 'ggml_cpy_impl', 'ggml_cross_entropy_loss',
    'ggml_cross_entropy_loss_back', 'ggml_custom1_op_t',
    'ggml_custom2_op_t', 'ggml_custom3_op_t', 'ggml_custom_4d',
    'ggml_custom_inplace', 'ggml_custom_op_t', 'ggml_cycles',
    'ggml_cycles_per_ms', 'ggml_diag', 'ggml_diag_mask_inf',
    'ggml_diag_mask_inf_impl', 'ggml_diag_mask_inf_inplace',
    'ggml_diag_mask_zero', 'ggml_diag_mask_zero_impl',
    'ggml_diag_mask_zero_inplace', 'ggml_div', 'ggml_div_impl',
    'ggml_div_inplace', 'ggml_dup', 'ggml_dup_impl',
    'ggml_dup_inplace', 'ggml_dup_tensor', 'ggml_dup_tensor_layout',
    'ggml_element_size', 'ggml_elu', 'ggml_elu_inplace', 'ggml_exp',
    'ggml_exp_inplace', 'ggml_flash_attn_back', 'ggml_flash_attn_ext',
    'ggml_flash_attn_ext_get_prec', 'ggml_flash_attn_ext_set_prec',
    'ggml_fopen', 'ggml_format_name', 'ggml_fp16_t',
    'ggml_fp16_to_fp32', 'ggml_fp16_to_fp32_row', 'ggml_fp32_to_bf16',
    'ggml_fp32_to_bf16_row', 'ggml_fp32_to_bf16_row_ref',
    'ggml_fp32_to_fp16', 'ggml_fp32_to_fp16_row', 'ggml_free',
    'ggml_from_float_t', 'ggml_ftype', 'ggml_ftype_to_ggml_type',
    'ggml_gallocr', 'ggml_gallocr_alloc_graph', 'ggml_gallocr_free',
    'ggml_gallocr_get_buffer_size', 'ggml_gallocr_new',
    'ggml_gallocr_new_n', 'ggml_gallocr_p', 'ggml_gallocr_reserve',
    'ggml_gallocr_reserve_n', 'ggml_gallocr_t',
    'ggml_gated_linear_attn', 'ggml_gelu', 'ggml_gelu_erf',
    'ggml_gelu_erf_inplace', 'ggml_gelu_inplace', 'ggml_gelu_quick',
    'ggml_gelu_quick_inplace', 'ggml_get_data', 'ggml_get_data_f32',
    'ggml_get_f32_1d', 'ggml_get_f32_nd', 'ggml_get_first_tensor',
    'ggml_get_i32_1d', 'ggml_get_i32_nd', 'ggml_get_max_tensor_size',
    'ggml_get_mem_buffer', 'ggml_get_mem_size', 'ggml_get_name',
    'ggml_get_next_tensor', 'ggml_get_no_alloc', 'ggml_get_rel_pos',
    'ggml_get_rows', 'ggml_get_rows_back', 'ggml_get_tensor',
    'ggml_get_type_traits', 'ggml_get_type_traits_cpu',
    'ggml_get_unary_op', 'ggml_graph_add_node', 'ggml_graph_clear',
    'ggml_graph_compute', 'ggml_graph_compute_with_ctx',
    'ggml_graph_cpy', 'ggml_graph_dump_dot',
    'ggml_graph_dump_dot_leaf_edge', 'ggml_graph_dump_dot_node_edge',
    'ggml_graph_dup', 'ggml_graph_export', 'ggml_graph_find',
    'ggml_graph_get_grad', 'ggml_graph_get_grad_acc',
    'ggml_graph_get_parent', 'ggml_graph_get_tensor',
    'ggml_graph_import', 'ggml_graph_n_nodes', 'ggml_graph_nbytes',
    'ggml_graph_node', 'ggml_graph_nodes', 'ggml_graph_overhead',
    'ggml_graph_overhead_custom', 'ggml_graph_plan',
    'ggml_graph_print', 'ggml_graph_reset', 'ggml_graph_size',
    'ggml_graph_view', 'ggml_group_norm', 'ggml_group_norm_impl',
    'ggml_group_norm_inplace', 'ggml_guid', 'ggml_guid_matches',
    'ggml_guid_t', 'ggml_hardsigmoid', 'ggml_hardswish',
    'ggml_hash_map_free', 'ggml_hash_set', 'ggml_hash_set_free',
    'ggml_hash_set_new', 'ggml_hash_set_p', 'ggml_hash_set_reset',
    'ggml_hash_size', 'ggml_im2col', 'ggml_im2col_back', 'ggml_init',
    'ggml_init_params', 'ggml_init_params_p', 'ggml_is_3d',
    'ggml_is_contiguous', 'ggml_is_contiguous_0',
    'ggml_is_contiguous_1', 'ggml_is_contiguous_2',
    'ggml_is_contiguous_channels', 'ggml_is_contiguous_n',
    'ggml_is_contiguously_allocated', 'ggml_is_empty',
    'ggml_is_matrix', 'ggml_is_numa', 'ggml_is_padded_1d',
    'ggml_is_permuted', 'ggml_is_quantized', 'ggml_is_scalar',
    'ggml_is_transposed', 'ggml_is_vector', 'ggml_is_view_op',
    'ggml_l2_norm', 'ggml_l2_norm_impl', 'ggml_l2_norm_inplace',
    'ggml_leaky_relu', 'ggml_log', 'ggml_log_callback',
    'ggml_log_callback_default', 'ggml_log_impl', 'ggml_log_inplace',
    'ggml_log_internal', 'ggml_log_internal_v', 'ggml_log_level',
    'ggml_log_set', 'ggml_logger_state', 'ggml_logger_state_p',
    'ggml_malloc', 'ggml_map_custom1', 'ggml_map_custom1_impl',
    'ggml_map_custom1_inplace', 'ggml_map_custom2',
    'ggml_map_custom2_impl', 'ggml_map_custom2_inplace',
    'ggml_map_custom3', 'ggml_map_custom3_impl',
    'ggml_map_custom3_inplace', 'ggml_mean', 'ggml_mul',
    'ggml_mul_impl', 'ggml_mul_inplace', 'ggml_mul_mat',
    'ggml_mul_mat_id', 'ggml_mul_mat_set_prec', 'ggml_n_dims',
    'ggml_nbytes', 'ggml_nbytes_pad', 'ggml_neg', 'ggml_neg_inplace',
    'ggml_nelements', 'ggml_new_buffer', 'ggml_new_f32',
    'ggml_new_graph', 'ggml_new_graph_custom', 'ggml_new_hash_map',
    'ggml_new_i32', 'ggml_new_object', 'ggml_new_tensor',
    'ggml_new_tensor_1d', 'ggml_new_tensor_2d', 'ggml_new_tensor_3d',
    'ggml_new_tensor_4d', 'ggml_new_tensor_impl', 'ggml_norm',
    'ggml_norm_impl', 'ggml_norm_inplace', 'ggml_nrows',
    'ggml_numa_init', 'ggml_numa_strategy', 'ggml_object',
    'ggml_object_p', 'ggml_object_type', 'ggml_op', 'ggml_op_desc',
    'ggml_op_name', 'ggml_op_pool', 'ggml_op_symbol',
    'ggml_opt_context', 'ggml_opt_context_p', 'ggml_opt_dataset',
    'ggml_opt_dataset_p', 'ggml_opt_dataset_t',
    'ggml_opt_epoch_callback', 'ggml_opt_optimizer_params',
    'ggml_opt_optimizer_params_adamw',
    'ggml_opt_optimizer_params_adamw_p',
    'ggml_opt_optimizer_params_p', 'ggml_opt_result',
    'ggml_opt_result_p', 'ggml_opt_result_t', 'ggml_opt_step_adamw',
    'ggml_out_prod', 'ggml_pad', 'ggml_pad_reflect_1d',
    'ggml_permute', 'ggml_pool_1d', 'ggml_pool_2d',
    'ggml_pool_2d_back', 'ggml_prec', 'ggml_print_backtrace',
    'ggml_print_backtrace_symbols', 'ggml_print_object',
    'ggml_print_objects', 'ggml_quantize_chunk', 'ggml_quantize_free',
    'ggml_quantize_init', 'ggml_quantize_requires_imatrix',
    'ggml_relu', 'ggml_relu_inplace', 'ggml_repeat', 'ggml_repeat_4d',
    'ggml_repeat_back', 'ggml_reset', 'ggml_reshape',
    'ggml_reshape_1d', 'ggml_reshape_2d', 'ggml_reshape_3d',
    'ggml_reshape_4d', 'ggml_rms_norm', 'ggml_rms_norm_back',
    'ggml_rms_norm_impl', 'ggml_rms_norm_inplace', 'ggml_rope',
    'ggml_rope_custom', 'ggml_rope_custom_inplace', 'ggml_rope_ext',
    'ggml_rope_ext_back', 'ggml_rope_ext_inplace', 'ggml_rope_impl',
    'ggml_rope_inplace', 'ggml_rope_multi', 'ggml_rope_multi_back',
    'ggml_rope_yarn_corr_dim', 'ggml_rope_yarn_corr_dims',
    'ggml_row_size', 'ggml_rwkv_wkv6', 'ggml_rwkv_wkv7', 'ggml_scale',
    'ggml_scale_impl', 'ggml_scale_inplace', 'ggml_scale_mode',
    'ggml_sched_priority', 'ggml_set', 'ggml_set_1d',
    'ggml_set_1d_inplace', 'ggml_set_2d', 'ggml_set_2d_inplace',
    'ggml_set_f32', 'ggml_set_f32_1d', 'ggml_set_f32_nd',
    'ggml_set_i32', 'ggml_set_i32_1d', 'ggml_set_i32_nd',
    'ggml_set_impl', 'ggml_set_inplace', 'ggml_set_input',
    'ggml_set_loss', 'ggml_set_name', 'ggml_set_no_alloc',
    'ggml_set_output', 'ggml_set_param', 'ggml_set_zero', 'ggml_sgn',
    'ggml_sgn_inplace', 'ggml_sigmoid', 'ggml_sigmoid_inplace',
    'ggml_silu', 'ggml_silu_back', 'ggml_silu_inplace', 'ggml_sin',
    'ggml_sin_impl', 'ggml_sin_inplace', 'ggml_soft_max',
    'ggml_soft_max_ext', 'ggml_soft_max_ext_back',
    'ggml_soft_max_ext_back_impl', 'ggml_soft_max_ext_back_inplace',
    'ggml_soft_max_impl', 'ggml_soft_max_inplace', 'ggml_sort_order',
    'ggml_sqr', 'ggml_sqr_impl', 'ggml_sqr_inplace', 'ggml_sqrt',
    'ggml_sqrt_impl', 'ggml_sqrt_inplace', 'ggml_ssm_conv',
    'ggml_ssm_scan', 'ggml_status', 'ggml_status_to_string',
    'ggml_step', 'ggml_step_inplace', 'ggml_sub', 'ggml_sub_impl',
    'ggml_sub_inplace', 'ggml_sub_or_set', 'ggml_sum',
    'ggml_sum_rows', 'ggml_tallocr', 'ggml_tallocr_alloc',
    'ggml_tallocr_new', 'ggml_tallocr_p', 'ggml_tanh',
    'ggml_tanh_inplace', 'ggml_tensor', 'ggml_tensor_flag',
    'ggml_tensor_overhead', 'ggml_tensor_p', 'ggml_threadpool',
    'ggml_threadpool_free', 'ggml_threadpool_get_n_threads',
    'ggml_threadpool_new', 'ggml_threadpool_p',
    'ggml_threadpool_params', 'ggml_threadpool_params_default',
    'ggml_threadpool_params_init', 'ggml_threadpool_params_match',
    'ggml_threadpool_params_p', 'ggml_threadpool_pause',
    'ggml_threadpool_resume', 'ggml_threadpool_t', 'ggml_time_init',
    'ggml_time_ms', 'ggml_time_us', 'ggml_timestep_embedding',
    'ggml_to_float_t', 'ggml_top_k', 'ggml_transpose', 'ggml_type',
    'ggml_type_name', 'ggml_type_size', 'ggml_type_sizef',
    'ggml_type_traits', 'ggml_type_traits_cpu',
    'ggml_type_traits_cpu_p', 'ggml_type_traits_p', 'ggml_unary',
    'ggml_unary_impl', 'ggml_unary_inplace', 'ggml_unary_op',
    'ggml_unary_op_name', 'ggml_unravel_index', 'ggml_upscale',
    'ggml_upscale_ext', 'ggml_upscale_impl', 'ggml_used_mem',
    'ggml_validate_row_data', 'ggml_vec_dot_bf16', 'ggml_vec_dot_f16',
    'ggml_vec_dot_f32', 'ggml_vec_dot_t', 'ggml_view_1d',
    'ggml_view_2d', 'ggml_view_3d', 'ggml_view_4d', 'ggml_view_impl',
    'ggml_view_tensor', 'ggml_visit_parents', 'ggml_win_part',
    'ggml_win_unpart', 'graph_copy_dup_tensor',
    'graph_copy_init_tensor', 'incr_ptr_aligned', 'int32_t',
    'int64_t', 'llama_adapter_lora_free', 'llama_adapter_lora_init',
    'llama_add_bos_token', 'llama_add_eos_token',
    'llama_apply_adapter_cvec', 'llama_attach_threadpool',
    'llama_attention_type', 'llama_backend_free',
    'llama_backend_init', 'llama_batch', 'llama_batch_free',
    'llama_batch_get_one', 'llama_batch_init',
    'llama_chat_apply_template', 'llama_chat_builtin_templates',
    'llama_chat_message', 'llama_clear_adapter_lora',
    'llama_context_default_params', 'llama_copy_state_data',
    'llama_decode', 'llama_detach_threadpool', 'llama_detokenize',
    'llama_encode', 'llama_free', 'llama_free_model', 'llama_ftype',
    'llama_get_embeddings', 'llama_get_embeddings_ith',
    'llama_get_embeddings_seq', 'llama_get_kv_self',
    'llama_get_logits', 'llama_get_logits_ith', 'llama_get_model',
    'llama_get_state_size', 'llama_init_from_model',
    'llama_kv_self_can_shift', 'llama_kv_self_clear',
    'llama_kv_self_defrag', 'llama_kv_self_n_tokens',
    'llama_kv_self_seq_add', 'llama_kv_self_seq_cp',
    'llama_kv_self_seq_div', 'llama_kv_self_seq_keep',
    'llama_kv_self_seq_pos_max', 'llama_kv_self_seq_pos_min',
    'llama_kv_self_seq_rm', 'llama_kv_self_update',
    'llama_kv_self_used_cells', 'llama_load_model_from_file',
    'llama_load_session_file', 'llama_log_set', 'llama_logit_bias',
    'llama_max_devices', 'llama_max_parallel_sequences',
    'llama_model_chat_template', 'llama_model_decoder_start_token',
    'llama_model_default_params', 'llama_model_desc',
    'llama_model_free', 'llama_model_get_vocab',
    'llama_model_has_decoder', 'llama_model_has_encoder',
    'llama_model_is_recurrent', 'llama_model_kv_override_type',
    'llama_model_load_from_file', 'llama_model_load_from_splits',
    'llama_model_meta_count', 'llama_model_meta_key_by_index',
    'llama_model_meta_val_str', 'llama_model_meta_val_str_by_index',
    'llama_model_n_ctx_train', 'llama_model_n_embd',
    'llama_model_n_head', 'llama_model_n_head_kv',
    'llama_model_n_layer', 'llama_model_n_params',
    'llama_model_quantize', 'llama_model_quantize_default_params',
    'llama_model_quantize_params',
    'llama_model_rope_freq_scale_train', 'llama_model_rope_type',
    'llama_model_save_to_file', 'llama_model_size', 'llama_n_batch',
    'llama_n_ctx', 'llama_n_ctx_train', 'llama_n_embd',
    'llama_n_head', 'llama_n_layer', 'llama_n_seq_max',
    'llama_n_threads', 'llama_n_threads_batch', 'llama_n_ubatch',
    'llama_n_vocab', 'llama_new_context_with_model',
    'llama_numa_init', 'llama_opt_epoch', 'llama_opt_init',
    'llama_opt_param_filter', 'llama_opt_param_filter_all',
    'llama_perf_context', 'llama_perf_context_print',
    'llama_perf_context_reset', 'llama_perf_sampler',
    'llama_perf_sampler_print', 'llama_perf_sampler_reset',
    'llama_pooling_type', 'llama_pos', 'llama_print_system_info',
    'llama_progress_callback', 'llama_rm_adapter_lora',
    'llama_rope_scaling_type', 'llama_rope_type',
    'llama_sampler_accept', 'llama_sampler_apply',
    'llama_sampler_chain_add', 'llama_sampler_chain_default_params',
    'llama_sampler_chain_get', 'llama_sampler_chain_init',
    'llama_sampler_chain_n', 'llama_sampler_chain_params',
    'llama_sampler_chain_remove', 'llama_sampler_clone',
    'llama_sampler_context_t', 'llama_sampler_free',
    'llama_sampler_get_seed', 'llama_sampler_init',
    'llama_sampler_init_dist', 'llama_sampler_init_dry',
    'llama_sampler_init_grammar', 'llama_sampler_init_grammar_lazy',
    'llama_sampler_init_grammar_lazy_patterns',
    'llama_sampler_init_greedy', 'llama_sampler_init_infill',
    'llama_sampler_init_logit_bias', 'llama_sampler_init_min_p',
    'llama_sampler_init_mirostat', 'llama_sampler_init_mirostat_v2',
    'llama_sampler_init_penalties', 'llama_sampler_init_softmax',
    'llama_sampler_init_temp', 'llama_sampler_init_temp_ext',
    'llama_sampler_init_top_k', 'llama_sampler_init_top_n_sigma',
    'llama_sampler_init_top_p', 'llama_sampler_init_typical',
    'llama_sampler_init_xtc', 'llama_sampler_name',
    'llama_sampler_reset', 'llama_sampler_sample',
    'llama_save_session_file', 'llama_seq_id',
    'llama_set_abort_callback', 'llama_set_adapter_lora',
    'llama_set_causal_attn', 'llama_set_embeddings',
    'llama_set_n_threads', 'llama_set_state_data', 'llama_set_warmup',
    'llama_split_mode', 'llama_split_path', 'llama_split_prefix',
    'llama_state_get_data', 'llama_state_get_size',
    'llama_state_load_file', 'llama_state_save_file',
    'llama_state_seq_get_data', 'llama_state_seq_get_size',
    'llama_state_seq_load_file', 'llama_state_seq_save_file',
    'llama_state_seq_set_data', 'llama_state_set_data',
    'llama_supports_gpu_offload', 'llama_supports_mlock',
    'llama_supports_mmap', 'llama_supports_rpc', 'llama_synchronize',
    'llama_time_us', 'llama_token', 'llama_token_attr',
    'llama_token_bos', 'llama_token_cls', 'llama_token_data',
    'llama_token_data_array', 'llama_token_eos', 'llama_token_eot',
    'llama_token_fim_mid', 'llama_token_fim_pad',
    'llama_token_fim_pre', 'llama_token_fim_rep',
    'llama_token_fim_sep', 'llama_token_fim_suf',
    'llama_token_get_attr', 'llama_token_get_score',
    'llama_token_get_text', 'llama_token_is_control',
    'llama_token_is_eog', 'llama_token_nl', 'llama_token_pad',
    'llama_token_sep', 'llama_token_to_piece', 'llama_token_type',
    'llama_tokenize', 'llama_vocab_bos', 'llama_vocab_cls',
    'llama_vocab_eos', 'llama_vocab_eot', 'llama_vocab_fim_mid',
    'llama_vocab_fim_pad', 'llama_vocab_fim_pre',
    'llama_vocab_fim_rep', 'llama_vocab_fim_sep',
    'llama_vocab_fim_suf', 'llama_vocab_get_add_bos',
    'llama_vocab_get_add_eos', 'llama_vocab_get_attr',
    'llama_vocab_get_score', 'llama_vocab_get_text',
    'llama_vocab_is_control', 'llama_vocab_is_eog',
    'llama_vocab_n_tokens', 'llama_vocab_nl', 'llama_vocab_pad',
    'llama_vocab_pre_type', 'llama_vocab_sep', 'llama_vocab_type',
    'size_t', 'struct__0', 'struct__IO_FILE', 'struct__IO_codecvt',
    'struct__IO_marker', 'struct__IO_wide_data',
    'struct___va_list_tag', 'struct_c__SA_ggml_bf16_t',
    'struct_hash_map', 'struct_llama_adapter_lora',
    'struct_llama_batch', 'struct_llama_chat_message',
    'struct_llama_context', 'struct_llama_context_params',
    'struct_llama_kv_cache', 'struct_llama_logit_bias',
    'struct_llama_model', 'struct_llama_model_kv_override',
    'struct_llama_model_params', 'struct_llama_model_quantize_params',
    'struct_llama_model_tensor_buft_override',
    'struct_llama_opt_params', 'struct_llama_perf_context_data',
    'struct_llama_perf_sampler_data', 'struct_llama_sampler',
    'struct_llama_sampler_chain_params', 'struct_llama_sampler_i',
    'struct_llama_token_data', 'struct_llama_token_data_array',
    'struct_llama_vocab', 'uint32_t', 'uint64_t', 'uint8_t',
    'union_llama_model_kv_override_0', 'va_list']
