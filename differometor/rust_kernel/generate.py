"""Translate the existing scalar component formulas to straight-line Rust at build time."""
import math
import types
from pathlib import Path
import numpy as np
from differometor import components


def generate():
    """Trace each registered formula, hoist invariant nodes, and emit its transpose."""

    def literal(value):
        if isinstance(value, Expr):
            return value.name
        value = complex(value)
        return f"C::new({value.real!r}, {value.imag!r})"

    class Expr:
        def __init__(self, name):
            self.name = name

        def __add__(self, y):
            return emit("+", self, y)

        def __radd__(self, y):
            return emit("+", y, self)

        def __sub__(self, y):
            return emit("-", self, y)

        def __rsub__(self, y):
            return emit("-", y, self)

        def __mul__(self, y):
            return emit("*", self, y)

        def __rmul__(self, y):
            return emit("*", y, self)

        def __truediv__(self, y):
            return emit("/", self, y)

        def __rtruediv__(self, y):
            return emit("/", y, self)

        def __neg__(self):
            return emit("neg", self)

        def __pow__(self, y):
            if y != 2:
                raise NotImplementedError("Only squared expression nodes are supported")
            return self * self

    def emit(op, x, y=None):
        key = (op, literal(x), None if y is None else literal(y))
        if key in cache:
            return cache[key]
        a = literal(x)
        rhs = (
            f"{a} {op} {literal(y)}"
            if y is not None
            else (f"-{a}" if op == "neg" else f"{a}.{op}()")
        )
        v = Expr(f"v{len(lines)}")
        lines.append(rhs)
        tape.append((v, op, x, y))
        cache[key] = v
        return v

    class Math:
        pi = math.pi
        e = math.e
        array = staticmethod(lambda v: np.asarray(v, dtype=object))
        radians = staticmethod(lambda x: x * (math.pi / 180))

        def __getattr__(self, k):
            if k in ("conjugate", "conj"):
                k = "conj"
            if k == "arcsin":
                k = "asin"
            if k == "log10":
                return math.log10
            return lambda x: emit(k, x)

    ns = dict(vars(components), jnp=Math())
    for k, f in vars(components).items():
        if isinstance(f, types.FunctionType) and f.__module__ == components.__name__:
            ns[k] = types.FunctionType(f.__code__, ns, k, f.__defaults__, f.__closure__)
    ns["standardize_output"] = lambda v: np.asarray(
        v + [0] * (4 - len(v)), dtype=object
    )

    src = [
        "use num_complex::Complex64 as C;",
        f'#[no_mangle] pub extern "C" fn feyer_function_count() -> usize {{ {len(components.FUNCTIONS)} }}',
    ]

    def gradname(x):
        return "d" + x.name.replace("p[", "p").replace("]", "")

    for n, fn in enumerate(components.FUNCTIONS):
        lines = []
        cache = {}
        tape = []
        args = [Expr(f"p[{i}]") for i in range(7)]
        out = ns[fn.__name__](args)
        masks = {arg.name: 1 << j for j, arg in enumerate(args)}
        pre = []
        cached_lines = []
        for i, (v, op, x, y) in enumerate(tape):
            mask = masks.get(literal(x), 0) | (
                masks.get(literal(y), 0) if y is not None else 0
            )
            masks[v.name] = mask
            rhs = lines[i]
            pre.append(
                f"let {v.name} = if M & {mask} == 0 {{ {rhs} }} else {{ C::default() }};"
            )
            pre.append(f"cache[{i}]={v.name};")
            cached_lines.append(
                f"let {v.name}=if M & {mask} == 0 {{ cache[{i}] }} else {{ {rhs} }};"
            )
        src += [f"fn pre{n}<const M:u8>(p:&[C;7],cache:&mut [C;128]) {{", *pre, "}"]
        if len(tape) > 128:
            raise ValueError(f"Scratch space exceeded by {fn.__name__}")
        src += [
            f"fn eval{n}<const M:u8>(p: &[C;7],cache:&[C;128]) -> [C;4] {{",
            *cached_lines,
            "[" + ", ".join(map(literal, out)) + "]",
            "}",
        ]
        # JAX complex cotangents use the bilinear transpose, including conjugation for conj.
        used = set(literal(o) for o in out)
        reverse = []
        for v, op, x, y in reversed(tape):
            if v.name not in used:
                continue
            g = gradname(v)
            a = literal(x)
            b = literal(y) if y is not None else ""
            if op == "+":
                parts = [g, g]
            elif op == "-":
                parts = [g, "-" + g]
            elif op == "*":
                parts = [g + "*" + b, g + "*" + a]
            elif op == "/":
                parts = [g + "/" + b, f"-{g}*{v.name}/{b}"]
            elif op == "neg":
                parts = ["-" + g]
            elif op == "conj":
                parts = [g + ".conj()"]
            else:
                slope = {
                    "exp": v.name,
                    "sqrt": f"C::new(0.5,0.0)/{v.name}",
                    "cos": "-" + a + ".sin()",
                    "sin": a + ".cos()",
                    "cosh": a + ".sinh()",
                    "sinh": a + ".cosh()",
                    "asin": f"C::new(1.0,0.0)/(C::new(1.0,0.0)-{a}*{a}).sqrt()",
                }[op]
                parts = [f"{g}*({slope})"]
            for var, term in zip((x, y), parts):
                if isinstance(var, Expr):
                    reverse.append(f"{gradname(var)} += {term};")
                    used.add(var.name)
        vars_ = [x for x in args + [r[0] for r in tape] if x.name in used]
        src += [
            f"fn back{n}<const M:u8>(p:&[C;7], g:&[C;4],cache:&[C;128]) -> [C;7] {{",
            *cached_lines,
            *[f"let mut {gradname(x)}=C::default();" for x in vars_],
            *[
                f"{gradname(x)}+=g[{i}];"
                for i, x in enumerate(out)
                if isinstance(x, Expr)
            ],
            *reverse,
            "["
            + ", ".join(gradname(x) if x.name in used else "C::default()" for x in args)
            + "]",
            "}",
        ]
    for name, sig, typ, call in [
        ("pre", "p:&[C;7],cache:&mut [C;128]", "()", "p,cache"),
        ("eval", "p:&[C;7],cache:&[C;128]", "[C;4]", "p,cache"),
        ("back", "p:&[C;7],g:&[C;4],cache:&[C;128]", "[C;7]", "p,g,cache"),
    ]:
        src += [
            f"fn {name}<const M:u8>(kind:i32,{sig}) -> {typ} {{ match kind {{",
            *[
                f"{i} => {name}{i}::<M>({call}),"
                for i in range(len(components.FUNCTIONS))
            ],
            "_ => unreachable!(),",
            "}}",
        ]
    Path("src/formulas.rs").write_text("\n".join(src))


if __name__ == "__main__":
    generate()
