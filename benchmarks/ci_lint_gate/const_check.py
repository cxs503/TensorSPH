"""Compare every string-literal constant (in source order) between two versions
of a file, using the AST.  Used to prove that the E501 line splitting kept the
documents byte-identical.
"""
import ast
import sys


def constants(path):
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = []

    class V(ast.NodeVisitor):
        def visit_Constant(self, node):
            if isinstance(node.value, str):
                out.append(node.value)

    V().visit(tree)
    return out


def main():
    a_path, b_path = sys.argv[1], sys.argv[2]
    a, b = constants(a_path), constants(b_path)
    if a == b:
        print(f"OK  {b_path}: {len(a)} string constants identical to {a_path}")
        return 0
    print(f"DIFF {b_path}: {len(a)} vs {len(b)} constants")
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            print(f"  first difference at #{i} (len {len(x)} vs {len(y)})")
            print(f"  A: {x[:200]!r}")
            print(f"  B: {y[:200]!r}")
            break
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
