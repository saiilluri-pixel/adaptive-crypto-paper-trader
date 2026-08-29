"""
Safety verification for the adaptive/ module (section 25 of the adaptive
spec): no reachable real-order route anywhere in the active execution
path. Public Binance Spot REST endpoints only, no API keys, no
authenticated trading, no margin, no futures.
"""
import ast
import os

ADAPTIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "adaptive")

BANNED_TOKENS = ("create_order", "createOrder", "cancel_order", "cancelOrder",
                  "withdraw", "transfer(", "apiKey", "api_key", "private_post", "sign(")
# margin/futures are checked separately (see below) since "future" is a
# common English word that appears legitimately in comments about roadmap
# intent -- the AST strip already removes comments/docstrings, so a literal
# match here is a real code reference, not prose


def _source_without_docstrings_and_comments(path):
    with open(path) as f:
        src = f.read()
    tree = ast.parse(src)
    doc_lines = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if (node.body and isinstance(node.body[0], ast.Expr)
                    and isinstance(node.body[0].value, ast.Constant)
                    and isinstance(node.body[0].value.value, str)):
                d = node.body[0]
                doc_lines.update(range(d.lineno, d.end_lineno + 1))
    kept = []
    for i, line in enumerate(src.splitlines(), start=1):
        if i in doc_lines:
            continue
        kept.append(line.split("#", 1)[0])
    return "\n".join(kept)


def _adaptive_py_files():
    return [os.path.join(ADAPTIVE_DIR, f) for f in sorted(os.listdir(ADAPTIVE_DIR)) if f.endswith(".py")]


def test_no_authenticated_order_code_path_in_adaptive_module():
    for path in _adaptive_py_files():
        code = _source_without_docstrings_and_comments(path)
        for token in BANNED_TOKENS:
            assert token not in code, f"{path} contains banned authenticated-order token {token!r} in live code"


MARGIN_TRADING_TOKENS = ("marginmode", "margin_mode", "isolated_margin", "cross_margin",
                          "add_margin", "reduce_margin", "setmargin", "setleverage", "leverage=")


def test_no_margin_or_futures_code_path_in_adaptive_module():
    # bare "margin" is deliberately NOT checked here -- adaptive/dashboard.py
    # embeds a CSS stylesheet as a Python string constant, and the CSS
    # `margin:`/`margin-top:` properties are legitimate styling, not a
    # margin-TRADING code path. Precise ccxt/margin-trading identifiers
    # below are what actually indicate a reachable margin order route.
    for path in _adaptive_py_files():
        code = _source_without_docstrings_and_comments(path)
        code_lower = code.lower()
        for token in MARGIN_TRADING_TOKENS:
            assert token not in code_lower, f"{path} references margin-trading token {token!r} in live code"
        assert "defaultType" not in code, f"{path} sets a ccxt defaultType override (spot-only expected)"


def test_market_data_builds_plain_spot_exchange():
    from adaptive.market_data import build_exchange
    ex = build_exchange()
    assert ex.id == "binance"
    assert not ex.apiKey
    assert not ex.secret
    assert ex.options.get("defaultType", "spot") == "spot"


def test_all_symbols_are_plain_spot_usdt_pairs():
    from adaptive.market_data import SYMBOLS
    for sym in SYMBOLS:
        assert sym.endswith("/USDT")
        assert ":" not in sym  # ccxt perpetual/futures symbols carry a ":SETTLE" suffix; spot never does
