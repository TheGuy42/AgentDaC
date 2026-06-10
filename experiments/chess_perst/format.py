import chess


INSTRUCTION = (
    "You are playing as {side}. Choose the single strongest legal move for {side}.\n"
    "Reply with the move in UCI long-algebraic notation: the source square followed by the "
    "destination square, plus a promotion piece letter if the move promotes a pawn "
    "(e.g. g1f3, e2e4, or e7e8q).\n"
    "Your final answer's Text must contain ONLY that move and nothing else."
)


def render_board(board: chess.Board) -> str:
    """ASCII board from White's point of view (uppercase = White, lowercase = Black)."""
    return str(board)


def format_prompt(sample: dict) -> str:
    board = chess.Board(sample["fen"])
    side = "White" if board.turn == chess.WHITE else "Black"
    legal_moves = " ".join(sorted(move.uci() for move in board.legal_moves))

    parts = [
        f"You are given a chess position. It is {side} to move.",
        "",
        "Board (uppercase = White, lowercase = Black, '.' = empty square):",
        render_board(board),
        "",
        f"FEN: {board.fen()}",
        f"Legal moves (UCI): {legal_moves}",
        "",
        INSTRUCTION.format(side=side),
    ]
    return "\n".join(parts)
