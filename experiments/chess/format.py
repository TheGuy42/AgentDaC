import chess


# TODO: biased instruction towards sub-tasks
INSTRUCTION = (
    "You are given a chess position. Find the best legal move for the side to move. Important: don't reason too long, you have token limits. "
    "Use sub-tasks to break down the problem if needed and break down the reasoning into smaller steps."
)

ANSWER_FORMAT = (
    "The final answer must be exactly one legal move in UCI notation. The final move should appear within \\boxed{<uci_move>} as a single move. "
)


def render_board(board: chess.Board) -> str:
    """ASCII board from White's point of view.

    Uppercase = White pieces
    Lowercase = Black pieces
    . = empty square
    """
    lines: list[str] = []

    for rank in range(7, -1, -1):  # 8 -> 1
        row: list[str] = []

        for file in range(8):  # a -> h
            square = chess.square(file, rank)
            piece = board.piece_at(square)
            row.append(piece.symbol() if piece is not None else ".")

        lines.append(f"{rank + 1}  " + " ".join(row))

    lines.append("   a b c d e f g h")
    return "\n".join(lines)


def format_prompt(sample: dict) -> str:
    board = chess.Board(sample["fen"])
    side = "White" if board.turn == chess.WHITE else "Black"
    legal_moves = " ".join(sorted(move.uci() for move in board.legal_moves))

    parts = [
        "Task:",
        INSTRUCTION,
        "",
        "Current FEN:",
        sample["fen"],
        "",
        "Side to move:",
        side,
        "",
        "Board, White at bottom (uppercase = White, lowercase = Black, '.' = empty square):",
        render_board(board),
        "",
        "Legal moves (UCI):",
        legal_moves,
        "",
        "Final answer format:",
        ANSWER_FORMAT,
    ]
    return "\n".join(parts)
