import chess


INSTRUCTION = (
    "You are given a chess position. Find the best legal move for the side to move. "
    "Analyze the position thoroughly and consider different strategies, but the final answer must be only one move."
)

ANSWER_FORMAT = "The final answer must be exactly one legal move in UCI notation. It should contain only the move and nothing else."


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
        "Answer format:",
        ANSWER_FORMAT,
    ]
    return "\n".join(parts)
