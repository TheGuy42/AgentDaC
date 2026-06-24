import chess


INSTRUCTION = (
    "You are given a chess position. Find the best move for your side to play. Analyze thoroughly the position and consider different strategies."
)

ANSWER_FORMAT = (
    "Reply with the move in UCI long-algebraic notation. When You answer, your final Text must contain ONLY the best move and nothing else."
)


def render_board(board: chess.Board) -> str:
    """ASCII board from White's point of view (uppercase = White, lowercase = Black)."""
    return str(board)


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
        "Board (uppercase = White, lowercase = Black, '.' = empty square):",
        render_board(board),
        "",
        "Legal moves (UCI):",
        legal_moves,
        "",
        "Answer format:",
        ANSWER_FORMAT,
    ]
    return "\n".join(parts)
