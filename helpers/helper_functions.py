# ---------------------
# Helper Functions
# ---------------------
def extract_xml_answer(text: str) -> str:
    """
    1) Truncate 'text' at <|endoftext|> if present.
    2) Find the FIRST fully-completed <answer>...</answer> block in that truncated text.
    3) Return that block's content, or None if not found.
    """
    try:
        # 1) Truncate at <|endoftext|>
        eot_index = text.find("<|endoftext|>")
        truncated_text = text[:eot_index] if eot_index != -1 else text

        # 2) Find the FIRST <answer> tag
        start = truncated_text.find("<answer>")
        if start == -1:
            return None

        # 3) Find the NEXT </answer> after this <answer>
        end = truncated_text.find("</answer>", start)
        if end == -1:
            return None

        return truncated_text[start+len("<answer>"):end].strip()

    except Exception:
        return None

def execute_prolog_code(prolog_code: str) -> str:
    """
    Executes the given Prolog code in SWI-Prolog, calling solve(X),
    and returns the printed solution as a string (e.g., "12000").
    Returns None if there's an error or no output.
    """
    try:
        # Write the Prolog code to a temporary file
        with open("temp.pl", "w") as f:
            f.write(prolog_code)

        # Run SWI-Prolog: load 'temp.pl', call solve(X), print X, then halt
        result = subprocess.run(
            ["swipl", "-q", "-f", "temp.pl", "-g", "solve(X), writeln(X), halt"],
            capture_output=True,
            text=True,
            timeout=5,  # optional: 5-second timeout
        )

        # If there's any error output, we can check result.stderr or result.returncode
        if result.returncode != 0 or not result.stdout:
            return None

        # result.stdout is whatever got printed by writeln(X)
        lines = result.stdout.strip().splitlines()
        return lines[-1].strip() if lines else None

    except Exception as e:
        print(f"Error executing Prolog code: {e}")
        return None
