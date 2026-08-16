"""Apply a vi->en line-substitution dictionary to a target .py file.

Usage: python apply_translation.py <vi_lines_file> <en_lines_file> <target_py_file>

Reads two parallel line-list files (same line count, one-to-one positional
correspondence), zips them into a {vi_line: en_line} dict, then rewrites the
target file by replacing any physical line that exactly matches a vi_line key.
Lines that don't match any key (i.e. already-English / pure-code lines) pass
through unchanged. Prints a summary of how many lines were substituted, and
lists any vi_lines that were never matched in the target file (helps catch
stray whitespace differences).
"""
import io
import sys


def load_lines(path):
    with io.open(path, encoding="utf-8") as f:
        content = f.read()
    lines = content.split("\n")
    while lines and lines[-1] == "":
        lines.pop()
    return lines


def main():
    vi_path, en_path, target_path = sys.argv[1], sys.argv[2], sys.argv[3]

    vi_lines = load_lines(vi_path)
    en_lines = load_lines(en_path)
    if len(vi_lines) != len(en_lines):
        print(f"ERROR: line count mismatch: {len(vi_lines)} vi vs {len(en_lines)} en")
        sys.exit(1)

    mapping = dict(zip(vi_lines, en_lines))

    with io.open(target_path, encoding="utf-8") as f:
        target_lines = f.read().split("\n")

    matched_keys = set()
    n_sub = 0
    out_lines = []
    for line in target_lines:
        if line in mapping:
            out_lines.append(mapping[line])
            matched_keys.add(line)
            n_sub += 1
        else:
            out_lines.append(line)

    unmatched = [k for k in vi_lines if k not in matched_keys]

    with io.open(target_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out_lines))

    print(f"Substituted {n_sub} line-occurrences out of {len(target_lines)} total lines.")
    print(f"Dictionary had {len(vi_lines)} unique vi lines; {len(unmatched)} never matched in target.")
    if unmatched:
        print("Unmatched vi lines (first 20):")
        for u in unmatched[:20]:
            print(repr(u))


if __name__ == "__main__":
    main()
