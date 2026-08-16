import json
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

in_path = r'E:\05deep L\DeepLearing_13.7\DL_gui_thay_EN\_translate\chunk_4'
out_path = r'E:\05deep L\DeepLearing_13.7\DL_gui_thay_EN\_translate\chunk_4.jsonl'

with open(in_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')

# The file has 479 lines; if split produces a trailing empty string due to final newline, drop it
if lines and lines[-1] == '':
    lines = lines[:-1]

print(len(lines))
