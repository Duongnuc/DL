"""Extract candidate Vietnamese lines from a target file that still need translation.

A line is a candidate if either:
  (a) it contains any non-ASCII character (accented Vietnamese), or
  (b) it exactly matches a line already known to be Vietnamese from the
      original (pre-translation) sibling file in DL_gui_thay/ at the same
      relative path, AND that original-language line is unchanged in the
      target (i.e. not yet translated) -- this catches diacritic-free
      Vietnamese (ASCII-only Vietnamese words like "khong", "toan bo", etc).

Usage: python extract_vn.py <orig_file_in_DL_gui_thay> <target_file_in_DL_gui_thay_EN> <out_candidates_file> <master_dict_file(optional, TSV vi\ten)>

Prints counts of: lines already covered by master dict (auto-applied) vs new
lines needing manual translation (written to out_candidates_file, one per line,
in first-seen order, deduplicated).
"""
import io
import sys


def load_lines(path):
    with io.open(path, encoding="utf-8") as f:
        content = f.read()
    return content.split("\n")


def is_candidate(orig_line, new_line):
    if orig_line != new_line:
        return False  # already translated (differs from original)
    if not new_line.strip():
        return False
    if any(ord(ch) > 127 for ch in new_line):
        return True
    # diacritic-free Vietnamese heuristic word list
    vn_words = set('''khong tu la va cua cho duoc mot nay tren voi den nhu neu thi dung ma da o nen bi ra vao
trong ngoai sau truoc gop toan bo ket qua chay xong huan luyen mo hinh sieu tham so du lieu ve tao doc lai
buoc phan luu y chuan bi dinh nghia kien truc tim uoc luong tinh trung binh gia tri hinh anh tot nhat gian
phai an tai bai nhat lam nhieu ngay them qua sap xep giam dan chi nhung goi bang giua moi con neu day
day du con lai lay xoa them bot muc dich dieu kien truong hop ham so bien so gia su vi du chang han
thuc te ly do can thiet quan trong dac biet chinh xac sai so uu tien mac dinh khoi tao ket thuc bat dau
danh gia so sanh phan tich tong hop rieng chung toan the tat ca hau het phan lon it nhat nhieu nhat
truoc do sau do dong thoi cung luc lien tuc lap lai kiem tra xac nhan bao gio cung khong bao gio
duong dan tap tin thu muc bien luu tru doc ghi mo dong tao ra xoa bo cap nhat toan thoi gian chay phut
nguon diem mau kich thuoc so luong loai hinh nhom rieng biet khac nhau giong nhau tuong tu vong theo'''.split())
    low = new_line.lower()
    words = set(low.replace("(", " ").replace(")", " ").replace(",", " ")
                .replace(".", " ").replace(":", " ").replace("-", " ")
                .replace('"', " ").split())
    return bool(words & vn_words)


def main():
    orig_path, target_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    master_path = sys.argv[4] if len(sys.argv) > 4 else None

    orig_lines = load_lines(orig_path)
    target_lines = load_lines(target_path)

    master = {}
    if master_path:
        with io.open(master_path, encoding="utf-8") as f:
            for line in f:
                if "\t" in line:
                    k, v = line.rstrip("\n").split("\t", 1)
                    master[k] = v

    candidates = []
    seen = set()
    auto_applied = 0
    n = min(len(orig_lines), len(target_lines))
    out_target = list(target_lines)
    for i in range(n):
        o, t = orig_lines[i], target_lines[i]
        if is_candidate(o, t):
            if o in master:
                out_target[i] = master[o]
                auto_applied += 1
            elif o not in seen:
                seen.add(o)
                candidates.append(o)

    with io.open(target_path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(out_target))

    with io.open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(candidates))

    print(f"auto-applied from master dict: {auto_applied}")
    print(f"new candidates needing translation: {len(candidates)} -> {out_path}")


if __name__ == "__main__":
    main()
