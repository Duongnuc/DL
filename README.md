# DL_gui_thay — Pipeline du doan co ngot Be tong tu dam (SCC)

## Huong dan cho cac lan lam viec sau

Doc ba file nay truoc khi doc toan bo thu muc:

- `PROJECT_CONTEXT.md`: muc tieu, cau truc, trang thai va lich su cap nhat.
- `PARAMETER_SOURCE_AUDIT.md`: nguon tham khao cho cac tham so rieng cua tung mo hinh.
- `TODO_NOTES.md`: cac viec con lai va diem can kiem tra.

Khong dua cac file tu sinh vao Git: `_cache/`, `tmp/` va `__pycache__/`.

Ban rut gon cua thu muc `DL/` (26 file `buoc_XX_*.py` + 3 module dung
chung), gop lai thanh **15 file doc lap**. Moi file `NN_ten.py` TU CHAY
DUOC (khong import file `.py` nao khac trong bo nay ngoai thu vien
ngoai), tu doc du lieu tho trong `data_raw/`, tu tien xu ly, tu huan
luyen/danh gia, va IN KET QUA RA CONSOLE (khong chi ghi file).

## Cau truc thu muc

```
DL_gui_thay/
  data_raw/                  # du lieu tho (Database SCC.xlsx) - da co san
  01_annbasic.py .. 15_do_nhay_optuna.py
  requirements.txt
  README.md
  _cache/                    # (tu sinh khi chay) ket qua trung gian: k-fold summary,
                              #   oof predictions, best_params.json, ... theo tung model
  output_figures/            # (tu sinh khi chay) hinh cho bai bao (PNG)
  output_tables/             # (tu sinh khi chay) bang cho bai bao (CSV)
```

`_cache/` la noi luu ket qua trung gian (khong can xem, chi de doi
chieu/debug). `output_figures/` va `output_tables/` la cac deliverable
chinh can dua vao bai bao.

## Danh sach 15 file

| File | Noi dung |
|---|---|
| `01_annbasic.py` | Mo hinh ANN co ban (baseline MLP) + Optuna k-fold |
| `02_mlp_tapered.py` | MLP tapered (giam dan so chieu an) + Optuna k-fold |
| `03_resmlp.py` | ResMLP (khoi residual) + Optuna k-fold + kiem tra data leakage |
| `04_ft_transformer.py` | FT-Transformer (tabular transformer) + Optuna k-fold |
| `05_tabnet.py` | TabNet (attention-based tabular) + Optuna k-fold |
| `06_dcnv2.py` | DCNv2 (Deep & Cross Network v2) + Optuna k-fold |
| `07_gru.py` | GRU (mo hinh chuoi theo thoi gian) + Optuna k-fold |
| `08_lstm.py` | LSTM (mo hinh chuoi theo thoi gian) + Optuna k-fold |
| `09_tcn.py` | TCN (Temporal Convolutional Network) + Optuna k-fold |
| `10_gbdt.py` | GBDT moc so sanh: XGBoost + LightGBM, cong kiem tra do ben (fold theo nguon tai lieu, doi seed) |
| `11_cong_thuc_kinh_nghiem.py` | Cong thuc kinh nghiem ACI 209, B3, B4, MC2010 lam moc so sanh phi-DL |
| `12_tong_hop_so_sanh.py` | Tong hop bang so sanh tat ca mo hinh + bootstrap CI (theo cum GID) + khoang du doan conformal |
| `13_giai_thich_resmlp.py` | Giai thich mo hinh ResMLP: SHAP, Partial Dependence Plot, learning curves |
| `14_hinh_ve_bao_cao.py` | Toan bo hinh/bang cho bai bao: mo ta du lieu, phan bo, tuong quan, duong phat trien co ngot theo thoi gian |
| `15_do_nhay_optuna.py` | Do nhay ngan sach Optuna: chay lai top-3 mo hinh DL (ResMLP, GRU, LSTM) voi 30 trial/fold (thay vi 8) de kiem tra ket luan co ben vung khong |

## Cai dat

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Yeu cau Python 3.11 tro len. Cac model DL chay tren CPU theo mac
dinh (khong bat buoc GPU/CUDA).

## Chay

Moi file chay doc lap, vi du:

```bash
python 03_resmlp.py
python 15_do_nhay_optuna.py
```

Khuyen nghi thu tu chay: 01 -> 10 (huan luyen tung mo hinh, sinh
`_cache/<model>/...`), roi 11 (cong thuc kinh nghiem), 12 (tong hop
so sanh, can ket qua `_cache/` cua 01-11), 13-14 (giai thich + hinh
bao cao), cuoi cung 15 (do nhay Optuna, ton nhieu thoi gian nhat).

**Luu y ve thoi gian chay**: cac file Optuna k-fold day du (01-10,
15) co the mat tu vai phut den vai chuc phut moi file tuy cau hinh
`N_TRIALS`/`N_SPLITS`/so epoch trong tung file — day la cau hinh
goc dung cho ket qua chinh thuc trong bai bao, khong rut gon.

## Nguon goc

Duoc gop tu thu muc `DL/` goc (26 file `buoc_XX_*.py` +
`common_tabular.py`, `common_sequence_group.py`,
`common_console_summary.py`, `buoc_01_code_loc_data.py`). Moi file
gop ghi ro trong docstring dau file no duoc gop tu file goc nao.
