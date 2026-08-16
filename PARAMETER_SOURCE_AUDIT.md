# Audit nguồn tham số cho các thuật toán

Last reviewed: 2026-08-15

## Cách dùng tài liệu này trong bài báo

Mỗi mô hình cần phân biệt ba lớp, không gộp chung thành “tham số của thuật toán”:

1. **Kiến trúc/mô hình**: tham số quyết định cấu trúc, lấy từ bài báo gốc hoặc API chính thức.
2. **Tối ưu/regularization**: learning rate, dropout, weight decay, early stopping; lấy từ
   tài liệu phương pháp chung và được tune trên validation.
3. **Protocol thí nghiệm**: seed, số fold, trial budget, epoch cap, batch size; là quyết định
   tái lập/chi phí tính toán của nghiên cứu, không phải tham số riêng của thuật toán.

Nguồn gốc thường **không** quy định một dải số đúng cho mọi bộ dữ liệu. Vì vậy các cận trong
code là dải thực nghiệm của project, cần được báo cáo là “search space” và được kiểm chứng bằng
validation, không được mô tả là sao chép nguyên văn từ bài báo gốc.

## Nguồn dùng chung

- Dropout: Srivastava et al. (2014), [Dropout: A Simple Way to Prevent Neural Networks from Overfitting](https://jmlr.org/papers/v15/srivastava14a.html).
- AdamW/weight decay: Loshchilov & Hutter (2019), [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101).
- Hyperparameter optimization: Akiba et al. (2019), [Optuna](https://arxiv.org/abs/1907.10902).
- Các dải `lr` và `wd` được search theo log scale vì chúng có độ lớn thay đổi theo cấp số nhân;
  đây là quy ước search của project, không phải một giá trị chuẩn cố định.

## Thuật toán ML/DL

| File / model | Tham số đang tune trong code | Nguồn kiến trúc và lý do chọn nhóm tham số | Trạng thái dải số hiện tại |
|---|---|---|---|
| `01_annbasic.py` - ANNBasic | `lr`, `wd`, `dropout`, `ann_h`, `ann_layers` | MLP cơ bản; width/depth là năng lực mô hình, dropout/AdamW là regularization/tối ưu dùng chung. | Dải `lr 2e-4-3e-3`, `wd 1e-6-5e-4`, dropout `0.05-0.35`, width `32-256`, depth `1-4` là **thực nghiệm nội bộ**; không có bài gốc riêng cho ANNBasic. |
| `02_mlp_tapered.py` - MLP Tapered | `lr`, `wd`, `dropout`, `taper_h1` | Biến thể MLP do project định nghĩa (hidden layers giảm dần), không phải thuật toán chuẩn có paper gốc riêng. | Các cận, gồm `taper_h1 64-512`, là **thiết kế/tuning nội bộ**. Chỉ trích dẫn nguồn chung cho MLP, dropout và AdamW. |
| `03_resmlp.py` - ResMLP | `lr`, `wd`, `dropout`, `res_d`, `res_blocks` | Skip connections dựa trên [He et al. (2016), ResNet](https://arxiv.org/abs/1512.03385); ResNet-like tabular baseline từ [Gorishniy et al. (2021)](https://arxiv.org/abs/2106.11959). | `res_d 96-224`, `res_blocks 2-4` là dải nội bộ; không được gán là thông số nguyên bản của ResNet/FT-Transformer paper. |
| `04_ft_transformer.py` - FT-Transformer | `lr`, `wd`, `dropout`, `d_token`, `n_heads`, `n_layers`, `ff_mult` | [Gorishniy et al. (2021)](https://arxiv.org/abs/2106.11959): token dimension, attention heads, Transformer depth và FFN width là các tham số kiến trúc chính. | `d_token 32/48/64`, heads `2/4/8` (chỉ khi chia hết `d_token`), layers `1-3`, FFN multiplier `2/4` là dải nội bộ có ràng buộc toán học hợp lệ. |
| `05_tabnet.py` - TabNet | `lr`, `wd`, `dropout`, `n_d`, `n_steps` | [Arik & Pfister (2021), TabNet](https://arxiv.org/abs/1908.07442): decision/attention dimension và số decision steps là tham số cốt lõi. Code đặt `n_a = n_d`. | `n_d 16/24/32/48`, `n_steps 3-6` và dải regularization là dải nội bộ; paper là nguồn cho **loại** tham số. |
| `06_dcnv2.py` - DCNv2 | `lr`, `wd`, `dropout`, `n_cross`, `deep_h`, `deep_layers` | [Wang et al. (2021), DCN V2](https://arxiv.org/abs/2008.13535): số cross layers và kích thước/độ sâu deep branch là tham số cốt lõi. | `n_cross 1-2`, `deep_h 48-128`, `deep_layers 1-2` là dải nội bộ, cố ý nhỏ để hạn chế overfit. |
| `07_gru.py` - GRU | `hidden_dim`, `num_layers`, `dropout`, `lr`, `wd` | [Cho et al. (2014), GRU](https://arxiv.org/abs/1406.1078): hidden state/layer là kiến trúc; regularization và optimizer thuộc huấn luyện. | `hidden 32/48/64`, layers `1-2`, dropout `0.05-0.30`, `lr 1e-4-3e-3`, `wd 1e-7-1e-3` là dải nội bộ; đồng nhất với LSTM để so sánh công bằng. |
| `08_lstm.py` - LSTM | `hidden_dim`, `num_layers`, `dropout`, `lr`, `wd` | [Greff et al. (2017), LSTM: A Search Space Odyssey](https://arxiv.org/abs/1503.04069): learning rate quan trọng nhất, tiếp theo là hidden size; xem thêm API [PyTorch LSTM](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html). | Dải như GRU là dải nội bộ, không phải giá trị sao chép. `input_size` được suy ra từ dữ liệu; không tune `bidirectional` vì dự báo theo thời gian không được dùng thông tin tương lai. |
| `09_tcn.py` - TCN | `num_channels`, `levels`, `kernel_size`, `dropout`, `lr`, `wd` | [Bai et al. (2018), TCN](https://arxiv.org/abs/1803.01271): channels, convolution kernel và số temporal blocks xác định receptive field/capacity. | channels `16/24/32/40`, levels `2-4`, kernel `2/3` là dải nội bộ, giới hạn nhỏ do số đường cong không lớn. |
| `10_gbdt.py` - XGBoost | `learning_rate`, `max_depth`, `min_child_weight`, `subsample`, `colsample_bytree`, `reg_lambda`, `reg_alpha` | [Chen & Guestrin (2016), XGBoost](https://arxiv.org/abs/1603.02754) và [XGBoost parameter documentation](https://xgboost.readthedocs.io/en/stable/parameter.html). | Dải trong `xgb_search_space()` là dải nội bộ; nguồn API giải thích nghĩa tham số, không xác nhận từng cận. |
| `10_gbdt.py` - LightGBM | `learning_rate`, `num_leaves`, `min_child_samples`, `feature_fraction`, `bagging_fraction`, `lambda_l2`, `lambda_l1` | [Ke et al. (2017), LightGBM](https://proceedings.neurips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html) và [LightGBM parameter tuning guide](https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html). | Dải trong `lgbm_search_space()` là dải nội bộ; guide xác định `num_leaves` và `min_data_in_leaf` là kiểm soát độ phức tạp/overfit. |

## Công thức kinh nghiệm

| File / model | “Tham số” và nguồn |
|---|---|
| `11_empirical_formula.py` - ACI 209R-92 | Hệ số và biểu thức lấy từ ACI Committee 209. Các giả định thiếu dữ liệu (slump/air coefficient = 1) phải được nêu rõ. |
| `11_empirical_formula.py` - B3 | Bažant & Baweja / RILEM recommendation. |
| `11_empirical_formula.py` - B4 | RILEM TC-242-MDC, *Materials and Structures* 48 (2015), 753-770. |
| `11_empirical_formula.py` - fib MC2010 | fib *Model Code 2010*, Eqs. 5.1-76..83. |

Các hằng số vật lý/kinh nghiệm chuẩn thuộc công thức phải được trích từ tài liệu gốc. Hệ số
“calibrated” fit bằng least squares trong code là kết quả theo dữ liệu project, không phải tham số chuẩn.

## Điểm cần báo cáo trung thực

- Các paper/API trên biện minh **vì sao tune nhóm tham số đó**, không biện minh chính xác mọi
  cận nhỏ/lớn trong code.
- Để biện minh cận số, báo cáo `search space`, số trial, objective validation, tham số tốt nhất
  từng fold và kiểm tra độ nhạy 8 so với 30 trial. Không nói “lấy đúng cận từ paper” nếu không có
  bảng số tương ứng trong paper.
- `SEED`, `N_SPLITS`, `N_TRIALS`, max epoch, patience và batch size phải để trong mục
  *experimental protocol*, tách khỏi bảng hyperparameter từng thuật toán.
