# Vietnam National Exam 2023–2025 Analysis & Forecast 2026 (Python Project for Data science)

Dự án này xây dựng **pipeline dữ liệu + phân tích + phát hiện điểm gãy + dự báo** trên dữ liệu điểm thi THPT giai đoạn **2023–2025**, với mục tiêu dự báo xu hướng **2026**.  
> Lưu ý: thư mục **Dashboard/** không nằm trong phạm vi đánh giá (có thể bỏ qua khi chấm môn).

---

## 1) Mục tiêu & đầu ra

### Mục tiêu
- Chuẩn hóa dữ liệu điểm thi THPT (Raw → Clean) theo cấu trúc thống nhất.
- Phân tích mô tả (EDA) theo **môn / khối / tỉnh**.
- Kiểm định thống kê (ANOVA / t-test) để kiểm tra khác biệt giữa nhóm.
- Phát hiện **điểm gãy (change point)** (đặc biệt năm 2025).
- Dự báo **2026** theo:
  - **môn** (Subject)
  - **khối/tổ hợp** (Block/Combination share)

### Đầu ra chính
- Thư mục **Clean_Data_2023-2025/**:
  - `Subject_Data/`, `Block_Data/`, `Province_Data/`
  - mỗi thư mục con `CleanData_<name>/` chứa:
    - `Export_Analysis_*.csv` (summary stats)
    - `Export_Distribution_*.csv` (phân phối)

- Notebook:
  - `Notebook/EDA.ipynb`
  - `Notebook/ChangePoint.ipynb`
  - `Notebook/Forecast2026.ipynb`

- Report:
  - `Report/ReportProject.ipynb` (file **duy nhất** để trình bày với giảng viên)

---

## 2) Cấu trúc thư mục

```
PythonProject/
├─ Raw_Data/                       # dữ liệu thô 2023–2025
├─ Clean_Data_2023-2025/            # dữ liệu đã export theo cấu trúc chuẩn
├─ Module/                          # ETL + Stats (Load/Process/Analysis/Export/ANOVA)
│  ├─ Load_Data.py
│  ├─ Processor_Data.py
│  ├─ Analysis.py
│  ├─ Export.py
│  └─ ANOVA_ttest.py
├─ Model/                           # Modeling layer
│  ├─ ChangePoint/                  # phát hiện & phân tích điểm gãy
│  │  ├─ __init__.py
│  │  ├─ ChangePointPreparer.py
│  │  ├─ ChangePointDetector.py
│  │  └─ ChangePointAnalyzer.py
│  └─ Forecast/                     # dự báo 2026 (theo môn/khối)
│     ├─ __init__.py
│     ├─ ForecastSubjectModel.py
│     └─ ForecastBlockModel.py
├─ Notebook/                        # notebook làm việc/triển khai
│  ├─ EDA.ipynb
│  ├─ ChangePoint.ipynb
│  └─ Forecast.ipynb
├─ Report/                          # notebook báo cáo
│  └─ ReportProject.ipynb
├─ run_pipeline.py                  # chạy end-to-end: Load → Process → Export
├─ requirements.txt                 # dependencies
└─ installation.txt                 # hướng dẫn cài đặt

```

---

## 3) Cài đặt môi trường

Khuyến nghị: **Python 3.10+** (Windows/Mac/Linux đều được).

### Cách 1 — venv (khuyên dùng)
```bash
cd PythonProject
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

### Cách 2 — conda
```bash
conda create -n thpt python=3.10 -y
conda activate thpt
pip install -r requirements.txt
```

> Nếu bạn chỉ chạy code `.py` (không mở notebook), có thể bỏ `jupyter`/`notebook`.

---

## 4) Chạy pipeline dữ liệu (Raw → Clean)

Chạy từ thư mục `PythonProject/`:

```bash
python run_pipeline.py
```

Kết quả export sẽ được ghi vào:
- `PythonProject/Clean_Data_2023-2025/`

---

## 5) Chạy notebook & report

### Mở notebook
```bash
jupyter notebook
# hoặc
jupyter lab
```

Sau đó mở:
- `Notebook/EDA.ipynb`
- `Notebook/ChangePoint.ipynb`
- `Notebook/Forecast2026.ipynb`

### Report để nộp/báo cáo
- `Report/ReportProject.ipynb`

**Mẹo quan trọng:**  
Trong Report, nên có 1 cell đầu “Setup paths” để tự nhận `project_root` và `clean_root` (tránh lỗi đường dẫn khi người khác chạy).

---

## 6) Liên hệ giữa report và code (để dễ trả lời khi thầy hỏi)

| Nội dung | File/Module liên quan |
|---|---|
| Load & chuẩn hóa | `Module/Load_Data.py`, `Module/Processor_Data.py` |
| Export clean | `Module/Export.py`, `run_pipeline.py` |
| EDA theo môn/khối/tỉnh | `Module/Analysis.py` |
| ANOVA / t-test | `Module/ANOVA_ttest.py` |
| Change point | `Model/ChangePointPreparer.py`, `ChangePointDetector.py`, `ChangePointAnalyzer.py` |
| Forecast 2026 | `Model/ForecastSubjectModel.py`, `Model/ForecastBlockModel.py` |

---

## 7) Troubleshooting (lỗi hay gặp)

### `NameError: project_root is not defined`
Bạn cần chạy cell “Setup paths” **trước** các cell dùng `project_root/clean_root`.

### Thiếu thư viện khi import (ví dụ `xgboost`, `ruptures`)
Chạy lại:
```bash
pip install -r requirements.txt
```

### Không thấy file export trong Clean_Data_2023-2025
Chạy lại pipeline:
```bash
python run_pipeline.py
```

---

## 8) Ghi chú
- **Dashboard/** không yêu cầu cho phần chấm môn (có thể bỏ qua).
- Report có thể “lược bớt” bớt hình so với notebook; miễn là report vẫn:
  1) bám đúng pipeline & code
  2) có minh họa đủ để giải thích khi bị hỏi.
