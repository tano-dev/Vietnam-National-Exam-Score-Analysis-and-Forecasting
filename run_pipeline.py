from Module.Load_Data import DataLoader
from Module.Processor_Data import DataProcessor
from Module.Analysis import Analysis
from Module.Export import Export
from pathlib import Path

import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np


def main():
    """Chạy toàn bộ pipeline: Load → Process → Analyze → Export (Clean Data)."""
    start = time.perf_counter()

    # 1. Xác định thư mục gốc của project (chứa folder Module, data/raw, ...)
    project_root = Path(__file__).resolve().parent

    # 2. Khởi tạo DataProcessor
    #    - Bên trong tự tạo DataLoader(project_root) và load toàn bộ file điểm
    processor = DataProcessor(project_root)

    # 3. Chạy toàn bộ quy trình tiền xử lý & chuẩn hóa dữ liệu
    #    - Chuẩn tên cột
    #    - Xử lý NaN, trùng
    #    - Gộp 4 năm thành combined_data
    #    - Kiểm tra điều kiện điểm [0, 10]
    processor.process_all()

    # (Tuỳ chọn) Kiểm tra nhanh kích thước dữ liệu sau xử lý
    combined = processor.get_processed_data()
    print(f"[INFO] Combined data shape: {combined.shape}")

    # 4. Khởi tạo Export
    #    - Export sẽ tự tạo Analysis(processor) bên trong
    #    - root_path là thư mục Clean_Data_2023-2025 trong project
    output_root = project_root / "Clean_Data_2023-2025"
    exporter = Export(processor=processor, root_path=str(output_root))

    # 5. Chạy export full:
    #    - Subject_Data / CleanData_<mon> / Export_Analysis_<mon>.csv
    #    - Block_Data   / CleanData_<khoi> / Export_Analysis_<khoi>.csv
    #    - Province_Data/ CleanData_<tinh_khong_dau>/ Export_Analysis_<tinh>.csv
    exporter.run_export_all()

    end = time.perf_counter()
    print(f"[DONE] Pipeline completed. Clean data saved at: {output_root}")
    print(f"Elapsed: {end - start:.6f} s")


if __name__ == "__main__":
    main()
