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

    # 1. Xác định thư mục gốc của project
    project_root = Path(__file__).resolve().parent

    # 2. Khởi tạo DataProcessor
    processor = DataProcessor(project_root)

    # 3. Chạy tiền xử lý & chuẩn hóa dữ liệu
    processor.process_all()

    # Kiểm tra nhanh dữ liệu sau xử lý
    combined = processor.get_processed_data()
    print(f"[INFO] Combined data shape: {combined.shape}")

    # 4. Export dữ liệu sạch
    output_root = project_root / "Clean_Data_2023-2025"
    exporter = Export(processor=processor, root_path=str(output_root))
    exporter.run_export_all()

    end = time.perf_counter()
    print(f"[DONE] Pipeline completed. Clean data saved at: {output_root}")
    print(f"Elapsed: {end - start:.6f} s")

if __name__ == "__main__":
    main()
