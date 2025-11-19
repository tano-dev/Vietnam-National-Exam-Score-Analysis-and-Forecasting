from Module.Load_Data import DataLoader
from Module.Processor_Data import DataProcessor
from Module.Analysis import Analysis
from Module.Export import Export
from pathlib import Path

import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np

# def main():
#     # start = time.perf_counter()
#     # # Khởi tạo processor và gán loader (setter sẽ ok)
#     # ROOT_DIR = Path(__file__).resolve().parent           # thư mục PythonProject
#     # CLEAN_DIR = ROOT_DIR / "Data" / "Clean_data_2023-2025"
#     # CLEAN_DIR.mkdir(parents=True, exist_ok=True)         # tạo nếu chưa tồn tại

#     # proc = DataProcessor()

#     # # Gọi pipeline xử lý rồi lấy dữ liệu
#     # proc.process_all()
#     # df = proc.get_processed_data()

#     # # Phân tích dữ liệu
#     # analysis = Analysis(proc)

#     # # ================== MÔN HỌC ==================
#     # # map tên hiển thị -> tên cột trong DataFrame
#     # subject_map = {
#     #     "Toan":  "toan",
#     #     "Van":   "ngu_van",
#     #     "Anh":   "ngoai_ngu",
#     #     "Ly":    "vat_li",
#     #     "Hoa":   "hoa_hoc",
#     #     "Sinh":  "sinh_hoc",
#     #     "Su":    "lich_su",
#     #     "Dia":   "dia_li",
#     #     "GDCD":  "gdcd",
#     # }

#     # for short_name, col_name in subject_map.items():
#     #     # dùng phân phối điểm theo môn (theo từng năm)
#     #     df_subject = analysis.get_arregate_by_exam_subsections(col_name)
    
#     #     filename = CLEAN_DIR / f"Export_Analysis_Subject_{short_name}_2023-2025.csv"
#     #     df_subject.to_csv(filename, index=False)

#     # # ================== KHỐI THI ==================
#     # blocks = ["A", "B", "C", "D"]

#     # for block in blocks:
#     #     # phân phối tổng điểm theo khối & năm
#     #     df_block = analysis.analyze_scores_by_exam_block(block)
    
#     #     filename = CLEAN_DIR / f"Export_Analysis_Block_{block}_2023-2025.csv"
#     #     df_block.to_csv(filename, index=False)

#     # end = time.perf_counter()
#     # print(f"Elapsed: {end - start:.6f} s")

if __name__ == "__main__":
    main()
