from pathlib import Path
import time

from Module.Processor_Data import DataProcessor
from Module.Export import Export


def main() -> None:
    """Chạy toàn bộ pipeline: Load → Process → Analyze → Export (Clean Data)."""
    start = time.perf_counter()

    # 1. Xác định thư mục gốc của project (thư mục chứa file run_pipeline.py)
    project_root = Path(__file__).resolve().parent
    print(f"[INFO] Project root: {project_root}")

    # 2. Khởi tạo DataProcessor (bên trong tự khởi tạo DataLoader)
    processor = DataProcessor(project_root=project_root)

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
    print(f"[INFO] Elapsed: {end - start:.2f} s")


if __name__ == "__main__":
    main()
