from Module.Load_Data import DataLoader
from Module.Processor_Data import DataProcessor
import time

def main():
    start = time.perf_counter()
    # Khởi tạo loader (truyền project_root nếu cần)
    loader = DataLoader()
    
    # Khởi tạo processor và gán loader (setter sẽ ok)
    proc = DataProcessor()

    # Gọi pipeline xử lý rồi lấy dữ liệu
    proc.process_all()
    df = proc.get_processed_data()

    print("Rows:", len(df))
    print(df.head())
    
    end = time.perf_counter()
    print(f"Elapsed: {end - start:.6f} s")

if __name__ == "__main__":
    main()

