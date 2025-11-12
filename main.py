from Module.Load_Data import DataLoader
from Module.Processor_Data import DataProcessor

def main():
    # Khởi tạo loader (truyền project_root nếu cần)
    loader = DataLoader()
    
    # Khởi tạo processor và gán loader (setter sẽ ok)
    proc = DataProcessor()
    proc.loader = loader

    # Gọi pipeline xử lý rồi lấy dữ liệu
    proc.process_all()
    df = proc.get_processed_data()

    print("Rows:", len(df))
    print(df.head())

if __name__ == "__main__":
    main()
