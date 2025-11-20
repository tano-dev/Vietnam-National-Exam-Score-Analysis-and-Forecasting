from Module.Load_Data import DataLoader
from Module.Processor_Data import DataProcessor
from Module.Analysis import Analysis

import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np

def main():
    start = time.perf_counter()
    # Khởi tạo processor và gán loader (setter sẽ ok)
    
    # Phân tích dữ liệu
    analysis = Analysis(proc)
    
    # In ra kết quả thống kê
    print(analysis.get_aggregate_by_exam_subsections(subject="toan"))
        
    end = time.perf_counter()
    print(f"Elapsed: {end - start:.6f} s")
    print("Done.")
    print("hi")
    
    
    print("Duy skibidi")
    print("Duy skibidi")
    print("Duy skibidi")
    print("Duy skibidi")
    print("Duy skibidi")



if __name__ == "__main__":
    main()
