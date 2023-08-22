import os
from collections import defaultdict

def count_class_occurrences(folder_path):
    class_count = defaultdict(int)

    # 遍历labels文件夹中的所有txt文件
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder_path, file_name)

            # 读取txt文件的内容并统计类别数出现的次数
            with open(file_path, 'r') as file:
                lines = file.readlines()

            for line in lines:
                parts = line.split()
                if len(parts) > 0:
                    class_count[int(parts[0])] += 1

    # 打印统计结果
    print("类别数\t出现次数")
    for class_num, count in class_count.items():
        print(f"{class_num}\t{count}")

if __name__ == "__main__":
    folder_path = "train/labels"  # 请将此处替换为您的labels文件夹的路径
    count_class_occurrences(folder_path)
