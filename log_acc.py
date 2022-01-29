import pandas as pd

acc = []
path = r"D:\download\work_dir\ntu\xview\aagcn_joint__3gcn_3alpha_fix\log.txt"
# path = path.replace('\\', '/')
with open(path, 'r') as f:
    # data = f.read()
    for line in f.readlines():
        if "Top1: " in line:
            line = line.split("Top1: ")
            line = line[1].split('%')
            acc.append(line[0])
dataframe = pd.DataFrame(
    {'acc': acc})  # 二维表
dataframe.to_csv(path + r"\..\单独acc.csv", sep=',')
print(acc)
