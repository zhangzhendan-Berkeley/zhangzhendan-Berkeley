import os

root = "."   # 你的项目根目录

output = []

for dirpath, dirnames, filenames in os.walk(root):
    for f in filenames:
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".html")):
            full = os.path.join(dirpath, f)
            output.append(full.replace("\\", "/"))

# 保存结果
with open("file_list.txt", "w", encoding="utf-8") as fp:
    fp.write("\n".join(sorted(output)))

print("生成完成：file_list.txt")
