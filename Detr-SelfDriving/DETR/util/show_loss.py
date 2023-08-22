import re
import json
import matplotlib.pyplot as plt

# 读取日志文件
with open('../Output/log.txt', 'r') as file:
    log_text = file.read()

# 初始化列表以存储损失数据
train_losses = []
test_losses = []

# 使用正则表达式逐行匹配JSON数据并解析
pattern = r'\{.*?\}'
matches = re.findall(pattern, log_text)

for match in matches:
    data = json.loads(match)
    train_loss = data.get("train_loss")
    test_loss = data.get("test_loss")
    if train_loss is not None and test_loss is not None:
        train_losses.append(train_loss)
        test_losses.append(test_loss)

# 绘制训练和测试损失曲线
plt.plot(train_losses, label='Train Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
# 保存图像到文件
plt.savefig('../Output/loss_plot.png')

plt.show()
