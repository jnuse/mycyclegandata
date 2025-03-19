import re
import matplotlib.pyplot as plt
from collections import defaultdict

# 初始化数据结构
losses_by_epoch = defaultdict(lambda: {
    'D_A': [], 'G_A': [], 'cycle_A': [], 'idt_A': [],
    'D_B': [], 'G_B': [], 'cycle_B': [], 'idt_B': []
})

# 读取并解析日志文件
with open('loss_log.txt', 'r') as f:
    for line in f:
        line = line.strip()
        if line.startswith('(epoch:'):
            # 提取epoch和损失值
            epoch = int(re.search(r'epoch: (\d+)', line).group(1))
            loss_matches = re.findall(
                r'([A-Za-z_]+): ([+-]?\d+\.?\d*)', 
                line.split(') ')[1]
            )
            
            # 存储损失值到对应epoch
            for key, value in loss_matches:
                if key in losses_by_epoch[epoch]:
                    losses_by_epoch[epoch][key].append(float(value))

# 计算每个epoch的平均损失
epochs = sorted(losses_by_epoch.keys())
avg_losses = {
    key: [] for key in ['D_A', 'G_A', 'cycle_A', 'idt_A', 'D_B', 'G_B', 'cycle_B', 'idt_B']
}

for epoch in epochs:
    for key in avg_losses:
        values = losses_by_epoch[epoch][key]
        avg = sum(values)/len(values) if values else 0
        avg_losses[key].append(avg)

# 创建带子图的画布
fig, axs = plt.subplots(4, 1, figsize=(12, 20))

# 绘制判别器损失
axs[0].plot(epochs, avg_losses['D_A'], 'o-', label='D_A')
axs[0].plot(epochs, avg_losses['D_B'], 's-', label='D_B')
axs[0].set_title('Discriminator Loss')
axs[0].set_ylabel('Loss')
axs[0].legend()

# 绘制生成器损失
axs[1].plot(epochs, avg_losses['G_A'], 'o-', label='G_A')
axs[1].plot(epochs, avg_losses['G_B'], 's-', label='G_B')
axs[1].set_title('Generator Loss')
axs[1].set_ylabel('Loss')
axs[1].legend()

# 绘制循环一致性损失
axs[2].plot(epochs, avg_losses['cycle_A'], 'o-', label='cycle_A')
axs[2].plot(epochs, avg_losses['cycle_B'], 's-', label='cycle_B')
axs[2].set_title('Cycle Consistency Loss')
axs[2].set_ylabel('Loss')
axs[2].legend()

# 绘制身份损失
axs[3].plot(epochs, avg_losses['idt_A'], 'o-', label='idt_A')
axs[3].plot(epochs, avg_losses['idt_B'], 's-', label='idt_B')
axs[3].set_title('Identity Loss')
axs[3].set_xlabel('Epoch')
axs[3].set_ylabel('Loss')
axs[3].legend()

plt.tight_layout()
plt.show()