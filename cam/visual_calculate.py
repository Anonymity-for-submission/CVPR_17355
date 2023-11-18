import numpy as np
from matplotlib import pyplot as plt

from matplotlib.pyplot import MultipleLocator
from scipy.ndimage import gaussian_filter1d
#定义变量
x = np.arange(1, 151, 1)
noisy_clean = np.load("./results/noisy_clean.npy")
noisy_ot = np.load("./results/noisy_ot.npy")
print(noisy_clean)
noisy_clean = 1/noisy_clean
print(noisy_clean)
#绘制折线图
# y_space = MultipleLocator(1)
y_smoothed_nc = gaussian_filter1d(noisy_clean[:150], sigma=5)
y_smoothed_ot = gaussian_filter1d(noisy_ot[:150], sigma=5)
# ax=plt.gca()
# ax.yaxis.set_major_locator(y_space)

plt.title("feature ratio when noisy rate=0.4")
plt.axhline(1, 0, 150,color="red",linestyle=":",label="feature ratio=1")#横线
plt.plot(x, y_smoothed_nc, label="clean/human")
plt.plot(x,y_smoothed_ot,label = "human/random")
plt.legend()
#展示
plt.savefig("./results/visual_calculate_hline.png")
