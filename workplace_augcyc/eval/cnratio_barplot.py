from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_pkl

sns.set_theme()
sns.set(font_scale=1.4)
sns.set_style("whitegrid")

reals = load_pkl("cnratio_real.pkl")
c = Counter(reals)
p = {k: v / sum(c.values()) for k, v in c.items()}
print(p)

pp = {}
for k in p:
    near_ps = [kk for kk in pp if abs(kk - k) < 1e-3]
    if len(near_ps) == 0:
        pp[k] = p[k]
    else:
        pp[near_ps[0]] += p[k]

print(pp)
print(sum(pp.values()))
p = pp
x = [k for k in p]
y = [v for v in p.values()]

f, (ax, ax2) = plt.subplots(2, 1, sharex='all')

width = 0.05
ax.bar(x, y, width=width, facecolor="k", label=r"$C_B$")
ax2.bar(x, y, width=width, facecolor="k")

ax.set_ylim(.8, 0.86)  # outliers only
ax2.set_ylim(0, .06)  # most of the data

ax.spines['bottom'].set_visible(False)
ax2.spines['top'].set_visible(False)
ax.xaxis.tick_top()
ax.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = .015  # how big to make the diagonal lines in axes coordinates
# arguments to pass to plot, just so we don't keep repeating them
kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
ax.plot((-d, +d), (-d, +d), **kwargs)  # top-left diagonal
ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal

kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal
plt.xlim([0., 4.1])
ax2.set_xlabel("C/N ratio")
ax2.set_ylabel("Probability", loc="top")
ax2.yaxis.set_label_coords(-0.12, 1.4)
ax.legend()
ax.tick_params(length=0.2, top="off", pad=8)
ax2.tick_params(length=0, pad=8)
plt.savefig("cnratio_barplot.tiff", dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
plt.savefig("cnratio_barplot.eps", dpi=600, bbox_inches='tight', pil_kwargs={"compression": "tiff_lzw"})
