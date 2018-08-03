import matplotlib.pyplot as plt

plt.rcParams["patch.force_edgecolor"] = True
# rc("text", usetex=True)
import seaborn as sns

sns.set(style="whitegrid", palette="muted", color_codes=True)
sns.set_style("whitegrid", {"grid.linestyle": "--"})
red = sns.xkcd_rgb["light red"]
green = sns.xkcd_rgb["medium green"]
blue = sns.xkcd_rgb["denim blue"]