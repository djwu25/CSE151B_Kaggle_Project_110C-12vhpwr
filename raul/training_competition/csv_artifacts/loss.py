#!/usr/bin/env python3

import pandas as pd
from matplotlib import pyplot as plt
plt.rcParams["figure.figsize"] = [7.00, 3.50]
plt.rcParams["figure.autolayout"] = True
columns = ["epoch", "loss"]
df = pd.read_csv("loss.csv", usecols=columns)
print("Contents in csv file:", df)

fig, ax = plt.subplots()
#ax.plot(range(0,70,1),range(40000,100000,2000))
ax.ticklabel_format(useOffset=False, style='plain')

plt.title("Epoch vs Loss")
plt.ylabel("MSE Loss")
plt.xlabel("Epoch")

plt.plot(df.epoch, df.loss)
#plt.show()
plt.savefig("image.jpg")
