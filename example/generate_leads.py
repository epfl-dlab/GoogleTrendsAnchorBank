import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import gtab

t = gtab.GTAB()

query_swaziland = t.new_query("swaziland")
query_facebook = t.new_query("Facebook")
query_google = t.new_query("Google")

# result1.png

plt.plot(query_swaziland.index,
         np.ceil(query_swaziland.max_ratio.values / 1.52) * 1,
         color="#F44336", label="Swaziland Before Calibration", ls=":")
plt.plot(query_swaziland.max_ratio, color="#F44336", label="Swaziland")

plt.plot(query_facebook.max_ratio, color="#2196F3", label="Facebook")
plt.yscale("log")
plt.xlabel("Date")
plt.ylabel("Popularity")
plt.legend()
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10, 3.5)
plt.title("gtab-corrected trends")
plt.xlim([pd.to_datetime("2019-09-01"), pd.to_datetime("2020-08-01")])
plt.show()

# result2.png

plt.plot(query_facebook.max_ratio, color="#2196F3", label="Facebook")
plt.plot(query_swaziland.max_ratio, color="#F44336", label="Swaziland")
plt.plot(query_google.max_ratio, color="#F4B402", label="Google")
plt.yscale("log")
plt.xlabel("Date")
plt.ylabel("Popularity")
plt.legend()
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(10, 3.5)
plt.title("gtab-corrected trends")
plt.xlim([pd.to_datetime("2019-09-01"), pd.to_datetime("2020-08-01")])
plt.show()
