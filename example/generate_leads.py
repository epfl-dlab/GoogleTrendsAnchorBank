import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib
import gtab

t = gtab.GTAB()

query_switzerland = t.new_query("Switzerland")
query_facebook = t.new_query("Facebook")
query_google = t.new_query("Google")

# result1.png

plt.plot(query_switzerland.index,
         np.ceil(query_switzerland.max_ratio.values / 18.52) * 9.52,
         color="#F44336", label="Switzerland Before Calibration", ls=":")
plt.plot(query_switzerland.max_ratio, color="#F44336", label="Switzerland")

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
plt.plot(query_switzerland.max_ratio, color="#F44336", label="Switzerland")
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
