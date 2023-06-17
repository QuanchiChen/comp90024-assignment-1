import matplotlib.pyplot as plt
import numpy as np

# Runtimes for various SPARTAN resources
one_node_one_core = 800.7738356590271
one_node_eight_cores = 108.79034471511841
two_nodes_eight_cores = 114.48109245300293

# Create bar plot
x = ["1 Node and 1 Core", "1 Node and 8 Cores", "2 Nodes and 8 Cores"]
y = [one_node_one_core, one_node_eight_cores, two_nodes_eight_cores]
plt.figure(figsize=(8, 8))
plt.bar(x, y, width=0.4)
plt.xlabel("SPARTAN Resource", fontsize=14)
plt.ylabel("Time Taken (Seconds)", fontsize=14)
plt.title("Runtimes for Execution of Solution on Various SPARTAN Resources", fontsize=20)
plt.savefig("bar_chart.png", facecolor="white", bbox_inches="tight")
plt.show()
plt.close()
