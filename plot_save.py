import matplotlib.pyplot as plt

# Sample data
x = [1, 2, 3, 4, 5]
y = [10, 20, 25, 30, 40]


plt.plot(x, y, marker='o')
plt.title("Sample Line Chart")
plt.xlabel("X Axis")
plt.ylabel("Y Axis")


plt.savefig("my_plot.png")  
print("Plot saved as 'my_plot.png'")
