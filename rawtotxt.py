from matplotlib import pyplot as plt
f = open("filename.raw", "rb").read() 
plt.imshow(f) 
plt.show()