import matplotlib.pyplot as plt
import datetime

threads = 25
t_secuencial = 23.64
data = [23.081, 13.388, 16.922, 13.179, 14.184, 13.118, 13.518, 15.082, 15.014, 14.876, 14.939, 14.929, 14.891, 14.932, 14.845, 14.847, 15.086, 14.827, 14.885, 14.971, 14.933, 14.901, 14.947, 14.888]
s_n = []

print(len(data))
print(data[0])

for i in range(len(data)):
	s_n.append((t_secuencial/data[i]))

hilos = range(24)

fig = plt.figure()
plt.plot(hilos, s_n)
fig.suptitle('Speed Up')
plt.xlabel("Número de hilos")
plt.ylabel("S(n)")
fig.savefig('./test{}.jpg'.format(datetime.datetime.now()))