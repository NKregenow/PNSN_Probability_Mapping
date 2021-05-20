import matplotlib.pyplot as plt
import numpy as np
import math

c2 = 1.11
delta = 0.1

max_mag = 4.5
min_mag = -0.27
x = round(min_mag, 1) - 0.05
mag_bins = np.arange(x, max_mag, 0.1)
#print(mag_bins[8 + 1] - 0.000002)

dist_bins = [0, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]

eq_dist = [765, 1800, 1222, 498, 500, 1000, 1200, 1150, 2002, 2350, 72, 2000, 2000, 2000]
eq_mag = [1.2, 1.5, 2.4, 3.2, 4.1, 3.8, 2.7, 0.5, 0.6, 1.7, 3.2, -0.27, -0.27, -0.27]
eq_on_off = [True, True, False, True, False, False, False, True, True, False, True, True, True, False]

#making counter
#right now get an extra loop with [ybin +1], maybe to a while loop to end after reaching 2560?
#How do we want to increase the area being counted if total_count doesn't reach specified amount (10)?
for ybin in range(len(dist_bins)):
    for xbin in range(len(mag_bins)):
        mag_bins[xbin] = round(mag_bins[xbin], 1)
        #print(mag_bins[xbin])
        total_count = 0
        true_count = 0
        false_count = 0
        #print(dist_bins[ybin])
        for event in range(len(eq_dist)):
            if (eq_dist[event] >= dist_bins[ybin] and eq_dist[event] < dist_bins[ybin + 1]) and (eq_mag[event] >= round(mag_bins[xbin], 1) and eq_mag[event] < round(mag_bins[xbin + 1], 1)) and eq_on_off[event] == True:
                total_count = total_count + 1
                true_count = true_count + 1
                '''print(eq_dist[event])
                print(eq_mag[event])
                print(dist_bins[ybin])
                print(dist_bins[ybin + 1])
                print(mag_bins[xbin])
                print(mag_bins[xbin + 1])
                print('\n')'''
            if (eq_dist[event] >= dist_bins[ybin] and eq_dist[event] < dist_bins[ybin + 1]) and (eq_mag[event] >= round(mag_bins[xbin], 1) and eq_mag[event] < round(mag_bins[xbin + 1], 1)) and eq_on_off[event] == False:
                total_count = total_count + 1
                false_count = false_count + 1
        extra_bin = 1
        while total_count < 3: #increase bins being searched
            total_count = 0
            for event in range(len(eq_dist)):
                if dist_bins[ybin + 1 + extra_bin] < range(dist_bins): #Just spitballing here... trying to make 
                if eq_dist[event] >= dist_bins[ybin - extra_bin] and eq_dist[event] < dist_bins[ybin + 1 + extra_bin] and eq_mag[event] >= \
                        mag_bins[xbin - extra_bin] and (eq_mag[event] < mag_bins[xbin + 1 + extra_bin] - 0.00000002) and eq_on_off[
                    event] == True:
                    total_count = total_count + 1
                    true_count = true_count + 1
                if eq_dist[event] >= dist_bins[ybin - extra_bin] and eq_dist[event] < dist_bins[ybin + 1 + extra_bin] and eq_mag[event] >= \
                        mag_bins[xbin - extra_bin] and (eq_mag[event] < mag_bins[xbin + 1 + extra_bin] - 0.00000002) and eq_on_off[
                    event] == False:
                    total_count = total_count + 1
                    false_count = false_count + 1
                extra_bin = extra_bin + 1
        print(total_count)
        print(true_count)
        print(false_count)
        print('\n')


plt.figure()
(H, xedges, yedges, image) = plt.hist2d(eq_mag, eq_dist, bins=(mag_bins, dist_bins))
'''print(xedges)
print(yedges)
print(H)'''
plt.show()

