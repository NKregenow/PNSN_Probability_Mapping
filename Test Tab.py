import matplotlib.pyplot as plt
import numpy as np


c2 = 1.11
delta = 0.1

max_mag = 4.5
min_mag = -0.27
x = round(min_mag, 1) - 0.05
mag_bins = np.arange(x, max_mag, 0.1)

dist_bins = [0, 10, 20, 40, 80, 160, 320, 640, 1280, 2560]

#if eq_dist is at dist_bin marker, it counts towards the higher bin --> eq_dist = 160 --> dist_bin = 160-320 range
eq_dist = [765, 1800, 1222, 498, 500, 1000, 1200, 1150, 2002, 2350, 72, 2000, 50, 50, 50, 115, 150, 160, 170, 170, 170, 25, 1150, 35, 35, 30]
eq_mag = [1.2, 1.5, 2.4, 3.2, 4.1, 3.8, 2.7, 0.5, 0.6, 1.7, 3.2, -0.27, -0.27, -0.27, -0.27, -0.27, -0.27, -0.27, -0.27, -0.27, -0.27, 1.7, 1.7, -0.27, -0.27, -0.27]
eq_on_off = [True, True, False, True, False, False, False, True, True, False, True, False, True, True, False, True, False, True, True, True, True, True, False, False, False, False]

def prob(true,false):
    return(true/(false + true))

all_bin_boundaries = []
all_column_data = []
for xbin in range(len(mag_bins)):
    total_count = 0
    true_count = 0
    false_count = 0
    bin_check = 0
    column_boundaries = [0]
    column_data = [round(mag_bins[xbin], 2)]
    for ybin in range(len(dist_bins)):
        mag_bins[xbin] = round(mag_bins[xbin], 2)
        bin_data = []
        if total_count >= 3: #loops over distances until we have enough events, then resets to 0 and starts again
            probability = prob(true_count, false_count)

            print(total_count) #save all of these values in a variable to go off of for the next loop (list/dictionary)
            print(true_count)
            print(false_count)
            bin_data.append(probability)
            bin_data.append(total_count)
            bin_data.append(true_count)
            bin_data.append(false_count)
            bin_data.append(dist_bins[ybin])
            column_data.append(bin_data)
            print(mag_bins[xbin])
            print(dist_bins[ybin]) #this is where we are starting to count up to 3 again
            lower_bounds = dist_bins[ybin]
            column_boundaries.append(lower_bounds)
            bin_check = bin_check + 1
            total_count = 0
            true_count = 0
            false_count = 0
            #if... Make an if statement where if there are not enough events in the last cell, add them to the previous one (may not be wanted because that would expand the distances and misrepresent the longer distances)
            #Or, if there aren't enough events but we reach final distance bin, print out results
        if dist_bins[ybin] == 2560 and total_count < 3:
            bin_data.append(total_count)
            bin_data.append(true_count)
            bin_data.append(false_count)
            bin_data.append(dist_bins[ybin])
            column_data.append(bin_data)
            total_count = 0
            true_count = 0
            false_count = 0
        for event in range(len(eq_dist)):
            if (eq_dist[event] >= dist_bins[ybin] and eq_dist[event] < dist_bins[ybin + 1]) and (eq_mag[event] >= round(mag_bins[xbin], 1) and eq_mag[event] < round(mag_bins[xbin + 1], 1)) and eq_on_off[event] == True:
                total_count = total_count + 1
                true_count = true_count + 1
            elif (eq_dist[event] >= dist_bins[ybin] and eq_dist[event] < dist_bins[ybin + 1]) and (eq_mag[event] >= round(mag_bins[xbin], 1) and eq_mag[event] < round(mag_bins[xbin + 1], 1)) and eq_on_off[event] == False:
                total_count = total_count + 1
                false_count = false_count + 1
    if bin_check == 0:
        print(mag_bins[xbin])  # Save this information as well (list)
        print('Not enough events in this magnitude bin')
        print('\n')
    if column_boundaries[-1] != 2560:
        column_boundaries.append(2560)
    all_bin_boundaries.append(column_boundaries)
    all_column_data.append(column_data)
print(all_bin_boundaries)
print(all_column_data)

plt.figure()
(H, xedges, yedges, image) = plt.hist2d(eq_mag, eq_dist, bins=(mag_bins, dist_bins))
'''print(xedges)
print(yedges)
print(H)'''
plt.show()

