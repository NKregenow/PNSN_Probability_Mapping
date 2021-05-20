import numpy as np
import pandas as pd
import datetime
import pyproj
import math
import matplotlib.pyplot as plt

header_list = ['Network', 'Station', 'Lat', 'Long', 'Elevation', 'Location', 'Channels', 'On Time', 'Off Time']
df = pd.read_csv('high_gain_stations.csv', names=header_list)

df = df.drop_duplicates(keep='first', ignore_index=True)
df.to_csv(r'C:\Users\Owner\PycharmProjects\PNSN\updated_stations.csv')

df['On Time'] = pd.to_datetime(df['On Time'], format='%Y %m %d %H:%M:%S')
df['Off Time'] = pd.to_datetime(df['Off Time'], format='%Y %m %d %H:%M:%S', errors='coerce')

df_dict = {}
for index in range(len(df['Station'])):
    df_dict[(df['Network'][index].strip(), df['Station'][index].strip())] = {'Network': df['Network'][index],
                                                                             'Station': df['Station'][index],
                                                                             'Lat': df['Lat'][index],
                                                                             'Long': df['Long'][index],
                                                                             'Elevation':df['Elevation'][index],
                                                                             'On Time': df['On Time'][index],
                                                                             'Off Time': df['Off Time'][index]}

#print(df_dict['BK', 'GASB'])
with open("all_local_earthquakes_phasedata_without_SM_and_horizontals.txt") as file:
    f = file.readlines()#116 #503 #1502 #10004    #[next(file) for line in range(116)]
    f = f[-302:] #-5005, not enough #-10003
    line_list = []
    all_picks = []
    for line in f:
        if len(line) > 125:
            line_list.append(line)
            station_picks = []
        elif len(line) < 125 and line[0] != ' ':
            station_picks.append(line[0:7])
        else:
            all_picks.append(station_picks)

    print('\n')
    file.close()
#print(all_picks)
#print(len(all_picks))
#print(len(line_list))
all_pick_list_earthquake = []
for earthquake in range(len(all_picks)):
    pick_list_earthquake = []
    for station_pick in range(len(all_picks[earthquake])):
        net_sta = []
        network = all_picks[earthquake][station_pick][5:]
        station = all_picks[earthquake][station_pick][:5]
        pick_tester = "('" + network.strip() + "', '" + station.strip() + "')"
        #net_sta.append(network.strip())
        #net_sta.append(station.strip())
        #pick_tuple = tuple(net_sta)
        #pick_list_earthquake.append(pick_tuple)
        pick_list_earthquake.append(pick_tester)
    all_pick_list_earthquake.append(pick_list_earthquake)

# Comparing times with single line
eq_times = []
for line in line_list:
    time = datetime.datetime.strptime(line[:16], "%Y%m%d%H%M%S%f")
    eq_times.append(time)
print(len(eq_times))
print(len(all_pick_list_earthquake))
list_of_list = []
for event in range(len(eq_times)):
    was_on = []
    station_list = []
    for key in df_dict:
        if all_pick_list_earthquake[event].count(str(key)) > 0 and df_dict[key]['On Time'] < eq_times[event] and df_dict[key]['Off Time'] > eq_times[event]:
            was_on.append(True)
            station_list.append(key)
        elif df_dict[key]['On Time'] > eq_times[event] or df_dict[key]['Off Time'] < eq_times[event]:
            was_on.append('Off')
            station_list.append(key)
        else:
            was_on.append(False)
            station_list.append(key)

    list_of_list.append(was_on)
eq_df = pd.DataFrame.from_records(list_of_list)
eq_df = pd.DataFrame.transpose(eq_df)
#print(eq_df)

eq_df.to_csv(r'C:\Users\Owner\PycharmProjects\PNSN\test.csv')


#Getting the EQ lat/long values
eq_lat_all = []
eq_long_all = []
eq_depth_all = []
eq_mag_all = []
for line in line_list:
    depth = float(line[31:36].strip())*10 #divide by 100 (decimal in line) multiply by 1000 (km to m) and remember that they are negative values
    eq_depth_all.append(depth)
    mag = float(line[147:150])/100
    eq_mag_all.append(mag)
    if line[19] == ' ':
        line_conversion = list(line)
        line_conversion[19] = '0'
        line = ''.join(line_conversion)
    lat_degree = int(line[16:18])
    lat_minute = int(line[19:21])
    lat_second = int(line[21:23])
    lat_eq = lat_degree + lat_minute/60 + lat_second/3600
    eq_lat_all.append(lat_eq)
    if line[27] == ' ':
        line_conversion = list(line)
        line_conversion[27] = '0'
        line = ''.join(line_conversion)
    long_degree = int(line[23:26])
    long_minute = int(line[27:29])
    long_second = int(line[29:31])
    long_eq = (long_degree + long_minute/60 + long_second/3600) * (-1) #westward is negative, this now matches the station longitudes
    eq_long_all.append(long_eq)
eq_coordinates = np.array([eq_lat_all, eq_long_all])

#Getting station lat/long values
station_lat = []
station_long = []
station_elevation = []
for key in df_dict:
    station_lat.append(df_dict[key]['Lat'])
    station_long.append(df_dict[key]['Long'])
    station_elevation.append(df_dict[key]['Elevation'])
station_coordinates = np.array([station_lat, station_long])

P = pyproj.Proj(proj='utm', zone=10, ellps='WGS84')
x_eq,y_eq = P(eq_coordinates[1], eq_coordinates[0]) #x=longitude, y=latitude
eq_xy = np.array([x_eq,y_eq])
x_station,y_station = P(station_coordinates[1], station_coordinates[0])
station_xy = np.array([x_station,y_station])


depth_diff_per_station = []
for station in range(len(station_elevation)):
    depth_diff = []
    for event in range(len(eq_depth_all)):
        depth_difference = eq_depth_all[event] + station_elevation[station]
        depth_diff.append(depth_difference)
    depth_diff_per_station.append(depth_diff)
df_depth_diff = pd.DataFrame.from_records(depth_diff_per_station)

x_diff_all_eq = []
for station in range(len(station_xy[0])):
    x_difference = []
    for event in range(len(eq_xy[0])):
        x_diff = abs(station_xy[0][station] - eq_xy[0][event])
        x_difference.append(x_diff)
    x_diff_all_eq.append(x_difference)

y_diff_all_eq = []
for station in range(len(station_xy[1])):
    y_difference = []
    for event in range(len(eq_xy[1])):
        y_diff = abs(station_xy[1][station] - eq_xy[1][event])
        y_difference.append(y_diff)
    y_diff_all_eq.append(y_difference)

#turn into dataframes (may not be needed)
df_x_diff = pd.DataFrame.from_records(x_diff_all_eq)
df_y_diff = pd.DataFrame.from_records(y_diff_all_eq)

diff_2d_all = []
for row in range(len(df_x_diff)):
    diff_2d = []
    for column in range(len(df_x_diff.iloc[row])):
        dist_diff_2d = math.sqrt(df_x_diff.iloc[row, column] ** 2 + df_y_diff.iloc[row, column] ** 2)
        diff_2d.append(dist_diff_2d)
    diff_2d_all.append(diff_2d)
df_diff_2d = pd.DataFrame.from_records(diff_2d_all)

diff_3d_all = []
for row in range(len(df_diff_2d)):
    diff_3d = []
    for column in range(len(df_diff_2d.iloc[row])):
        dist_diff_3d = math.sqrt(df_diff_2d.iloc[row, column] ** 2 + df_depth_diff.iloc[row, column] ** 2)
        diff_3d.append(dist_diff_3d)
    diff_3d_all.append(diff_3d)
df_3d_dist = pd.DataFrame.from_records(diff_3d_all) #dataframe with distances from each station to each eq in meters

info_list_all = []

for station in range(len(station_list)):
    info_list_station = []
    for column in range(len(eq_df.iloc[0])):
        station_info = []
        station_info.append(eq_df.iloc[station, column])
        station_info.append(df_3d_dist.iloc[station, column])
        station_info.append(eq_mag_all[column])
        info_list_station.append(station_info)
    info_tuple_station = [tuple(index) for index in info_list_station]
    info_list_all.append(info_tuple_station)
station_eq_df = pd.DataFrame.from_records(info_list_all)
station_eq_df['Station'] = station_list
station_eq_df = station_eq_df.set_index('Station')

#True/False = On/Off during event, distance to event in meters, magnitude of event (rows = stations, columns = events)
#print(station_eq_df.iloc[804])
indices = [0, 1, 2]
def on_off(list):
    return [item[0] for item in list]
def distances(list): #selecting first of each index in list
    return [item[1] for item in list]
def magnitudes(list):
    return [item[2] for item in list]


station_on_dists_mags = []
for station in range(len(station_eq_df)):
    single_station = []
    for earthquake in range(len(station_eq_df.iloc[0])):
        on_dist_mag = [station_eq_df.iloc[station, earthquake][index] for index in indices]
        single_station.append(on_dist_mag)
    station_on_dists_mags.append(single_station)

all_on_off = []
all_station_mags = []
all_station_dist = []
for station in range(len(station_on_dists_mags)):
    colors = on_off(station_on_dists_mags[station])
    all_on_off.append(colors)
    distances_station = distances(station_on_dists_mags[station])
    all_station_dist.append(distances_station)
    magnitudes_station = magnitudes(station_on_dists_mags[station])
    all_station_mags.append(magnitudes_station)
#print(all_station_mags)
#print(max(max(all_station_mags)))
#print(min(min(all_station_mags)))
#print(all_on_off)
for station in range(len(all_station_dist)):
    for earthquake in range(len(all_station_dist[station])):
        all_station_dist[station][earthquake] = all_station_dist[station][earthquake]/1000



#Divide data into true and false --> I don't think I should do this for
all_color = []
all_mag = []
all_dist = []
for station in range(len(all_on_off)):
    station_color = []
    station_mag = []
    station_dist = []
    for index in range(len(all_on_off[station])):
        if all_on_off[station][index] == True and all_station_mags[station][index] != 0:
            station_color.append('Green')
            station_mag.append(all_station_mags[station][index])
            station_dist.append(all_station_dist[station][index])
        elif all_on_off[station][index] == False and all_station_mags[station][index] != 0:
            station_color.append('Red')
            station_mag.append(all_station_mags[station][index])
            station_dist.append(all_station_dist[station][index])
    all_color.append(station_color)
    all_dist.append(station_dist)
    all_mag.append(station_mag)
mag_max = max(max(all_station_mags))
print(mag_max)
mag_min = min(min(all_station_mags))
print(round(mag_min))
#x_bins_list = np.linspace(round(mag_min), round(mag_max), 0.1)
#print(x_bins_list)
dist_max = 2500
c2 = 1.11
delta = -0.1

def y_incrementation(L2):
    return (10 ** (delta/c2) * L2)

#print(y_incrementation(2500))
y_bins = [2500, 0]
while dist_max > 10:
    #print(y_incrementation(dist_max))
    dist_max = y_incrementation(dist_max)
    y_bins.append(dist_max)
y_bins.sort()
for index in range(len(y_bins)):
    #print(y_bins[index])
    y_bins[index] = int(round(y_bins[index]))
y_bins_list = []
for i in y_bins:
    if i not in y_bins_list:
        y_bins_list.append(i)
print(y_bins_list)
#Divide the data by each bins (mix of mag and length)

'''for xbin in range(len(int(mag_max - mag_min) * 10)):
    for ybin in range(len(y_bins)):
        counter = 0
        station = 0
        for key in df_dict:
            if all_mag[station] and all_dist[station] in (xbin and ybin):
                counter = counter + 1'''


station = 0
xbinwidth = 0.1
for key in df_dict:
    for xbin in range(int(round(mag_max) - round(mag_min)) * 10): #will this make enough bins? make a list like I did for the ybins beforehand
        for ybin in range(len(y_bins)):
            counter_total = 0
            if all_mag[station] and all_dist[station] in (xbin and (y_bins_list[ybin] and y_bins_list[ybin + 1])): #this doesn't work because xbin/ybin are 0, 1, 2, 3..., not the bin boundary values
                counter_total = counter_total + 1
            print(counter_total)

    plt.figure()
    if len(all_color[station]) != 0:
        #plt.scatter(all_mag[station], all_dist[station], c=all_color[station])
        plt.hist2d(all_mag[station], all_dist[station], bins=(int(round(mag_max) - round(mag_min)) * 10, y_bins))
    #if len(all_color[station]) != 0:
        #plt.scatter(all_false_mag[station], all_false_dist[station], c=all_false_color[station], label='Unpicked', alpha=0.25)
    plt.xlabel('Earthquake Magnitude')
    plt.ylabel('Distance to Station [km]')
    plt.title('Station ' + str(key) + ' Picks')
    plt.colorbar()
    plt.show()
    station = station + 1

#sort data triplets by increasing magnitude --> sorted
#standardize bins for every graph
#whit if: make the bins the areas to try and get N+ and N- to produce probability --> make bin size = Lm = 0.1 (or whatever value we end up going with)
'''def deltaM(M, M_prime): #M_prime is from data, M is value we want to compare
    return abs(M - M_prime)

def deltaL(L, L_prime): #same concept as deltaM
    return abs(L - L_prime)

#c2 = 1.11
#make deltaMStar = 0.1 by using log of L values
Allow Lm to be 0.14... instead of 0.1 for convenience?

def deltaMStar(c2, L1, L2): #for EQ that gave same amplitude to the station at different distances --> Do I need to get the amplitude reading for every eq at every station as well?
    return abs(c2* math.log(L1) -c2 * math.log(L2))

L_Mag = math.sqrt(deltaM ** 2 + deltaMStar ** 2)

#general idea...
selected_list = []
N = 0
for earthquake in range(len(L_Mag)): #this may not be the right thing to iterate over as L_Mag is a single value, unless deltaM and deltaMStar are both arrays
    if L_Mag[earthquake] <= 0.1:
        selected_list.append(L_Mag[earthquake])
        N = N + 1
    if N < 10:'''