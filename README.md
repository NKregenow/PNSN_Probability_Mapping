Test Tab is a dummy set of data while Test2 is based on the real data set. I was having errors running the code because it said I didn't have the pyproj module downloaded despite double checking that I did and re-downloading/importing it a number of times. The ultimate goal was to eventually merge the Test2 back with the main script, although Test2 has been serving as my 'main' for a while meaning that may not be necessary at the end.

Things to look over:
Making sure the Test2 script works since I couldn't figure out what was wrong with my pyproj module (aka. I wasn't able to run it to ensure I copied things over appropriately and used all the correct variable names)
Getting familiarized with the variables
Finding a better system for organizing the ybins. As of now it goes from 0-10km, and then goes through the y_incrementation function making the bins increase by smaller amounts initially.

Next Step:
Using the stored information for each "new" bin to display the probability for the new bins, rather than just the number of events per bin.
