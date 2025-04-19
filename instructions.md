today we are working on the files in fifty_one/re (2)
please walk through the files to uncderstand what's going on, before dointg something ask before that i see you understand

1. use fifty_one/measurements/merging_measurements/uncertatinty_dist.py to calcualte uncertainty based on fifty_one/re (2)/Carapace20times.csv, fifty_one/re (2)/Square20times.csv, fifty_one/re (2)/Total20times.csv each one seperatly save to the figures to a file
2. lets make a new dataframe out of the fifty_one/re (2)/Scales.csv and the ones starting with measurements-fixed.... look at the structure the scales contatin the image name called 'Label'. each of the measurements-fixed one contain the image_name,prawn_id,meas_1,meas_2,meas_3,avg,std,dev_pct,flag we need to use the scale first with scales which you need to convert to  1/(Lenght/10) and multiplly with each meas_1,meas_2,meas_3, avg combine them to a dataframe.

3.  now we need to combine the new stats with the script with the files in updated_filtered_data_with_lengths_body-all.xlsx etc. using the part of the script in fifty_one/measurements/merging_measurements/derving_length.py

