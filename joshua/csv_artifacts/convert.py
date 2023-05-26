#!/usr/bin/env python3

# "TRIP_ID","CALL_TYPE","ORIGIN_CALL","ORIGIN_STAND","TAXI_ID","TIMESTAMP","DAY_TYPE","MISSING_DATA","POLYLINE"

from datetime import datetime

def getPolyTime(polyline):
  return (polyline.count('[') - 2) * 15;

# I am too lazy to setup arguments so here you go
training = 0
outName = 'outfile.csv'

# Define if we are converting training or not
if training == 1:
  fileName = 'train.csv'
else:
  fileName = 'test_public.csv'

flag = 0
with open(fileName, 'r') as f:
  outFile = open(outName, 'w')

  # Setup headers and columns
  if training == 1:
    columns = 8
    outFile.write('"CALL_TYPE","TAXI_ID","MONTH","WEEKDAY","DAY","SECOND","DAY_TYPE","TRIP_TIME"\n')
  else:
    columns = 7
    outFile.write('"CALL_TYPE","TAXI_ID","MONTH","WEEKDAY","DAY","SECOND","DAY_TYPE"\n')
  
  for i in f:
    # skip first line
    if flag == 0:
      flag = 1;
    else:
      line = i.strip().upper().replace('"', '').split(',', columns)
      if line[7] == 'FALSE' or training != 1:
        # calculate polyline time
        if training == 1:
          timeTaken = (getPolyTime(line[8]))
  
        # Date object
        date = datetime.utcfromtimestamp(int(line[5]))
  
        # Make day types into numbers
        if line[6] == 'A':
          dayType = 0
        elif line[6] == 'B':
          dayType = 1
        else:
          dayType = 2
  
        # Make call types into values
        if line[1] == 'A':
          callType = 64
        elif line[1] == 'C':
          callType = 65
        elif line[1] == 'B':
          if line[3] == '':
            callType = 0
          else:
            callType = int(line[3])

        # Taxi ID
        if line[4] == '':
          taxiId = 0
        else:
          taxiId = int(line[4])
        
        # Data row
        if training == 1:
          row = '{},{},{},{},{},{},{},{}\n'.format(callType, taxiId, date.month, date.weekday(), date.day, date.hour * 60 * 60 + date.minute * 60 + date.second, dayType, timeTaken)
        else:
          row = '{},{},{},{},{},{},{}\n'.format(callType, taxiId, date.month, date.weekday(), date.day, date.hour * 60 * 60 + date.minute * 60 + date.second, dayType)

        # Write out
        outFile.write(row)
