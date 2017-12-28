import csv

wr = open('artists.txt', 'w+', encoding="utf8")

with open('artists.dat', 'r', encoding="utf8") as f:
    reader = csv.reader(f, delimiter='\t')
    for line in reader:
    	wr.write(str(line[0]) + " " + str(line[1]) + "\n")