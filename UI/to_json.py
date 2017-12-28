import csv

wr = open('artists.json', 'w+', encoding="utf8")

with open('artists.dat', 'r', encoding="utf8") as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    line = next(reader)
    wr.write("[{\"id\":" + str(line[0]) + ", \"name\":\"" + str(line[1]) + "\" }")

    for line in reader:
    	wr.write(", {\"id\":" + 
    		str(line[0]) + ", \"name\":\"" + str(line[1]) + 
    		"\" }")

    wr.write("]")







