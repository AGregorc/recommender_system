import csv

wr = open('artists.html', 'w+', encoding="utf8")

with open('artists.dat', 'r', encoding="utf8") as f:
    reader = csv.reader(f, delimiter='\t')
    header = next(reader)
    wr.write("<table style=\"width:100%\">\n")
    wr.write("<tr> <th>" + str(header[0]) + "</th> <th>" + str(header[1]) + "</th></tr>\n")
    for line in reader:
    	wr.write("<tr> <th>" + str(line[0]) + "</th> <th>" + str(line[1]) + "  </th> <th> <input type=\"number\" step=\"0.01\" ></th></tr>\n")








