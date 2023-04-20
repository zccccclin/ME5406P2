import csv

# Open the input and output CSV files
with open('progress.csv', newline='') as csvfile, open('output_file.csv', 'w', newline='') as outfile:
    reader = csv.reader(csvfile)
    writer = csv.writer(outfile)
    row_num = 1
    test = 0 
    for row in reader:
        row_num +=1
        # check if the column exist
        if len(row) > 7 and row_num > 60:
            if int(row[2]) % 200 == 0:
                test += 1
                if test % 2 == 0:
                    # Extract the column
                    succ = row[8]
                    # irs = row[9]
                    print(row[5], row[8])
                    # Write the column to the output file
                    writer.writerow([succ])

