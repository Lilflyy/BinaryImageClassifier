import csv
import os
import imghdr
filepath = 'art_real.csv'
img_ext = ['jpeg', 'jpg', 'bmp', 'png']

with open(filepath, 'w', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write header
    #csv_writer.writerow(['Path', 'Class'])
    
    # Write data to the CSV file
    for imgclass in os.listdir('data'):
        if imgclass[0] != '.':
            for img in os.listdir(os.path.join('data', imgclass)):
                path = os.path.join('data', imgclass, img)
                tip = imghdr.what(path)
                if imgclass == 'art':
                    i_class = 1
                elif(imgclass == 'real_life'):
                    i_class = 0
                if tip in img_ext:
                    data = [path, i_class]
                    csv_writer.writerow(data)
                else:
                    os.remove(path)