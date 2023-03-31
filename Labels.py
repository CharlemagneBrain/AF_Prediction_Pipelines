import os
import csv

class XMLtoCSV:
    def __init__(self, folder_name):
        self.folder_name = folder_name

    def convert(self, output_file):

        if not os.path.exists(output_file):
          os.makedirs(output_file)

        files = os.listdir(self.folder_name)
        files.sort() # trier les fichiers dans l'ordre alphabétique
        
        data = [["AF"]] # initialiser la liste des données CSV avec l'en-tête
        for file_name in files:
            if file_name.endswith("0.XML"):
                data.append(["0"])
            elif file_name.endswith("1.XML"):
                data.append(["1"])
            else:
              continue # ignorer les fichiers qui ne finissent pas par "0" ou "1"
        
        with open(output_file, "w", newline="") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerows(data)
