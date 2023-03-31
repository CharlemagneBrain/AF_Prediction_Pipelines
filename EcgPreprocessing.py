import os
import xmltodict
import h5py
import numpy as np
from scipy.signal import resample



class ECGDataset:

    def __init__(self, input_dir, output_file):
        self.input_dir = input_dir
        self.output_file = output_file

    @staticmethod
    def extract_lead_values(ecg):
        names = [
            "lead_1", "lead_2", "lead_3", "lead_avr", "lead_avl", "lead_avf",
            "lead_v1", "lead_v2", "lead_v3", "lead_v4", "lead_v5", "lead_v6"
        ]
        lead_values = {}
        for i in range(1, 13):
            lead_name = names[i-1]
            try:
                lead_values[lead_name] = [int(digit) for digit in ecg['AnnotatedECG']['component']['series']['component']
                                          ['sequenceSet']['component'][i]['sequence']['value']['digits'].split()]
            except:
                pass
        return lead_values

    @staticmethod 
    def normalize_and_resample_ecg_signal(ecg_signal):
      ecg_signal = np.array(ecg_signal)
      
      # Rééchantillonner le signal à 4096 Hz
      resampled_signal = resample(ecg_signal, 4096)

      # Normaliser le signal rééchantillonné
      max_val = max(resampled_signal)
      min_val = min(resampled_signal)
      ecg_signal_normalized = (resampled_signal - min_val) / (max_val - min_val)
      return ecg_signal_normalized

    def xml_to_tensor(self):
        # noms des dérivations
        names = [
            "lead_1", "lead_2", "lead_3", "lead_avr", "lead_avl", "lead_avf",
            "lead_v1", "lead_v2", "lead_v3", "lead_v4", "lead_v5", "lead_v6"
        ]

        # nombre de dérivations
        num_leads = len(names)
        data = []

        # On parcourt chaque fichier xml du dossier, en stockant les valeurs de chaque dérivation dans un dictionnaire
        for filename in os.listdir(self.input_dir):
            if filename.endswith('.XML'):
                with open(os.path.join(self.input_dir, filename)) as f:
                    ecg = xmltodict.parse(f.read())
                lead_dict = self.extract_lead_values(ecg)
                lead_data = []

                # On réordonne les dérivations
                for lead_name in names:
                    lead_values = lead_dict.get(lead_name, [])
                    lead_data.append(lead_values)
                data.append(lead_data)

        # nombre de patients
        num_patients = len(data)

        # longueur maximale des signaux
        max_samples = 4096

        tensor_shape = (num_patients, max_samples, num_leads)
        tensor = np.zeros(tensor_shape)

        """
              les valeurs de chaque dérivation pour chaque patient sont stockées dans la matrice
              de tenseurs en utilisant les indices appropriés.

        """

        for i, lead_data in enumerate(data):
            for j, lead_values in enumerate(lead_data):
                normalized_lead_values = self.normalize_and_resample_ecg_signal(lead_values)
                tensor[i, :len(normalized_lead_values), j] = normalized_lead_values
                

        print(tensor.shape)  
        with h5py.File(self.output_file, 'w') as f:
            f.create_dataset('tracings', data=tensor)
