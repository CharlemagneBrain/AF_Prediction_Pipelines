import numpy as np
import os
import json
import xmltodict
from scipy.signal import medfilt
from scipy.signal import resample

class ECGPreprocessor:
    def __init__(self, input_dir, output_file):
        self.input_dir = input_dir
        self.output_file = output_file

    def extract_lead_values(self, ecg):
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
        
        # Rééchantillonner le signal 
        resampled_signal = resample(ecg_signal, 4096)

        # Normaliser le signal rééchantillonné
        max_val = max(resampled_signal)
        min_val = min(resampled_signal)
        ecg_signal_normalized = (resampled_signal - min_val) / (max_val - min_val)
        
        #Arrondir les valeurs à deux décimales après la virgule
        ecg_signal_normalized = np.round(ecg_signal_normalized, 3)
    
        return ecg_signal_normalized

    def preprocess(self):
        
        # names = [
        #     "lead_1", "lead_2", "lead_3", "lead_avr", "lead_avl", "lead_avf",
        #     "lead_v1", "lead_v2", "lead_v3", "lead_v4", "lead_v5", "lead_v6"
        # ]
        
        preprocessed_data = []
       
        for filename in os.listdir(self.input_dir):
            if filename.endswith('.XML'):
                with open(os.path.join(self.input_dir, filename)) as f:
                    ecg = xmltodict.parse(f.read())
                lead_dict = self.extract_lead_values(ecg)
                
                # Ajout de la normalisation des dérivations
                for lead_name, lead_values in lead_dict.items():
                    lead_values = np.array(lead_values)
                    lead_values = (lead_values - np.mean(lead_values)) / np.std(lead_values)
                    lead_resampled=self.normalize_and_resample_ecg_signal(lead_values)
                    lead_dict[lead_name] = lead_resampled.tolist()
                
                preprocessed_data.append(lead_dict)

        with open(self.output_file, 'w') as f:
            json.dump(preprocessed_data, f)

