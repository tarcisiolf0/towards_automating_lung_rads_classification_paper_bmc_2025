from .nodule import Nodule
import numpy as np

class LungRADSClassifier:
    def __init__(self, nodule: Nodule):
        self.nodule = nodule

    def classifier(self):
        # print("Classifying nodule...")
        # print(f"Attenuation: {self.nodule.attenuation}")
        # print(f"Edges: {self.nodule.edges}")
        # print(f"Calcification: {self.nodule.calcification}")
        # print(f"Size: {self.nodule.size}")
        # print("-"*50)

        if self.nodule.calcification:
            return "1"
        
        elif self.nodule.attenuation == "Sólido":
            return self.classify_solid_nodule()
        elif self.nodule.attenuation == "Partes Moles":
            return self.classify_part_solid_nodule()
        elif self.nodule.attenuation == "Vidro Fosco":
            return self.classify_ground_glass_nodule()
        elif self.nodule.size == -1:
            return "1"
        else: 
            return "0"
        
    def classify_solid_nodule(self):
        if self.nodule.size < 6:
            return "2"
        elif 6 <= self.nodule.size < 8:
            return self.evaluate_edges(3, 4)
        elif 8 <= self.nodule.size < 15:
            return self.evaluate_edges("4A", 4)
        elif self.nodule.size >= 15:
            return self.evaluate_edges("4B", 4)
        else: 
            return "0"

    def classify_part_solid_nodule(self):
        if self.nodule.size < 6:
            return "2"
        elif 6 <= self.nodule.size and self.nodule.solid_componet_size < 6:
            return self.evaluate_edges(3, 4)
        elif self.nodule.solid_componet_size >= 6 and self.nodule.solid_componet_size < 8:
            return self.evaluate_edges("4A", 4)
        elif self.nodule.solid_componet_size >= 8:
            return self.evaluate_edges("4B", 4)
        else: 
            return "0"

    def classify_ground_glass_nodule(self):
        if self.nodule.size < 30:
            return "2"
        elif self.nodule.size >= 30:
            return "3"
        else: 
            return "0"
            # return self.evaluate_edges(3, 4)

    def evaluate_edges(self, non_spiculation_category, spiculation_category):
        if self.nodule.edges == "Espiculada":
            return f"{spiculation_category}X"
        else:
            return f"{non_spiculation_category}"
