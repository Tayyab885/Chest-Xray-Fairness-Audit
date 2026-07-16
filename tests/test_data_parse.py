import pandas as pd
from src.data import parse_labels, age_to_bin

LABELS = ["Atelectasis","Cardiomegaly","Consolidation","Edema","Effusion",
          "Emphysema","Fibrosis","Hernia","Infiltration","Mass","Nodule",
          "Pleural_Thickening","Pneumonia","Pneumothorax"]

def test_parse_labels_multi():
    v = parse_labels("Cardiomegaly|Effusion", LABELS)
    assert len(v) == 14
    assert v[LABELS.index("Cardiomegaly")] == 1
    assert v[LABELS.index("Effusion")] == 1
    assert sum(v) == 2

def test_parse_labels_no_finding():
    assert sum(parse_labels("No Finding", LABELS)) == 0

def test_age_to_bin():
    bins = [0, 40, 60, 80, 200]
    bl = ["<40","40-60","60-80","80+"]
    assert age_to_bin(35, bins, bl) == "<40"
    assert age_to_bin(70, bins, bl) == "60-80"
    assert age_to_bin(95, bins, bl) == "80+"
