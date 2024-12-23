import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
import numpy as np
from inference import run_inference_from_checkpoint
from os.path import join, realpath, dirname

SCRIPT_DIR = dirname(realpath(__file__))
XLABELS = ["Init. aléatoire", "Pré-entr. Original", "Pré-entr. Passe-Bas", "Pré-entr. PCA"]
YLABELS = ["MSE", "Écart-type"]
PATHS = [
    "finetune/2024-12-20_12-02-19",
    "finetune/2024-12-22_01-36-12",
    "finetune/2024-12-22_04-28-18",
    "finetune/2024-12-22_07-28-06"
]

def compute_metrics(path):
    labels, outputs = run_inference_from_checkpoint(path)
    mean_error = np.mean((labels -  outputs) ** 2)
    std_error = np.std((labels - outputs) ** 2)
    return [mean_error, std_error]

def generate_latex_table(xlabels, ylabels, errors):
    table = "\\begin{table}[h!]\n"
    table += "\setlength{\\tabcolsep}{3pt}\n"
    table += "\\renewcommand{\\arraystretch}{1.3}\n"
    table += "\\centering\n"
    table += "\\begin{tabular}{c" + "C{1.5cm}" * len(xlabels) + "}\n"
    table += "\\hline\n"
    table += " & " + " & ".join(xlabels) + " \\\\\n"
    table += "\\hline\n"
    
    for i, ylabel in enumerate(ylabels):
        row = ylabel + " & " + " & ".join(f"{error:.4f}" for error in errors[i]) + " \\\\\n"
        table += row
        table += "\\hline\n"
    
    table += "\\end{tabular}\n"
    table += "\\caption{Erreur quadratique moyenne du réseau de neurones sur la prédiction de vibrations avec et sans pré-entraînement.}\n"
    table += "\\label{tab:downstream}\n"
    table += "\\end{table}"
    return table

def main():
    metrics = []
    for path in PATHS:
        full_path = join(SCRIPT_DIR, "..", "output", path)
        out = compute_metrics(full_path)
        metrics.append(out)
    latex_table = generate_latex_table(XLABELS, YLABELS, np.array(metrics).T)
    print(latex_table)

if __name__ == "__main__":
    main()