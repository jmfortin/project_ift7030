import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "scripts"))
import numpy as np
from inference import run_inference_from_checkpoint
from os.path import join, realpath, dirname

SCRIPT_DIR = dirname(realpath(__file__))
XLABELS = ["Original", "Passe-Bas", "PCA"]
YLABELS = ["MSE", "Écart-type"]
PATHS = [
    "training/2024-12-22_00-09-38",
    "training/2024-12-22_03-05-09",
    "training/2024-12-22_05-57-16"
]

def compute_metrics(path):
    labels, outputs = run_inference_from_checkpoint(path)
    mean_error = np.mean((labels -  outputs) ** 2)
    std_error = np.std((labels - outputs) ** 2)
    return [mean_error, std_error]

def generate_latex_table(xlabels, ylabels, errors):
    table = "\\begin{table}[h!]\n"
    table += "\setlength{\\tabcolsep}{10pt}\n"
    table += "\\renewcommand{\\arraystretch}{1.3}\n"
    table += "\\centering"
    table += "\n\\begin{tabular}{c" + "c" * len(xlabels) + "}\n"
    table += "\\hline\n"
    table += " & " + " & ".join(xlabels) + " \\\\\n"
    table += "\\hline\n"
    
    for i, ylabel in enumerate(ylabels):
        row = ylabel + " & " + " & ".join(f"{error:.4f}" for error in errors[i]) + " \\\\\n"
        table += row
        table += "\\hline\n"
    
    table += "\\end{tabular}\n"
    table += "\\caption{Erreur quadratique moyenne de la prédiction du réseau de neurones après avoir appliqué différentes méthodes de filtrage.}\n"
    table += "\\label{tab:filtering}\n"
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