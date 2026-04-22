import nbformat as nbf
import os
import sys

def py_to_nb(py_file, nb_file, title):
    if not os.path.exists(py_file):
        print(f"{py_file} introuvable.")
        return
        
    with open(py_file, 'r', encoding='utf-8') as f:
        code = f.read()
        
    nb = nbf.v4.new_notebook()
    nb.cells.append(nbf.v4.new_markdown_cell(f"# {title}"))
    nb.cells.append(nbf.v4.new_code_cell(code))
    
    with open(nb_file, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Converti {py_file} en {nb_file}")

if __name__ == "__main__":
    py_to_nb('01_Regularisation_Vision.py', '01_Regularisation_Vision.ipynb', "Module 4 : Apprentissage Profond (Régularisation par l'Aléa Visuel)")
