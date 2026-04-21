import nbformat as nbf
import os

def py_to_nb(py_file, nb_file, title):
    if not os.path.exists(py_file):
        print(f"{py_file} introuvable.")
        return
        
    with open(py_file, 'r', encoding='utf-8') as f:
        code = f.read()
        
    nb = nbf.v4.new_notebook()
    # Ajout d'une cellule markdown pour le titre
    nb.cells.append(nbf.v4.new_markdown_cell(f"# {title}"))
    # Ajout du code
    nb.cells.append(nbf.v4.new_code_cell(code))
    
    with open(nb_file, 'w', encoding='utf-8') as f:
        nbf.write(nb, f)
    print(f"Converti {py_file} en {nb_file}")

py_to_nb('01_EDA_et_Pretraitement.py', '01_EDA_et_Pretraitement.ipynb', "Phase 1 : Analyse Exploratoire et Prétraitement")
py_to_nb('02_Modelisation_Stochastique.py', '02_Modelisation_Stochastique.ipynb', "Phase 2 : Ingénierie et Modélisation Stochastique")
