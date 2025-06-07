import nbformat

input_path = "notebook/KnowGPT_colab.ipynb"
output_path = "notebook/KnowGPT_colab2.ipynb"

nb = nbformat.read(input_path, as_version=4)

if "widgets" in nb.metadata and "state" not in nb.metadata["widgets"]:
    nb.metadata["widgets"]["state"] = {}

nbformat.write(nb, output_path)
print("✅ Saved cleaned notebook to:", output_path)
