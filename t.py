import nbformat

input_path = "notebook/KnowGPT_colab.ipynb"
output_path = "notebook/KnowGPT_colab2.ipynb"


def clean_notebook_widgets(input_path, output_path):
    nb = nbformat.read(input_path, as_version=4)

    # Xoá metadata.widgets ở cấp notebook
    if "widgets" in nb.metadata:
        print("🧹 Removing notebook-level widgets...")
        del nb.metadata["widgets"]

    # Xoá metadata.widgets trong từng cell
    for cell in nb.cells:
        if "metadata" in cell and "widgets" in cell["metadata"]:
            print("🧹 Removing widgets from a cell...")
            del cell["metadata"]["widgets"]

    nbformat.write(nb, output_path)
    print(f"✅ Saved cleaned notebook to: {output_path}")

# Sử dụng:
clean_notebook_widgets(input_path, output_path)