from docling.document_converter import DocumentConverter
import os
from pathlib import Path

# source = "https://arxiv.org/pdf/2408.09869"  # document per local path or URL
source = "https://arxiv.org/pdf/1507.05717"

converter = DocumentConverter()
conv_result = converter.convert(source)
print(conv_result.document.export_to_markdown())  # output: "## Docling Technical Report[...]"

output_dir = './output'
doc_filename = conv_result.input.file.stem

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Export Markdown format (fixed):
output_path = Path(f"{output_dir}/{doc_filename}.md")
with open(output_path, "w", encoding="utf-8") as fp:
    fp.write(conv_result.document.export_to_markdown())
