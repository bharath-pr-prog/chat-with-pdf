import time
from utils.pdf_extractor import extract_text_from_pdf

# Path to your PDF
pdf_path = "D:\pc\Bharath Ram.pdf"

start = time.time()
pdf_text = extract_text_from_pdf(pdf_path)
end = time.time()

print("✅ PDF Extraction Time:", end - start, "seconds")
print("\n📄 Extracted Text (Preview):\n", pdf_text[:500])
