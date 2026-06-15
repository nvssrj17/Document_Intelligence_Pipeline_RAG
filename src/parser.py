import fitz
from tqdm import tqdm

class PDFParser:
    def __init__(self, file_path):
        self.file_path = file_path

    def parse(self):
        doc = fitz.open(self.file_path)
        pages = []
        for page_num in tqdm(
            range(len(doc)),
            desc = "Extracting Pages"
        ):
            
            page = doc[page_num]
            text = page.get_text()

            pages.append(
                {
                    "page": page_num + 1,
                    "text": text
                }
            )
        doc.close()
        return pages