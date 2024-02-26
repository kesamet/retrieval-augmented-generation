import io
import os
from typing import List

import fitz
from PIL import Image
from langchain.schema import Document

from src import logger


def extract_images(filename: str, image_output_dir_path: str) -> None:
    """Extract images from PDF."""
    pdf_file = fitz.open(filename)
    for page in pdf_file:
        image_list = page.get_images()
        if image_list:
            logger.info(f"[+] Found {len(image_list)} images on page {page.number}")
        else:
            logger.info(f"[!] No images found on page {page.number}")

        for image_index, img in enumerate(page.get_images(), start=1):
            xref = img[0]
            base_image = pdf_file.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            image = Image.open(io.BytesIO(image_bytes))
            image.save(
                os.path.join(
                    image_output_dir_path,
                    f"figure-{page.number}-{image_index}.{image_ext}",
                )
            )


def save_pages_as_images(filename: str, image_output_dir_path: str) -> None:
    pdf_file = fitz.open(filename)
    for page in pdf_file:
        pix = page.get_pixmap()
        pix.save(os.path.join(image_output_dir_path, f"page-{page.number}.png"))


def extract_tables(filename: str) -> List[Document]:
    """Extract tables from PDF."""
    pdf_file = fitz.open(filename)
    table_docs = list()
    for page in pdf_file:
        tabs = page.find_tables()
        logger.info(f"[+] Found {len(tabs.tables)} table(s) on page {page.number}")

        for tab in tabs:
            try:
                df = tab.to_pandas()
                if df.shape == (1, 1):
                    logger.info("  [!] dataframe shape is (1, 1)")
                    continue
                d = Document(
                    page_content=df.to_json(),
                    metadata={"source": filename, "page": page.number},
                )
                table_docs.append(d)
            except Exception:
                logger.info("  [!] unable to convert to dataframe")
    return table_docs
