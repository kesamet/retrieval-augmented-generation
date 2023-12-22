import io
import os
import logging

import fitz
from PIL import Image

logging.basicConfig(level=logging.INFO)


def extract_images(filename: str, image_output_dir_path: str) -> None:
    """Extract images from PDF."""
    pdf_file = fitz.open(filename)
    for page_index, page in enumerate(pdf_file):
        image_list = page.get_images()
        # printing number of images found in this page
        if image_list:
            logging.info(
                f"[+] Found a total of {len(image_list)} images in page {page_index}"
            )
        else:
            logging.info(f"[!] No images found on page {page_index}")
        for image_index, img in enumerate(page.get_images(), start=1):
            # get the XREF of the image
            xref = img[0]
            # extract the image bytes
            base_image = pdf_file.extract_image(xref)
            image_bytes = base_image["image"]
            # get the image extension
            image_ext = base_image["ext"]
            # load it to PIL
            image = Image.open(io.BytesIO(image_bytes))
            # save it to local disk
            image.save(
                os.path.join(
                    image_output_dir_path,
                    f"figure-{page_index+1}-{image_index}.{image_ext}",
                )
            )
