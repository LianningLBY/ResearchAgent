import os
import fitz  # PyMuPDF
import base64


def pdf_to_images(pdf_path: str, out_dir: str = None, dpi: int = 200, fmt: str = "png",
                  password: str = None, transparent: bool = False, keep_images=False):
    if out_dir is None:
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        out_dir = os.path.join(os.path.dirname(pdf_path), f"{base}_pages")
    os.makedirs(out_dir, exist_ok=True)

    doc = fitz.open(pdf_path)
    if doc.needs_pass:
        if not password or not doc.authenticate(password):
            raise RuntimeError("PDF 受密码保护，认证失败。")

    zoom = dpi / 72.0
    mat = fitz.Matrix(zoom, zoom)
    n_pages = doc.page_count

    for i in range(n_pages):
        page = doc.load_page(i)
        pix = page.get_pixmap(matrix=mat, alpha=transparent)
        fpath = os.path.join(out_dir, f"page-{i+1:03d}.{fmt.lower()}")
        pix.save(fpath)

    images = []
    for i in range(n_pages):
        fpath = os.path.join(out_dir, f"page-{i+1:03d}.{fmt.lower()}")
        with open(fpath, "rb") as file:
            data = file.read()
        images.append(base64.b64encode(data).decode('utf-8'))

    if not keep_images:
        for i in range(n_pages):
            fpath = os.path.join(out_dir, f"page-{i+1:03d}.{fmt.lower()}")
            if os.path.exists(fpath):
                os.remove(fpath)

    return images
