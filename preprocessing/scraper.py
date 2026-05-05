import re
from pathlib import Path

import requests

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}


def guess_extension(response, fallback_url):
    content_disposition = response.headers.get("content-disposition", "")
    if "filename=" in content_disposition:
        filename = content_disposition.split("filename=")[-1].strip('"')
        suffix = Path(filename).suffix
        if suffix:
            return suffix

    content_type = (
        response.headers.get("content-type", "").split(";")[0].strip().lower()
    )
    if content_type in ("application/pdf", "application/x-pdf"):
        return ".pdf"
    if content_type in ("application/msword", "application/vnd.ms-word"):
        return ".doc"
    if content_type in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ):
        return ".docx"
    if content_type.startswith("text/plain"):
        return ".txt"

    url_suffix = Path(fallback_url).suffix
    if url_suffix:
        return url_suffix

    return ".bin"


def extract_dbn_links(page_url):
    try:
        response = requests.get(page_url, headers=HEADERS, timeout=10)
        response.encoding = "utf-8"
        content = response.text
        matches = re.findall(
            r'<div class="eTitle"[^>]*>\s*<a href="([^"]+)">(.*?)</a>\s*</div>',
            content,
            flags=re.IGNORECASE | re.DOTALL,
        )
        return matches
    except Exception as e:
        print(f"Error while fetching {page_url}: {e}")
        return []


def download_pdf_from_dbn_page(dbn_page_url, filename_prefix=""):
    try:
        response = requests.get(dbn_page_url, headers=HEADERS, timeout=10)
        response.encoding = "utf-8"
        content = response.text

        match = re.search(r'href="(/load/\d+-\d+-\d+-\d+-\d+)"', content)

        if match:
            pdf_link = match.group(1)
            full_url = f"https://dbn.co.ua{pdf_link}"

            pdf_response = requests.get(full_url, headers=HEADERS, timeout=30)

            if pdf_response.status_code == 200:
                if "content-disposition" in pdf_response.headers:
                    filename = (
                        pdf_response.headers["content-disposition"]
                        .split("filename=")[-1]
                        .strip('"')
                    )
                else:
                    pdf_name = pdf_link.split("/")[-1]
                    extension = guess_extension(pdf_response, full_url)
                    filename = (
                        f"{filename_prefix}{pdf_name}{extension}"
                        if filename_prefix
                        else f"{pdf_name}{extension}"
                    )

                filepath = Path(__file__).parent / "data" / filename
                filepath.parent.mkdir(exist_ok=True)

                with open(filepath, "wb") as f:
                    f.write(pdf_response.content)

                return filepath
            else:
                print(f"Error downloading PDF: {pdf_response.status_code}")
                return None
        else:
            print(f"PDF link not found")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None


def download_file(url):
    try:
        response = requests.get(url, headers=HEADERS, timeout=10)
        response.encoding = "utf-8"
        content = response.text
        match = re.search(r'href="(/load/\d+-\d+-\d+-\d+-\d+)"', content)

        if match:
            pdf_link = match.group(1)
            full_url = f"https://dbn.co.ua{pdf_link}"

            pdf_response = requests.get(full_url, headers=HEADERS, timeout=30)

            if pdf_response.status_code == 200:

                if "content-disposition" in pdf_response.headers:
                    filename = (
                        pdf_response.headers["content-disposition"]
                        .split("filename=")[-1]
                        .strip('"')
                    )
                else:
                    pdf_name = pdf_link.split("/")[-1]
                    extension = guess_extension(pdf_response, full_url)
                    filename = f"{pdf_name}{extension}"

                filepath = Path(__file__).parent / "data" / filename
                filepath.parent.mkdir(exist_ok=True)

                with open(filepath, "wb") as f:
                    f.write(pdf_response.content)

                return filepath
            else:
                print(f"Error downloading PDF: {pdf_response.status_code}")
                return None
        else:
            print(f"PDF link not found")
            return None

    except Exception as e:
        print(f"Error: {e}")
        return None
