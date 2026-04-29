"""Download paper metadata and PDFs from arXiv."""
import urllib.request
import xml.etree.ElementTree as ET
import os
import sys

PAPER_IDS = {
    "v-jepa-2.1": "2603.14482",
    "lejepa": "2511.08544",
}

os.makedirs("research/vjepa2.1", exist_ok=True)
os.makedirs("research/lejepa", exist_ok=True)

for name, pid in PAPER_IDS.items():
    url = f"http://export.arxiv.org/api/query?id_list={pid}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    resp = urllib.request.urlopen(req, timeout=30)
    xml_data = resp.read().decode()

    ns = {"atom": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}
    root = ET.fromstring(xml_data)
    entry = root.find("atom:entry", ns)
    if entry is None:
        print(f"No entry found for {name} ({pid})")
        continue

    title = entry.find("atom:title", ns).text.strip()
    summary = entry.find("atom:summary", ns).text.strip()
    authors = [a.find("atom:name", ns).text for a in entry.findall("atom:author", ns)]
    published = entry.find("atom:published", ns).text
    pdf_url = None
    for link in entry.findall("atom:link", ns):
        if link.get("title") == "pdf":
            pdf_url = link.get("href")
            break
    abs_url = f"https://arxiv.org/abs/{pid}"

    # Save metadata
    md_path = f"research/{name}/metadata.txt"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(f"Title: {title}\n")
        f.write(f"ArXiv ID: {pid}\n")
        f.write(f"URL: {abs_url}\n")
        f.write(f"PDF: {pdf_url}\n")
        f.write(f"Published: {published}\n")
        f.write(f"Authors: {', '.join(authors)}\n")
        f.write(f"\nAbstract:\n{summary}\n")
    print(f"Saved metadata to {md_path}")

    # Download PDF
    if pdf_url:
        pdf_path = f"research/{name}/paper.pdf"
        try:
            pdf_req = urllib.request.Request(pdf_url, headers={"User-Agent": "Mozilla/5.0"})
            pdf_resp = urllib.request.urlopen(pdf_req, timeout=120)
            with open(pdf_path, "wb") as f:
                f.write(pdf_resp.read())
            print(f"Downloaded PDF to {pdf_path} ({os.path.getsize(pdf_path)} bytes)")
        except Exception as e:
            print(f"Failed to download PDF for {name}: {e}")

print("\nDone!")
