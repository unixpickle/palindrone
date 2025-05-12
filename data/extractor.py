import os
import re
from lxml import etree as ET  # instead of xml.etree.ElementTree
from tqdm import tqdm
import mwparserfromhell
import hashlib
import bz2


# Input and output paths
XML_PATH = "enwiki-latest-pages-articles.xml.bz2"  # Replace with your actual file
OUTPUT_DIR = "wikipedia_texts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Handle the default namespace
NS = {"mw": "http://www.mediawiki.org/xml/export-0.11/"}


def clean_wikitext(wikitext):
    code = mwparserfromhell.parse(wikitext)
    text = code.strip_code(normalize=True, collapse=True)
    text = re.sub(r"\n{2,}", "\n\n", text.strip())

    # Remove leftover image captions and formatting artifacts
    text = re.sub(r"(?m)^thumb\|.*$", "", text)  # lines starting with 'thumb|'
    text = re.sub(r"(?m)^File:[^\n]*$", "", text)  # lines with "File:..." alone
    text = re.sub(r"(?:^|\n)Category:[^\n]+", "", text)  # remove Category: lines
    text = re.sub(
        r"(?m)^\w{1,20}$", "", text
    )  # remove stray one-word lines like "Novel"

    # Remove empty or whitespace-only parentheses
    text = re.sub(r"\(\s*\)", "", text)

    text = re.sub(r"\n{2,}", "\n\n", text.strip())  # re-collapse newlines
    return text


def extract_texts(xml_path):
    with bz2.open(xml_path, "rb") as f:
        context = ET.iterparse(
            f, events=("end",), tag="{http://www.mediawiki.org/xml/export-0.11/}page"
        )

        for _, elem in tqdm(context, desc="Processing articles"):
            title = elem.find("mw:title", NS)
            revision = elem.find("mw:revision", NS)
            if title is None or revision is None:
                continue
            text_node = revision.find("mw:text", NS)
            if text_node is None or not text_node.text:
                continue

            raw_text = text_node.text
            cleaned = clean_wikitext(raw_text)

            if cleaned:
                # Compute MD5 hash of title
                title_hash = hashlib.md5(title.text.encode("utf-8")).hexdigest()
                shard_dir = os.path.join(OUTPUT_DIR, title_hash[:2])
                os.makedirs(shard_dir, exist_ok=True)

                filename = os.path.join(shard_dir, f"{title_hash}.txt")
                with open(filename, "w", encoding="utf-8") as f:
                    f.write(cleaned)

            elem.clear()  # free memory


if __name__ == "__main__":
    extract_texts(XML_PATH)
