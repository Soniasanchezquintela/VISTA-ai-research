#!/usr/bin/env python3

import csv
import re
from pathlib import Path
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import requests


API_URL = (
    "https://www.ametllerorigen.com/mobify/proxy/api/search/"
    "shopper-search/v1/organizations/f_ecom_blzv_prd/product-search"
)

OUTPUT_DIR = Path("ametller_dataset")
CSV_PATH = OUTPUT_DIR / "ametller_dataset.csv"

LIMIT = 48


# Paste a fresh Bearer token here when needed.
BEARER_TOKEN = "eyJ2ZXIiOiIxLjAiLCJqa3UiOiJzbGFzL3Byb2QvYmx6dl9wcmQiLCJraWQiOiI5ZjQ1MjdlNC0zZjI0LTQzNmMtYjg5ZS0wMzA4NDRhMGRiY2UiLCJ0eXAiOiJqd3QiLCJjbHYiOiJKMi4zLjQiLCJhbGciOiJFUzI1NiJ9.eyJhdXQiOiJHVUlEIiwic2NwIjoic2ZjYy5zaG9wcGVyLW15YWNjb3VudC5iYXNrZXRzIGNfc2l0ZVByZWZlcmVuY2VzIHNmY2Muc2hvcHBlci1kaXNjb3Zlcnktc2VhcmNoIHNmY2Muc2hvcHBlci1jdXN0b21lcnMubG9naW4gc2ZjYy5zaG9wcGVyLWV4cGVyaWVuY2Ugc2ZjYy5zaG9wcGVyLW15YWNjb3VudC5vcmRlcnMgc2ZjYy5zaG9wcGVyLXByb2R1Y3RsaXN0cyBzZmNjLnNob3BwZXItcHJvbW90aW9ucyBjX3Bvc3RhbENvZGVzIGNfY2FuY2VsT3JkZXIgc2ZjYy5zZXNzaW9uX2JyaWRnZSBjX2hlcm9rdUxvZ2luIHNmY2Muc2hvcHBlci1teWFjY291bnQucGF5bWVudGluc3RydW1lbnRzLnJ3IGNfaGVyb2t1UmVnaXN0ZXIgY191cGRhdGVQYXNzd29yZCBjX2xveWFsR3VydVNjb3JlIHNmY2Muc2hvcHBlci1jYXRlZ29yaWVzIGNfbGVnYWN5T3JkZXJzIHNmY2Muc2hvcHBlci1wcm9kdWN0cyBzZmNjLnNob3BwZXItbXlhY2NvdW50LnJ3IHNmY2Muc2hvcHBlci1zdG9yZXMgc2ZjYy5zaG9wcGVyLWN1c3RvbWVycy5yZWdpc3RlciBzZmNjLnNob3BwZXItbXlhY2NvdW50LmFkZHJlc3Nlcy5ydyBzZmNjLnNob3BwZXItbXlhY2NvdW50LnByb2R1Y3RsaXN0cy5ydyBjX3BheW1lbnRJbnN0cnVtZW50cyBzZmNjLnNob3BwZXItYmFza2V0cy1vcmRlcnMucncgc2ZjYy5zaG9wcGVyLWdpZnQtY2VydGlmaWNhdGVzIGNfdGltZVNsb3RzIHNmY2Muc2hvcHBlci1wcm9kdWN0LXNlYXJjaCBjX2hlcm9rdUdldEN1c3RvbWVyIHNmY2Muc2hvcHBlci1zZW8iLCJzdWIiOiJjYy1zbGFzOjpibHp2X3ByZDo6c2NpZDpmZDNjOWRiOC0yYTBkLTRmNGItOWU3NC0yOTRlMDY4ZjlhZTQ6OnVzaWQ6NTBjNGE3MjQtMGU5MS00ZDM2LThhMTQtNWNhZjhkYjliOTRhIiwic3NjIjoiNGpwcHQzN2EiLCJjdHgiOiJzbGFzIiwiaXNzIjoic2xhcy9wcm9kL2JsenZfcHJkIiwiaXN0IjoxLCJkbnQiOiIxIiwiYXVkIjoiY29tbWVyY2VjbG91ZC9wcm9kL2JsenZfcHJkIiwibmJmIjoxNzgwNjA0Njc0LCJzdHkiOiJVc2VyIiwiaXNiIjoidWlkbzpzbGFzOjp1cG46R3Vlc3Q6OnVpZG46R3Vlc3QgVXNlcjo6Z2NpZDphY2xyYUhsZWMxa0hvUmtlczNrcVlZeGJrMDo6Y2hpZDphbWV0bGxlciIsImV4cCI6MTc4MDYwNjUwNCwiaWF0IjoxNzgwNjA0NzA0LCJqdGkiOiJDMkMtMTk4MzM5ODY5MTAtNjQ3MzA0ODQyMjkxNDA4MDMzMTQ0NTI2MiJ9.6o1hs93AtxCQC3oKl7i9O_Ap4wzbHmez-sU6yjpB8z3rpzgYdry5C3cUCDdS73HkHA_89Tr1oc7iY4VTiDijnw"

def build_headers() -> dict:
    return {
        "accept": "*/*",
        "accept-language": "es-ES,es;q=0.9,en;q=0.8",
        "authorization": f"Bearer {BEARER_TOKEN}",
        "user-agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/148.0.0.0 Safari/537.36"
        ),
        "referer": "https://www.ametllerorigen.com/es/bebidas/bebidas-vegetales",
    }


def build_params(offset: int) -> dict:
    return {
        "siteId": "ametller",
        "refine": "cgid=101",
        "currency": "EUR",
        "locale": "es",
        "expand": (
            "availability,images,prices,represented_products,variations,"
            "promotions,custom_properties,page_meta_tags"
        ),
        "allImages": "true",
        "perPricebook": "true",
        "allVariationProperties": "true",
        "offset": str(offset),
        "limit": str(LIMIT),
    }


def force_jpg_url(image_url: str) -> str:
    """
    The API/image CDN may return webp by default.
    Replace fmt=webp with fmt=jpg.
    """
    parsed = urlparse(image_url)
    query = dict(parse_qsl(parsed.query))

    query["fmt"] = "jpg"

    return urlunparse(parsed._replace(query=urlencode(query)))


def clean_filename_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = text.strip("_")
    return text[:80]


def get_product_description(product: dict) -> str:
    return (
        product.get("productName")
        or product.get("name")
        or product.get("product_name")
        or ""
    ).strip()


def get_product_image_url(product: dict) -> str | None:
    """
    Salesforce Commerce Cloud search responses often expose the main image as:

        product["image"]["link"]

    But depending on expansion/configuration, images may also appear under:

        product["image"]["disBaseLink"]
        product["images"][...]

    This function tries several common shapes.
    """

    image = product.get("image")

    if isinstance(image, dict):
        for key in ("link", "disBaseLink", "absUrl", "url"):
            value = image.get(key)
            if value:
                return value

    images = product.get("images")

    if isinstance(images, list):
        for item in images:
            if isinstance(item, dict):
                for key in ("link", "disBaseLink", "absUrl", "url"):
                    value = item.get(key)
                    if value:
                        return value

    if isinstance(images, dict):
        for group in images.values():
            if isinstance(group, list):
                for item in group:
                    if isinstance(item, dict):
                        for key in ("link", "disBaseLink", "absUrl", "url"):
                            value = item.get(key)
                            if value:
                                return value

    return None


def download_image(
    session: requests.Session,
    image_url: str,
    output_path: Path,
) -> None:
    response = session.get(image_url, timeout=30)
    response.raise_for_status()

    output_path.write_bytes(response.content)


def fetch_products_page(
    session: requests.Session,
    offset: int,
) -> tuple[list[dict], int | None]:
    response = session.get(
        API_URL,
        params=build_params(offset),
        headers=build_headers(),
        timeout=30,
    )

    response.raise_for_status()

    data = response.json()

    products = data.get("hits", [])
    total = data.get("total")

    if not isinstance(products, list):
        raise RuntimeError("Unexpected API response: 'hits' is not a list")

    return products, total


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    session = requests.Session()

    rows = []
    product_counter = 1
    offset = 0
    total = None

    while True:
        print(f"Fetching offset={offset} limit={LIMIT}")

        products, total = fetch_products_page(session, offset)

        if not products:
            break

        for product in products:
            description = get_product_description(product)
            image_url = get_product_image_url(product)

            if not description:
                print("Skipping product without description")
                continue

            if not image_url:
                print(f"Skipping product without image: {description}")
                continue

            image_url = force_jpg_url(image_url)

            filename = f"product_{product_counter:04d}.jpg"
            image_path = OUTPUT_DIR / filename

            print(f"Downloading {filename}: {description}")

            try:
                download_image(session, image_url, image_path)
            except requests.RequestException as exc:
                print(f"Failed to download image for {description}: {exc}")
                continue

            rows.append(
                {
                    "filename": filename,
                    "description": description,
                    "category": "",
                }
            )

            product_counter += 1

        offset += LIMIT

        if total is not None and offset >= total:
            break

    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["filename", "description", "category"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Downloaded {len(rows)} products")
    print(f"CSV written to: {CSV_PATH}")
    print(f"Images written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
