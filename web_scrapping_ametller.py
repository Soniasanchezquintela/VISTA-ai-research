
from playwright.sync_api import sync_playwright
import requests
import pandas as pd
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# =====================================================
# CONFIG
# =====================================================

BASE_URL = (
    "https://www.ametllerorigen.com"
    "/mobify/proxy/api/search/shopper-search/v1/"
    "organizations/f_ecom_blzv_prd/product-search"
)

SHOP_URL = "https://www.ametllerorigen.com/ca/online"

OUTPUT_FOLDER = "ametller_images"

CSV_NAME = "ametller_full_catalogue.csv"

PAGE_SIZE = 200

MAX_WORKERS = 20

# =====================================================
# CAPTURE AUTHENTICATED HEADERS
# =====================================================

captured_headers = {}

print("Launching browser...")

with sync_playwright() as p:

    browser = p.chromium.launch(
        headless=False
    )

    context = browser.new_context()

    page = context.new_page()

    # ---------------------------------------------
    # INTERCEPT API REQUESTS
    # ---------------------------------------------

    def handle_request(request):

        global captured_headers

        url = request.url

        if "product-search" in url:

            print("\nCaptured authenticated request!")

            captured_headers = dict(request.headers)

    page.on("request", handle_request)

    # ---------------------------------------------
    # OPEN WEBSITE
    # ---------------------------------------------

    page.goto(
        SHOP_URL,
        wait_until="domcontentloaded",
        timeout=120000
    )

    page.wait_for_timeout(5000)

    print("\nIMPORTANT:")
    print("- click categories")
    print("- scroll products")
    print("- wait until products appear")

    input("\nPress ENTER when done...")

    browser.close()

# =====================================================
# CLEAN HEADERS
# =====================================================

REMOVE_HEADERS = [
    "host",
    "content-length",
    "connection",
    "sec-fetch-site",
    "sec-fetch-mode",
    "sec-fetch-dest",
    "accept-encoding"
]

for h in REMOVE_HEADERS:

    captured_headers.pop(h, None)

# =====================================================
# CREATE SESSION
# =====================================================

session = requests.Session()

session.headers.update(captured_headers)

# =====================================================
# DISCOVER CATEGORY IDS
# =====================================================

print("\nDiscovering categories...")

params = {
    "siteId": "ametller",
    "refine": "cgid=6",
    "currency": "EUR",
    "locale": "ca",
    "offset": 0,
    "limit": 1
}

r = session.get(
    BASE_URL,
    params=params,
    timeout=60
)

print("Status:", r.status_code)

r.raise_for_status()

data = r.json()

category_map = {}

def extract_categories(values):

    for item in values:

        cgid = item.get("value")

        label = item.get("label")

        if cgid and label:

            category_map[cgid] = label

        children = item.get("values", [])

        if children:

            extract_categories(children)

for refinement in data.get("refinements", []):

    if refinement.get("attributeId") == "cgid":

        extract_categories(
            refinement.get("values", [])
        )

print("\nCATEGORIES FOUND:\n")

for cgid, label in category_map.items():

    print(cgid, "->", label)

# =====================================================
# FETCH ALL PRODUCTS
# =====================================================

all_products = []

seen_products = set()

for cgid, category_name in category_map.items():

    print("\n====================================")
    print(f"CATEGORY {cgid}: {category_name}")

    offset = 0

    while True:

        params = {
            "siteId": "ametller",
            "refine": f"cgid={cgid}",
            "currency": "EUR",
            "locale": "ca",
            "expand": (
                "availability,"
                "images,"
                "prices,"
                "represented_products,"
                "variations,"
                "promotions,"
                "custom_properties,"
                "page_meta_tags"
            ),
            "allImages": "true",
            "perPricebook": "true",
            "allVariationProperties": "true",
            "offset": offset,
            "limit": PAGE_SIZE
        }

        r = session.get(
            BASE_URL,
            params=params,
            timeout=60
        )

        print(
            f"Offset {offset} -> "
            f"{r.status_code}"
        )

        if r.status_code != 200:

            break

        data = r.json()

        hits = data.get("hits", [])

        print(f"Products: {len(hits)}")

        if not hits:

            break

        for hit in hits:

            try:

                product_id = hit.get(
                    "productId"
                )

                if product_id in seen_products:
                    continue

                seen_products.add(product_id)

                name = hit.get(
                    "productName"
                )

                price = hit.get(
                    "price"
                )

                image_url = None

                image = hit.get("image")

                if image:

                    image_url = image.get(
                        "link"
                    )

                # HIGH QUALITY IMAGE
                if image_url:

                    if "?" in image_url:

                        image_url += (
                            "&sw=2000&q=95"
                        )

                    else:

                        image_url += (
                            "?sw=2000&q=95"
                        )

                all_products.append({
                    "id": product_id,
                    "name": name,
                    "price": price,
                    "image_url": image_url,
                    "category_id": cgid,
                    "category_name": category_name
                })

            except Exception as e:

                print("Parse error:", e)

        offset += PAGE_SIZE

# =====================================================
# SAVE CSV
# =====================================================

df = pd.DataFrame(all_products)

df.drop_duplicates(inplace=True)

df.to_csv(
    CSV_NAME,
    index=False,
    encoding="utf-8-sig"
)

print("\n====================================")
print("TOTAL PRODUCTS:", len(df))
print("CSV SAVED:", CSV_NAME)

# =====================================================
# DOWNLOAD IMAGES
# =====================================================

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def safe_filename(text):

    return "".join(
        c for c in str(text)
        if c.isalnum() or c in (" ", "-", "_")
    ).strip()[:150]


def download_image(row):

    try:

        url = row["image_url"]

        if not url:

            return "NO IMAGE"

        filename = (
            safe_filename(row["name"])
            + ".jpg"
        )

        filepath = os.path.join(
            OUTPUT_FOLDER,
            filename
        )

        if os.path.exists(filepath):

            return f"SKIP {filename}"

        r = requests.get(
            url,
            timeout=30
        )

        if r.status_code == 200:

            with open(filepath, "wb") as f:

                f.write(r.content)

            return f"OK {filename}"

        return f"FAILED {filename}"

    except Exception as e:

        return f"ERROR {e}"

print("\nDownloading images...")

with ThreadPoolExecutor(
    max_workers=MAX_WORKERS
) as executor:

    futures = [
        executor.submit(
            download_image,
            row
        )
        for _, row in df.iterrows()
    ]

    for future in as_completed(futures):

        print(future.result())

print("\nDONE")