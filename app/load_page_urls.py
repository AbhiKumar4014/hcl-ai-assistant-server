import requests
import xml.etree.ElementTree as ET
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_hcl_sitemap(sitemap_urls):
    # sitemap_urls = [
    #     "https://www.hcl-software.com/sitemap_en.xml",
    #     "https://www.hcl-software.com/sitemap_blog.xml"
    # ]
    base_url = "https://www.hcl-software.com"
    namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}

    result = {}

    for sitemap_url in sitemap_urls:
        try:
            logging.info(f"Fetching sitemap from: {sitemap_url}")
            response = requests.get(sitemap_url)
            response.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Failed to fetch sitemap: {sitemap_url}, error: {e}")
            continue  # Skip this sitemap and move on

        try:
            root = ET.fromstring(response.content)
        except ET.ParseError as e:
            logging.error(f"Failed to parse sitemap XML: {sitemap_url}, error: {e}")
            continue

        for url in root.findall("ns:url", namespaces=namespace):
            loc = url.find("ns:loc", namespaces=namespace)
            if loc is not None:
                link = loc.text.strip()
                if link == base_url:
                    key = "home"
                else:
                    endpoint = link.replace(base_url, "").strip("/")
                    key = endpoint or "home"
                result.setdefault(key, []).append(link)

    return result
