import requests
import xml.etree.ElementTree as ET
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_hcl_sitemap():
    sitemap_url = "https://www.hcl-software.com/sitemap_en.xml"
    base_url = "https://www.hcl-software.com"

    try:
        logging.info(f"Fetching sitemap from: {sitemap_url}")
        response = requests.get(sitemap_url)
        response.raise_for_status()
    except requests.RequestException as e:
        logging.error(f"Failed to fetch sitemap: {e}")
        return {}

    root = ET.fromstring(response.content)
    namespace = {'ns': 'http://www.sitemaps.org/schemas/sitemap/0.9'}
    
    result = {}

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

    # return dict(list(result.items())[:5])
    # logging.info(f"Successfully loaded sitemap with {len(result)} entries")
    return result
