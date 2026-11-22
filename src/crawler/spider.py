"""
NITP Institutional Category-Driven Spider
=========================================
Robust, carefully architected crawler that classifies input endpoints 
into structured institutional categories:

  Category 1: NOTICE FEEDS      (/Notices/All, /Notices/Academic, /Notices/Exam)
              -> Powered by admin.nitp.ac.in/api/notice?type={all|academics|exam}
              -> Extracts official circulars, fee notifications, orders

  Category 2: EVENT CIRCULARS   (/Notices/Events)
              -> Powered by admin.nitp.ac.in/api/events?type=all
              -> Extracts workshops, conferences, seminars & brochure PDFs

  Category 3: DEPARTMENT SYLLABI (/Department/<DEPT>/syllabus/<COURSE>)
              -> Parses syllabus table & Google Docs/Drive course exports
              -> Extracts course code, subject name, units, and PDFs

  Category 4: DIRECT PDF / HTML (/Others/..., or direct .pdf URLs)
              -> Safe fallback for direct PDF links with dedup & error handling
"""

import os
import re
import json
import hashlib
import requests
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
from urllib.parse import urlparse, urljoin


MANIFEST_FILE = "data/crawler_manifest.json"
NITP_API_BASE = "https://admin.nitp.ac.in/api"


class CategorySpider:
    """
    Modular, category-aware institutional spider.
    Auto-detects the domain/endpoint category and routes to the exact scraper.
    """

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.crawled_dir = os.path.join(data_dir, "crawled")
        os.makedirs(self.crawled_dir, exist_ok=True)

        self.session = requests.Session()
        self.session.verify = False
        self.session.headers.update({
            "User-Agent": "NITP-RAG-Crawler/2.0 (NIT Patna Institutional Research)"
        })

        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

        self.manifest = self._load_manifest()

    # ================================================================
    #  MANIFEST & DEDUPLICATION REGISTRY
    # ================================================================

    def _load_manifest(self) -> dict:
        if os.path.exists(MANIFEST_FILE):
            with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return {"downloaded": {}}

    def _save_manifest(self):
        with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
            json.dump(self.manifest, f, indent=2, ensure_ascii=False)

    def _is_downloaded(self, url: str) -> bool:
        return url in self.manifest["downloaded"]

    def _register(self, url: str, filepath: str, title: str = "", category: str = "General", extra: dict = None):
        entry = {
            "filename": os.path.basename(filepath),
            "title": title,
            "category": category,
            "indexed_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        if extra:
            entry.update(extra)
        self.manifest["downloaded"][url] = entry

    # ================================================================
    #  CATEGORY CLASSIFIER
    # ================================================================

    def classify_endpoint(self, url: str) -> str:
        """
        Classifies an endpoint into one of the designated categories:
        1. DIRECT_PDF: Direct .pdf file link (including S3 or document URLs ending in .pdf)
        2. GDRIVE_SYLLABUS: Google Drive or Docs link (syllabus or curriculum documents)
        3. EVENT: Event circulars (/Notices/Events or /events)
        4. NOTICE: General, academic, or exam notice feeds (/Notices/...)
        5. SYLLABUS: Department syllabus webpage (/Department/.../syllabus/...)
        6. GENERIC_HTML: Any standard HTML webpage containing document/PDF links
        """
        clean_url = url.strip().lower()
        parsed = urlparse(clean_url)
        netloc = parsed.netloc
        path = parsed.path

        # 1. Any direct PDF link takes priority (even if URL contains '/notices/')
        if path.endswith(".pdf") or clean_url.endswith(".pdf"):
            return "DIRECT_PDF"
        # 2. Google Drive / Docs documents
        elif "drive.google.com" in netloc or "docs.google.com" in netloc:
            return "GDRIVE_SYLLABUS"
        # 3. Events feed
        elif "/notices/events" in path or "/event" in path:
            return "EVENT"
        # 4. Notice feeds (All, Academic, Exam, Tender, etc.)
        elif "/notices" in path:
            return "NOTICE"
        # 5. Department syllabus webpage
        elif "/department/" in path and "/syllabus" in path:
            return "SYLLABUS"
        # 6. Fallback generic HTML
        else:
            return "GENERIC_HTML"

    # ================================================================
    #  MAIN ENTRY POINT: CRAWL
    # ================================================================

    def crawl(self, url: str, days: int = 2, max_files: int = 10) -> List[Dict[str, Any]]:
        """
        Main crawling method. Classifies URL and dispatches to the right pipeline.
        Intelligently handles date lookback:
        - NOTICE: Strict lookback window (default 2-7 days)
        - EVENT: Extended lookback window (events are published infrequently; auto-defaults to 90 days if days < 30)
        - SYLLABUS / GDRIVE_SYLLABUS / DIRECT_PDF / GENERIC_HTML: Evergreen static documents; date lookback is not applicable
        """
        category = self.classify_endpoint(url)
        print(f"\n{'='*65}")
        print(f"  NITP CATEGORY SPIDER: {category}")
        print(f"  Target: {url}")
        
        # Explain date lookback relevance
        if category == "NOTICE":
            print(f"  Lookback: {days} days (time-sensitive notices) | Max Files: {max_files}")
        elif category == "EVENT":
            actual_days = max(days, 90) if days < 30 else days
            print(f"  Lookback: {actual_days} days (events published periodically) | Max Files: {max_files}")
        else:
            print(f"  Lookback: N/A (Static curriculum/document) | Max Files: {max_files}")
        print(f"{'='*65}")

        if category == "GDRIVE_SYLLABUS":
            results = self._crawl_direct_gdrive_syllabus(url)
        elif category == "NOTICE":
            results = self._crawl_notices(url, days=days, max_files=max_files)
        elif category == "EVENT":
            event_days = max(days, 90) if days < 30 else days
            results = self._crawl_events(url, days=event_days, max_files=max_files)
        elif category == "SYLLABUS":
            results = self._crawl_syllabus(url, max_files=max_files)
        elif category == "DIRECT_PDF":
            results = self._crawl_direct_pdf(url)
        else:
            results = self._crawl_generic_html(url, max_files=max_files)

        self._save_manifest()
        print(f"\n[Summary] Successfully retrieved {len(results)} new file(s) for ingestion.")
        return results

    # ================================================================
    #  CATEGORY: DIRECT GOOGLE DRIVE / DOCS SYLLABUS DOCUMENT
    # ================================================================

    def _crawl_direct_gdrive_syllabus(self, url: str) -> List[Dict[str, Any]]:
        print(f"  [Category: GDRIVE_SYLLABUS] Processing direct Google Drive/Docs syllabus link")
        clean_url = re.sub(r'/(view|edit|preview|pub).*$', '', url).strip()
        if self._is_downloaded(clean_url):
            print("  [SKIP] Direct Drive link already downloaded in manifest.")
            return []

        file_id = self._extract_gdrive_file_id(clean_url)
        if not file_id:
            print(f"  [ERROR] Cannot parse Google Drive file ID from {url}")
            return []

        # Download PDF bytes
        downloaded = self._export_gdrive_pdf(clean_url, title_hint=f"Syllabus_{file_id[:8]}")
        if not downloaded:
            return []

        filepath = downloaded["filepath"]

        # Inspect PDF cover/first page to detect Department & Course title
        detected_title = f"Course Structure Syllabus ({file_id[:8]})"
        detected_dept = "General"
        try:
            import pypdf
            reader = pypdf.PdfReader(filepath)
            if reader.pages:
                first_text = reader.pages[0].extract_text()
                # Search for Department name
                dept_match = re.search(r'Department\s+of\s+([A-Za-z\s]{3,50})', first_text)
                if dept_match:
                    detected_dept = re.sub(r'\s+', ' ', dept_match.group(1)).strip()

                # Search for Course Structure / Syllabus title
                title_match = re.search(
                    r'(?:Course\s+Structure|Syllabus|Curriculum)[\s\S]{0,140}?(?:B\.?\s*Tech|M\.?\s*Tech|Dual\s+Degree|Doctor|Degree)[^\n]*',
                    first_text,
                    re.IGNORECASE
                )
                if title_match:
                    detected_title = re.sub(r'\s+', ' ', title_match.group(0)).strip()
                elif dept_match:
                    detected_title = f"Syllabus - Department of {detected_dept}"
        except Exception as e:
            print(f"  [NOTE] Cover page parsing skipped: {e}")

        # Update entry
        downloaded.update({
            "title": detected_title,
            "department": detected_dept,
            "category": "Syllabus",
            "source_url": clean_url,
        })
        self._register(clean_url, filepath, title=detected_title, category="Syllabus", extra={"department": detected_dept})
        return [downloaded]

    # ================================================================
    #  CATEGORY 1: NOTICE FEEDS (/Notices/All, /Notices/Academic, /Notices/Exam)
    # ================================================================

    def _crawl_notices(self, url: str, days: int = 2, max_files: int = 10) -> List[Dict[str, Any]]:
        path = urlparse(url).path.lower().strip("/")
        category_slug = path.split("/")[-1]

        # Map frontend slug to backend API parameter
        type_map = {
            "all": "all",
            "academic": "academics",
            "academics": "academics",
            "exam": "exam",
            "tender": "tender",
            "general": "general",
        }
        notice_type = type_map.get(category_slug, "all")
        print(f"  [Category: NOTICE] Backend query: type={notice_type}")

        cutoff = datetime.now() - timedelta(days=days)
        cutoff_ms = int(cutoff.timestamp() * 1000)

        results = []
        page = 1

        while len(results) < max_files:
            try:
                resp = self.session.get(
                    f"{NITP_API_BASE}/notice",
                    params={"type": notice_type, "page": page, "limit": 20},
                    timeout=15
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"  [ERROR] Notice API error: {e}")
                break

            items = data.get("data", [])
            if not items:
                break

            for item in items:
                ts = int(item.get("timestamp", 0))
                # Skip if older than lookback cutoff
                if ts > 0 and ts < cutoff_ms:
                    continue

                title = item.get("title", "Untitled Notice").strip()
                dt_str = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d")

                for att in item.get("attachments", []):
                    if len(results) >= max_files:
                        break
                    pdf_url = att.get("url", "")
                    if not pdf_url.lower().endswith(".pdf"):
                        continue

                    if self._is_downloaded(pdf_url):
                        continue

                    downloaded = self._download_binary(
                        pdf_url,
                        title_hint=f"{dt_str}_{title}",
                        category="Notice"
                    )
                    if downloaded:
                        downloaded.update({
                            "title": title,
                            "date": dt_str,
                            "notice_id": item.get("id", ""),
                            "category": "Notice",
                            "source_url": pdf_url,
                        })
                        self._register(pdf_url, downloaded["filepath"], title=title, category="Notice", extra={"date": dt_str})
                        results.append(downloaded)

            if page >= data.get("totalPages", 1):
                break
            page += 1

        return results

    # ================================================================
    #  CATEGORY 2: EVENT CIRCULARS (/Notices/Events)
    # ================================================================

    def _crawl_events(self, url: str, days: int = 60, max_files: int = 10) -> List[Dict[str, Any]]:
        print(f"  [Category: EVENT] Backend query: admin.nitp.ac.in/api/events?type=all")

        cutoff = datetime.now() - timedelta(days=days)
        cutoff_ms = int(cutoff.timestamp() * 1000)

        results = []
        page = 1

        while len(results) < max_files:
            try:
                resp = self.session.get(
                    f"{NITP_API_BASE}/events",
                    params={"type": "all", "page": page, "limit": 20},
                    timeout=15
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                print(f"  [ERROR] Events API error: {e}")
                break

            items = data.get("data", [])
            if not items:
                break

            for item in items:
                if len(results) >= max_files:
                    break

                ts = int(item.get("timestamp", 0))
                title = item.get("title", "Untitled Event").strip()
                venue = item.get("venue", "NIT Patna")
                dt_str = datetime.fromtimestamp(ts / 1000).strftime("%Y-%m-%d")

                # If days cutoff is provided, enforce it (unless days > 180)
                if days < 180 and ts < cutoff_ms:
                    continue

                for att in item.get("attachments", []):
                    if len(results) >= max_files:
                        break
                    pdf_url = att.get("url", "")
                    if not pdf_url.lower().endswith(".pdf"):
                        continue

                    if self._is_downloaded(pdf_url):
                        continue

                    downloaded = self._download_binary(
                        pdf_url,
                        title_hint=f"Event_{dt_str}_{title}",
                        category="Event"
                    )
                    if downloaded:
                        downloaded.update({
                            "title": f"Event: {title}",
                            "date": dt_str,
                            "venue": venue,
                            "category": "Event",
                            "source_url": pdf_url,
                        })
                        self._register(pdf_url, downloaded["filepath"], title=title, category="Event", extra={"venue": venue, "date": dt_str})
                        results.append(downloaded)

            if page >= data.get("totalPages", 1):
                break
            page += 1

        return results

    # ================================================================
    #  CATEGORY 3: DEPARTMENT SYLLABUS (/Department/<DEPT>/syllabus/<COURSE>)
    # ================================================================

    def _crawl_syllabus(self, url: str, max_files: int = 10) -> List[Dict[str, Any]]:
        # Extract department from URL (e.g. /Department/CSE/syllabus/...)
        parts = [p for p in urlparse(url).path.split("/") if p]
        dept = "CSE"
        if "Department" in parts:
            d_idx = parts.index("Department")
            if d_idx + 1 < len(parts):
                dept = parts[d_idx + 1].upper()

        print(f"  [Category: SYLLABUS] Target Department: {dept}")

        try:
            resp = self.session.get(url, timeout=20)
            resp.raise_for_status()
            html = resp.text
        except Exception as e:
            print(f"  [ERROR] Cannot fetch syllabus page: {e}")
            return []

        # Find all anchor tags containing Google Docs or Drive links
        anchor_pattern = r'<a[^>]+href=["\'](https?://(?:drive|docs)\.google\.com/[^"\']+)["\'][^>]*>(.*?)</a>'
        matches = re.findall(anchor_pattern, html, re.IGNORECASE | re.DOTALL)

        results = []
        seen_urls = set()

        for href, text in matches:
            if len(results) >= max_files:
                break

            clean_href = re.sub(r'/(view|edit|preview|pub).*$', '', href).strip()
            clean_title = re.sub(r'<[^>]+>', '', text).strip()
            clean_title = re.sub(r'\s+', ' ', clean_title).strip()

            # Ignore generic navigation text
            if not clean_title or clean_title.lower() in {"view", "download", "click here", "syllabus", "pan/gst", "organization chart"}:
                continue

            if clean_href in seen_urls or self._is_downloaded(clean_href):
                continue
            seen_urls.add(clean_href)

            # Export from Google Drive / Docs as PDF
            downloaded = self._export_gdrive_pdf(clean_href, title_hint=f"{dept}_{clean_title}")
            if downloaded:
                downloaded.update({
                    "title": f"{dept} Syllabus: {clean_title}",
                    "department": dept,
                    "category": "Syllabus",
                    "source_url": clean_href,
                })
                self._register(clean_href, downloaded["filepath"], title=clean_title, category="Syllabus", extra={"department": dept})
                results.append(downloaded)

        return results

    # ================================================================
    #  CATEGORY 4: DIRECT PDF LINK
    # ================================================================

    def _crawl_direct_pdf(self, url: str) -> List[Dict[str, Any]]:
        print(f"  [Category: DIRECT_PDF] Downloading single PDF target")
        if self._is_downloaded(url):
            print("  [SKIP] Already indexed.")
            return []

        clean_name = os.path.basename(urlparse(url).path) or "document.pdf"
        downloaded = self._download_binary(url, title_hint=clean_name, category="Document")
        if downloaded:
            detected_title = clean_name.replace(".pdf", "").replace("_", " ")
            try:
                import pypdf
                reader = pypdf.PdfReader(downloaded["filepath"])
                if reader.pages:
                    txt = reader.pages[0].extract_text() or ""
                    lines = [l.strip() for l in txt.split("\n") if l.strip() and len(l.strip()) > 5]
                    if lines:
                        detected_title = lines[0][:80]
            except Exception:
                pass

            downloaded.update({
                "title": detected_title,
                "category": "Document",
                "source_url": url,
            })
            self._register(url, downloaded["filepath"], title=downloaded["title"], category="Document")
            return [downloaded]
        return []

    # ================================================================
    #  FALLBACK: GENERIC HTML
    # ================================================================

    def _crawl_generic_html(self, url: str, max_files: int = 10) -> List[Dict[str, Any]]:
        print(f"  [Category: GENERIC_HTML] Scanning page for PDF links...")
        try:
            resp = self.session.get(url, timeout=20)
            resp.raise_for_status()
            html = resp.text
        except Exception as e:
            print(f"  [ERROR] Cannot fetch page: {e}")
            return []

        # Find direct PDF links
        pattern = r'<a[^>]*href=["\']([^"\']*\.pdf[^"\']*)["\'][^>]*>(.*?)</a>'
        matches = re.findall(pattern, html, re.IGNORECASE | re.DOTALL)

        results = []
        for href, text in matches:
            if len(results) >= max_files:
                break
            abs_url = urljoin(url, href)
            if self._is_downloaded(abs_url):
                continue

            clean_text = re.sub(r'<[^>]+>', '', text).strip()
            clean_text = re.sub(r'\s+', ' ', clean_text).strip() or "document"

            downloaded = self._download_binary(abs_url, title_hint=clean_text, category="Document")
            if downloaded:
                downloaded.update({
                    "title": clean_text,
                    "category": "Document",
                    "source_url": abs_url,
                })
                self._register(abs_url, downloaded["filepath"], title=clean_text, category="Document")
                results.append(downloaded)

        return results

    # ================================================================
    #  FILE DOWNLOAD & G-DRIVE EXPORT HELPERS
    # ================================================================

    def _extract_gdrive_file_id(self, url: str) -> Optional[str]:
        match = re.search(r'/(?:file/d/|document/d/|spreadsheets/d/|d/|id=)([a-zA-Z0-9_-]+)', url)
        return match.group(1) if match else None

    def _download_binary(self, url: str, title_hint: str = "", category: str = "General") -> Optional[Dict[str, Any]]:
        try:
            resp = self.session.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            print(f"  [ERROR] HTTP fetch failed for {url[:80]}: {e}")
            return None

        # Build clean filename
        safe_name = re.sub(r'[^\w\s-]', '', title_hint).strip().replace(' ', '_')[:60]
        url_hash = hashlib.md5(url.encode()).hexdigest()[:6]
        filename = f"{safe_name}_{url_hash}.pdf" if safe_name else f"doc_{url_hash}.pdf"
        filepath = os.path.join(self.crawled_dir, filename)

        with open(filepath, "wb") as f:
            f.write(resp.content)

        print(f"  [DOWNLOADED] {filename} ({len(resp.content)//1024} KB)")
        return {"filepath": filepath}

    def _export_gdrive_pdf(self, gdrive_url: str, title_hint: str = "") -> Optional[Dict[str, Any]]:
        file_id_match = re.search(r'/d/([a-zA-Z0-9_-]+)', gdrive_url)
        if not file_id_match:
            return None
        file_id = file_id_match.group(1)

        if "docs.google.com/document" in gdrive_url:
            export_url = f"https://docs.google.com/document/d/{file_id}/export?format=pdf"
        elif "docs.google.com/spreadsheets" in gdrive_url:
            export_url = f"https://docs.google.com/spreadsheets/d/{file_id}/export?format=pdf"
        else:
            export_url = f"https://drive.google.com/uc?export=download&id={file_id}"

        try:
            resp = self.session.get(export_url, timeout=30, allow_redirects=True)
            content_type = resp.headers.get("Content-Type", "")
            if resp.status_code != 200 or ("text/html" in content_type and len(resp.content) < 5000):
                print(f"  [SKIP] Private or unavailable Google file: {file_id}")
                return None

            safe_name = re.sub(r'[^\w\s-]', '', title_hint).strip().replace(' ', '_')[:60]
            filename = f"{safe_name}_{file_id[:6]}.pdf" if safe_name else f"gdrive_{file_id[:8]}.pdf"
            filepath = os.path.join(self.crawled_dir, filename)

            with open(filepath, "wb") as f:
                f.write(resp.content)

            print(f"  [EXPORTED] {filename} ({len(resp.content)//1024} KB)")
            return {"filepath": filepath}
        except Exception as e:
            print(f"  [ERROR] Google export failed for {file_id}: {e}")
            return None


# Backward-compatible alias
UniversalSpider = CategorySpider


if __name__ == "__main__":
    import sys
    test_url = sys.argv[1] if len(sys.argv) > 1 else "https://www.nitp.ac.in/Notices/All"
    days_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    max_arg = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    spider = CategorySpider()
    docs = spider.crawl(test_url, days=days_arg, max_files=max_arg)
    print(f"\nTotal Crawled: {len(docs)}")
