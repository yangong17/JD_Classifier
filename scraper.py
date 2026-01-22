"""
Job Description Scraper for City of Eastvale Class Specifications.
Uses Playwright for dynamic content and exports to CSV.
"""

import re
from playwright.sync_api import sync_playwright
import pandas as pd

BASE_URL = "https://www.governmentjobs.com"
START_URL = f"{BASE_URL}/careers/eastvaleca/classspecs?keywords="


def get_all_job_urls(page):
    """Navigate through all pages and collect job detail URLs."""
    job_urls = []
    current_page = 1

    while True:
        # Build URL with page parameter
        url = f"{START_URL}&page={current_page}" if current_page > 1 else START_URL
        page.goto(url)

        try:
            page.wait_for_selector("a.item-details-link", timeout=15000)
        except:
            # No more results on this page
            break

        # Collect job links on the current page
        links = page.query_selector_all("a.item-details-link")
        if not links:
            break

        print(f"  Page {current_page}: found {len(links)} jobs")

        for link in links:
            href = link.get_attribute("href")
            if href:
                full_url = href if href.startswith("http") else BASE_URL + href
                job_urls.append(full_url)

        # Check if there's a next page
        next_disabled = page.query_selector("li.next.disabled")
        if next_disabled:
            break

        current_page += 1

    return list(set(job_urls))  # Remove duplicates


def extract_job_details(page, url):
    """Extract job details from a single job specification page."""
    page.goto(url)
    page.wait_for_selector("h1.entity-title, dt.term-description", timeout=30000)

    job_data = {"url": url}

    # Extract title from h1.entity-title
    title_el = page.query_selector("h1.entity-title")
    job_data["title"] = title_el.inner_text().strip() if title_el else ""

    # Extract class code and salary from dt/dd pairs
    term_descriptions = page.query_selector_all("dt.term-description")
    for dt in term_descriptions:
        label = dt.inner_text().strip()
        dd = dt.evaluate_handle("el => el.nextElementSibling")
        if dd:
            value = dd.evaluate("el => el ? el.textContent.trim() : ''")
            if "Class Code" in label:
                job_data["class_code"] = value
            elif "Salary" in label:
                job_data["salary"] = value

    # Set defaults if not found
    job_data.setdefault("class_code", "")
    job_data.setdefault("salary", "")

    # Extract full description from the main content area
    spec_info = page.query_selector("#class-spec-info, #class-spec-definition, .class-spec-body")
    if spec_info:
        description = spec_info.inner_text()
    else:
        # Fallback: get body text
        description = page.query_selector("body").inner_text()

    # Exclude "EQUAL EMPLOYMENT OPPORTUNITY & ACCOMODATION" section and everything after
    eeo_markers = [
        "EQUAL EMPLOYMENT OPPORTUNITY & ACCOMODATION",
        "EQUAL EMPLOYMENT OPPORTUNITY & ACCOMMODATION",
        "EQUAL EMPLOYMENT OPPORTUNITY AND ACCOMMODATION",
    ]
    for marker in eeo_markers:
        if marker in description:
            description = description.split(marker)[0]
            break

    job_data["description"] = description.strip()

    return job_data



def main():
    """Main entry point for the scraper."""
    print("Starting scraper...")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        page = context.new_page()

        # Step 1: Collect all job URLs
        print("Collecting job URLs from list pages...")
        job_urls = get_all_job_urls(page)
        print(f"Found {len(job_urls)} job listings.")

        # Step 2: Extract details from each job page
        all_jobs = []
        for i, url in enumerate(job_urls, 1):
            print(f"Scraping job {i}/{len(job_urls)}: {url}")
            try:
                job_data = extract_job_details(page, url)
                all_jobs.append(job_data)
            except Exception as e:
                print(f"  Error scraping {url}: {e}")

        browser.close()

    # Step 3: Export to CSV
    if all_jobs:
        df = pd.DataFrame(all_jobs)
        output_file = "job_descriptions.csv"
        df.to_csv(output_file, index=False)
        print(f"\nSaved {len(all_jobs)} jobs to {output_file}")
    else:
        print("No jobs were scraped.")


if __name__ == "__main__":
    main()
