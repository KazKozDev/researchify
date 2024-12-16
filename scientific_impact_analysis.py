import requests
import time
from typing import Dict, List
import logging
from xml.etree import ElementTree as ET
from datetime import datetime
import json


class CitationsAnalyzer:
    def __init__(self):
        self.headers = {
            'User-Agent': 'CitationsAnalyzer/1.0'
        }
        self.arxiv_namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def analyze_with_gemma(self, analysis_data: Dict, total_citations: int, paper_title: str) -> str:
        """Analyze citation data using Gemma2B through Ollama API"""
        prompt = """Analyze scientific significance of "{title}" ({total_citations} total citations, {sample_size} analyzed):
    
Analyze the scientific article using its DOI or arXiv ID by providing total citations, year-over-year citation trends, types of citing publications, and geographic distribution (top 5 countries). Evaluate the article's scientific significance through quantitative metrics: citation volume, diversity of research domains citing the work, and international research engagement. Conclude with a concise assessment (max 150 words) highlighting the article's contribution to the field, research impact, potential scholarly influence, and whether it merits attention within the academic community.
"""
        prompt = prompt.format(
            title=paper_title,
            total_citations=total_citations,
            sample_size=analysis_data['total_count']
        )
        prompt += json.dumps(analysis_data, indent=2)

        try:
            response = requests.post(
                'http://localhost:11434/api/generate',
                json={
                    "model": "gemma2:9b",
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            return response.json()['response']
        except Exception as e:
            self.logger.error(f"Error getting Gemma-2B analysis: {e}")
            return "Failed to generate AI analysis"

    def calculate_metrics(self, analysis: Dict) -> Dict:
        """Calculate additional scientific impact metrics"""
        metrics = {}

        # Calculate year-over-year growth rates
        years = sorted(analysis['by_year'].keys())
        if len(years) > 1:
            metrics['yoy_growth'] = {}
            for i in range(1, len(years)):
                prev_year = years[i - 1]
                curr_year = years[i]
                prev_count = analysis['by_year'][prev_year]
                curr_count = analysis['by_year'][curr_year]
                if prev_count > 0:
                    growth = ((curr_count - prev_count) / prev_count) * 100
                    metrics['yoy_growth'][curr_year] = growth

        # Calculate research type ratios
        total_pubs = sum(analysis['by_type'].values())
        if total_pubs > 0:
            metrics['type_ratios'] = {
                ptype: (count / total_pubs) * 100
                for ptype, count in analysis['by_type'].items()
            }

        # Calculate geographic concentration
        total_countries = len(analysis['by_country'])
        if total_countries > 0:
            top_3_countries = sorted(
                analysis['by_country'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:3]
            metrics['geographic_concentration'] = sum(
                count for _, count in top_3_countries
            ) / sum(analysis['by_country'].values()) * 100

        return metrics

    def is_doi(self, identifier: str) -> bool:
        """Checks if the identifier is a DOI"""
        return identifier.startswith('10.') and '/' in identifier

    def clean_arxiv_id(self, arxiv_id: str) -> str:
        """Cleans the arXiv ID from the version and normalizes the format"""
        if 'v' in arxiv_id:
            arxiv_id = arxiv_id.split('v')[0]
        return arxiv_id

    def is_recent_paper(self, published_date: str, threshold_days: int = 90) -> bool:
        """Checks if the paper is recently published"""
        try:
            pub_date = datetime.strptime(published_date, '%Y-%m-%d')
            days_since_pub = (datetime.now() - pub_date).days
            return days_since_pub <= threshold_days
        except Exception:
            return False

    def get_arxiv_info(self, arxiv_id: str) -> Dict:
        """Gets information about the paper from arXiv, including DOI if available"""
        cleaned_id = self.clean_arxiv_id(arxiv_id)
        url = f"http://export.arxiv.org/api/query?id_list={cleaned_id}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            root = ET.fromstring(response.text)
            entry = root.find('.//atom:entry', self.arxiv_namespaces)

            if entry is None:
                return {}

            doi = entry.find('arxiv:doi', self.arxiv_namespaces)
            doi_text = doi.text if doi is not None else None

            title = entry.find('atom:title', self.arxiv_namespaces)
            authors = entry.findall('atom:author/atom:name', self.arxiv_namespaces)
            published = entry.find('atom:published', self.arxiv_namespaces)

            return {
                'arxiv_id': cleaned_id,
                'title': title.text.strip() if title is not None else None,
                'authors': [author.text for author in authors],
                'published': published.text[:10] if published is not None else None,
                'doi': doi_text
            }

        except Exception as e:
            self.logger.error(f"Error fetching information from arXiv: {e}")
            return {}

    def search_openalex(self, title: str, year: str = None) -> Dict:
        """Searches for a work in OpenAlex by title and year"""
        url = "https://api.openalex.org/works"
        params = {
            'filter': f'title.search:"{title}"'
        }
        if year:
            params['filter'] += f',publication_year:{year}'

        try:
            response = requests.get(url, params=params, headers=self.headers)
            response.raise_for_status()
            data = response.json()

            if data.get('results'):
                # Find the most accurate title match
                for work in data['results']:
                    if work['title'].lower().strip() == title.lower().strip():
                        return work
                # If no exact match, return the first result
                return data['results'][0]
            return {}

        except Exception as e:
            self.logger.error(f"Error searching in OpenAlex: {e}")
            return {}

    def get_openalex_data(self, identifier: str, is_doi: bool = False, title: str = None, year: str = None) -> Dict:
        """Gets data from OpenAlex by DOI or arXiv ID with fallback search by title"""
        if is_doi:
            url = f"https://api.openalex.org/works/doi:{identifier}"
        else:
            url = f"https://api.openalex.org/works/arxiv:{identifier}"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404 and title:
                print("\nAttempting to find the paper by title...")
                return self.search_openalex(title, year)
            else:
                self.logger.error(f"Error fetching data from OpenAlex: {e}")
            return {}
        except Exception as e:
            self.logger.error(f"Unexpected error fetching data from OpenAlex: {e}")
            return {}

    def get_citing_works(self, openalex_id: str, max_pages: int = 5) -> List[Dict]:
        """Gets a list of citing works through the OpenAlex API"""
        all_citations = []
        page = 1
        per_page = 50

        while page <= max_pages:
            url = "https://api.openalex.org/works"
            params = {
                'filter': f'cites:{openalex_id}',
                'per-page': per_page,
                'page': page
            }

            try:
                response = requests.get(url, params=params, headers=self.headers)
                response.raise_for_status()
                data = response.json()

                items = data.get('results', [])
                if not items:
                    break

                all_citations.extend(items)
                print(f"Retrieved {len(all_citations)} citations...")

                page += 1
                time.sleep(0.1)

            except Exception as e:
                self.logger.error(f"Error fetching page {page}: {e}")
                break

        return all_citations

    def analyze_citations(self, citations: List[Dict]) -> Dict:
        """Analyzes citations and returns statistics"""
        analysis = {
            'total_count': len(citations),
            'by_year': {},
            'by_journal': {},
            'by_type': {},
            'by_country': {}
        }

        for citation in citations:
            # Analysis by year
            year = citation.get('publication_year')
            if year:
                analysis['by_year'][year] = analysis['by_year'].get(year, 0) + 1

            # Analysis by journal
            journal = citation.get('host_venue', {}).get('display_name')
            if journal:
                analysis['by_journal'][journal] = analysis['by_journal'].get(journal, 0) + 1

            # Analysis by publication type
            pub_type = citation.get('type')
            if pub_type:
                analysis['by_type'][pub_type] = analysis['by_type'].get(pub_type, 0) + 1

            # Analysis by countries
            countries = set()
            for author in citation.get('authorships', []):
                for inst in author.get('institutions', []):
                    country = inst.get('country_code')
                    if country:
                        countries.add(country)

            for country in countries:
                analysis['by_country'][country] = analysis['by_country'].get(country, 0) + 1

        return analysis


def main():
    print("=" * 50)
    print("Scientific Impact Analyzer (arXiv + OpenAlex + Gemma-2B)")
    print("=" * 50)

    analyzer = CitationsAnalyzer()

    while True:
        paper_id = input("\nEnter the arXiv ID or DOI of the paper (or 'q' to quit): ").strip()

        if paper_id.lower() == 'q':
            print("\nExiting...")
            break

        is_doi = analyzer.is_doi(paper_id)
        paper_title = None

        if is_doi:
            print("\nProcessing DOI...")
            openalex_data = analyzer.get_openalex_data(paper_id, is_doi=True)
            if openalex_data:
                paper_title = openalex_data.get('title')
        else:
            print("\nFetching information from arXiv...")
            arxiv_info = analyzer.get_arxiv_info(paper_id)

            if not arxiv_info:
                print(f"\nPaper with ID {paper_id} not found in arXiv")
                continue

            paper_title = arxiv_info['title']
            print(f"\nTitle: {paper_title}")
            print(f"Authors: {', '.join(arxiv_info['authors'])}")
            print(f"Publication Date: {arxiv_info['published']}")

            year = arxiv_info['published'][:4] if arxiv_info['published'] else None

            if analyzer.is_recent_paper(arxiv_info['published']):
                print("\nNote: This paper was published recently (less than 90 days ago).")
                print("OpenAlex may not have indexed it yet. Please try checking later.")

            if arxiv_info.get('doi'):
                print(f"Found DOI: {arxiv_info['doi']}")
                print("\nFetching data from OpenAlex by DOI...")
                openalex_data = analyzer.get_openalex_data(arxiv_info['doi'], is_doi=True)
            else:
                print("\nFetching data from OpenAlex...")
                cleaned_id = analyzer.clean_arxiv_id(paper_id)
                openalex_data = analyzer.get_openalex_data(
                    cleaned_id,
                    is_doi=False,
                    title=paper_title,
                    year=year
                )

        if not openalex_data:
            print("\nFailed to retrieve citation data")
            print("Possible reasons:")
            print("1. The paper is not yet indexed in OpenAlex")
            print("2. The paper is too new")
            print("3. Issues with API connection")
            continue

        total_citations = openalex_data.get('cited_by_count', 0)
        print(f"\nTotal citations: {total_citations}")

        if total_citations > 0:
            print("\nFetching list of citations...")
            citations = analyzer.get_citing_works(openalex_data['id'])

            if citations:
                analysis = analyzer.analyze_citations(citations)
                metrics = analyzer.calculate_metrics(analysis)
                analysis['metrics'] = metrics

                print(f"\nAnalyzed citing works: {analysis['total_count']}")

                if analysis['by_year']:
                    print("\nCitation Timeline:")
                    for year, count in sorted(analysis['by_year'].items()):
                        percentage = (count / analysis['total_count']) * 100
                        growth = metrics.get('yoy_growth', {}).get(year, None)
                        if growth is not None:
                            print(f"{year}: {count} ({percentage:.1f}%) | Growth: {growth:+.1f}%")
                        else:
                            print(f"{year}: {count} ({percentage:.1f}%)")

        if analysis['by_type']:
            print("\nResearch Output Types:")
            for pub_type, count in sorted(analysis['by_type'].items()):
                percentage = (count / analysis['total_count']) * 100
                print(f"- {pub_type}: {count} ({percentage:.1f}%)")

        if analysis['by_country']:
            print("\nGeographic Distribution (Top 10):")
            countries = sorted(analysis['by_country'].items(),
                               key=lambda x: x[1], reverse=True)[:10]
            total_countries = len(analysis['by_country'])
            print(f"Total contributing countries: {total_countries}")

            for country, count in countries:
                percentage = (count / analysis['total_count']) * 100
                print(f"- {country}: {count} ({percentage:.1f}%)")

        if metrics.get('geographic_concentration'):
            print(f"\nGeographic concentration (top 3 countries): "
                  f"{metrics['geographic_concentration']:.1f}%")

        if analysis['by_journal']:
            print("\nTop Research Venues:")
            journals = sorted(analysis['by_journal'].items(),
                              key=lambda x: x[1], reverse=True)[:10]
            for journal, count in journals:
                percentage = (count / analysis['total_count']) * 100
                print(f"- {journal}: {count} ({percentage:.1f}%)")

        print("\nGenerating Scientific Impact Analysis...")
        ai_analysis = analyzer.analyze_with_gemma(analysis, total_citations, paper_title)
        print("\nScientific Impact Analysis:")
        print(ai_analysis)
    else:
        print("\nFailed to retrieve detailed citation information")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()