import os
import fitz
import requests
import tempfile
import ollama
from typing import Dict, Optional, List
import logging
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from vector_store import VectorStore, RAGSystem, Document

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PaperAnalysis:
    """Data class for storing paper analysis results."""
    paper_id: str
    title: str
    sections: Dict[str, str]
    main_findings: List[str]
    methodology: str
    conclusions: str
    limitations: Optional[str]
    future_work: Optional[str]
    analysis_date: datetime

    def to_dict(self):
        return {
            'paper_id': self.paper_id,
            'title': self.title,
            'sections': self.sections,
            'main_findings': self.main_findings,
            'methodology': self.methodology,
            'conclusions': self.conclusions,
            'limitations': self.limitations,
            'future_work': self.future_work,
            'analysis_date': self.analysis_date.isoformat()
        }

class PaperAnalyzer:
    def __init__(
        self,
        vector_store: VectorStore,
        rag_system: RAGSystem,
        model_name: str = "gemma2:9b",
        cache_dir: str = "paper_cache"
    ):
        """
        Initialize the paper analyzer.
        
        Arguments:
            vector_store: An initialized instance of VectorStore
            rag_system: An initialized instance of RAGSystem
            model_name: Name of the LLM model to use
            cache_dir: Directory for caching downloaded papers
        """
        self.vector_store = vector_store
        self.rag_system = rag_system
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _download_pdf(self, url: str) -> Optional[str]:
        """Download PDF from a URL and return the path to the temporary file."""
        try:
            # Create a hash of the URL for caching
            cache_file = os.path.join(self.cache_dir, f"{hash(url)}.pdf")
            
            # Check cache first
            if os.path.exists(cache_file):
                return cache_file
            
            # Download if not in cache
            response = requests.get(url)
            response.raise_for_status()
            
            with open(cache_file, 'wb') as f:
                f.write(response.content)
            
            return cache_file
            
        except Exception as e:
            logger.error(f"Error downloading PDF: {e}")
            return None
    
    def _extract_text_from_pdf(self, pdf_path: str) -> Optional[Dict[str, str]]:
        """Extract text from PDF and organize it by sections."""
        try:
            doc = fitz.open(pdf_path)
            sections = {}
            current_section = "Abstract"
            current_text = []
            
            for page in doc:
                text = page.get_text()
                
                # Simple section detection based on common headers
                section_headers = [
                    "Abstract", "Introduction", "Related Work",
                    "Methodology", "Methods", "Experiments",
                    "Results", "Discussion", "Conclusion",
                    "References"
                ]
                
                for line in text.split('\n'):
                    # Check if the line can be a section header
                    clean_line = line.strip().lower()
                    for header in section_headers:
                        if header.lower() in clean_line and len(clean_line) < 50:
                            # Save the previous section
                            if current_text:
                                sections[current_section] = '\n'.join(current_text).strip()
                            current_section = header
                            current_text = []
                            break
                    else:
                        current_text.append(line)
            
            # Save the last section
            if current_text:
                sections[current_section] = '\n'.join(current_text).strip()
            
            return sections
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return None
    
    def _analyze_section(self, section_name: str, section_text: str) -> str:
        """Analyze a specific section of the paper using LLM."""
        prompts = {
            "Abstract": "Summarize the key points of this abstract clearly and concisely.",
            "Introduction": "Identify the main research questions and objectives presented in the introduction.",
            "Methodology": "Explain the key methodological approaches and techniques used in this study.",
            "Results": "What are the main findings and results presented in this section?",
            "Discussion": "What are the key implications and interpretations of the results?",
            "Conclusion": "Summarize the main conclusions and their significance."
        }
        
        prompt = prompts.get(section_name, f"Summarize the key points from the {section_name} section.")
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=f"Section: {section_text}\n\nTask: {prompt}\n\nAnalysis:"
            )
            return response['response'].strip()
        except Exception as e:
            logger.error(f"Error analyzing section: {e}")
            return f"Error analyzing the {section_name} section."
    
    def _extract_main_findings(self, sections: Dict[str, str]) -> List[str]:
        """Extract the main findings from the paper."""
        relevant_sections = [
            sections.get('Abstract', ''),
            sections.get('Results', ''),
            sections.get('Discussion', ''),
            sections.get('Conclusion', '')
        ]
        
        combined_text = '\n'.join(relevant_sections)
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=(
                    "Based on the following text, list the main findings and contributions "
                    f"of this research work:\n\n{combined_text}\n\nMain Findings:"
                )
            )
            
            findings = response['response'].strip().split('\n')
            return [f.strip('- ') for f in findings if f.strip()]
        except Exception as e:
            logger.error(f"Error extracting findings: {e}")
            return ["Error extracting main findings"]
    
    def analyze_paper(self, pdf_url: str, paper_metadata: Dict) -> Optional[PaperAnalysis]:
        """
        Analyze a scientific paper using its PDF URL.
        
        Arguments:
            pdf_url: URL of the paper's PDF file
            paper_metadata: Dictionary containing the paper's metadata
            
        Returns:
            PaperAnalysis object on success, None on failure
        """
        try:
            # Download PDF
            pdf_path = self._download_pdf(pdf_url)
            if not pdf_path:
                return None
            
            # Extract text and sections
            sections = self._extract_text_from_pdf(pdf_path)
            if not sections:
                return None
            
            # Analyze each section
            analyzed_sections = {}
            for section_name, section_text in sections.items():
                analyzed_sections[section_name] = self._analyze_section(section_name, section_text)
            
            # Extract main findings
            main_findings = self._extract_main_findings(sections)
            
            # Create a document for the vector store
            doc = Document(
                id=paper_metadata.get('arxiv_id', str(datetime.now().timestamp())),
                title=paper_metadata['title'],
                authors=paper_metadata['authors'],
                abstract=paper_metadata['abstract'],
                pdf_link=pdf_url,
                arxiv_link=paper_metadata.get('arxiv_link'),
                published=paper_metadata['published'],
                categories=paper_metadata.get('categories', ''),
                embedding=None  # To be generated by the vector store
            )
            
            # Add full text to the vector store
            full_text = '\n'.join(sections.values())
            doc_with_full_text = doc
            doc_with_full_text.abstract = full_text
            self.vector_store.add_documents([doc_with_full_text])
            
            # Create the analysis object
            analysis = PaperAnalysis(
                paper_id=doc.id,
                title=doc.title,
                sections=analyzed_sections,
                main_findings=main_findings,
                methodology=analyzed_sections.get('Methodology', 'Not found'),
                conclusions=analyzed_sections.get('Conclusion', 'Not found'),
                limitations=self._extract_limitations(sections),
                future_work=self._extract_future_work(sections),
                analysis_date=datetime.now()
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing paper: {e}")
            return None
    
    def _extract_limitations(self, sections: Dict[str, str]) -> Optional[str]:
        """Extract the limitations discussed in the paper."""
        relevant_sections = [
            sections.get('Discussion', ''),
            sections.get('Conclusion', '')
        ]
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=(
                    "What limitations and issues are discussed in this paper? "
                    f"Text: {' '.join(relevant_sections)}\n\nLimitations:"
                )
            )
            return response['response'].strip()
        except Exception as e:
            logger.error(f"Error extracting limitations: {e}")
            return None
    
    def _extract_future_work(self, sections: Dict[str, str]) -> Optional[str]:
        """Extract future work suggestions from the paper."""
        relevant_sections = [
            sections.get('Discussion', ''),
            sections.get('Conclusion', '')
        ]
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=(
                    "What future research directions are proposed in this paper? "
                    f"Text: {' '.join(relevant_sections)}\n\nFuture Work:"
                )
            )
            return response['response'].strip()
        except Exception as e:
            logger.error(f"Error extracting future work suggestions: {e}")
            return None
