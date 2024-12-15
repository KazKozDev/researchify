import os
import fitz  # PyMuPDF
import requests
import tempfile
import ollama  # Добавлен импорт ollama
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
    """Класс данных для хранения результатов анализа статьи."""
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
        model_name: str = "gemma2",
        cache_dir: str = "paper_cache"
    ):
        """
        Инициализация анализатора статей.
        
        Аргументы:
            vector_store: Инициализированный экземпляр VectorStore
            rag_system: Инициализированный экземпляр RAGSystem
            model_name: Название используемой LLM модели
            cache_dir: Директория для кэширования загруженных статей
        """
        self.vector_store = vector_store
        self.rag_system = rag_system
        self.model_name = model_name
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def _download_pdf(self, url: str) -> Optional[str]:
        """Загрузка PDF по URL и возврат пути к временному файлу."""
        try:
            # Создаем хэш URL для кэширования
            cache_file = os.path.join(self.cache_dir, f"{hash(url)}.pdf")
            
            # Сначала проверяем кэш
            if os.path.exists(cache_file):
                return cache_file
            
            # Загружаем, если нет в кэше
            response = requests.get(url)
            response.raise_for_status()
            
            with open(cache_file, 'wb') as f:
                f.write(response.content)
            
            return cache_file
            
        except Exception as e:
            logger.error(f"Ошибка загрузки PDF: {e}")
            return None
    
    def _extract_text_from_pdf(self, pdf_path: str) -> Optional[Dict[str, str]]:
        """Извлечение текста из PDF и организация по разделам."""
        try:
            doc = fitz.open(pdf_path)
            sections = {}
            current_section = "Abstract"
            current_text = []
            
            for page in doc:
                text = page.get_text()
                
                # Простое определение разделов на основе общих заголовков
                section_headers = [
                    "Abstract", "Introduction", "Related Work",
                    "Methodology", "Methods", "Experiments",
                    "Results", "Discussion", "Conclusion",
                    "References"
                ]
                
                for line in text.split('\n'):
                    # Проверяем, может ли строка быть заголовком раздела
                    clean_line = line.strip().lower()
                    for header in section_headers:
                        if header.lower() in clean_line and len(clean_line) < 50:
                            # Сохраняем предыдущий раздел
                            if current_text:
                                sections[current_section] = '\n'.join(current_text).strip()
                            current_section = header
                            current_text = []
                            break
                    else:
                        current_text.append(line)
            
            # Сохраняем последний раздел
            if current_text:
                sections[current_section] = '\n'.join(current_text).strip()
            
            return sections
            
        except Exception as e:
            logger.error(f"Ошибка извлечения текста из PDF: {e}")
            return None
    
    def _analyze_section(self, section_name: str, section_text: str) -> str:
        """Анализ определенного раздела статьи с использованием LLM."""
        prompts = {
            "Abstract": "Обобщите ключевые моменты этой аннотации ясно и кратко.",
            "Introduction": "Определите основные исследовательские вопросы и цели, представленные во введении.",
            "Methodology": "Объясните ключевые методологические подходы и техники, использованные в этом исследовании.",
            "Results": "Каковы основные выводы и результаты, представленные в этом разделе?",
            "Discussion": "Каковы ключевые последствия и интерпретации результатов?",
            "Conclusion": "Обобщите основные выводы и их значимость."
        }
        
        prompt = prompts.get(section_name, f"Обобщите ключевые моменты из раздела {section_name}.")
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=f"Section: {section_text}\n\nTask: {prompt}\n\nAnalysis:"
            )
            return response['response'].strip()
        except Exception as e:
            logger.error(f"Ошибка анализа раздела: {e}")
            return f"Ошибка анализа раздела {section_name}."
    
    def _extract_main_findings(self, sections: Dict[str, str]) -> List[str]:
        """Извлечение основных результатов из статьи."""
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
                    "На основе следующего текста перечислите основные результаты и вклад "
                    f"этой исследовательской работы:\n\n{combined_text}\n\nОсновные результаты:"
                )
            )
            
            findings = response['response'].strip().split('\n')
            return [f.strip('- ') for f in findings if f.strip()]
        except Exception as e:
            logger.error(f"Ошибка извлечения результатов: {e}")
            return ["Ошибка извлечения основных результатов"]
    
    def analyze_paper(self, pdf_url: str, paper_metadata: Dict) -> Optional[PaperAnalysis]:
        """
        Анализ научной статьи по её PDF URL.
        
        Аргументы:
            pdf_url: URL PDF-файла статьи
            paper_metadata: Словарь с метаданными статьи
            
        Возвращает:
            Объект PaperAnalysis в случае успеха, None в случае ошибки
        """
        try:
            # Загрузка PDF
            pdf_path = self._download_pdf(pdf_url)
            if not pdf_path:
                return None
            
            # Извлечение текста и разделов
            sections = self._extract_text_from_pdf(pdf_path)
            if not sections:
                return None
            
            # Анализ каждого раздела
            analyzed_sections = {}
            for section_name, section_text in sections.items():
                analyzed_sections[section_name] = self._analyze_section(section_name, section_text)
            
            # Извлечение основных результатов
            main_findings = self._extract_main_findings(sections)
            
            # Создание документа для векторного хранилища
            doc = Document(
                id=paper_metadata.get('arxiv_id', str(datetime.now().timestamp())),
                title=paper_metadata['title'],
                authors=paper_metadata['authors'],
                abstract=paper_metadata['abstract'],
                pdf_link=pdf_url,
                arxiv_link=paper_metadata.get('arxiv_link'),
                published=paper_metadata['published'],
                categories=paper_metadata.get('categories', ''),
                embedding=None  # Будет сгенерирован векторным хранилищем
            )
            
            # Добавление полного текста в векторное хранилище
            full_text = '\n'.join(sections.values())
            doc_with_full_text = doc
            doc_with_full_text.abstract = full_text
            self.vector_store.add_documents([doc_with_full_text])
            
            # Создание объекта анализа
            analysis = PaperAnalysis(
                paper_id=doc.id,
                title=doc.title,
                sections=analyzed_sections,
                main_findings=main_findings,
                methodology=analyzed_sections.get('Methodology', 'Не найдено'),
                conclusions=analyzed_sections.get('Conclusion', 'Не найдено'),
                limitations=self._extract_limitations(sections),
                future_work=self._extract_future_work(sections),
                analysis_date=datetime.now()
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Ошибка анализа статьи: {e}")
            return None
    
    def _extract_limitations(self, sections: Dict[str, str]) -> Optional[str]:
        """Извлечение ограничений, обсуждаемых в статье."""
        relevant_sections = [
            sections.get('Discussion', ''),
            sections.get('Conclusion', '')
        ]
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=(
                    "Какие ограничения и проблемы обсуждаются в этой статье? "
                    f"Текст: {' '.join(relevant_sections)}\n\nОграничения:"
                )
            )
            return response['response'].strip()
        except Exception as e:
            logger.error(f"Ошибка извлечения ограничений: {e}")
            return None
    
    def _extract_future_work(self, sections: Dict[str, str]) -> Optional[str]:
        """Извлечение предложений по будущей работе из статьи."""
        relevant_sections = [
            sections.get('Discussion', ''),
            sections.get('Conclusion', '')
        ]
        
        try:
            response = ollama.generate(
                model=self.model_name,
                prompt=(
                    "Какие направления будущих исследований предлагаются в этой статье? "
                    f"Текст: {' '.join(relevant_sections)}\n\nБудущая работа:"
                )
            )
            return response['response'].strip()
        except Exception as e:
            logger.error(f"Ошибка извлечения предложений по будущей работе: {e}")
            return None