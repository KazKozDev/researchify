import os
import re
import mimetypes
import chardet
import requests
import urllib.parse
import feedparser
import PyPDF2
import io
from flask import Flask, request, render_template, session, redirect, url_for, jsonify, send_from_directory
from flask_session import Session
import ollama
import asyncio
from typing import Dict, List, Union, Tuple
from vector_store import VectorStore, RAGSystem, Document
from document_processor import DocumentProcessor
from research_bot import ResearchBot
from paper_analyzer import PaperAnalyzer, PaperAnalysis
from scientific_impact_analysis import CitationsAnalyzer
import fitz  # PyMuPDF
import tempfile
from datetime import datetime
import logging
from werkzeug.utils import secure_filename

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['PAPER_ANALYSIS_CACHE'] = {}
app.config['MAX_CACHE_SIZE'] = 100  # Максимальное количество кэшированных анализов

# Создаем папки, если их нет
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('paper_cache', exist_ok=True)
os.makedirs('vector_store', exist_ok=True)

Session(app)

# MIME types mapping
MIME_TYPES = {
    'txt': 'text/plain',
    'pdf': 'application/pdf',
    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
    'doc': 'application/msword',
    'xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
    'xls': 'application/vnd.ms-excel',
    'csv': 'text/csv',
    'md': 'text/markdown',
    'rst': 'text/x-rst'
}

# Allowed extensions
ALLOWED_EXTENSIONS = set(MIME_TYPES.keys())

def get_mime_type(filename):
    """Определяет MIME-тип файла по расширению."""
    ext = filename.rsplit('.', 1)[1].lower() if '.' in filename else ''
    return MIME_TYPES.get(ext, 'application/octet-stream')

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Инициализация компонентов
vector_store = VectorStore()
rag_system = RAGSystem(vector_store)
research_bot = ResearchBot(vector_store, rag_system)
document_processor = DocumentProcessor(
    vector_store=vector_store,
    rag_system=rag_system,
    upload_folder=app.config['UPLOAD_FOLDER']
)
paper_analyzer = PaperAnalyzer(
    vector_store=vector_store,
    rag_system=rag_system,
    cache_dir='paper_cache'
)

# Initialize CitationsAnalyzer
citations_analyzer = CitationsAnalyzer()

# Определяем системный промпт
SYSTEM_PROMPT = """
You are Researchify, an AI research assistant created by Artem Kazakov Kozlov, designed to help users find and understand scientific papers across all academic fields while maintaining clear communication and academic integrity. When searching for papers, identify key terms, use academic databases, and present results with title, authors, date, and a brief summary explaining relevance to the query. Present all information in order of relevance, using clear language and complete citations, while offering to refine searches if needed. Maintain a professional tone, respect user privacy, and acknowledge any limitations in your knowledge or uncertainty in research findings.

Inform the user when he asks about search that there is a search function. 'Enter 'find [keyword]' or 'search [keyword]' in the chat window (for example, 'find LLM' or 'search quantum computing') to activate the search function. I will search archive.org and provide relevant information with sources and dates.'

Remember, your purpose is to make scientific research more accessible and efficient for the user. Always strive to be helpful, informative, and supportive in your interactions.

You can now also analyze uploaded documents and provide insights based on their content.
"""

def generate_response(user_input):
    """Generates a conversational response using the Ollama model."""
    chat_history = session.get('chat_history', [])
    conversation = SYSTEM_PROMPT + "\n\n"
    for message in chat_history:
        if 'user' in message:
            conversation += f"User: {message['user']}\n"
        elif 'bot' in message:
            conversation += f"Researchify: {message['bot']}\n"
    conversation += f"User: {user_input}\nResearchify:"

    try:
        response = ollama.generate(model='gemma2', prompt=conversation)
        assistant_message = response['response'].strip()
        return assistant_message
    except Exception as e:
        logger.error(f"Error generating response with Ollama: {str(e)}")
        return "Извините, я не могу сейчас ответить на ваш вопрос."

def generate_response_with_context(user_input: str, system_prompt: str) -> str:
    """Generates a response considering the context of uploaded files."""
    chat_history = session.get('chat_history', [])
    conversation = system_prompt + "\n\n"
    for message in chat_history:
        if 'user' in message:
            conversation += f"User: {message['user']}\n"
        elif 'bot' in message:
            conversation += f"Assistant: {message['bot']}\n"
    conversation += f"User: {user_input}\nAssistant:"

    try:
        response = ollama.generate(model='gemma2', prompt=conversation)
        return response['response'].strip()
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return "I apologize, but I cannot answer your question right now."

def process_user_input(user_input):
    """Processes user input, distinguishes between research intent, search intent, and general conversation."""
    # Если это исследовательский запрос
    if any(cmd in user_input.lower() for cmd in [
        'find papers', 'summarize papers', 'compare papers',
        'latest research', 'explain', 'research', 'study',
        'найти статьи', 'обобщить статьи', 'сравнить статьи',
        'последние исследования', 'объяснить'
    ]):
        return {'type': 'research', 'query': user_input}
    # Если это поисковый запрос
    elif any(word in user_input.lower() for word in ['search', 'find', 'look for', 'поиск', 'найди', 'искать']):
        search_query = user_input.lower()
        for word in ['search', 'find', 'look for', 'поиск', 'найди', 'искать']:
            search_query = search_query.replace(word, '')
        search_query = search_query.strip()
        final_query = generate_arxiv_query(search_query)
        return {'type': 'search', 'query': final_query}
    else:
        return {'type': 'chat', 'message': user_input}

def generate_arxiv_query(input_text):
    """Generates an arXiv API search query based on user text."""
    input_text = input_text.strip()
    acronym_expansions = {
        "LLM": "large language model",
    }

    if input_text.upper() in acronym_expansions:
        expansion = acronym_expansions[input_text.upper()]
        return f'abs:"{expansion}" OR ti:"{expansion}"'

    prompt = SYSTEM_PROMPT + f"""
Generate an arXiv search query based on the following description: "{input_text}"

Use standard search format with prefixes:
- ti: for title search
- abs: for abstract search
- au: for author search
- cat: for category search

Example query: abs:"deep learning" AND cat:cs.LG

Query:
"""
    try:
        response = ollama.generate(model='gemma2', prompt=prompt)
        query = response['response'].strip()
        logger.info(f"Generated arXiv query: {query}")
        return query
    except Exception as e:
        logger.error(f"Error generating arXiv query with Ollama: {str(e)}")
        return f'all:"{input_text}"'

def get_arxiv_results(query, max_results=5):
    """Fetches search results from arXiv API and stores them in the RAG system."""
    base_url = 'http://export.arxiv.org/api/query'
    params = {
        'search_query': query,
        'start': 0,
        'max_results': max_results,
    }

    try:
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        feed = feedparser.parse(response.content)

        if len(feed.entries) == 0:
            return []

        results = []
        for entry in feed.entries:
            try:
                title = entry.title.replace('\n', ' ').strip() if hasattr(entry, 'title') else None
                if not title:
                    continue

                authors = []
                if hasattr(entry, 'authors'):
                    authors = [author.get('name', '') for author in entry.authors]
                elif hasattr(entry, 'author'):
                    authors = [entry.author]
                authors_str = ', '.join(filter(None, authors))

                abstract = entry.summary if hasattr(entry, 'summary') else ''
                abstract = abstract.replace('\n', ' ').strip()

                pdf_link = None
                arxiv_link = None
                for link in entry.get('links', []):
                    if link.get('type', '') == 'application/pdf':
                        pdf_link = link.get('href')
                    elif link.get('rel') == 'alternate':
                        arxiv_link = link.get('href')

                published = entry.published if hasattr(entry, 'published') else entry.get('updated', '')
                
                categories = []
                if hasattr(entry, 'tags'):
                    categories = [tag.get('term', '') for tag in entry.tags]
                category_str = ', '.join(filter(None, categories))

                # Extract arxiv_id from the links
                arxiv_id = None
                if arxiv_link:
                    match = re.search(r'abs/([^/]+)$', arxiv_link)
                    if match:
                        arxiv_id = match.group(1)

                paper_data = {
                    'title': title,
                    'authors': authors_str,
                    'abstract': abstract,
                    'pdf_link': pdf_link,
                    'arxiv_link': arxiv_link,
                    'arxiv_id': arxiv_id,  # Add arxiv_id to the paper data
                    'published': published,
                    'categories': category_str,
                    'comments': entry.get('arxiv_comment', ''),
                    'journal_ref': entry.get('arxiv_journal_ref', ''),
                    'primary_category': entry.get('arxiv_primary_category', {}).get('term', '')
                }

                results.append(paper_data)
                
                # Сохраняем статью в RAG системе
                try:
                    rag_system.add_paper(paper_data)
                except Exception as e:
                    logger.error(f"Error adding paper to RAG system: {str(e)}")

            except Exception as e:
                logger.error(f"Warning: Error parsing paper: {str(e)}")
                continue

        return results

    except Exception as e:
        logger.error(f"Error fetching results: {str(e)}")
        return []

def print_search_tips():
    """Returns helpful tips for using the search."""
    tips = """
    <p><strong>Search Tips:</strong></p>
    <ol>
        <li>For author searches, use the format: au:"LastName, FirstName" or au:"LastName, F."</li>
        <li>For exact phrases, use quotes: "quantum computing"</li>
        <li>Use wildcards (* or ?), but not at the beginning of a word: comput*</li>
        <li>Use categories, for example: cat:cs.AI for Artificial Intelligence</li>
        <li>Use AND/OR/NOT: quantum AND computing</li>
        <li>Find papers about specific topics: "find papers about quantum computing"</li>
        <li>Get paper summaries: "summarize papers on machine learning"</li>
        <li>Compare research: "compare papers on neural networks"</li>
        <li>Check latest trends: "latest research in NLP"</li>
    </ol>
    """
    return tips

def format_analysis_for_chat(analysis: Dict, paper_data: Dict) -> Dict:
    """Format paper analysis for chat display."""
    formatted_html = f"""
    <div class="analysis-section bg-gray-50 p-4 rounded-lg">
        <h3 class="text-xl font-semibold mb-4">Analysis: {paper_data['title']}</h3>
        
        {format_section('Main Findings', analysis['main_findings'], is_list=True) if analysis['main_findings'] else ''}
        {format_section('Methodology', analysis['methodology']) if analysis['methodology'] else ''}
        {format_section('Conclusions', analysis['conclusions']) if analysis['conclusions'] else ''}
        {format_section('Limitations', analysis['limitations']) if analysis['limitations'] else ''}
        {format_section('Future Work', analysis['future_work']) if analysis['future_work'] else ''}
        
        <div class="mt-4 text-sm text-gray-500">
            <p>Authors: {paper_data['authors']}</p>
            <p>Published: {paper_data['published']}</p>
            <p>Categories: {paper_data['categories']}</p>
        </div>
        
        <div class="mt-4">
            <a href="{paper_data['pdf_link']}" target="_blank" 
               class="text-blue-600 hover:underline mr-4">View PDF</a>
            <a href="{paper_data['arxiv_link']}" target="_blank" 
               class="text-blue-600 hover:underline">View on arXiv</a>
        </div>
    </div>
    """
    
    return {
        'type': 'analysis',
        'html': formatted_html,
        'raw': analysis
    }

def format_section(title: str, content: Union[str, List[str]], is_list: bool = False) -> str:
    """Format a section of the analysis."""
    if not content:
        return ''
    
    if is_list and isinstance(content, list):
        items = '\n'.join([f'<li class="ml-4">{item}</li>' for item in content])
        return f"""
        <div class="mb-4">
            <h4 class="text-lg font-medium mb-2">{title}</h4>
            <ul class="list-disc">
                {items}
            </ul>
        </div>
        """
    else:
        return f"""
        <div class="mb-4">
            <h4 class="text-lg font-medium mb-2">{title}</h4>
            <p class="text-gray-700">{content}</p>
        </div>
        """

def format_search_results(results, query):
    """Formats search results into HTML with proper meta data."""
    if not results:

        return "<p>No results found for your query.</p>"
        
    html = f"<p><strong>Using query:</strong> {query}</p><h2>Search Results:</h2>"
    
    for paper in results:
        html += f"""
        <div class="paper mb-4 p-4 border rounded-lg">
            <h3 class="text-lg font-medium">{paper['title']}</h3>
            <p class="text-gray-600 mb-2">{paper['authors']}</p>
            <div class="paper-meta">
                <p class="published"><strong>Published:</strong> {paper['published']}</p>
                <p class="categories"><strong>Categories:</strong> {paper['categories']}</p>
                <p class="abstract"><strong>Abstract:</strong> {paper['abstract']}</p>
                <div class="links">
                    <a href="{paper['pdf_link']}" target="_blank" class="text-blue-600 hover:underline">PDF</a>
                    <a href="{paper['arxiv_link']}" target="_blank" class="text-blue-600 hover:underline ml-4">arXiv</a>
                </div>
            </div>
        </div>
        """
        
    return html

@app.route('/upload', methods=['POST'])
def upload_file():
    """Обработка загрузки файла."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': f'File type not allowed. Supported types: {", ".join(ALLOWED_EXTENSIONS)}'}), 400
    
    try:
        # Обрабатываем файл через DocumentProcessor
        result = document_processor.process_file(file)
        
        # Добавляем информацию о файле в сессию
        if 'uploaded_files' not in session:
            session['uploaded_files'] = []
        session['uploaded_files'].append({
            'filename': result['filename'],
            'format': result['format'],
            'metadata': result['metadata']
        })
        
        return jsonify(result)
    
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Отдача загруженных файлов."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/analyze_paper', methods=['POST'])
def analyze_paper():
    """Handle paper analysis requests and format results for chat display."""
    try:
        paper_data = request.get_json()
        
        if not paper_data or 'pdf_link' not in paper_data:
            logger.error("No PDF link provided in request")
            return jsonify({
                'error': 'No PDF link provided'
            }), 400
        
        # Check cache
        cache_key = f"paper_analysis_{hash(paper_data['pdf_link'])}"
        cached_analysis = app.config.get('PAPER_ANALYSIS_CACHE', {}).get(cache_key)
        
        if cached_analysis:
            logger.info(f"Returning cached analysis for {paper_data['title']}")
            return jsonify(format_analysis_for_chat(cached_analysis, paper_data))
        
        logger.info(f"Starting analysis for paper: {paper_data['title']}")
        
        # Perform analysis
        analysis = paper_analyzer.analyze_paper(
            pdf_url=paper_data['pdf_link'],
            paper_metadata=paper_data
        )
        
        if not analysis:
            logger.error(f"Analysis failed for paper: {paper_data['title']}")
            return jsonify({
                'error': 'Failed to analyze paper'
            }), 500
        
        # Cache the result
        if 'PAPER_ANALYSIS_CACHE' not in app.config:
            app.config['PAPER_ANALYSIS_CACHE'] = {}
            
        analysis_dict = analysis.to_dict()
        app.config['PAPER_ANALYSIS_CACHE'][cache_key] = analysis_dict
        
        # Clean old cache entries if there are too many
        if len(app.config['PAPER_ANALYSIS_CACHE']) > app.config.get('MAX_CACHE_SIZE', 100):
            oldest_key = min(app.config['PAPER_ANALYSIS_CACHE'].keys(), 
                           key=lambda k: app.config['PAPER_ANALYSIS_CACHE'][k].get('analysis_date', ''))
            app.config['PAPER_ANALYSIS_CACHE'].pop(oldest_key)
            
        logger.info(f"Analysis completed for paper: {paper_data['title']}")
        
        return jsonify(format_analysis_for_chat(analysis_dict, paper_data))
    
    except Exception as e:
        logger.error(f"Error in paper analysis: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/scientific_impact', methods=['POST'])
def analyze_scientific_impact():
    """Handle scientific impact analysis requests."""
    try:
        paper_data = request.get_json()
        
        if not paper_data:
            return jsonify({
                'error': 'No paper data provided'
            }), 400
        
        # Get paper arXiv ID or title
        arxiv_id = paper_data.get('arxiv_id')
        title = paper_data.get('title')
        
        if not (arxiv_id or title):
            return jsonify({
                'error': 'No arXiv ID or title provided'
            }), 400
        
        # Get paper info from arXiv if needed
        arxiv_info = {}
        if arxiv_id:
            arxiv_info = citations_analyzer.get_arxiv_info(arxiv_id)
            if not arxiv_info:
                return jsonify({
                    'error': f'Paper with ID {arxiv_id} not found in arXiv'
                }), 404
            
        # Get OpenAlex data
        if arxiv_info.get('doi'):
            openalex_data = citations_analyzer.get_openalex_data(arxiv_info['doi'], is_doi=True)
        else:
            cleaned_id = citations_analyzer.clean_arxiv_id(arxiv_id) if arxiv_id else None
            openalex_data = citations_analyzer.get_openalex_data(
                cleaned_id,
                is_doi=False,
                title=arxiv_info.get('title', title)
            )
            
        if not openalex_data:
            return jsonify({
                'error': 'Failed to retrieve citation data'
            }), 500
        
        # Get citation details and analyze
        total_citations = openalex_data.get('cited_by_count', 0)
        citations = citations_analyzer.get_citing_works(openalex_data['id'])
        
        if not citations:
            return jsonify({
                'response': f"""
                <div class="mb-4">
                    <h4 class="text-lg font-medium mb-2">Citation Analysis</h4>
                    <p>Total citations: {total_citations}</p>
                    <p>No detailed citation data available for analysis.</p>
                </div>
                """
            })
        
        # Analyze citations
        analysis = citations_analyzer.analyze_citations(citations)
        metrics = citations_analyzer.calculate_metrics(analysis)
        analysis['metrics'] = metrics
        
        # Get AI analysis
        ai_analysis = citations_analyzer.analyze_with_gemma(analysis, total_citations, paper_data['title'])
        
        # Format response
        response = f"""
        <div class="space-y-6">
            <div class="mb-4">
                <h4 class="text-lg font-medium mb-2">Citation Overview</h4>
                <p>Total citations: {total_citations}</p>
                <p>Analyzed citations: {len(citations)}</p>
            </div>

            <div class="mb-4">
                <h4 class="text-lg font-medium mb-2">Citation Timeline</h4>
                <div class="space-y-1">
                    {format_timeline(analysis['by_year'], metrics.get('yoy_growth', {}))}
                </div>
            </div>

            <div class="mb-4">
                <h4 class="text-lg font-medium mb-2">Geographic Distribution</h4>
                <p>Total contributing countries: {len(analysis['by_country'])}</p>
                {format_geographic_distribution(analysis['by_country'])}
                <p class="mt-2">Geographic concentration (top 3 countries): {metrics.get('geographic_concentration', 0):.1f}%</p>
            </div>

            <div class="mb-4">
                <h4 class="text-lg font-medium mb-2">Publication Types</h4>
                {format_publication_types(analysis['by_type'], analysis['total_count'])}
            </div>

            <div class="mb-4">
                <h4 class="text-lg font-medium mb-2">Research Venues</h4>
                {format_venues(analysis['by_journal'], analysis['total_count'])}
            </div>

            <div class="mb-4">
                <h4 class="text-lg font-medium mb-2">AI Analysis</h4>
                <div class="whitespace-pre-wrap">{ai_analysis}</div>
            </div>
        </div>
        """
        
        return jsonify({'response': response})
    
    except Exception as e:
        logger.error(f"Error in scientific impact analysis: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

def format_timeline(by_year: Dict, yoy_growth: Dict) -> str:
    """Format citation timeline."""
    timeline = []
    for year, count in sorted(by_year.items()):
        growth = yoy_growth.get(year)
        growth_text = f" | Growth: {growth:+.1f}%" if growth is not None else ""
        timeline.append(f"<p>{year}: {count} citations{growth_text}</p>")
    return "\n".join(timeline)

def format_geographic_distribution(by_country: Dict) -> str:
    """Format geographic distribution of citations."""
    total = sum(by_country.values())
    countries = sorted(by_country.items(), key=lambda x: x[1], reverse=True)[:10]
    
    distribution = []
    for country, count in countries:
        percentage = (count / total) * 100
        distribution.append(f"<p>{country}: {count} ({percentage:.1f}%)</p>")
    return "\n".join(distribution)

def format_publication_types(by_type: Dict, total: int) -> str:
    """Format publication type distribution."""
    types = []
    for pub_type, count in sorted(by_type.items()):
        percentage = (count / total) * 100
        types.append(f"<p>{pub_type}: {count} ({percentage:.1f}%)</p>")
    return "\n".join(types)

def format_venues(by_journal: Dict, total: int) -> str:
    """Format publication venues distribution."""
    venues = sorted(by_journal.items(), key=lambda x: x[1], reverse=True)[:10]
    formatted = []
    for journal, count in venues:
        percentage = (count / total) * 100
        formatted.append(f"<p>{journal}: {count} ({percentage:.1f}%)</p>")
    return "\n".join(formatted)

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handle main page requests."""
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'awaiting_confirmation' not in session:
        session['awaiting_confirmation'] = False
    if 'uploaded_files' not in session:
        session['uploaded_files'] = []
        
    if request.method == 'POST':
        if request.is_json:
            data = request.get_json()
            user_input = data.get('user_input', '')
            num_results = data.get('num_results', 5)
        else:
            user_input = request.form.get('user_input', '')
            num_results = request.form.get('num_results', 5)
            
        try:
            num_results = int(num_results)
            if num_results < 1 or num_results > 50:
                num_results = 5
        except ValueError:
            num_results = 5
            
        # Add incoming message to history
        session['chat_history'].append({'user': user_input})
        
        # Check if request is related to uploaded files
        if any(keyword in user_input.lower() for keyword in ['analyze file', 'analyze document', 'analyze uploaded']):
            if session['uploaded_files']:
                latest_file = session['uploaded_files'][-1]
                response = f"Analyzing {latest_file['filename']}...\n\n"
                
                try:
                    analysis = document_processor.analyze_content(latest_file['filename'])
                    response += analysis
                except Exception as e:
                    response += f"Error analyzing file: {str(e)}"
            else:
                response = "No files have been uploaded yet. Please upload a file first."
                
            session['chat_history'].append({'bot': response})
            return jsonify({'response': response}) if request.is_json else redirect(url_for('index'))
        
        # Handle normal requests
        if session['awaiting_confirmation']:
            if user_input.lower() in ['yes', 'y', 'да']:
                final_query = session['pending_query']
                results = get_arxiv_results(final_query, max_results=num_results)
                response = format_search_results(results, final_query)
                session['chat_history'].append({'bot': response})
                session['awaiting_confirmation'] = False
                session.pop('pending_query', None)
                session.pop('pending_num_results', None)
            else:
                response = "Please enter your modified query:"
                session['chat_history'].append({'bot': response})
                session['awaiting_modification'] = True
                session['awaiting_confirmation'] = False
        elif session.get('awaiting_modification', False):
            final_query = user_input.strip()
            results = get_arxiv_results(final_query, max_results=num_results)
            response = format_search_results(results, final_query)
            session['chat_history'].append({'bot': response})
            session['awaiting_modification'] = False
        else:
            if user_input.lower() in ['help', 'помощь']:
                tips = print_search_tips()
                session['chat_history'].append({'bot': tips})
            else:
                result = process_user_input(user_input)
                if result['type'] == 'search':
                    final_query = result['query']
                    response = f"<p><strong>Generated query:</strong> {final_query}</p>"
                    response += "<p>Type 'no' to open a field for entering a query (yes/no)</p>"
                    session['chat_history'].append({'bot': response})
                    session['awaiting_confirmation'] = True
                    session['pending_query'] = final_query
                    session['pending_num_results'] = num_results
                elif result['type'] == 'research':
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        research_response = loop.run_until_complete(research_bot.process_query(result['query']))
                        loop.close()

                        # Если ответ содержит запрос на подтверждение
                        if "Would you like me to proceed with this search?" in research_response:
                            session['chat_history'].append({'bot': research_response})
                            # Сохраняем состояние ожидания подтверждения для research_bot
                            session['awaiting_research_confirmation'] = True
                        elif "please enter your search query" in research_response.lower():
                            session['chat_history'].append({'bot': research_response})
                            session['awaiting_research_query'] = True
                        else:
                            # Обычный ответ с результатами
                            session['chat_history'].append({'bot': research_response})
                            
                    except Exception as e:
                        logger.error(f"Error processing research query: {str(e)}")
                        response = "I apologize, but I encountered an error while processing your research query."
                        session['chat_history'].append({'bot': response})
                else:
                    if session['uploaded_files']:
                        modified_prompt = SYSTEM_PROMPT + "\n\nCurrently uploaded files:\n"
                        for file in session['uploaded_files']:
                            modified_prompt += f"- {file['filename']} ({file['format']})\n"
                        response = generate_response_with_context(user_input, modified_prompt)
                    else:
                        response = generate_response(user_input)
                    session['chat_history'].append({'bot': response})
                    
        if request.is_json:
            return jsonify({'response': response})
        return redirect(url_for('index'))
    
    return render_template('chat.html', chat_history=session.get('chat_history', []))

@app.route('/clear', methods=['GET'])
def clear_chat():
    """Clear chat history and session data."""
    session.pop('chat_history', None)
    session.pop('awaiting_confirmation', None)
    session.pop('awaiting_modification', None)
    session.pop('pending_query', None)
    session.pop('pending_num_results', None)
    session.pop('uploaded_files', None)
    return jsonify({'status': 'success'})

@app.route('/save_papers', methods=['POST'])
def save_papers():
    """Save papers to the RAG system manually."""
    try:
        data = request.get_json()
        papers = data.get('papers', [])
        
        for paper in papers:
            rag_system.add_paper(paper)
            
        return jsonify({
            'status': 'success',
            'message': f'Successfully saved {len(papers)} papers to the RAG system'
        })
    except Exception as e:
        logger.error(f"Error saving papers: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/query_papers', methods=['POST'])
def query_papers():
    """Query papers from the RAG system directly."""
    try:
        data = request.get_json()
        query = data.get('query', '')
        max_results = data.get('max_results', 5)
        category_filter = data.get('category', None)
        
        date_filter = None
        start_date = data.get('start_date')
        end_date = data.get('end_date')
        if start_date and end_date:
            date_filter = (
                datetime.strptime(start_date, '%Y-%m-%d'),
                datetime.strptime(end_date, '%Y-%m-%d')
            )
            
        results = rag_system.query(
            query=query,
            k=max_results,
            category_filter=category_filter,
            date_filter=date_filter
        )
        
        return jsonify({
            'status': 'success',
            'results': results
        })
    except Exception as e:
        logger.error(f"Error querying papers: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/research_chat', methods=['POST'])
def research_chat():
    """Direct interface to the research bot."""
    try:
        data = request.get_json()
        query = data.get('query', '')
        
        if not query:
            return jsonify({
                'status': 'error',
                'message': 'Query is required'
            }), 400
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response = loop.run_until_complete(research_bot.process_query(query))
        loop.close()
        
        return jsonify({
            'status': 'success',
            'response': response
        })
    except Exception as e:
        logger.error(f"Error in research chat: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors."""
    return jsonify({
        'status': 'error',
        'message': 'Resource not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({
        'status': 'error',
        'message': 'Internal server error'
    }), 500

if __name__ == "__main__":
    try:
        app.run(debug=True)
    except Exception as e:
        logger.error(f"Error starting the application: {str(e)}")
