import tiktoken
import openai
import logging
import os
from datetime import datetime
import time
import json
import PyPDF2
import copy
import asyncio
import pymupdf
from io import BytesIO
from dotenv import load_dotenv
load_dotenv()
import logging
import yaml
from pathlib import Path
from types import SimpleNamespace as config

CHATGPT_API_KEY = os.getenv("CHATGPT_API_KEY")

def count_tokens(text, model=None):
    if not text:
        return 0
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)
    return len(tokens)

def ChatGPT_API_with_finish_reason(model, prompt, api_key=CHATGPT_API_KEY, chat_history=None):
    max_retries = 10
    client = openai.OpenAI(api_key=api_key)

    # Add system message to enforce JSON output
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant that ALWAYS responds in valid JSON format. "
                   "Output ONLY the JSON object with no additional text or markdown."
    }

    for i in range(max_retries):
        try:
            if chat_history:
                messages = [system_message] + chat_history
                messages.append({"role": "user", "content": prompt})
            else:
                messages = [system_message, {"role": "user", "content": prompt}]

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )
            if response.choices[0].finish_reason == "length":
                return response.choices[0].message.content, "max_output_reached"
            else:
                return response.choices[0].message.content, "finished"

        except Exception as e:
            error_str = str(e).lower()
            if '429' in str(e) or 'rate' in error_str or 'too many' in error_str:
                wait_time = min(5 * (2 ** i), 60)  # Start at 5s, max 60s
                print(f'************* Rate limited, waiting {wait_time}s ({i+1}/{max_retries}) *************')
                time.sleep(wait_time)
            else:
                print(f'************* Retrying ({i+1}/{max_retries}) *************')
                logging.error(f"Error: {e}")
                time.sleep(2)

            if i >= max_retries - 1:
                logging.error('Max retries reached for prompt: ' + prompt[:200])
                return "", "error"  # Return tuple with empty string and error status



def ChatGPT_API(model, prompt, api_key=CHATGPT_API_KEY, chat_history=None):
    max_retries = 10
    client = openai.OpenAI(api_key=api_key)
    logging.info(f"ChatGPT_API: Calling model={model}, prompt_length={len(prompt)}")

    # Add system message to enforce JSON output for non-OpenAI models
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant that ALWAYS responds in valid JSON format. "
                   "When asked to return JSON, output ONLY the JSON object with no additional text, "
                   "no markdown formatting, no explanations before or after. "
                   "Start your response with { and end with }."
    }

    for i in range(max_retries):
        try:
            if chat_history:
                messages = [system_message] + chat_history
                messages.append({"role": "user", "content": prompt})
            else:
                messages = [system_message, {"role": "user", "content": prompt}]

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
            )

            result = response.choices[0].message.content
            logging.info(f"ChatGPT_API: Response received, length={len(result) if result else 0}")
            logging.info(f"ChatGPT_API: Response preview: {result[:300] if result else 'None'}...")
            return result
        except Exception as e:
            error_str = str(e).lower()
            # Handle rate limiting with exponential backoff
            if '429' in str(e) or 'rate' in error_str or 'too many' in error_str:
                wait_time = min(2 ** i, 30)  # Exponential backoff, max 30 seconds
                print(f'************* Rate limited, waiting {wait_time}s ({i+1}/{max_retries}) *************')
                time.sleep(wait_time)
            else:
                print(f'************* Retrying ({i+1}/{max_retries}) *************')
                logging.error(f"ChatGPT_API Error: {e}")
                time.sleep(2)

            if i >= max_retries - 1:
                logging.error('Max retries reached for prompt: ' + prompt[:200])
                return ""  # Return empty string instead of "Error" to trigger fallback
            

async def ChatGPT_API_async(model, prompt, api_key=CHATGPT_API_KEY):
    max_retries = 10

    # Add system message to enforce JSON output
    system_message = {
        "role": "system",
        "content": "You are a helpful assistant that ALWAYS responds in valid JSON format. "
                   "When asked to return JSON, output ONLY the JSON object with no additional text, "
                   "no markdown formatting, no explanations before or after. "
                   "Start your response with { and end with }."
    }
    messages = [system_message, {"role": "user", "content": prompt}]

    logging.info(f"ChatGPT_API_async: Calling model={model}, prompt_length={len(prompt)}")
    for i in range(max_retries):
        try:
            async with openai.AsyncOpenAI(api_key=api_key) as client:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=0,
                )
                result = response.choices[0].message.content
                logging.info(f"ChatGPT_API_async: Response received, length={len(result) if result else 0}")
                return result
        except Exception as e:
            error_str = str(e).lower()
            if '429' in str(e) or 'rate' in error_str or 'too many' in error_str:
                wait_time = min(5 * (2 ** i), 60)  # Start at 5s, max 60s
                print(f'************* Rate limited async, waiting {wait_time}s ({i+1}/{max_retries}) *************')
                await asyncio.sleep(wait_time)
            else:
                print(f'************* Retrying async ({i+1}/{max_retries}) *************')
                logging.error(f"ChatGPT_API_async Error: {e}")
                await asyncio.sleep(2)

            if i >= max_retries - 1:
                logging.error('Max retries reached for prompt: ' + prompt[:200])
                return ""  # Return empty string instead of "Error"  
            
            
def get_json_content(response):
    start_idx = response.find("```json")
    if start_idx != -1:
        start_idx += 7
        response = response[start_idx:]
        
    end_idx = response.rfind("```")
    if end_idx != -1:
        response = response[:end_idx]
    
    json_content = response.strip()
    return json_content
         

def extract_json(content):
    """
    Extract JSON from LLM response with multiple fallback strategies.
    Uses robust parsing to handle various LLM output formats.
    """
    import re

    # Handle empty or None content
    if not content or not content.strip():
        logging.warning("extract_json: Empty content received")
        return _get_default_json_response('')

    logging.info(f"extract_json: Processing content (first 200 chars): {content[:200]}")

    # Strategy 1: Try to extract JSON from markdown code blocks
    json_content = None

    # Check for ```json blocks
    json_block_match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
    if json_block_match:
        json_content = json_block_match.group(1).strip()
        logging.info("extract_json: Found JSON in markdown code block")

    # Strategy 2: Try to find a JSON object with balanced braces
    if not json_content:
        # Find all potential JSON objects
        brace_start = content.find('{')
        if brace_start != -1:
            # Find matching closing brace
            depth = 0
            for i, char in enumerate(content[brace_start:], brace_start):
                if char == '{':
                    depth += 1
                elif char == '}':
                    depth -= 1
                    if depth == 0:
                        json_content = content[brace_start:i+1]
                        logging.info("extract_json: Extracted JSON using brace matching")
                        break

    # Strategy 3: Try to find JSON array
    if not json_content:
        bracket_start = content.find('[')
        if bracket_start != -1:
            depth = 0
            for i, char in enumerate(content[bracket_start:], bracket_start):
                if char == '[':
                    depth += 1
                elif char == ']':
                    depth -= 1
                    if depth == 0:
                        json_content = content[bracket_start:i+1]
                        logging.info("extract_json: Extracted JSON array using bracket matching")
                        break

    # Strategy 4: Use the entire content
    if not json_content:
        json_content = content.strip()
        logging.info("extract_json: Using entire content as JSON")

    # Clean up the JSON content
    json_content = _clean_json_string(json_content)

    # Try to parse
    result = _try_parse_json(json_content)

    if result is not None:
        # Ensure common keys exist
        result = _ensure_default_keys(result, content)
        logging.info(f"extract_json: Successfully parsed JSON with keys: {list(result.keys()) if isinstance(result, dict) else 'array'}")
        return result

    # All parsing strategies failed, return defaults based on content analysis
    logging.warning(f"extract_json: All parsing strategies failed, using content analysis")
    return _get_default_json_response(content)


def _clean_json_string(json_str):
    """Clean up common JSON formatting issues."""
    import re

    if not json_str:
        return json_str

    # Remove any leading/trailing whitespace
    json_str = json_str.strip()

    # Replace Python None with JSON null
    json_str = re.sub(r'\bNone\b', 'null', json_str)

    # Replace Python True/False with JSON true/false
    json_str = re.sub(r'\bTrue\b', 'true', json_str)
    json_str = re.sub(r'\bFalse\b', 'false', json_str)

    # Replace single quotes with double quotes (careful with apostrophes)
    # Only replace if it looks like a JSON key/value delimiter
    json_str = re.sub(r"(?<=[{,:\[])\s*'([^']*?)'\s*(?=[},:}\]])", r'"\1"', json_str)

    # Remove trailing commas before } or ]
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)

    # Normalize whitespace (but preserve spaces in strings)
    # This is tricky, so we'll be conservative
    json_str = re.sub(r'\n\s*', ' ', json_str)

    return json_str


def _try_parse_json(json_str):
    """Try multiple strategies to parse JSON string. Always returns dict or None."""
    import re

    if not json_str:
        return None

    def ensure_dict(result):
        """Ensure result is a dictionary."""
        if result is None:
            return None
        if isinstance(result, dict):
            return result
        if isinstance(result, list):
            return {'table_of_contents': result, 'data': result}
        if isinstance(result, str):
            return {'value': result, 'thinking': result}
        return {'value': str(result)}

    # Strategy 1: Direct parse
    try:
        result = json.loads(json_str)
        return ensure_dict(result)
    except json.JSONDecodeError:
        pass

    # Strategy 2: Try to fix common issues
    try:
        # Remove any text before the first { or [
        match = re.search(r'[\[{]', json_str)
        if match:
            trimmed = json_str[match.start():]
            result = json.loads(trimmed)
            return ensure_dict(result)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Try to extract just the first valid JSON object
    try:
        # Use a more lenient approach - find content between first { and last }
        first_brace = json_str.find('{')
        last_brace = json_str.rfind('}')
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            potential_json = json_str[first_brace:last_brace+1]
            result = json.loads(potential_json)
            return ensure_dict(result)
    except json.JSONDecodeError:
        pass

    # Strategy 4: Try line by line for multi-line responses
    try:
        lines = json_str.split('\n')
        for line in lines:
            line = line.strip()
            if line.startswith('{') and line.endswith('}'):
                result = json.loads(line)
                return ensure_dict(result)
    except json.JSONDecodeError:
        pass

    return None


def _ensure_default_keys(result, original_content):
    """Ensure common keys exist in the result dictionary."""
    if not isinstance(result, dict):
        return result

    content_lower = original_content.lower() if original_content else ''

    # Add default keys if missing
    default_mappings = {
        'toc_detected': 'yes' if 'yes' in content_lower and 'table' in content_lower else 'no',
        'answer': result.get('toc_detected', 'yes' if 'yes' in content_lower else 'no'),
        'completed': 'yes' if 'complete' in content_lower and 'yes' in content_lower else 'no',
        'page_index_given_in_toc': 'yes' if ('page' in content_lower or 'index' in content_lower) and 'yes' in content_lower else 'no',
    }

    for key, default_value in default_mappings.items():
        if key not in result:
            result[key] = default_value

    return result


def _get_default_json_response(content):
    """Generate a default JSON response based on sophisticated content analysis."""
    import re

    content_lower = content.lower() if content else ''
    content_stripped = content.strip() if content else ''

    # More sophisticated pattern matching for yes/no answers
    # Look for explicit yes/no answers, especially at the end or as standalone words

    # Check for clear "Yes" or "No" answers (case insensitive, standalone)
    yes_patterns = [
        r'\byes\b',
        r'\byes[,.]',
        r':\s*yes',
        r'answer[:\s]+yes',
        r'is[:\s]+yes',
        r'\baffirmative\b',
        r'\bcorrect\b',
        r'\btrue\b',
    ]

    no_patterns = [
        r'\bno\b(?!\w)',  # 'no' but not 'not', 'none', etc.
        r'\bno[,.]',
        r':\s*no\b',
        r'answer[:\s]+no',
        r'is[:\s]+no',
        r'\bnegative\b',
        r'\bfalse\b',
        r'\bnot found\b',
        r'\bnone\b',
        r'\bdoes not\b',
        r'\bno table of content\b',
    ]

    # Count yes/no occurrences
    yes_count = sum(1 for pattern in yes_patterns if re.search(pattern, content_lower))
    no_count = sum(1 for pattern in no_patterns if re.search(pattern, content_lower))

    # Check for TOC-related content
    has_numbered_list = bool(re.search(r'^\s*\d+[\.\)]\s+\w', content, re.MULTILINE))
    has_chapter_refs = bool(re.search(r'\b(chapter|section|part)\s+\d', content_lower))
    has_page_numbers = bool(re.search(r'\b(page|p\.?|pg\.?)\s*\d+', content_lower))

    # Determine toc_detected
    # If the response contains a numbered list, it might be showing TOC content
    if has_numbered_list or has_chapter_refs:
        toc_detected = 'yes'
    elif yes_count > no_count:
        toc_detected = 'yes'
    elif no_count > yes_count:
        toc_detected = 'no'
    else:
        # Default to no if unclear
        toc_detected = 'no'

    # Determine page_index_given_in_toc
    page_index_given = 'yes' if has_page_numbers else 'no'

    # Determine completed
    completed = 'yes' if yes_count > 0 or 'complete' in content_lower else 'no'

    logging.info(f"_get_default_json_response: yes_count={yes_count}, no_count={no_count}, "
                 f"has_numbered_list={has_numbered_list}, toc_detected={toc_detected}")

    return {
        'toc_detected': toc_detected,
        'answer': toc_detected,  # Use same logic
        'completed': completed,
        'page_index_given_in_toc': page_index_given,
        'thinking': content[:500] if content else 'No content provided',
        'table_of_contents': [],
        'physical_index': None,
        'start': 'no',
        'start_index': None,
    }

def write_node_id(data, node_id=0):
    if isinstance(data, dict):
        data['node_id'] = str(node_id).zfill(4)
        node_id += 1
        for key in list(data.keys()):
            if 'nodes' in key:
                node_id = write_node_id(data[key], node_id)
    elif isinstance(data, list):
        for index in range(len(data)):
            node_id = write_node_id(data[index], node_id)
    return node_id

def get_nodes(structure):
    if isinstance(structure, dict):
        structure_node = copy.deepcopy(structure)
        structure_node.pop('nodes', None)
        nodes = [structure_node]
        for key in list(structure.keys()):
            if 'nodes' in key:
                nodes.extend(get_nodes(structure[key]))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(get_nodes(item))
        return nodes
    
def structure_to_list(structure):
    if isinstance(structure, dict):
        nodes = []
        nodes.append(structure)
        if 'nodes' in structure:
            nodes.extend(structure_to_list(structure['nodes']))
        return nodes
    elif isinstance(structure, list):
        nodes = []
        for item in structure:
            nodes.extend(structure_to_list(item))
        return nodes

    
def get_leaf_nodes(structure):
    if isinstance(structure, dict):
        if not structure['nodes']:
            structure_node = copy.deepcopy(structure)
            structure_node.pop('nodes', None)
            return [structure_node]
        else:
            leaf_nodes = []
            for key in list(structure.keys()):
                if 'nodes' in key:
                    leaf_nodes.extend(get_leaf_nodes(structure[key]))
            return leaf_nodes
    elif isinstance(structure, list):
        leaf_nodes = []
        for item in structure:
            leaf_nodes.extend(get_leaf_nodes(item))
        return leaf_nodes

def is_leaf_node(data, node_id):
    # Helper function to find the node by its node_id
    def find_node(data, node_id):
        if isinstance(data, dict):
            if data.get('node_id') == node_id:
                return data
            for key in data.keys():
                if 'nodes' in key:
                    result = find_node(data[key], node_id)
                    if result:
                        return result
        elif isinstance(data, list):
            for item in data:
                result = find_node(item, node_id)
                if result:
                    return result
        return None

    # Find the node with the given node_id
    node = find_node(data, node_id)

    # Check if the node is a leaf node
    if node and not node.get('nodes'):
        return True
    return False

def get_last_node(structure):
    return structure[-1]


def extract_text_from_pdf(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    ###return text not list 
    text=""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text+=page.extract_text()
    return text

def get_pdf_title(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    meta = pdf_reader.metadata
    title = meta.title if meta and meta.title else 'Untitled'
    return title

def get_text_of_pages(pdf_path, start_page, end_page, tag=True):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    text = ""
    for page_num in range(start_page-1, end_page):
        page = pdf_reader.pages[page_num]
        page_text = page.extract_text()
        if tag:
            text += f"<start_index_{page_num+1}>\n{page_text}\n<end_index_{page_num+1}>\n"
        else:
            text += page_text
    return text

def get_first_start_page_from_text(text):
    start_page = -1
    start_page_match = re.search(r'<start_index_(\d+)>', text)
    if start_page_match:
        start_page = int(start_page_match.group(1))
    return start_page

def get_last_start_page_from_text(text):
    start_page = -1
    # Find all matches of start_index tags
    start_page_matches = re.finditer(r'<start_index_(\d+)>', text)
    # Convert iterator to list and get the last match if any exist
    matches_list = list(start_page_matches)
    if matches_list:
        start_page = int(matches_list[-1].group(1))
    return start_page


def sanitize_filename(filename, replacement='-'):
    # In Linux, only '/' and '\0' (null) are invalid in filenames.
    # Null can't be represented in strings, so we only handle '/'.
    return filename.replace('/', replacement)

def get_pdf_name(pdf_path):
    # Extract PDF name
    if isinstance(pdf_path, str):
        pdf_name = os.path.basename(pdf_path)
    elif isinstance(pdf_path, BytesIO):
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        meta = pdf_reader.metadata
        pdf_name = meta.title if meta and meta.title else 'Untitled'
        pdf_name = sanitize_filename(pdf_name)
    return pdf_name


class JsonLogger:
    def __init__(self, file_path):
        # Extract PDF name for logger name
        pdf_name = get_pdf_name(file_path)
            
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = f"{pdf_name}_{current_time}.json"
        os.makedirs("./logs", exist_ok=True)
        # Initialize empty list to store all messages
        self.log_data = []

    def log(self, level, message, **kwargs):
        if isinstance(message, dict):
            self.log_data.append(message)
        else:
            self.log_data.append({'message': message})
        # Add new message to the log data
        
        # Write entire log data to file
        with open(self._filepath(), "w") as f:
            json.dump(self.log_data, f, indent=2)

    def info(self, message, **kwargs):
        self.log("INFO", message, **kwargs)

    def error(self, message, **kwargs):
        self.log("ERROR", message, **kwargs)

    def debug(self, message, **kwargs):
        self.log("DEBUG", message, **kwargs)

    def exception(self, message, **kwargs):
        kwargs["exception"] = True
        self.log("ERROR", message, **kwargs)

    def _filepath(self):
        return os.path.join("logs", self.filename)
    



def list_to_tree(data):
    def get_parent_structure(structure):
        """Helper function to get the parent structure code"""
        if not structure:
            return None
        parts = str(structure).split('.')
        return '.'.join(parts[:-1]) if len(parts) > 1 else None
    
    # First pass: Create nodes and track parent-child relationships
    nodes = {}
    root_nodes = []
    
    for item in data:
        structure = item.get('structure')
        node = {
            'title': item.get('title'),
            'start_index': item.get('start_index'),
            'end_index': item.get('end_index'),
            'nodes': []
        }
        
        nodes[structure] = node
        
        # Find parent
        parent_structure = get_parent_structure(structure)
        
        if parent_structure:
            # Add as child to parent if parent exists
            if parent_structure in nodes:
                nodes[parent_structure]['nodes'].append(node)
            else:
                root_nodes.append(node)
        else:
            # No parent, this is a root node
            root_nodes.append(node)
    
    # Helper function to clean empty children arrays
    def clean_node(node):
        if not node['nodes']:
            del node['nodes']
        else:
            for child in node['nodes']:
                clean_node(child)
        return node
    
    # Clean and return the tree
    return [clean_node(node) for node in root_nodes]

def add_preface_if_needed(data):
    if not isinstance(data, list) or not data:
        return data

    if data[0]['physical_index'] is not None and data[0]['physical_index'] > 1:
        preface_node = {
            "structure": "0",
            "title": "Preface",
            "physical_index": 1,
        }
        data.insert(0, preface_node)
    return data



def get_page_tokens(pdf_path, model="gpt-4o-2024-11-20", pdf_parser="PyPDF2"):
    enc = tiktoken.encoding_for_model(model)
    if pdf_parser == "PyPDF2":
        pdf_reader = PyPDF2.PdfReader(pdf_path)
        page_list = []
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            page_text = page.extract_text()
            token_length = len(enc.encode(page_text))
            page_list.append((page_text, token_length))
        return page_list
    elif pdf_parser == "PyMuPDF":
        if isinstance(pdf_path, BytesIO):
            pdf_stream = pdf_path
            doc = pymupdf.open(stream=pdf_stream, filetype="pdf")
        elif isinstance(pdf_path, str) and os.path.isfile(pdf_path) and pdf_path.lower().endswith(".pdf"):
            doc = pymupdf.open(pdf_path)
        page_list = []
        for page in doc:
            page_text = page.get_text()
            token_length = len(enc.encode(page_text))
            page_list.append((page_text, token_length))
        return page_list
    else:
        raise ValueError(f"Unsupported PDF parser: {pdf_parser}")

        

def get_text_of_pdf_pages(pdf_pages, start_page, end_page):
    text = ""
    for page_num in range(start_page-1, end_page):
        text += pdf_pages[page_num][0]
    return text

def get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page):
    text = ""
    for page_num in range(start_page-1, end_page):
        text += f"<physical_index_{page_num+1}>\n{pdf_pages[page_num][0]}\n<physical_index_{page_num+1}>\n"
    return text

def get_number_of_pages(pdf_path):
    pdf_reader = PyPDF2.PdfReader(pdf_path)
    num = len(pdf_reader.pages)
    return num



def post_processing(structure, end_physical_index):
    # First convert page_number to start_index in flat list
    for i, item in enumerate(structure):
        item['start_index'] = item.get('physical_index')
        if i < len(structure) - 1:
            if structure[i + 1].get('appear_start') == 'yes':
                item['end_index'] = structure[i + 1]['physical_index']-1
            else:
                item['end_index'] = structure[i + 1]['physical_index']
        else:
            item['end_index'] = end_physical_index
    tree = list_to_tree(structure)
    if len(tree)!=0:
        return tree
    else:
        ### remove appear_start 
        for node in structure:
            node.pop('appear_start', None)
            node.pop('physical_index', None)
        return structure

def clean_structure_post(data):
    if isinstance(data, dict):
        data.pop('page_number', None)
        data.pop('start_index', None)
        data.pop('end_index', None)
        if 'nodes' in data:
            clean_structure_post(data['nodes'])
    elif isinstance(data, list):
        for section in data:
            clean_structure_post(section)
    return data

def remove_fields(data, fields=['text']):
    if isinstance(data, dict):
        return {k: remove_fields(v, fields)
            for k, v in data.items() if k not in fields}
    elif isinstance(data, list):
        return [remove_fields(item, fields) for item in data]
    return data

def print_toc(tree, indent=0):
    for node in tree:
        print('  ' * indent + node['title'])
        if node.get('nodes'):
            print_toc(node['nodes'], indent + 1)

def print_json(data, max_len=40, indent=2):
    def simplify_data(obj):
        if isinstance(obj, dict):
            return {k: simplify_data(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [simplify_data(item) for item in obj]
        elif isinstance(obj, str) and len(obj) > max_len:
            return obj[:max_len] + '...'
        else:
            return obj
    
    simplified = simplify_data(data)
    print(json.dumps(simplified, indent=indent, ensure_ascii=False))


def remove_structure_text(data):
    if isinstance(data, dict):
        data.pop('text', None)
        if 'nodes' in data:
            remove_structure_text(data['nodes'])
    elif isinstance(data, list):
        for item in data:
            remove_structure_text(item)
    return data


def check_token_limit(structure, limit=110000):
    list = structure_to_list(structure)
    for node in list:
        num_tokens = count_tokens(node['text'], model='gpt-4o')
        if num_tokens > limit:
            print(f"Node ID: {node['node_id']} has {num_tokens} tokens")
            print("Start Index:", node['start_index'])
            print("End Index:", node['end_index'])
            print("Title:", node['title'])
            print("\n")


def convert_physical_index_to_int(data):
    if isinstance(data, list):
        for i in range(len(data)):
            # Check if item is a dictionary and has 'physical_index' key
            if isinstance(data[i], dict) and 'physical_index' in data[i]:
                if isinstance(data[i]['physical_index'], str):
                    if data[i]['physical_index'].startswith('<physical_index_'):
                        data[i]['physical_index'] = int(data[i]['physical_index'].split('_')[-1].rstrip('>').strip())
                    elif data[i]['physical_index'].startswith('physical_index_'):
                        data[i]['physical_index'] = int(data[i]['physical_index'].split('_')[-1].strip())
    elif isinstance(data, str):
        if data.startswith('<physical_index_'):
            data = int(data.split('_')[-1].rstrip('>').strip())
        elif data.startswith('physical_index_'):
            data = int(data.split('_')[-1].strip())
        # Check data is int
        if isinstance(data, int):
            return data
        else:
            return None
    return data


def convert_page_to_int(data):
    for item in data:
        if 'page' in item and isinstance(item['page'], str):
            try:
                item['page'] = int(item['page'])
            except ValueError:
                # Keep original value if conversion fails
                pass
    return data


def add_node_text(node, pdf_pages):
    if isinstance(node, dict):
        start_page = node.get('start_index')
        end_page = node.get('end_index')
        node['text'] = get_text_of_pdf_pages(pdf_pages, start_page, end_page)
        if 'nodes' in node:
            add_node_text(node['nodes'], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text(node[index], pdf_pages)
    return


def add_node_text_with_labels(node, pdf_pages):
    if isinstance(node, dict):
        start_page = node.get('start_index')
        end_page = node.get('end_index')
        node['text'] = get_text_of_pdf_pages_with_labels(pdf_pages, start_page, end_page)
        if 'nodes' in node:
            add_node_text_with_labels(node['nodes'], pdf_pages)
    elif isinstance(node, list):
        for index in range(len(node)):
            add_node_text_with_labels(node[index], pdf_pages)
    return


async def generate_node_summary(node, model=None):
    prompt = f"""You are given a part of a document, your task is to generate a description of the partial document about what are main points covered in the partial document.

    Partial Document Text: {node['text']}
    
    Directly return the description, do not include any other text.
    """
    response = await ChatGPT_API_async(model, prompt)
    return response


async def generate_summaries_for_structure(structure, model=None):
    nodes = structure_to_list(structure)
    tasks = [generate_node_summary(node, model=model) for node in nodes]
    summaries = await asyncio.gather(*tasks)
    
    for node, summary in zip(nodes, summaries):
        node['summary'] = summary
    return structure


def create_clean_structure_for_description(structure):
    """
    Create a clean structure for document description generation,
    excluding unnecessary fields like 'text'.
    """
    if isinstance(structure, dict):
        clean_node = {}
        # Only include essential fields for description
        for key in ['title', 'node_id', 'summary', 'prefix_summary']:
            if key in structure:
                clean_node[key] = structure[key]
        
        # Recursively process child nodes
        if 'nodes' in structure and structure['nodes']:
            clean_node['nodes'] = create_clean_structure_for_description(structure['nodes'])
        
        return clean_node
    elif isinstance(structure, list):
        return [create_clean_structure_for_description(item) for item in structure]
    else:
        return structure


def generate_doc_description(structure, model=None):
    prompt = f"""Your are an expert in generating descriptions for a document.
    You are given a structure of a document. Your task is to generate a one-sentence description for the document, which makes it easy to distinguish the document from other documents.
        
    Document Structure: {structure}
    
    Directly return the description, do not include any other text.
    """
    response = ChatGPT_API(model, prompt)
    return response


def reorder_dict(data, key_order):
    if not key_order:
        return data
    return {key: data[key] for key in key_order if key in data}


def format_structure(structure, order=None):
    if not order:
        return structure
    if isinstance(structure, dict):
        if 'nodes' in structure:
            structure['nodes'] = format_structure(structure['nodes'], order)
        if not structure.get('nodes'):
            structure.pop('nodes', None)
        structure = reorder_dict(structure, order)
    elif isinstance(structure, list):
        structure = [format_structure(item, order) for item in structure]
    return structure


class ConfigLoader:
    def __init__(self, default_path: str = None):
        if default_path is None:
            default_path = Path(__file__).parent / "config.yaml"
        self._default_dict = self._load_yaml(default_path)

    @staticmethod
    def _load_yaml(path):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _validate_keys(self, user_dict):
        unknown_keys = set(user_dict) - set(self._default_dict)
        if unknown_keys:
            raise ValueError(f"Unknown config keys: {unknown_keys}")

    def load(self, user_opt=None) -> config:
        """
        Load the configuration, merging user options with default values.
        """
        if user_opt is None:
            user_dict = {}
        elif isinstance(user_opt, config):
            user_dict = vars(user_opt)
        elif isinstance(user_opt, dict):
            user_dict = user_opt
        else:
            raise TypeError("user_opt must be dict, config(SimpleNamespace) or None")

        self._validate_keys(user_dict)
        merged = {**self._default_dict, **user_dict}
        return config(**merged)