import json
import psycopg2
import psycopg2.extensions
import traceback
import os
import re
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv
import fitz  # PyMuPDF
from io import BytesIO
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from flask_cors import CORS
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes (restrict in production if needed)

load_dotenv()  # Load environment variables

# --- Configuration (from environment variables) ---
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_HOST = os.environ.get('DB_HOST')
DB_PORT = os.environ.get('DB_PORT', '5432')  # Default to 5432
DB_TABLE = os.environ.get('DB_TABLE', 'StockBook')  # Default table name
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# Validate environment variables
required_env_vars = ['DB_NAME', 'DB_USER', 'DB_PASSWORD', 'DB_HOST', 'GEMINI_API_KEY']
missing_vars = [var for var in required_env_vars if not os.environ.get(var)]
if missing_vars:
    logger.error(f"[CONFIG_ERROR] Missing environment variables: {', '.join(missing_vars)}")
    raise ValueError(f"Missing environment variables: {', '.join(missing_vars)}")

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Helper Functions ---

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_database_if_not_exists(dbname, user, password, host, port):
    try:
        logger.info(f"[DB_SETUP] Connecting to 'postgres' database at {host}:{port}...")
        with psycopg2.connect(dbname='postgres', user=user, password=password, host=host, port=port) as conn:
            conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
            logger.info("[DB_SETUP] Connected to default database.")
            with conn.cursor() as cur:
                logger.info(f"[DB_SETUP] Checking if database '{dbname}' exists...")
                cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
                exists = cur.fetchone()
                if not exists:
                    logger.info(f"[DB_SETUP] Creating database '{dbname}'...")
                    cur.execute(f'CREATE DATABASE "{dbname}";')
                    logger.info(f"[DB_SETUP] Database '{dbname}' created.")
                else:
                    logger.info(f"[DB_SETUP] Database '{dbname}' already exists.")
        return True
    except psycopg2.Error as e:
        logger.error(f"[DB_ERROR] Postgre++

SQL Error during DB creation: {e}")
        logger.error(traceback.format_exc())
        return False
    except Exception as e:
        logger.error(f"[DB_ERROR] General error during DB creation: {e}")
        logger.error(traceback.format_exc())
        return False

def insert_data_into_postgres(data_list, dbname, user, password, host, port, table_name):
    try:
        logger.info(f"[DB_INSERT] Connecting to database '{dbname}'...")
        with psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port) as conn:
            with conn.cursor() as cur:
                logger.info(f"[DB_INSERT] Ensuring table '{table_name}' exists...")
                create_table_query = f"""
                CREATE TABLE IF NOT EXISTS "{table_name}" (
                    "Entry_ID" SERIAL PRIMARY KEY,
                    "DATE" TEXT,
                    "PARTICULARS" TEXT,
                    "Voucher_BillNo" TEXT,
                    "RECEIPTS_Quantity" INTEGER,
                    "RECEIPTS_Amount" REAL,
                    "ISSUED_Quantity" INTEGER,
                    "ISSUED_Amount" REAL,
                    "BALANCE_Quantity" INTEGER,
                    "BALANCE_Amount" REAL
                );
                """
                cur.execute(create_table_query)
                conn.commit()
                logger.info(f"[DB_INSERT] Table '{table_name}' ensured.")

                if not data_list:
                    logger.info("[DB_INSERT] No data to insert.")
                    return True

                logger.info(f"[DB_INSERT] Inserting {len(data_list)} records...")
                insert_count = 0
                for i, record in enumerate(data_list, start=1):
                    sql = f"""
                    INSERT INTO "{table_name}" (
                        "DATE", "PARTICULARS", "Voucher_BillNo",
                        "RECEIPTS_Quantity", "RECEIPTS_Amount",
                        "ISSUED_Quantity", "ISSUED_Amount",
                        "BALANCE_Quantity", "BALANCE_Amount"
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
                    """
                    try:
                        cur.execute(sql, (
                            record.get("DATE"),
                            record.get("PARTICULARS"),
                            record.get("Voucher_BillNo"),
                            record.get("RECEIPTS_Quantity"),
                            record.get("RECEIPTS_Amount"),
                            record.get("ISSUED_Quantity"),
                            record.get("ISSUED_Amount"),
                            record.get("BALANCE_Quantity"),
                            record.get("BALANCE_Amount")
                        ))
                        insert_count += 1
                    except psycopg2.Error as e:
                        logger.error(f"[DB_INSERT_ERROR] Failed to insert record {i}: {e}")
                        conn.rollback()
                        continue

                conn.commit()
                logger.info(f"[DB_INSERT] Inserted {insert_count} records.")
                cur.execute(f'SELECT COUNT(*) FROM "{table_name}"')
                total_rows = cur.fetchone()[0]
                logger.info(f"[DB_INSERT] Total rows in table '{table_name}': {total_rows}")
                return True
    except psycopg2.Error as e:
        logger.error(f"[DB_ERROR] PostgreSQL error during insertion: {e}")
        logger.error(traceback.format_exc())
        return False
    except Exception as e:
        logger.error(f"[DB_ERROR] General error during insertion: {e}")
        logger.error(traceback.format_exc())
        return False

def process_uploaded_file(file_storage):
    filename = secure_filename(file_storage.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file_storage.save(filepath)
    logger.info(f"[UPLOAD] Saved file to: {filepath}")

    image_paths = []
    try:
        if filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
            image_paths.append(filepath)
        elif filepath.lower().endswith('.pdf'):
            logger.info(f"[PDF] Processing PDF {filepath}...")
            doc = fitz.open(filepath)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img_data = pix.tobytes("jpeg")
                img = Image.open(BytesIO(img_data))
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_pdf_page_{page_num}_{os.path.basename(filename)}.jpg")
                img.save(output_path, 'JPEG')
                image_paths.append(output_path)
                logger.info(f"[PDF] Converted page {page_num} to {output_path}")
            doc.close()
    except Exception as e:
        logger.error(f"[UPLOAD_ERROR] Failed to process file {filepath}: {e}")
        logger.error(traceback.format_exc())
        return []
    return image_paths

def extract_json_from_images_with_gemini(image_paths, api_key):
    if not api_key:
        logger.error("[GEMINI_ERROR] GEMINI_API_KEY is not set.")
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        logger.info(f"[GEMINI] Using model: {model.model_name}")
    except Exception as e:
        logger.error(f"[GEMINI_ERROR] Failed to configure Gemini: {e}")
        logger.error(traceback.format_exc())
        return None

    image_objects = []
    for img_path in image_paths:
        try:
            img = Image.open(img_path)
            image_objects.append(img)
            logger.info(f"[GEMINI] Loaded image: {img_path}")
        except Exception as e:
            logger.error(f"[GEMINI_ERROR] Failed to load image {img_path}: {e}")
            logger.error(traceback.format_exc())
            return None

    if not image_objects:
        logger.error("[GEMINI_ERROR] No images loaded.")
        return None

    prompt_text = """
    You are an assistant that analyzes images of handwritten or printed Stock Book pages and extracts transaction information into JSON format. Return ONLY the raw JSON list with no additional text, comments, or formatting. Do not include markdown (e.g., ```json) or any explanatory text before or after the JSON.

    Look at the following image(s) carefully. It contains rows detailing stock transactions. For each distinct transaction row you can identify in the image, extract the following fields:
    - "DATE": The date of the transaction.
    - "PARTICULARS": The description or details of the transaction, including any voucher or bill number if clearly associated within the 'Particulars' column. If there's a separate 'Voucher Bill No.' column, prioritize that.
    - "Voucher_BillNo": The voucher or bill number, if there is a separate column for it. If it's integrated into 'Particulars' and there's no separate column, try to extract it, otherwise use null or an empty string.
    - "RECEIPTS_Quantity": The quantity received (under the 'RECEIPTS' or 'आवक माल' section). If empty or not applicable, use null.
    - "RECEIPTS_Amount": The amount/value received (under the 'RECEIPTS' or 'आवक माल' section). If empty or not applicable, use null.
    - "ISSUED_Quantity": The quantity issued (under the 'ISSUED' or 'जावक माल' section). If empty or not applicable, use null.
    - "ISSUED_Amount": The amount/value issued (under the 'ISSUED' or 'जावक माल' section). If empty or not applicable, use null.
    - "BALANCE_Quantity": The quantity remaining in balance (under the 'BALANCE' or 'बची संख्या' section).
    - "BALANCE_Amount": The amount/value remaining in balance (under the 'BALANCE' or 'बची संख्या' section).

    Format the output as a JSON list of objects. Each object represents one transaction row found in the image. The JSON keys MUST be exactly: "DATE", "PARTICULARS", "Voucher_BillNo", "RECEIPTS_Quantity", "RECEIPTS_Amount", "ISSUED_Quantity", "ISSUED_Amount", "BALANCE_Quantity", "BALANCE_Amount".

    Example output (no extra text):
    [{"DATE": "11/01/14", "PARTICULARS": "प्रारंभिक शेष", "Voucher_BillNo": null, "RECEIPTS_Quantity": 33, "RECEIPTS_Amount": 6930.00, "ISSUED_Quantity": null, "ISSUED_Amount": null, "BALANCE_Quantity": 33, "BALANCE_Amount": 6930.00}]
    """

    request_payload = [prompt_text] + image_objects
    logger.info("[GEMINI] Sending prompt and images to Gemini API...")
    try:
        response = model.generate_content(request_payload)
        logger.info("[GEMINI] Received response from API.")
        raw_json_text = response.text.strip()

        # Clean up response
        if raw_json_text.startswith('```json
            raw_json_text = raw_json_text[len('```json'):-len('```')].strip()
        elif not (raw_json_text.startswith('[') or raw_json_text.startswith('{')):
            json_match = re.search(r'(\[.*\]|\{.*\})', raw_json_text, re.DOTALL)
            if json_match:
                raw_json_text = json_match.group(1)
            else:
                logger.error("[GEMINI_ERROR] No valid JSON structure in response.")
                return None

        if not raw_json_text:
            logger.error("[GEMINI_ERROR] Empty response from Gemini.")
            return None

        parsed_data = json.loads(raw_json_text)
        logger.info(f"[GEMINI] Extracted {len(parsed_data) if isinstance(parsed_data, list) else 1} records.")
        return parsed_data if isinstance(parsed_data, list) else [parsed_data]
    except json.JSONDecodeError as e:
        logger.error(f"[GEMINI_ERROR] Failed to decode JSON: {e}")
        logger.error(f"Raw JSON Text: {raw_json_text}")
        logger.error(traceback.format_exc())
        return None
    except Exception as e:
        logger.error(f"[GEMINI_ERROR] Error during Gemini API call: {e}")
        logger.error(traceback.format_exc())
        return None
    finally:
        for img_path in image_paths:
            if os.path.exists(img_path) and "temp_pdf_page_" in img_path:
                try:
                    os.remove(img_path)
                    logger.info(f"[CLEANUP] Removed temporary file: {img_path}")
                except Exception as e:
                    logger.error(f"[CLEANUP_ERROR] Could not remove file {img_path}: {e}")

# --- Flask Routes ---

@app.route('/')
def health_check():
    return jsonify({"message": "Stock Book API is running!"}), 200

@app.route('/process-stock-book', methods=['POST'])
def process_stock_book():
    logger.info("[API] Received request to /process-stock-book")

    if 'file' not in request.files:
        logger.error("[API_ERROR] No file part in request.")
        return jsonify({"error": "No file part in request."}), 400

    files = request.files.getlist('file')
    if not files or all(f.filename == '' for f in files):
        logger.error("[API_ERROR] No valid files selected.")
        return jsonify({"error": "No valid files selected."}), 400

    all_image_paths = []
    original_filenames = []
    for file_storage in files:
        if file_storage.filename == '':
            logger.warning("[API_WARNING] Skipping empty filename.")
            continue
        if not allowed_file(file_storage.filename):
            logger.error(f"[API_ERROR] Invalid file type: {file_storage.filename}")
            return jsonify({"error": f"Invalid file type: {file_storage.filename}"}), 400
        original_filenames.append(file_storage.filename)
        all_image_paths.extend(process_uploaded_file(file_storage))

    if not all_image_paths:
        logger.error("[API_ERROR] No valid files processed.")
        return jsonify({"error": "No valid image or PDF files processed."}), 500

    try:
        # Ensure database exists
        if not create_database_if_not_exists(DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT):
            logger.error("[API_ERROR] Failed to ensure database existence.")
            return jsonify({"error": "Failed to connect to or create database. Check configuration."}), 500

 Judiciously handle database insertions
        extracted_data = extract_json_from_images_with_gemini(all_image_paths, GEMINI_API_KEY)
        if extracted_data:
            if insert_data_into_postgres(extracted_data, DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_TABLE):
                logger.info("[API_SUCCESS] Data inserted successfully.")
                return jsonify({
                    "message": "Data successfully extracted and inserted.",
                    "total_records_extracted": len(extracted_data),
                    "data": extracted_data
                }), 200
            else:
                logger.error("[API_ERROR] Failed to insert data into database.")
                return jsonify({"error": "Failed to insert data into database."}), 500
        else:
            logger.warning("[API_WARNING] No data extracted from files.")
            return jsonify({"message": "No data extracted from the provided files."}), 200
    finally:
        # Clean up all files
        for path in all_image_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logger.info(f"[CLEANUP] Removed file: {path}")
                except Exception as e:
                    logger.error(f"[CLEANUP_ERROR] Could not remove file {path}: {e}")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
