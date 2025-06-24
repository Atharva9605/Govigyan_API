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
from flask_cors import CORS # Import Flask-CORS

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

load_dotenv()  # Load environment variables

# --- Configuration (from environment variables) ---
DB_NAME = os.environ.get('DB_NAME')
DB_USER = os.environ.get('DB_USER')
DB_PASSWORD = os.environ.get('DB_PASSWORD')
DB_HOST = os.environ.get('DB_HOST')
DB_PORT = os.environ.get('DB_PORT', 5432)
DB_TABLE = os.environ.get('DB_TABLE', 'StockBook') # Default table name

# Ensure GEMINI_API_KEY is loaded from .env or provided
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'pdf'}
UPLOAD_FOLDER = 'uploads' # Folder to temporarily store uploaded files
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# --- Helper Functions (moved from your original script, slightly adapted) ---

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_database_if_not_exists(dbname, user, password, host, port):
    try:
        app.logger.info(f"[DB_SETUP] Attempting to connect to default 'postgres' database at {host}:{port}...")
        conn = psycopg2.connect(dbname='postgres', user=user, password=password, host=host, port=port)
        conn.set_isolation_level(psycopg2.extensions.ISOLATION_LEVEL_AUTOCOMMIT)
        app.logger.info("[DB_SETUP] Connected successfully to default database.")
        cur = conn.cursor()
        app.logger.info(f"[DB_SETUP] Checking if database '{dbname}' exists...")

        cur.execute("SELECT 1 FROM pg_database WHERE datname = %s", (dbname,))
        exists = cur.fetchone()
        if not exists:
            app.logger.info(f"[DB_SETUP] Database '{dbname}' does not exist. Creating now...")
            cur.execute(f'CREATE DATABASE "{dbname}";')
            app.logger.info(f"[DB_SETUP] Database '{dbname}' created successfully.")
        else:
            app.logger.info(f"[DB_SETUP] Database '{dbname}' already exists. Skipping creation.")
        cur.close()
        conn.close()
        app.logger.info("[DB_SETUP] Closed connection to default database.")
        return True
    except psycopg2.Error as e:
        app.logger.error(f"[ERROR] PostgreSQL Error during DB creation check/create: {e}")
        app.logger.error(traceback.format_exc())
        return False
    except Exception as e:
        app.logger.error(f"[ERROR] General error creating database: {e}")
        app.logger.error(traceback.format_exc())
        return False

def insert_data_into_postgres(data_list, dbname, user, password, host, port, table_name):
    conn = None
    cur = None
    try:
        app.logger.info(f"[DB_INSERT] Connecting to target database '{dbname}'...")
        conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
        app.logger.info("[DB_INSERT] Connected to target database.")
        cur = conn.cursor()
        app.logger.info(f"[DB_INSERT] Ensuring table '{table_name}' exists...")

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
        app.logger.info(f"[DB_INSERT] Table '{table_name}' ensured.")

        if not data_list:
            app.logger.info("[DB_INSERT] No data provided to insert. Skipping insertion.")
            return True

        app.logger.info(f"[DB_INSERT] Inserting {len(data_list)} records into '{table_name}'...")
        insert_count = 0
        for i, record in enumerate(data_list, start=1):
            date_val = record.get("DATE")
            particulars_val = record.get("PARTICULARS")
            voucher_billno_val = record.get("Voucher_BillNo")
            receipts_qty = record.get("RECEIPTS_Quantity")
            receipts_amt = record.get("RECEIPTS_Amount")
            issued_qty = record.get("ISSUED_Quantity")
            issued_amt = record.get("ISSUED_Amount")
            balance_qty = record.get("BALANCE_Quantity")
            balance_amt = record.get("BALANCE_Amount")

            app.logger.info(f"[DB_INSERT.{i}] Preparing record: Date='{date_val}', Particulars='{particulars_val[:30]}...'")

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
                    date_val, particulars_val, voucher_billno_val,
                    receipts_qty, receipts_amt,
                    issued_qty, issued_amt,
                    balance_qty, balance_amt
                ))
                insert_count += 1
            except psycopg2.Error as insert_err:
                app.logger.error(f"[ERROR] Failed to insert record {i}: {record}")
                app.logger.error(f"       PostgreSQL Error: {insert_err}")
                conn.rollback()

        conn.commit()
        app.logger.info(f"[DB_INSERT] {insert_count} records processed and committed.")
        cur.execute(f'SELECT COUNT(*) FROM "{table_name}"')
        total_rows = cur.fetchone()[0]
        app.logger.info(f"[DB_INSERT] Total rows in table '{table_name}': {total_rows}")
        return True

    except psycopg2.Error as e:
        app.logger.error(f"[ERROR] PostgreSQL Error during data insertion setup/connection: {e}")
        app.logger.error(traceback.format_exc())
        if conn: conn.rollback()
        return False
    except Exception as e:
        app.logger.error(f"[ERROR] General error during data insertion process: {e}")
        app.logger.error(traceback.format_exc())
        if conn: conn.rollback()
        return False
    finally:
        if cur: cur.close(); app.logger.info("[DB_INSERT] Cursor closed.")
        if conn: conn.close(); app.logger.info("[DB_INSERT] Connection closed.")

def process_uploaded_file(file_storage):
    filename = secure_filename(file_storage.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file_storage.save(filepath)
    app.logger.info(f"[UPLOAD] Saved uploaded file to: {filepath}")

    image_paths = []
    if filepath.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        image_paths.append(filepath)
    elif filepath.lower().endswith('.pdf'):
        app.logger.info(f"[PDF] Processing PDF {filepath}...")
        try:
            doc = fitz.open(filepath)
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img_data = pix.tobytes("jpeg")
                img = Image.open(BytesIO(img_data))
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_pdf_page_{page_num}_{os.path.basename(filename)}.jpg")
                img.save(output_path, 'JPEG')
                image_paths.append(output_path)
                app.logger.info(f"  - Converted page {page_num} to {output_path}")
            doc.close()
        except Exception as e:
            app.logger.error(f"[ERROR] Failed to process PDF {filepath}: {e}")
            app.logger.error(traceback.format_exc())
            return []
    return image_paths

def extract_json_from_images_with_gemini(image_paths, api_key):
    if not api_key:
        app.logger.error("[ERROR] GEMINI_API_KEY is not set.")
        return None

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        app.logger.info(f"[GEMINI] Using model: {model.model_name}")
    except Exception as config_err:
        app.logger.error(f"[ERROR] Failed to configure Gemini: {config_err}")
        app.logger.error(traceback.format_exc())
        return None

    image_objects = []
    app.logger.info(f"[GEMINI] Loading {len(image_paths)} images...")
    for img_path in image_paths:
        try:
            app.logger.info(f"  - Loading: {img_path}")
            img = Image.open(img_path)
            image_objects.append(img)
        except Exception as img_err:
            app.logger.error(f"[ERROR] Failed to load image {img_path}: {img_err}")
            app.logger.error(traceback.format_exc())
            return None

    if not image_objects:
        app.logger.error("[ERROR] No images were successfully loaded.")
        return None

    app.logger.info("[GEMINI] Images loaded successfully.")

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

    If a value is not present in the image for a specific field in a row, use null for numeric fields and an empty string or null for text fields where appropriate. Ensure all keys are present in each JSON object.
    """

    request_payload = [prompt_text] + image_objects

    app.logger.info("[GEMINI] Sending prompt and images to Gemini API...")
    try:
        response = model.generate_content(request_payload)
        app.logger.info("[GEMINI] Received response from API.")

        raw_json_text = response.text.strip()
        app.logger.debug(f"[GEMINI] Raw response text (full): {raw_json_text}")

        # Try to clean the response by removing any non-JSON text
        if raw_json_text.startswith('```json') and raw_json_text.endswith('```'):
            raw_json_text = raw_json_text[len('```json'):-len('```')].strip()
        elif raw_json_text.startswith('[') or raw_json_text.startswith('{'):
            pass  # Already looks like JSON, proceed
        else:
            # More robust regex to find the JSON structure, allowing for potential leading/trailing garbage
            json_match = re.search(r'(\[.*\]|\{.*\})', raw_json_text, re.DOTALL)
            if json_match:
                raw_json_text = json_match.group(1).strip() # Capture the matched group
            else:
                app.logger.error("[ERROR] No valid JSON structure found in response.")
                app.logger.error(f"       Raw Text: {raw_json_text}")
                return None

        if not raw_json_text:
            app.logger.error("[ERROR] Empty response from Gemini after stripping.")
            return None

        app.logger.info("[GEMINI] Attempting to parse JSON...")
        parsed_data = json.loads(raw_json_text)
        app.logger.info("[GEMINI] JSON parsed successfully.")

        if isinstance(parsed_data, list):
            app.logger.info(f"[GEMINI] Extracted {len(parsed_data)} records.")
            return parsed_data
        elif isinstance(parsed_data, dict):
            app.logger.warning("[WARNING] Gemini returned a single JSON object, wrapping it in a list.")
            return [parsed_data]
        else:
            app.logger.error(f"[ERROR] Gemini output was not a JSON list or object. Type: {type(parsed_data)}")
            return None

    except json.JSONDecodeError as json_err:
        app.logger.error(f"[ERROR] Failed to decode JSON response from Gemini: {json_err}")
        app.logger.error(f"       Problematic JSON Text: {raw_json_text}")
        app.logger.error(traceback.format_exc())
        return None
    except Exception as e:
        app.logger.error(f"[ERROR] Error during Gemini API call or processing: {e}")
        app.logger.error(traceback.format_exc())
        if 'response' in locals() and hasattr(response, 'prompt_feedback'):
            app.logger.error(f"       Prompt Feedback: {response.prompt_feedback}")
        if 'response' in locals() and hasattr(response, 'candidates') and response.candidates:
            app.logger.error(f"       Finish Reason: {response.candidates[0].finish_reason}")
            app.logger.error(f"       Safety Ratings: {response.candidates[0].safety_ratings}")
        return None
    finally:
        # Clean up temporary image files converted from PDF
        for temp_file in image_paths:
            if "temp_pdf_page_" in temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                    app.logger.info(f"[CLEANUP] Removed temporary file: {temp_file}")
                except Exception as e:
                    app.logger.error(f"[ERROR] Could not remove temp file {temp_file}: {e}")

# --- Flask Routes ---

@app.route('/')
def health_check():
    return "Stock Book API is running!"

@app.route('/process-stock-book', methods=['POST'])
def process_stock_book():
    app.logger.info("[API] Received request to /process-stock-book")

    if 'file' not in request.files:
        app.logger.error("[API_ERROR] No file part in the request.")
        return jsonify({"error": "No file part"}), 400

    files = request.files.getlist('file') # Get all files with name 'file'
    if not files:
        app.logger.error("[API_ERROR] No selected file.")
        return jsonify({"error": "No selected file"}), 400

    all_image_paths = []
    original_filenames = []

    for file_storage in files:
        if file_storage.filename == '':
            app.logger.warning("[API_WARNING] Skipping file with no filename.")
            continue
        if file_storage and allowed_file(file_storage.filename):
            original_filenames.append(file_storage.filename)
            processed_paths = process_uploaded_file(file_storage)
            all_image_paths.extend(processed_paths)
        else:
            app.logger.error(f"[API_ERROR] File type not allowed for {file_storage.filename}")
            return jsonify({"error": f"File type not allowed for {file_storage.filename}"}), 400

    if not all_image_paths:
        app.logger.error("[API_ERROR] No valid image or PDF files were processed from upload.")
        return jsonify({"error": "No valid image or PDF files were processed from upload."}), 500

    # Ensure database exists
    if not create_database_if_not_exists(DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT):
        return jsonify({"error": "Failed to ensure database existence. Check DB connection details."}), 500

    # Extract data using Gemini
    extracted_data = extract_json_from_images_with_gemini(all_image_paths, GEMINI_API_KEY)

    # Clean up uploaded files (and temp PDF images)
    for path in all_image_paths:
        if os.path.exists(path):
            try:
                os.remove(path)
                app.logger.info(f"[CLEANUP] Removed uploaded/temp file: {path}")
            except Exception as e:
                app.logger.error(f"[CLEANUP_ERROR] Could not remove file {path}: {e}")

    # Insert data into PostgreSQL
    if extracted_data:
        app.logger.info("[API] Data extracted. Attempting to insert into DB.")
        if insert_data_into_postgres(extracted_data, DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_TABLE):
            app.logger.info("[API_SUCCESS] Data successfully extracted and inserted.")
            return jsonify({
                "message": "Data successfully extracted and inserted into PostgreSQL.",
                "total_records_extracted": len(extracted_data),
                "data": extracted_data # Optionally return the extracted data
            }), 200
        else:
            app.logger.error("[API_ERROR] Failed to insert data into PostgreSQL.")
            return jsonify({"error": "Failed to insert data into PostgreSQL."}), 500
    else:
        app.logger.warning("[API_WARNING] No data extracted from the provided files.")
        return jsonify({"message": "No data extracted from the provided files."}), 200

if __name__ == '__main__':
    # For local development, set host to '0.0.0.0' to be accessible externally
    # Render automatically sets the PORT environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True) # debug=True for local dev, set to False for production