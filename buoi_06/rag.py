import os
import sys
import glob
import json
import sqlite3
import chromadb
from dotenv import load_dotenv

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

# Đường dẫn dự án
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHUNKS_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "buoi_05", "output", "chunks"))
STORAGE_DIR = os.path.join(BASE_DIR, "storage")
SQLITE_DB_PATH = os.path.join(STORAGE_DIR, "chunks.db")
CHROMA_DIR = os.path.join(STORAGE_DIR, "chroma")

os.makedirs(STORAGE_DIR, exist_ok=True)

# Khởi tạo Gemini Client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai_client = None
if GEMINI_API_KEY:
    try:
        from google import genai
        genai_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception:
        genai_client = None


def get_postgres_conn():
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    dbname = os.getenv("POSTGRES_DB", "rag_db")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")

    if not password:
        return None
    try:
        import psycopg
        conn = psycopg.connect(
            host=host, port=port, dbname=dbname, user=user, password=password, connect_timeout=3
        )
        return conn
    except Exception:
        return None


def init_text_storage():
    pg_conn = get_postgres_conn()
    if pg_conn:
        try:
            with pg_conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS chunks (
                        id VARCHAR(255) PRIMARY KEY,
                        doc_id VARCHAR(255),
                        text TEXT
                    );
                """)
                pg_conn.commit()
            pg_conn.close()
            return "postgres"
        except Exception:
            pass

    # Fallback SQLite
    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY,
            doc_id TEXT,
            text TEXT
        );
    """)
    conn.commit()
    conn.close()
    return "sqlite"


def save_chunks_text(chunks_data):
    storage_type = init_text_storage()
    if storage_type == "postgres":
        pg_conn = get_postgres_conn()
        if pg_conn:
            try:
                with pg_conn.cursor() as cur:
                    for c in chunks_data:
                        cur.execute("""
                            INSERT INTO chunks (id, doc_id, text)
                            VALUES (%s, %s, %s)
                            ON CONFLICT (id) DO UPDATE SET text = EXCLUDED.text, doc_id = EXCLUDED.doc_id;
                        """, (c["id"], c.get("doc_id", "doc_1"), c["text"]))
                    pg_conn.commit()
                pg_conn.close()
                return
            except Exception:
                pass

    conn = sqlite3.connect(SQLITE_DB_PATH)
    cursor = conn.cursor()
    for c in chunks_data:
        cursor.execute("""
            INSERT OR REPLACE INTO chunks (id, doc_id, text)
            VALUES (?, ?, ?);
        """, (c["id"], c.get("doc_id", "doc_1"), c["text"]))
    conn.commit()
    conn.close()


def get_chunk_text(chunk_id):
    pg_conn = get_postgres_conn()
    if pg_conn:
        try:
            with pg_conn.cursor() as cur:
                cur.execute("SELECT text FROM chunks WHERE id = %s;", (chunk_id,))
                row = cur.fetchone()
                if row:
                    return row[0]
        except Exception:
            pass
        finally:
            try:
                pg_conn.close()
            except Exception:
                pass

    if os.path.exists(SQLITE_DB_PATH):
        try:
            conn = sqlite3.connect(SQLITE_DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT text FROM chunks WHERE id = ?;", (chunk_id,))
            row = cursor.fetchone()
            conn.close()
            if row:
                return row[0]
        except Exception:
            pass
    return ""


def get_chroma_collection():
    try:
        client = chromadb.HttpClient(host="localhost", port=8000)
        client.heartbeat()
    except Exception:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
    return client.get_or_create_collection(name="rag_chunks")


def generate_embedding(text):
    if genai_client:
        try:
            from google.genai import types
            res = genai_client.models.embed_content(
                model="text-embedding-004",
                contents=text,
                config=types.EmbedContentConfig(output_dimensionality=384)
            )
            return res.embeddings[0].values
        except Exception:
            pass
    return None


def index():
    if not os.path.exists(CHUNKS_DIR):
        print(f"Thư mục {CHUNKS_DIR} không tồn tại.")
        return 0

    json_files = glob.glob(os.path.join(CHUNKS_DIR, "*.json"))
    if not json_files:
        print(f"Không tìm thấy file JSON nào tại {CHUNKS_DIR}.")
        return 0

    all_chunks = []
    chunk_counter = 0

    for file_path in json_files:
        doc_name = os.path.basename(file_path)
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                items = data if isinstance(data, list) else [data]
                for item in items:
                    chunk_counter += 1
                    c_id = item.get("id") or item.get("chunk_id") or f"chunk_{chunk_counter}"
                    text = item.get("text") or item.get("content") or item.get("chunk_text") or str(item)
                    all_chunks.append({
                        "id": str(c_id),
                        "doc_id": item.get("doc_id", doc_name),
                        "text": text
                    })
        except Exception as e:
            print(f"Lỗi đọc file {file_path}: {e}")

    if not all_chunks:
        print("Không có dữ liệu chunk hợp lệ.")
        return 0

    # 1. Lưu text vào Postgres/SQLite
    save_chunks_text(all_chunks)

    # 2. Lưu embeddings vào ChromaDB
    collection = get_chroma_collection()

    ids = []
    embeddings = []
    documents = []
    metadatas = []

    for c in all_chunks:
        ids.append(c["id"])
        documents.append(c["text"])
        metadatas.append({"doc_id": c["doc_id"]})
        emb = generate_embedding(c["text"])
        if emb:
            embeddings.append(emb)

    if len(embeddings) == len(ids) and len(embeddings) > 0:
        collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas)
    else:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

    print(f"Đã index thành công {len(all_chunks)} chunks.")
    return len(all_chunks)


def ask(question, k=3):
    collection = get_chroma_collection()
    q_emb = generate_embedding(question)

    if q_emb:
        results = collection.query(query_embeddings=[q_emb], n_results=k)
    else:
        results = collection.query(query_texts=[question], n_results=k)

    retrieved_ids = results.get("ids", [[]])[0]
    retrieved_texts = []

    for cid in retrieved_ids:
        text = get_chunk_text(cid)
        if text:
            retrieved_texts.append(text)

    context = "\n---\n".join(retrieved_texts)

    if not genai_client:
        return {
            "answer": "Thiếu GEMINI_API_KEY. Chỉ thực hiện Retrieval thành công.",
            "context": retrieved_texts,
            "ids": retrieved_ids
        }

    prompt = f"Dựa vào ngữ cảnh sau để trả lời câu hỏi.\n\nNgữ cảnh:\n{context}\n\nCâu hỏi: {question}"

    try:
        response = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        answer = response.text
    except Exception as e:
        answer = f"Lỗi gọi Gemini LLM: {e}"

    return {
        "answer": answer,
        "context": retrieved_texts,
        "ids": retrieved_ids
    }


def status():
    collection = get_chroma_collection()
    total_chunks = collection.count()

    doc_ids = set()
    pg_conn = get_postgres_conn()
    if pg_conn:
        try:
            with pg_conn.cursor() as cur:
                cur.execute("SELECT DISTINCT doc_id FROM chunks;")
                rows = cur.fetchall()
                doc_ids = {r[0] for r in rows}
        except Exception:
            pass
        finally:
            try:
                pg_conn.close()
            except Exception:
                pass
    elif os.path.exists(SQLITE_DB_PATH):
        try:
            conn = sqlite3.connect(SQLITE_DB_PATH)
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT doc_id FROM chunks;")
            rows = cursor.fetchall()
            doc_ids = {r[0] for r in rows}
            conn.close()
        except Exception:
            pass

    total_docs = len(doc_ids)
    res = {"documents": total_docs, "chunks": total_chunks}
    print(f"Status: {res}")
    return res


if __name__ == "__main__":
    print("Trạng thái hiện tại:")
    status()
