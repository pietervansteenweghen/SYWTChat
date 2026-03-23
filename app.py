import os
import json
import pdfplumber
import anthropic
from flask import Flask, render_template, request, Response, stream_with_context, send_file
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

PDF_PATH = os.path.join(os.path.dirname(__file__), "So You Want To (1).pdf")


def extract_pdf_text(path):
    text_parts = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, 1):
            page_text = page.extract_text()
            if page_text:
                text_parts.append(f"--- Page {i} ---\n{page_text}")
    return "\n\n".join(text_parts)


print("Loading PDF document...")
PDF_CONTENT = extract_pdf_text(PDF_PATH)
print(f"Loaded {len(PDF_CONTENT):,} characters from PDF.")

SYSTEM_PROMPT = (
    "You are a helpful and friendly career advisor for PhD students and researchers "
    "working with Jesse Thaler at MIT. You answer questions based on the career "
    "advice document 'So You Want To...' written by Jesse Thaler. Be specific, "
    "practical, and encouraging. If the question isn't directly covered in the "
    "document, draw on your general knowledge but note that it goes beyond the "
    "document's content. Keep answers clear and conversational.\n\n"
    "IMPORTANT: When referencing specific content from the document, always cite "
    "the page number using the format (page X) — for example: (page 4). This "
    "helps users navigate directly to that section in the original document.\n\n"
    "Format your responses using markdown: use **bold** for emphasis, bullet lists "
    "where appropriate, and clear paragraph breaks.\n\n"
    "CAREER ADVICE DOCUMENT:\n"
    f"{PDF_CONTENT}"
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/pdf")
def serve_pdf():
    return send_file(PDF_PATH, mimetype="application/pdf")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    messages = data.get("messages", [])

    def generate():
        try:
            with client.messages.stream(
                model="claude-opus-4-6",
                max_tokens=1024,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    yield f"data: {json.dumps({'text': text})}\n\n"
            yield f"data: {json.dumps({'done': True})}\n\n"
        except anthropic.APIError as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    app.run(debug=True, port=5000)
