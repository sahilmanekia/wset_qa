from fastapi import FastAPI, UploadFile, HTTPException
from PyPDF2 import PdfReader
from transformers import pipeline
import uvicorn
import os

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, Render!"}

# Load the question generation pipeline
question_generator = pipeline("text2text-generation", model="t5-small")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file, max_pages=10):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages[:max_pages]:
            text += page.extract_text()
        return text
    except Exception as e:
        raise HTTPException(status_code=400, detail="Error reading PDF file.")

# Endpoint to generate questions
@app.post("/generate-questions/")
async def generate_questions(pdf: UploadFile):
    if pdf.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Please upload a valid PDF file.")

    # Extract text from the uploaded PDF
    pdf_text = extract_text_from_pdf(pdf.file)
    if not pdf_text.strip():
        raise HTTPException(status_code=400, detail="PDF contains no readable text.")

    # Generate questions (limit to 5 for simplicity)
    questions = question_generator("generate questions:" + pdf_text, max_length=512)[:5]

    # Format the output
    formatted_questions = [{"question": q["generated_text"]} for q in questions]
    return {"questions": formatted_questions}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
