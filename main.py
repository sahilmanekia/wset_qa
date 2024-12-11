from fastapi import FastAPI, UploadFile, HTTPException
from PyPDF2 import PdfReader
from transformers import pipeline

app = FastAPI()

# Load the question generation pipeline
#question_generator = pipeline("question-generation")
question_generator = pipeline("text2text-generation", model="valhalla/t5-small-qg-prepend")

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    try:
        reader = PdfReader(pdf_file)
        text = ""
        for page in reader.pages:
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

    # Generate questions (limit to 5 for simplicity)
    #questions = question_generator(pdf_text, max_length=512)[:5]
    questions = question_generator("generate questions:" + pdf_text, max_length=512)[:5]

    # Format the output
    formatted_questions = [
        {"question":q["generated_text"] for q in questions}
        #{"question": q["question"], "answer": q["answer"]} for q in questions
    ]
    return {"questions": formatted_questions}
