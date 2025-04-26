from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from docx import Document
import tensorflow as tf
import joblib
import uvicorn
import re
from sklearn.feature_extraction import text

# Load model and vectorizer
model = tf.keras.models.load_model('keyword_model.keras')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

app = FastAPI()

# Allow CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],   # Only one time
    allow_headers=["*"],
)

@app.post("/extract-keywords/")
async def extract_keywords(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Save temporarily
    with open("temp_uploaded.docx", "wb") as f:
        f.write(contents)

    document = Document("temp_uploaded.docx")
    text_content = "\n".join([para.text for para in document.paragraphs])

    stop_words = text.ENGLISH_STOP_WORDS
    words = re.findall(r'\b\w+\b', text_content.lower())
    filtered_words = [word for word in words if word not in stop_words and len(word) > 2]

    word_vectors = vectorizer.transform(filtered_words).toarray()
    predictions = model.predict(word_vectors)

    extracted = []
    for word, pred in zip(filtered_words, predictions):
        if pred[0] > 0.5:
            extracted.append(word)

    return {"keywords": list(set(extracted))}  # Remove duplicates

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
