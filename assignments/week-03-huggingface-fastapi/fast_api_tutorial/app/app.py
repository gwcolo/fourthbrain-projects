from fastapi import FastAPI
from pydantic import BaseModel
import transformers

app = FastAPI()

# load local model
model_path = 'model/' + "t5-small" # fuller path: 'fast_api_tutorial/app/model/' + "t5-small"
pipeline =  transformers.pipeline( task = 'translation_XX_to_YY' , model=model_path)

# pydantic model that accepts a single string
class TextToTranslate(BaseModel):
    input_text: str

# pydantic model that accepts a list strings
class TextsToTranslate(BaseModel):
    input_texts: list


@app.get("/")
def index():
    return {"message": "Hello World"}

@app.get("/ping")
def ping():
    return {"message": "pong"}

@app.post("/echo")
def echo(text_to_translate: TextToTranslate):
    return {"message": text_to_translate.input_text}

@app.post("/translate")
def translate(text_to_translate: TextToTranslate):
    return {"message": pipeline(text_to_translate.input_text)[0]['translation_text'] }

@app.post("/batch_translate")
def batch_translate(texts_to_translate: TextsToTranslate):
    return {"message": [x['translation_text'] for x in pipeline( texts_to_translate.input_texts )] }



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)