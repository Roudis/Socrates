from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Socrates API is running. Use Chainlit for the UI."}

@app.get("/health")
def health_check():
    return {"status": "ok"}