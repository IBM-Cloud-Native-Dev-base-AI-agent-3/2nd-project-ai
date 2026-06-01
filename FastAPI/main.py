from fastapi import FastAPI
from routers import hope_housing

app = FastAPI()

app.include_router(hope_housing.router, prefix="/predict", tags=["희망하우징"])

@app.get("/health")
def health():
    return {"status": "ok"}