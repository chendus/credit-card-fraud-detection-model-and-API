from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd

app = FastAPI()

class ScoringItem(BaseModel):
	V1: float
	V2: float #0.01,
	V3: float #0.001,
	V4: float #/0.4,
	V5: float #/0.5,
	V6: float #/0.6,
	V7: float #/0.7,
	V8: float #/0.8,
	V9: float #/0.9,
	V10: float #/0.11,
	V11: float #/0.22,
	V12: float #/0.23,
	V13: float #/0.43,
	V14: float #/0.55,
	V15: float #/0.66,
	V16: float #/0.22,
	V17: float #/0.33,
	V18: float #/0.11,
	V19: float #/0.67,
	V20: float #/0.45,
	V21: float #/0.22,
	V22: float #/0.551,
	V23: float #/0.003,
	V24: float #/0.002,
	V25: float #/0.44,
	V26: float #/0.921,
	V27: float #/0.234,
	V28: float #/0.65,
	Amount: float #/22

with open('dt_modell.pkl', 'rb') as f:
	model = pickle.load(f)

@app.post('/')
async def scoring_endpoint(item:ScoringItem):
	df = pd.DataFrame([item.dict().values()], columns=item.dict().keys())
	yhat = model.predict(df)
	return {"prediction":int(yhat)}
