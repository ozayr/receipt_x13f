from enum import Enum
from typing import Any, Dict, List
from openai import OpenAI
from pydantic import BaseModel
from .system_promt import system_promt


class PaymentMethod(Enum):
    CASH = 'cash'
    CARD = 'card'
    BOTH = 'both'

class Receipt(BaseModel):
    date: str
    time: str
    items: List[str]
    prices: List[float]
    paymentMethod: PaymentMethod
    cardNumber: str
    cardType: str
    quantities: List[int]
    totalItems: int
    totalPrice: float
    businessName: str
    businessAddress: str
    businessPostalCode: str


class GPTReceiptProcessor:
    def __init__(self, api_key: str, model_name: str):
        self.client = OpenAI( api_key = api_key)
        self.model_name = model_name

    def extract_data_from_text(self, receipt_extract: str) -> Dict[str, Any]:
        completion = self.client.beta.chat.completions.parse(
        model=self.model_name,
        messages=[
            {"role": "system", "content": system_promt},
            {"role": "user", "content": receipt_extract},
        ],
        response_format=Receipt,
    )
        return completion.choices[0].message.parsed.model_dump(mode='json')