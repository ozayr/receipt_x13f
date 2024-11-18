import json
from typing_extensions import TypedDict
from enum import Enum
import google.generativeai as genai
from typing import Any, Dict, List
from .system_promt import system_promt


class PaymentMethod(Enum):
    CASH = "cash"
    CARD = "card"
    BOTH = "both"


class Receipt(TypedDict):
    date: str
    time: str
    items: List[str]
    prices: List[float]
    paymentMethod: PaymentMethod
    quantities: List[int]
    totalItems: int
    totalPrice: float
    businessName: str
    businessAddress: str
    businessPostalCode: str


class GeminiReceiptProcessor:

    def __init__(self, api_key: str, model_name: str):
        genai.configure(api_key=api_key)

        self.model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_promt,
            generation_config={
                "response_mime_type": "application/json",
                "response_schema": Receipt,
            },
        )

    def extract_data_from_image(self, image_path: str) -> Dict[str, Any]:
        image_file = genai.upload_file(image_path)
        prompt = "Extract the following data from the receipt"
        response = self.model.generate_content([prompt, image_file])
        return json.loads(response.text)

    def extract_data_from_text(self, text: str) -> Dict[str, Any]:
        prompt = "Extract the following data from the receipt"
        response = self.model.generate_content([prompt, text])
        return json.loads(response.text)
