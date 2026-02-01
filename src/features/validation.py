from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List
import pandas as pd

class CustomerSchema(BaseModel):
    """Contrato de dados para 1 cliente."""
    # Configuração oficial do Pydantic V2
    model_config = ConfigDict(
        coerce_numbers_to_str=False, 
        extra='allow',
        strict=False # Permite que o Pydantic converta float 0.0 para int 0
    )

    Age: int = Field(gt=17, lt=100)
    Subscription_Length: int = Field(ge=1)
    Monthly_Spend: float = Field(ge=0)
    Support_Tickets_Raised: int = Field(ge=0)
    Estimated_LTV: float = Field(ge=0)
    Engagement_Score: float = Field(ge=0, le=10) 
    Gender: str
    Region: str
    Payment_Method: str

    @field_validator('Gender')
    @classmethod
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError("Gender deve ser Male ou Female")
        return v
from collections import Counter

def validate_dataframe(df: pd.DataFrame):
    """Valida o DataFrame e reporta a causa raiz dos erros."""
    errors = []
    sample_errors = []

    for i, record in enumerate(df.to_dict(orient="records")):
        try:
            CustomerSchema(**record)
        except Exception as e:
            # Captura a mensagem de erro do Pydantic
            err_msg = str(e).split('\n')[0] 
            errors.append(err_msg)
            if len(sample_errors) < 5: # Guarda os 5 primeiros exemplos reais
                sample_errors.append(f"Linha {i}: {e}")
    
    if errors:
        print("\n❌ RELATÓRIO DE ERROS DE VALIDAÇÃO")
        print("-" * 30)
        # Conta quantas vezes cada tipo de erro apareceu
        summary = Counter(errors)
        for err, count in summary.items():
            print(f"- {count} ocorrências de: {err}")
        
        print("\n🔍 EXEMPLO DETALHADO:")
        print(sample_errors[0])
        return False
    
    print("✅ Dados validados com sucesso!")
    return True
