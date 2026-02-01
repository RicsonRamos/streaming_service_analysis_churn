import sys
import logging
from src.config.loader import ConfigLoader
from src.pipelines.train import train_pipeline # Seu script de treino atualizado

logging.basicConfig(level=logging.INFO)

def main():
    if len(sys.argv) < 2:
        print("Uso: python run.py [train|predict]")
        return

    action = sys.argv[1]
    loader = ConfigLoader()
    cfg = loader.load_all()

    if action == "train":
        print("🚀 Iniciando Pipeline de Treinamento...")
        train_pipeline() 
    elif action == "predict":
        print("🔮 Modo de Predição (Simulação)...")
        # Exemplo de como usar o pipeline modular no futuro
        # pipeline = ChurnPipeline(cfg)
        # ... lógica de predição ...
    else:
        print("Comando inválido.")

if __name__ == "__main__":
    main()