from src.config.loader import ConfigLoader
try:
    cfg = ConfigLoader().load_all()
    print("✅ Configurações carregadas!")
    print(f"Caminho do modelo: {cfg['paths']['models']['churn_model']}")
except Exception as e:
    print(f"❌ Erro na configuração: {e}")