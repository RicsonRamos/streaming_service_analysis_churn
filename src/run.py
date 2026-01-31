import sys
import os
import subprocess
from pathlib import Path
from src.config.loader import ConfigLoader

# Carrega a configuração central
cfg = ConfigLoader().load_all()

def train_pipeline():
    """
    Treina o modelo e salva o pipeline completo.
    """
    from src.pipelines.train import train_pipeline as tp
    tp(cfg)  # usa o cfg global

def run_app():
    """
    Roda o dashboard Streamlit garantindo que o PYTHONPATH esteja correto.
    """
    # 1. Caminhos base
    root_dir = Path(__file__).parent.parent
    app_path = Path(__file__).parent / "app" / "dashboard.py" # Onde você moveu o arquivo

    if not app_path.exists():
        raise FileNotFoundError(f"Arquivo do app não encontrado em {app_path}")

    # 2. Injetar a raiz do projeto no PYTHONPATH do ambiente
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root_dir) + os.pathsep + env.get("PYTHONPATH", "")

    # 3. Executar o Streamlit
    print(f"🚀 Iniciando Dashboard a partir de: {root_dir}")
    try:
        subprocess.run(
            ["streamlit", "run", str(app_path)],
            cwd=root_dir,  # Define a raiz como diretório de trabalho
            env=env,       # Passa o PYTHONPATH configurado
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro ao rodar o Streamlit: {e}")

def run_train():
    """
    Executa o pipeline de treino do modelo.
    """
    print("Iniciando treinamento...")
    train_pipeline()  # não passa cfg, já é usado dentro
    print("Treinamento concluído.")

def main():
    """
    Ponto de entrada da aplicação.
    Uso:
        python -m src.run train   # para treinar o modelo
        python -m src.run app     # para rodar o Streamlit
    """
    if len(sys.argv) < 2:
        print("Use: python -m src.run [train|app]")
        return

    mode = sys.argv[1].lower()

    if mode == "train":
        run_train()
    elif mode == "app":
        run_app()  # não passa cfg
    else:
        print(f"Modo desconhecido: {mode}. Use 'train' ou 'app'.")

if __name__ == "__main__":
    main()
