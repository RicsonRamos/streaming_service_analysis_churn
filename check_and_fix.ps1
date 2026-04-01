# Entrar no ambiente virtual
Write-Host "Ativando .venv..." -ForegroundColor Cyan

# Corrigir imports
Write-Host "`n[1/5] Corrigindo imports com isort..." -ForegroundColor Cyan
isort src/ app/ tests/

# Formatar código com black
Write-Host "`n[2/5] Formatando código com black..." -ForegroundColor Cyan
black src/ app/ tests/ --line-length 79

# Checagem de estilo com flake8
Write-Host "`n[3/5] Checando estilo com flake8..." -ForegroundColor Cyan
flake8 src/ app/ tests/

# Checagem de tipos com mypy
Write-Host "`n[4/5] Checando tipos com mypy..." -ForegroundColor Cyan
mypy src/

# Rodar testes com pytest
Write-Host "`n[5/5] Rodando testes com pytest..." -ForegroundColor Cyan
pytest tests/ -v

Write-Host "`n✅ Todas as verificações concluídas!" -ForegroundColor Green