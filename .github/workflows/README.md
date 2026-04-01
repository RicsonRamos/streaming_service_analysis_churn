# GitHub Actions - Configuração

## Visão Geral

Este projeto usa pipelines desacoplados:

- CI (rápido): valida código e lógica com mocks
- PR Check: validação rápida em pull requests
- Train (real): treinamento com MLflow/DVC (manual ou agendado)
- Data Drift: monitoramento semanal

---

##  Secrets Necessários

Settings > Secrets and variables > Actions

### Docker (opcional)
- DOCKER_USERNAME
- DOCKER_PASSWORD

### AWS / DVC (apenas pipelines reais)
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY

### Deploy (opcional)
- STAGING_HOST
- STAGING_USER
- STAGING_SSH_KEY

### Notificações (opcional)
- SLACK_WEBHOOK

---

##  Variables

Settings > Secrets and variables > Actions > Variables

- DVC_REMOTE: ex: s3://my-bucket/dvc-storage

---

##  Importante

O CI principal NÃO usa:

- MLflow
- DVC
- dados reais

Esses são usados apenas em:

- train.yml
- data-drift.yml

Isso garante:

- pipelines rápidos
- estabilidade
- baixo custo

