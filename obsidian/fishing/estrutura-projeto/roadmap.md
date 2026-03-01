# 🚀 3️⃣ Roadmap Geral do Produto

---

# 🔹 FASE 1 – Motor de IA Offline (MVP Técnico)

Objetivo: medir peixe localmente.

### Features

- Extração de frames
    
- Detecção peixe
    
- Detecção régua
    
- Conversão pixel → cm
    
- Confidence score
    
- Retorno JSON
    

### Tasks

-  Criar dataset
    
-  Anotar imagens (fish + ruler)
    
-  Treinar YOLOv8n
    
-  Testar precisão
    
-  Implementar pipeline Python
    
-  Exportar modelo ONNX
    
-  Converter para TFLite
    

---

# 🔹 FASE 2 – App Mobile MVP

Objetivo: usuário enviar vídeo e medir offline.

### Features Participante

- Login
    
- Selecionar campeonato
    
- Gravar ou enviar vídeo
    
- Rodar IA offline
    
- Exibir resultado
    
- Salvar submissão local
    
- Sincronizar quando online
    

### Tasks

-  Criar projeto Flutter
    
-  Implementar autenticação
    
-  Implementar upload de vídeo
    
-  Integrar TFLite
    
-  Criar tela de resultado
    
-  Criar sistema de sync offline
    

---

# 🔹 FASE 3 – Backend SaaS

Objetivo: centralizar dados.

### Features Backend

- Cadastro usuário
    
- Cadastro campeonato
    
- Receber submissão
    
- Armazenar vídeo
    
- Salvar resultado IA
    
- Painel árbitro
    
- Aprovar / rejeitar
    

### Tasks

-  Criar FastAPI
    
-  Modelar banco PostgreSQL
    
-  Criar endpoints:
    
    - POST /login
        
    - POST /submit
        
    - GET /submissions
        
    - POST /review
        
-  Integrar storage S3
    
-  Criar painel web simples (React ou Next.js)
    

---

# 🔹 FASE 4 – Sistema de Treinamento Contínuo

Objetivo: melhorar IA com dados reais.

### Fluxo

1. Árbitro corrige medida
    
2. Backend salva correção
    
3. Dataset atualizado
    
4. Re-treinar modelo
    
5. Publicar nova versão
    
6. App baixa novo modelo
    

### Tasks

-  Criar tabela dataset_feedback
    
-  Criar pipeline de re-treinamento
    
-  Versionar modelos
    
-  Endpoint GET /model/version
    
-  Sistema de download de novo modelo no app
    

---

# 🔹 FASE 5 – Segurança Anti-Fraude

Features futuras:

- Detectar boca fora do zero
    
- Detectar edição de vídeo
    
- Detectar múltiplos peixes
    
- Detectar ausência de régua
    
- Detectar régua falsa
    

---

# 🗃️ 4️⃣ Estrutura do Banco PostgreSQL

## users

- id
    
- name
    
- email
    
- password_hash
    
- role
    

## championships

- id
    
- name
    
- start_date
    
- end_date
    

## submissions

- id
    
- user_id
    
- championship_id
    
- video_url
    
- ai_length_cm
    
- ai_confidence
    
- status
    
- final_length_cm
    
- reviewed_by
    
- reviewed_at
    
- created_at
    

## model_versions

- id
    
- version
    
- file_url
    
- created_at
    

---

# 🧠 5️⃣ Arquitetura da IA Offline

Pipeline:

Video  
 ↓  
Frame extraction  
 ↓  
Detect ruler  
 ↓  
Detect fish  
 ↓  
Align perspective (futuro)  
 ↓  
Calcular comprimento  
 ↓  
Confidence  
 ↓  
JSON

---

# 📐 6️⃣ Estratégia de Medição (Versão Inicial)

V1 simples:

- Detectar bounding box do peixe
    
- Detectar bounding box da régua
    
- Calcular proporção horizontal
    
- Ignorar OCR
    
- Ignorar perspectiva
    

Já entrega MVP funcional.

---

# 🔄 7️⃣ Estratégia Offline-First

App deve:

- Salvar tudo local (SQLite)
    
- Ter fila de sincronização
    
- Tentar enviar quando online
    
- Resolver conflito
    

---

# 🔥 8️⃣ Features Futuras Premium

- Ranking em tempo real
    
- Dashboard analytics
    
- Estatísticas por espécie
    
- Histórico do atleta
    
- Modo live tournament
    
- Detecção automática de espécie
    
- Integração com patrocinadores
    
- Assinatura SaaS organizador
    

---

# 📊 9️⃣ Ordem Recomendada de Execução

### 1️⃣ Treinar IA básica

### 2️⃣ Criar pipeline Python offline

### 3️⃣ Criar app Flutter básico

### 4️⃣ Criar backend simples

### 5️⃣ Integrar tudo

### 6️⃣ Melhorar precisão

### 7️⃣ Implementar anti-fraude