# Agente Substituto para Age of Empires II: DE (Campanhas)

Este projeto implementa um **agente substituto** capaz de jogar missões de campanha do *Age of Empires II: Definitive Edition* imitando o estilo do jogador humano.  
O agente aprende observando múltiplas partidas humanas e toma **decisões de alto nível** (macro), como build orders, escolhas de tecnologia, movimentação de tropas e execução de objetivos de cenário.

---

## ✨ Funcionalidades
- **Imitação de estilo humano:** treina a partir dos replays ou inputs do jogador.  
- **Ações em nível macro:** treinar vilarejos, subir de idade, construir edifícios, mover exércitos, capturar relíquias etc.  
- **Execução automática:** transforma ações previstas em comandos no jogo por meio de rotinas de hotkeys e scripts.  
- **Generalização:** capaz de completar diferentes missões de campanha, mesmo em estados não vistos durante o treino.  

---

## 📂 Estrutura do Repositório
```
├── data/               # Partidas gravadas e trajetórias em formato YAML/JSON
├── assets/             # Templates de HUD, ícones e OCR
├── src/
│   ├── agent/          # Modelo de política (Transformer, imitation learning)
│   ├── env/            # Conector com o jogo (leitura de estado e execução de ações)
│   └── utils/          # Funções auxiliares (parser, logging, métricas)
├── notebooks/          # Experimentos e análises
├── scripts/            # Automação de treino e execução
└── README.md
```

---

## ⚙️ Como Funciona
1. **Coleta de Dados:**  
   - Extração de replays do AoE2DE.  
   - Log de recursos e ações via OCR e hotkeys.  
   - Conversão de micro-ações em macros de alto nível.  

2. **Treinamento do Agente:**  
   - **Behavioral Cloning (BC):** modelo imita diretamente as decisões do jogador.  
   - **DAgger:** corrige estados fora da distribuição.  
   - **Offline RL (opcional):** melhora robustez adicionando recompensas de objetivos (ex.: relíquias capturadas).  

3. **Execução:**  
   - O agente recebe estado resumido (recursos, população, objetivos).  
   - Gera uma ação macro (ex.: “treinar 5 arqueiros”).  
   - O executor envia comandos ao jogo usando hotkeys/script.  

---

## 🚀 Instalação
Pré-requisitos:
- Python 3.9+  
- PyTorch  
- OpenCV (OCR)  
- Tesseract OCR  
- AutoHotkey (Windows) ou alternativa para emissão de hotkeys  

Instalação:
```bash
cd aoe2-agent-substituto
pip install -r requirements.txt
```

---

## ▶️ Uso
1. Colete replays e salve em `data/`.  
2. Pré-processe dados:
```bash
python scripts/preprocess_replays.py --input data/replays --output data/episodes
```
3. Treine o agente:
```bash
python scripts/train_agent.py --config configs/bc.yaml
```
4. Execute em campanha:
```bash
python scripts/run_agent.py --mission "William Wallace 4"
```

---

## 📊 Avaliação
- **Métricas:**  
  - % de missões concluídas.  
  - Tempo médio até o objetivo.  
  - Precisão de imitação (ações previstas vs humanas).  
- **Logs:** salvos em `logs/` com TensorBoard para visualização.  

---

## 📌 Roadmap
- [x] Coleta de dados de partidas próprias  
- [x] Definição de catálogo de ações macro  
- [ ] Treinamento inicial por Behavioral Cloning  
- [ ] Integração com AutoHotkey  
- [ ] DAgger + Offline RL  
- [ ] Suporte a múltiplas campanhas  

---

## 🤝 Contribuições
Contribuições são bem-vindas! Abra uma **issue** para discutir ideias ou envie um **pull request**.

---

## 📜 Licença
Este projeto está sob a licença MIT – veja o arquivo [LICENSE](LICENSE) para mais detalhes.
