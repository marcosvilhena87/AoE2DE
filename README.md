# AoE2 Behavioral Cloning

Treinamento de agentes para **Age of Empires II: Definitive Edition (AoE2DE)** usando **Behavioral Cloning (BC)** a partir de replays (`.aoe2record`).  
O objetivo é aprender políticas de alto nível (macro) a partir de partidas humanas em campanhas, evitando o nível de micro (cliques/tiles).

---

## 🚀 Visão geral

1. **Coletar dados**: extrair episódios de decisões a partir de replays `.aoe2record`.  
2. **Pré-processar**: reconstruir estados, mapear comandos para ações discretas, aplicar máscara de ações válidas.  
3. **Treinar modelo supervisionado**: política πθ(a|s) com *cross-entropy* mascarada.  
4. **Validar**: medir acurácia top-k, taxa de ações inválidas, métricas específicas de campanha (tempo até Feudal, composição de exército, etc.).  
5. **Inferência**: usar a política treinada em rollouts offline ou embutida num simulador.

---

## 📂 Estrutura de pastas
```
aoe2-bc/
  data/
    replays/                 # arquivos .aoe2record
    episodes/                # episódios pré-processados (*.jsonl)
  src/
    parsers/ao2record_parser.py
    data/dataset.py
    models/policy.py
    utils/action_space.py
    utils/mask.py
  preprocess.py
  train_bc.py
  rollout_offline.py
  README.md
```

---

## ⚙️ Pré-processamento

```bash
python preprocess.py   --in data/replays   --out data/episodes   --action-space v1   --drop-bad
```

Saída: `*.jsonl` contendo episódios com:
- `state`: recursos, população, techs, edifícios, tempo, etc.  
- `action_id`: índice discreto da ação executada.  
- `valid_action_mask`: quais ações eram possíveis naquele estado.  

---

## 🧠 Treinamento

```bash
python train_bc.py   --train data/episodes/train.jsonl   --val data/episodes/val.jsonl   --n-actions 128
```

Modelo: MLP ou Transformer pequeno, com máscara de ações aplicada em treino e inferência.  
Perda: *CrossEntropy* mascarada.  
Métricas:  
- **Acc@1**  
- **Acc@k** (k=3/5)  
- **Invalid-rate** (ações previstas mas inválidas)  

---

## 🎮 Rollout Offline

```bash
python rollout_offline.py   --policy checkpoints/best.pt   --episodes data/episodes/val.jsonl
```

Permite avaliar se a política reproduz build orders e timings próximos aos humanos.  

---

## 🛠️ Espaço de ações (v1)

- **Eco**: `Train(Villager)`, `Build(Farm)`, `Send(Vils→Wood)`, `AdvanceAge(Feudal)`, etc.  
- **Militar/Tech**: `Train(Militia)`, `Research(ManAtArms)`, `AttackMove(Sector3)` …  
- **Objetivos campanha**: `Capture(Relic)`, `Destroy(TargetTowerWest)` …  

Posições são discretizadas em *setores* (ex.: grade 3×3).  

---

## 📊 Roadmap

- [x] Parser `.aoe2record` para JSONL de episódios  
- [x] Dataset + máscara de ações válidas  
- [x] MLP Policy com PyTorch  
- [ ] Transformer Policy (contexto build order)  
- [ ] CLI de rollouts e métricas por campanha  
- [ ] Suporte a *action head* fatorada (verbo + argumento)  
- [ ] DAgger offline (agregar divergências)  

---

## 🔗 Referências
- [aoe2record](https://github.com/happyleavesaoc/aoe2record)  
- [AoE2ScenarioParser](https://github.com/KSneijders/AoE2ScenarioParser)  
- SethBling’s [MarI/O](https://www.youtube.com/watch?v=qv6UVOQ0F44) (inspiração de imitation learning)  
