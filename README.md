# 🏰 Age of Empires Human-Guided RL

Este repositório explora **Aprendizado por Reforço (Reinforcement Learning)** combinado com **Aprendizado por Demonstração (Imitation Learning / Inverse Reinforcement Learning)** aplicado ao jogo **Age of Empires II: Definitive Edition**.

A ideia central é **observar a performance humana** em *replays* para:
- Extrair *trajectories* (estado → ação macro).
- Treinar uma política inicial por **Imitação** (Behavioral Cloning / GAIL).
- Refinar essa política com **Reinforcement Learning** em ambiente controlado ou diretamente no AoE2.

---

## 🎯 Objetivos

- Aprender *build orders* e transições a partir de replays humanos.
- Criar agentes capazes de executar **estratégias plausíveis**, em vez de micro perfeito.
- Evoluir de **demonstrações** para **self-play com recompensas densas**.

---

## 🛠️ Stack Técnica

- **Parsing de Replays**: [aoc-mgz](https://github.com/happyleavesaoc/aoc-mgz) (suporta `.aoe2record`).
- **RL / IL**: [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3), [imitation](https://github.com/HumanCompatibleAI/imitation).
- **Ambientes**:
  - Prototipagem: [µRTS](https://github.com/vwxyzjn/gym-microrts) (RTS minimalista com interface Gym).
  - Execução real: automação via OCR/hotkeys ou [openage](https://github.com/SFTtech/openage).
- **Visualização**: [CaptureAge](https://www.captureage.com/) para depuração.

---

## 📂 Estrutura

```
├── data/                # Replays e trajectories processadas
├── notebooks/           # Análises e protótipos
├── src/
│   ├── parsing/         # Conversores .aoe2record -> dataset
│   ├── macros/          # Definição de ações em nível macro
│   ├── policies/        # Modelos (BC, GAIL, PPO, SAC)
│   ├── envs/            # Wrappers (µRTS, AoE2 OCR, openage)
│   └── rewards/         # Funções de recompensa densas
└── README.md
```

---

## 🚀 Pipeline

1. **Coleta** de replays humanos (`.aoe2record`).
2. **Parsing** → estados agregados (recursos, upgrades, army size, timings).
3. **Mapeamento de Ações Macro** → `Train(Villager)`, `AgeUp(Feudal)`, `Build(ArcheryRange)`, etc.
4. **Imitação**:
   - *Behavioral Cloning* (supervisionado).
   - *GAIL* para maior robustez.
5. **Reinforcement Learning**:
   - *Offline RL*: CQL/IQL sobre os replays.
   - *Online RL*: PPO/SAC com recompensas densas (eco, army, timings).
6. **Validação**:
   - Timings (Feudal, Castle, Imperial).
   - Uptime de TCs e Idle Villagers.
   - Execução de *build orders* plausíveis.

---

## 📊 Exemplo de Feature (estado)

```json
{
  "time": 410,
  "food": 350,
  "wood": 180,
  "gold": 0,
  "villagers": 19,
  "idle_tc": 0,
  "age": "Dark",
  "military": {"maa": 2, "archers": 0},
  "upgrades": ["loom"],
  "enemy_pressure": 0.2
}
```

---

## 📦 Instalação

```bash
git clone https://github.com/SEU_USUARIO/aoe2-human-rl.git
cd aoe2-human-rl
pip install -r requirements.txt
```

Requisitos principais:
- Python 3.9+
- `stable-baselines3`
- `imitation`
- `torch`
- `aoc-mgz`
- `gym-microrts` (opcional)

---

## 📅 Roadmap

- [ ] Parsing inicial de replays → CSV/JSON de trajectories.  
- [ ] Definição de espaço de ação macro.  
- [ ] Treino com Behavioral Cloning.  
- [ ] GAIL para robustez.  
- [ ] Recompensas densas macro-aware.  
- [ ] Integração com openage.  
- [ ] Self-play híbrido IL + RL.  

---

## 📚 Referências

- Silver et al., *Mastering the game of Go with deep neural networks and tree search*, Nature (2016).
- Ho & Ermon, *Generative Adversarial Imitation Learning* (2016).
- Gym-MicroRTS: https://github.com/vwxyzjn/gym-microrts
- Projeto openage: https://github.com/SFTtech/openage
- Biblioteca aoc-mgz: https://github.com/happyleavesaoc/aoc-mgz

---

## 🤝 Contribuições

Pull requests e sugestões são bem-vindos! Se quiser discutir ideias, abra uma issue.
