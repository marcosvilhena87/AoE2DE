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

