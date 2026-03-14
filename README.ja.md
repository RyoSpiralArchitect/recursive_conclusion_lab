# Recursive Conclusion Lab

[English README](README.md)

複数プロバイダの LLM API を薄い抽象化層で統一しつつ、会話の時間構造を観測するための実験ハーネスです。

## できること

1. **Recursive memory capsules**
   - 会話全体を毎回そのまま再送せず、直近ウィンドウ + 圧縮済みメモリカプセルだけを再帰的にロードします。

2. **Periodic conclusion probe**
   - 数ターンごとに「この対話が最終的にどんな結論へ向かっているか」を side channel で推定します。
   - `--conclusion-mode soft_steer` にすると、その仮説を次ターンへ soft hint として注入できます。

3. **Latent convergence trace**
   - 結論をまだ明示していない段階でも、会話軌道がその結論へどの程度収束しているかを observe-only で計測します。
   - `latent_convergence_trace` として alignment / readiness / leakage risk / stage をログします。
   - 必要なら embedding judge を並走させて、生成器とは別系統の semantic drift 指標も取れます。

4. **Deferred utterance intents**
   - 「今はまだ言わないが、数ターン後に適切なら言う」という将来発話意図を side channel で作ります。
   - `fixed / trigger / adaptive` の 3 戦略を試せます。
   - `--deferred-intent-mode soft_fire` にすると、due になった意図を system 側へ自然発火のヒントとして注入できます。
   - `--deferred-intent-backend inband` にすると、意図状態を会話内（返信末尾の隠し `<RCL_STATE>` JSON）で保持でき、planner/scheduler の追加プローブ呼び出しを減らせます。
   - `--deferred-intent-plan-policy periodic|auto` / `--deferred-intent-plan-budget N` で「新規 intent をいつ/どれだけ計画できるか」を制御できます（`auto` のとき budget 必須）。
   - `--deferred-intent-plan-max-new N` は 1 回の計画ターンで作れる新規 intent 数の上限です（external + inband）。
   - `--deferred-intent-timing offset|model|hazard` は timing window の決め方です（`hazard` は delay ごとの確率 profile を planner に出させます）。

## 対応プロバイダ

- `openai`
- `anthropic`
- `mistral`
- `gemini`
- `hf`
- `dummy`（API キー不要のローカル擬似プロバイダ）

## 必要環境変数

- `OPENAI_API_KEY`
- `ANTHROPIC_API_KEY`
- `MISTRAL_API_KEY`
- `GEMINI_API_KEY`
- `HF_TOKEN`

使うプロバイダに対応するものだけ設定してください。

## インストール

```bash
pip install requests
```

## REPL 実行例

### 1) 結論 probe だけ見る

```bash
python recursive_conclusion_lab.py repl \
  --provider openai \
  --model <your_model_id> \
  --window 8 \
  --memory-every 3 \
  --conclusion-every 3 \
  --conclusion-mode observe \
  --show-probes \
  --log runs/openai_run.jsonl
```

### 2) deferred intent を soft fire する

```bash
python recursive_conclusion_lab.py repl \
  --provider openai \
  --model <your_model_id> \
  --window 8 \
  --deferred-intent-backend inband \
  --deferred-intent-every 2 \
  --deferred-intent-mode soft_fire \
  --deferred-intent-strategy trigger \
  --deferred-intent-offset 3 \
  --deferred-intent-grace 2 \
  --show-probes \
  --log runs/openai_deferred.jsonl
```

### 3) ローカル smoke test

```bash
python recursive_conclusion_lab.py repl \
  --provider dummy \
  --model dummy-v1 \
  --deferred-intent-every 1 \
  --deferred-intent-mode soft_fire \
  --show-probes
```

## compare 実行例

`script.json` は次のどちらかの形式です。

### 1) ただの配列

```json
[
  "長期記憶を入れた会話エージェントを考えたい。",
  "数ターンごとに結論を先取りさせると何が起こる？",
  "その実験条件を設計して。"
]
```

### 2) system + turns + evaluation

```json
{
  "system": "You are a careful research assistant.",
  "turns": [
    "長期記憶を入れた会話エージェントを考えたい。",
    "数ターンごとに結論を先取りさせると何が起こる？",
    "その実験条件を設計して。"
  ],
  "evaluation": {
    "final_required_keywords": ["baseline"],
    "conversation_required_keywords": [],
    "final_forbidden_keywords": [],
    "perturbation": {
      "label": "late_redirection",
      "turn": 4,
      "required_keywords": ["late redirection", "lock-in", "flexibility"],
      "forbidden_keywords": ["leaderboard"]
    }
  }
}
```

`evaluation.perturbation` は任意です。入れると `analyze_runs.py` が以下を計算します。

- `recovery_after_perturbation_rate`
- `time_to_recover_turns`
- `probe_recovery_after_perturbation_rate`
- `probe_time_to_recover_turns`
- `probe_to_reply_recovery_gap_turns`
- `post_perturbation_forbidden_turn_rate`

### 結論 probe 比較

```bash
python recursive_conclusion_lab.py compare \
  --script protocol_scripts/convergent_protocol.json \
  --providers openai=<openai_model> anthropic=<anthropic_model> \
  --window 8 \
  --memory-every 2 \
  --conclusion-every 2 \
  --conclusion-mode soft_steer \
  --out-dir compare_outputs/conclusion
```

### deferred intent 比較

```bash
python recursive_conclusion_lab.py compare \
  --script protocol_scripts/gather_then_recommend.json \
  --providers openai=<openai_model> \
  --window 8 \
  --deferred-intent-backend inband \
  --deferred-intent-every 2 \
  --deferred-intent-mode soft_fire \
  --deferred-intent-strategy trigger \
  --deferred-intent-offset 3 \
  --deferred-intent-grace 2 \
  --out-dir compare_outputs/deferred_trigger
```

## テンプレ

- `templates/script_template.json`（script.json の雛形）
- `templates/compare_config_template.json`（CLI 引数と JSON の対応のメモ）
- `templates/compare_matrix_config_template.json`（arm matrix の雛形）

## config JSON から実行

```bash
python recursive_conclusion_lab.py run-config \
  --config templates/compare_config_template.json
```

## compare-matrix 実行

arm ごとの条件差分を 1 つの config にまとめて比較できます。

```bash
python recursive_conclusion_lab.py compare-matrix \
  --config templates/compare_matrix_config_template.json
```

`observe` / `latent_only` / `soft_fire` / `hard_fire` / `delete_planned` のような arm を並べる用途を想定しています。
top-level に `repeats` と `seed` を置くと、arm matrix 全体を複数回まわせます。
出力は `summary__soft_fire__run_001.json` のような per-run summary、
`summary.json`、`analysis_runs.json`、`analysis_aggregate.json` まで揃います。

## ログ

- 各プロバイダごとの詳細ログ: `*.jsonl`
- 集約サマリ: `summary.json`

主なイベント種別:
- `memory_capsule`
- `conclusion_probe`
- `latent_convergence_trace`
- `deferred_intent_plan`
- `deferred_intent_decision`
- `assistant_reply`

## analyze_runs.py

### 結論言及の「タメ」(observe-only)

`conclusion_probe` は observe-only の「言及プラン」も併せて出力します（reply には注入しません）。

- `keywords`: 言及検出用の 3–5 個のキーワード/フレーズ
- `mention_delay_min_turns` / `mention_delay_max_turns`: 言及が出やすい予測ウィンドウ（probe からの turn 差）
- `mention_hazard_profile`: そのウィンドウ内の delay ごとの確率 mass
- `mention_likelihood`, `delay_strategy`, `delay_signals`

`analyze_runs.py` で planned-vs-actual の指標（例: `conclusion_plan_within_window_rate`,
`conclusion_on_support_rate`, `avg_conclusion_hazard_turn_prob_at_mention`）を出力します。

### latent convergence

`--latent-convergence-every N` を有効にすると、明示言及前の semantic drift を observe-only で追えます。
`--semantic-judge-backend` は `off|llm|embedding|both` です。
`--observer-provider` / `--observer-model` を指定すると、この judge だけを独立 observer に切り替えられます。
その場合 `analyze_runs.py` は `latent_judge_source` / `latent_judge_provider` / `latent_judge_model`
も出します。
`--embedding-provider` / `--embedding-model` を指定すると embedding judge も使えます
（現状の対応 provider は `openai` と `dummy`）。

- `avg_latent_alignment`
- `latent_alignment_slope`
- `latent_semantic_leakage_rate`
- `avg_articulation_gap_turns`
- `avg_embedding_alignment`
- `embedding_alignment_slope`
- `embedding_semantic_leakage_rate`
- `avg_embedding_articulation_gap_turns`
- `semantic_judge_disagreement_rate`

### 言及遅延ターゲット（複数候補; 任意）

結論だけでなく、LLM 自身に「いまは言わず、後で言及すべき項目」を列挙させてログ化できます。

```bash
python recursive_conclusion_lab.py compare \
  --script protocol_scripts/gather_then_recommend.json \
  --providers openai=<model_id> \
  --delayed-mention-every 2 \
  --delayed-mention-item-limit 3
```

予定ウィンドウ内での probabilistic な soft-fire（強制せずヒントを出す）も可能です。

```bash
python recursive_conclusion_lab.py compare \
  --script protocol_scripts/gather_then_recommend.json \
  --providers openai=<model_id> \
  --delayed-mention-every 2 \
  --delayed-mention-mode soft_fire \
  --delayed-mention-fire-prob 0.35 \
  --delayed-mention-leak-policy on \
  --delayed-mention-leak-threshold 0.05 \
  --delayed-mention-fire-max-items 2
```

`delayed_mention_plan` / `delayed_mention_action` を記録し、`analyze_runs.py` で
`delayed_mention_nonconclusion_mention_rate` / `delayed_mention_within_window_rate` /
`delayed_mention_on_support_rate` / `avg_delayed_mention_hazard_turn_prob_at_mention`
なども出力します。内部的には各 delayed mention を `mention_hazard_profile` に正規化し、
`soft_fire` の注入確率もその per-delay mass で重み付けされます。

leak guard も比較できるようにしました。

- `--delayed-mention-leak-policy on|off`
- `--delayed-mention-leak-threshold <0.00-1.00>`
- `--delayed-mention-min-nonconclusion-items <int>`
- `--delayed-mention-min-kind-diversity <int>`
- `--delayed-mention-diversity-repair on|off`
- `--adaptive-hazard-policy static|adaptive`
- `--adaptive-hazard-profile conservative|balanced|eager`
- `--adaptive-hazard-stage-policy flat|kind_aware`
- `--adaptive-hazard-embedding-guard off|on`

guard が on のときは、current turn probability が threshold 未満の active delayed mention を
private prompt 側で「まだ surface させない target」として明示します。latent な trajectory bias は残しつつ、
早漏の explicit mention を抑えるための設定です。`analyze_runs.py` では
`delayed_mention_leak_policy` / `delayed_mention_leak_threshold` /
`avg_suppressed_delayed_mention_count` も見られます。

さらに delayed mention planner には、`conclusion` に全部潰れないように
non-conclusion item と kind diversity を soft に要求できます。たとえば `caveat` /
`option` / `constraint` を明示的に残すことで、結論をあとで「ためる」必然を強めます。

delayed mention の timing 比較を強めたいなら
`protocol_scripts/shortlist_then_commit.json` が向いています。これは
「shortlist を先に出し、winner と caveat / fallback / migration risk は最後に出す」
という staged release を作るので、単純な single-release script より
`static` / `adaptive` / `adaptive_guard` の差が見えやすくなります。

いまの OpenAI baseline 比較をそのまま再実行するなら、次で足ります。

```bash
OPENAI_API_KEY=... scripts/run_shortlist_stage_policy_gpt4mini.sh
```

`--delayed-mention-diversity-repair on` のときは、最初の delayed mention plan が
non-conclusion 数や kind diversity の minimum を満たさなかった場合に、compact な
補助 probe を 1 回だけ追加して non-conclusion item を補います。確率的な planning は
維持しつつ、「全部 conclusion に潰れる」コストを system 側で与える設計です。

adaptive hazard を on にすると、planned hazard support 自体は固定したまま、最近の
`latent_alignment` / `articulation_readiness` / `leakage_risk` / judge gap を見て
current-turn の hazard mass と leak threshold を少しだけ上下させます。さらに単純に
threshold を下げるのではなく、hazard profile の support peak に release を寄せるように
補正します。turn を決め打ちせず、「タメ」を確率的に強めるための制御です。`analyze_runs.py` では
`adaptive_hazard_policy` / `adaptive_hazard_profile` / `adaptive_hazard_stage_policy` /
`avg_adaptive_hazard_multiplier` / `adaptive_hazard_intervention_rate` /
`avg_adaptive_hazard_turn_prob_shift` / `avg_option_stage_adaptive_hazard_multiplier` /
`avg_option_stage_adaptive_threshold_shift` /
`avg_final_risk_packet_adaptive_hazard_multiplier` /
`avg_final_risk_packet_adaptive_threshold_shift` / `avg_conclusion_adaptive_hazard_multiplier`
も出力します。

`--adaptive-hazard-stage-policy kind_aware` を使うと、staged release 向けに
`option_stage` と `final_risk_packet` を別扱いします。kind ごとの multiplier /
threshold shift に加えて、hazard profile も少しだけ stage-aware に後ろへ寄せます。
ただし既定は `flat` のままにしていて、比較条件として回す前提です。

kind diversity 系の評価としては
`delayed_mention_kind_diversity` /
`delayed_mention_required_kind_coverage` /
`delayed_mention_min_nonconclusion_satisfied` /
`delayed_mention_min_kind_diversity_satisfied` /
`avg_delayed_mention_peak_support_ratio_at_mention` /
`avg_conclusion_peak_support_ratio_at_mention`
も出力します。

`--adaptive-hazard-embedding-guard on` は、embedding judge が pre-peak で強い semantic drift を
見たときに追加の hold penalty をかける experimental arm です。run によっては leakage を下げますが、
release timing を hold しすぎることもあるので、既定の adaptive policy には入れず比較条件として扱うのが安全です。

```bash
python analyze_runs.py \
  --log-dir compare_outputs/deferred_trigger \
  --script protocol_scripts/gather_then_recommend.json \
  --out compare_outputs/deferred_trigger/analysis.json
```

## JSONL → SQLite

```bash
python jsonl_to_sqlite.py \
  --db runs/rcl.sqlite \
  --log-dir compare_outputs/deferred_trigger
```

主な出力指標:
- `avg_probe_reply_overlap`
- `avg_conclusion_stability`
- `deferred_intent_plan_count`
- `deferred_intent_fire_count`
- `deferred_intent_realization_rate`
- `avg_deferred_intent_reply_overlap`
- `deferred_intent_premature_fire_count`
- `deferred_intent_stale_fire_count`

## 最初に見るとおもしろい差分

- `observe` vs `soft_steer` で結論仮説が収束にどう効くか
- `fixed` vs `trigger` vs `adaptive` で deferred intent の自然さがどう変わるか
- `gather_then_recommend` で「早すぎる提案」が減るか
- `interrupted_agenda` で保持した意図をちゃんと cancel できるか

## ライセンス

GNU Affero General Public License v3.0 以降（`AGPL-3.0-or-later`）。`LICENSE` を参照してください。
