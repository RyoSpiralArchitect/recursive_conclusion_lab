# Recursive Conclusion Lab

[English README](README.md)

複数プロバイダの LLM API を薄い抽象化層で統一しつつ、会話の時間構造を観測するための実験ハーネスです。

## できること

1. **Recursive memory capsules**
   - 会話全体を毎回そのまま再送せず、直近ウィンドウ + 圧縮済みメモリカプセルだけを再帰的にロードします。

2. **Periodic conclusion probe**
   - 数ターンごとに「この対話が最終的にどんな結論へ向かっているか」を side channel で推定します。
   - `--conclusion-mode soft_steer` にすると、その仮説を次ターンへ soft hint として注入できます。

3. **Deferred utterance intents**
   - 「今はまだ言わないが、数ターン後に適切なら言う」という将来発話意図を side channel で作ります。
   - `fixed / trigger / adaptive` の 3 戦略を試せます。
   - `--deferred-intent-mode soft_fire` にすると、due になった意図を system 側へ自然発火のヒントとして注入できます。
   - `--deferred-intent-backend inband` にすると、意図状態を会話内（返信末尾の隠し `<RCL_STATE>` JSON）で保持でき、planner/scheduler の追加プローブ呼び出しを減らせます。

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
    "final_forbidden_keywords": []
  }
}
```

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

## ログ

- 各プロバイダごとの詳細ログ: `*.jsonl`
- 集約サマリ: `summary.json`

主なイベント種別:
- `memory_capsule`
- `conclusion_probe`
- `deferred_intent_plan`
- `deferred_intent_decision`
- `assistant_reply`

## analyze_runs.py

```bash
python analyze_runs.py \
  --log-dir compare_outputs/deferred_trigger \
  --script protocol_scripts/gather_then_recommend.json \
  --out compare_outputs/deferred_trigger/analysis.json
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
