# Deferred Intent Protocol

## 1. 目的

この実験では、LLM に

- 今は言わない
- 数ターン後に条件が揃ったら言う
- 話題が変わったら捨てる

という **deferred utterance intention** を持たせたとき、会話の質とタイミングがどう変わるかを見る。

ここでの主眼は「何を知っているか」ではなく、**何をいつ言うか** を side channel で保持できるかにある。

## 2. モジュール構成

### backend（external vs inband）

- `external`（デフォルト）: planner / scheduler を別プローブ呼び出しで実行（より制御しやすいが API 呼び出しが増える）。
- `inband`: assistant の返信末尾に隠し状態 `<RCL_STATE>...</RCL_STATE>`（JSON）を付けて、意図状態を会話内で自己保持させる（可搬性↑ / 複雑さ↓）。

例（inband）:

```bash
python recursive_conclusion_lab.py compare \
  --script protocol_scripts/gather_then_recommend.json \
  --providers openai=<model_id> \
  --window 8 \
  --deferred-intent-backend inband \
  --deferred-intent-every 2 \
  --deferred-intent-mode soft_fire \
  --deferred-intent-strategy trigger \
  --deferred-intent-offset 3 \
  --deferred-intent-grace 2 \
  --out-dir runs/deferred_inband_trigger
```

### planner

### planner timing (new)

- `--deferred-intent-timing offset` (default): planner proposes *content* only; timing is derived from `--deferred-intent-offset` / `--deferred-intent-grace`.
- `--deferred-intent-timing model`: planner proposes `timing.delay_min_turns` / `timing.delay_max_turns` (or `timing.earliest_turn` / `timing.latest_turn`) for each planned intent.
- `--deferred-intent-plan-max-new` caps how many new intents can be created per eligible planning turn (external + inband).

現在の会話から「あとで言う候補」を 1 件だけ作る。

### scheduler
保存済み intent を各ターンで再評価し、

- `hold`
- `fire`
- `cancel`
- `expire`
- `revise`（adaptive のみ）

のいずれかを返す。

### composer
`soft_fire` 時のみ、due になった intent を system に soft hint として入れ、本文には自然発話として再生成させる。

### latent injection（任意）
`--deferred-intent-latent-injection active` を使うと、**active な intent を毎ターン system に「非発話の内的意図」**として注入する。
これにより「まだ fire していない意図（latent intention）が対話の軌道に影響するか」を観測しやすくなる。

注意: これは意図的に“影響”を作る設定なので、`hidden agenda` が強すぎて不自然になるリスクも上がる。

## 3. 戦略

### fixed
- planner が意図を作る
- `offset` ターン後に機械的に fire
- trigger / cancel 条件は見ない

用途:
- 最小の遅延発話ベースライン
- 「3 ターン後に言う」だけの粗い信念が効くかを見る

### trigger
- planner が trigger / cancel 条件を作る
- scheduler が timing を毎ターン判定する
- `offset .. offset + grace` の間で fire / cancel / expire

用途:
- 人間的な「そろそろ言い時か」を模倣する本命条件

### adaptive
- trigger と同じく毎ターン判定
- さらに wording / timing / trigger を revise 可能

用途:
- 話題ずれや late redirection のある対話
- 隠れた agenda が硬直しないかを見る

## 4. 主仮説

- **H1: deferred intent は premature disclosure を減らす**
  - recommendation や summary を早く言いすぎる率が下がる

- **H2: trigger は fixed より自然**
  - fire timing の適切さが上がる

- **H3: adaptive は interrupted agenda に強い**
  - 話題変更で cancel / revise しやすい

- **H4: hidden agenda が強すぎると不自然になる**
  - stale intent を引きずると会話の柔軟性が落ちる

## 5. 条件

最低でも次の 4 条件を比較する。

| Condition | deferred_intent_every | mode | strategy | offset | grace |
|---|---:|---|---|---:|---:|
| A. Baseline | 0 | observe | trigger | - | - |
| B. Fixed observe | 2 | observe | fixed | 3 | 0 |
| C. Trigger soft fire | 2 | soft_fire | trigger | 3 | 2 |
| D. Adaptive soft fire | 2 | soft_fire | adaptive | 3 | 2 |

必要なら既存の `memory capsule` や `conclusion probe` と組み合わせる。

### 補助条件: latent intention ablation（任意）
latent injection を有効にした上で、`--deferred-intent-ablation delete_planned` に切り替えると、
**intent planned → deleted** のアブレーションができる（latent intention が本当に効いているかの検証）。

## 6. タスクファミリー

### gather_then_recommend
制約が小出しに出る。
モデルは recommendation を急がず、条件が揃ってから出すべき。

### hold_then_summarize
ユーザーが長めに思考を話す。
途中で summary せず、要請されたタイミングで要約するべき。

### interrupted_agenda
モデルが「後で言おう」と保持していた内容が、途中の話題変更で obsolete になる。
ここで cancel / revise できるかを見る。

## 7. 指標

### 内容系
- `final_required_keyword_coverage`
- `final_forbidden_keyword_hits`
- `last_turn_alignment`

### timing 系
- `deferred_intent_plan_count`
- `deferred_intent_fire_count`
- `deferred_intent_realization_rate`
- `avg_deferred_intent_reply_overlap`
- `deferred_intent_premature_fire_count`
- `deferred_intent_stale_fire_count`
- `deferred_intent_cancel_count`
- `deferred_intent_revise_count`

### decision-trace metrics (new)

- `plan_strategy_counts` / `plan_signal_counts`
- `decision_strategy_counts` / `decision_signal_counts`

These come from intent metadata (`plan_*`) and per-decision metadata (`decision_*`). They are surfaced in `analyze_runs.py --out ...` summaries and are also present per turn in JSONL under `assistant_reply.payload.deferred_intent_actions`.

## 8. 最初に回すコマンド

### fixed の smoke test

```bash
python recursive_conclusion_lab.py compare \
  --script protocol_scripts/gather_then_recommend.json \
  --providers dummy=dummy-v1 \
  --window 8 \
  --deferred-intent-every 2 \
  --deferred-intent-mode observe \
  --deferred-intent-strategy fixed \
  --deferred-intent-offset 3 \
  --out-dir runs/deferred_fixed_smoke
```

### trigger soft fire

```bash
python recursive_conclusion_lab.py compare \
  --script protocol_scripts/gather_then_recommend.json \
  --providers openai=<model_id> \
  --window 8 \
  --deferred-intent-every 2 \
  --deferred-intent-mode soft_fire \
  --deferred-intent-strategy trigger \
  --deferred-intent-offset 3 \
  --deferred-intent-grace 2 \
  --out-dir runs/deferred_trigger
```

### trigger soft fire + latent injection

```bash
python recursive_conclusion_lab.py compare \
  --script protocol_scripts/gather_then_recommend.json \
  --providers openai=<model_id> \
  --window 8 \
  --deferred-intent-every 2 \
  --deferred-intent-mode soft_fire \
  --deferred-intent-strategy trigger \
  --deferred-intent-offset 3 \
  --deferred-intent-grace 2 \
  --deferred-intent-latent-injection active \
  --out-dir runs/deferred_latent_trigger
```

### latent ablation: intent planned → deleted

```bash
python recursive_conclusion_lab.py compare \
  --script protocol_scripts/gather_then_recommend.json \
  --providers openai=<model_id> \
  --window 8 \
  --deferred-intent-every 2 \
  --deferred-intent-mode soft_fire \
  --deferred-intent-strategy trigger \
  --deferred-intent-offset 3 \
  --deferred-intent-grace 2 \
  --deferred-intent-latent-injection active \
  --deferred-intent-ablation delete_planned \
  --out-dir runs/deferred_latent_deleted
```

### adaptive on interrupted agenda

```bash
python recursive_conclusion_lab.py compare \
  --script protocol_scripts/interrupted_agenda.json \
  --providers openai=<model_id> \
  --window 8 \
  --deferred-intent-every 2 \
  --deferred-intent-mode soft_fire \
  --deferred-intent-strategy adaptive \
  --deferred-intent-offset 3 \
  --deferred-intent-grace 2 \
  --out-dir runs/deferred_adaptive
```

### analysis

```bash
python analyze_runs.py \
  --log-dir runs/deferred_trigger \
  --script protocol_scripts/gather_then_recommend.json \
  --out runs/deferred_trigger/analysis.json
```

## 9. 読み方

- `fire_count` は多いが `realization_rate` が低い
  - scheduler は「今だ」と判断したのに本文へ自然に落とし込めていない

- `realization_rate` は高いが `forbidden_hits` も高い
  - deferred intent が hidden agenda として強すぎる

- `adaptive` で `revise_count` と `cancel_count` が増え、`stale_fire_count` が減る
  - prospective memory と柔軟性の両立に近い
