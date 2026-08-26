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

## Model profile と GPT-5.6

provider adapter は wire protocol を担当し、exact-ID の model profile は model family ごとの
capability と lab default を担当します。利用側の安定した契約は引き続き `BaseAdapter` と
`build_adapter(provider, model)` で、その内側に adapter registry と profile resolver があります。

OpenAI profile は現在、次の model ID だけを exact match で認識します。

- `gpt-5.6`（GPT-5.6 Sol を指す OpenAI alias）
- `gpt-5.6-sol`
- `gpt-5.6-terra`
- `gpt-5.6-luna`

この family で扱える設定は次の通りです。

- reasoning effort: `none | low | medium | high | xhigh | max`
- reasoning mode: `standard | pro`
- reasoning context: `current_turn | all_turns`
- text verbosity: `low | medium | high`

CLI では各設定に `auto` も指定できます。`auto` は profile 解決用の sentinel で、provider へは
送信しません。GPT-5.6 に対するこの lab の既定値は effort=`none`、mode=`standard`、
context=`current_turn` です。text verbosity は明示指定しない限り未設定のままです。これは
visible token の余裕と turn-local な実験条件を保つための lab 固有の既定値であり、OpenAI service
全体の既定値を説明するものではありません。

adapter は既存の stateless な Responses 動作を維持します。`store=false`、chat message の手動 replay、
`previous_response_id` なしのままです。`all_turns` を選ぶと request field は設定されますが、この PR で
call 間や arm 間の persisted reasoning reuse が追加されるわけではありません。

reply と probe は別々に設定できます。

```bash
python recursive_conclusion_lab.py repl \
  --provider openai \
  --model gpt-5.6-terra \
  --reasoning-effort low \
  --probe-reasoning-effort none \
  --reasoning-mode standard \
  --probe-reasoning-mode standard \
  --reasoning-context current_turn \
  --probe-reasoning-context current_turn \
  --text-verbosity medium \
  --probe-text-verbosity low \
  --max-tokens 1200 \
  --probe-max-tokens 360
```

GPT-5.6 profile は送信する reasoning effort が `none` のときだけ `temperature` を送ります。
effort が `none` 以外なら `temperature` を payload から外し、その省略を adapter metadata に
記録します。また OpenAI の `max_output_tokens` は visible output と reasoning token の両方を
含む上限です。reasoning 条件に余裕が必要なら `--max-tokens` または `--probe-max-tokens` を
増やしてください。

Playtest session は reply、probe、独立 observer の解決済み control と、exact profile ID / version を
保存します。後から profile が一致しなくなった場合は、異なる生成条件で黙って再開せず、session list に
復元エラーを出して停止します。
独立 observer が generator と異なる profile を使う場合、generator 専用の probe control は継承しません。
明示する場合は `--observer-reasoning-*` と `--observer-text-verbosity` を使います。既存の version-1
Playtest snapshot は、従来どおりの generic profile か、すでに完全な pin を持つ場合だけ初回 load 時に
移行します。現在 exact profile に一致する未固定の旧 snapshot は、過去の wire semantics を安全に
復元できないため fail closed します。

同じ request shape を使う OpenAI Responses の model family を足すときは `ModelProfile` を
`register_model_profile(...)` で登録します。別 provider や別 request shape には `BaseAdapter` の実装も
必要で、`ADAPTER_REGISTRY.register(...)` へ登録します。既存の `build_adapter(...)` 呼び出し側は
変更不要です。embedding adapter は引き続き別 registry で管理します。capability value は lowercase の
canonical string とし、`auto` は profile default を選ぶ予約語です。

profile 登録、request construction、offline parser test は、特定 API account の live access、quota、
必要な service tier を保証せず、live provider validation を行ったことも意味しません。実行前に
OpenAI 公式の [model catalog](https://developers.openai.com/api/docs/models)、
[GPT-5.6 guidance](https://developers.openai.com/api/docs/guides/latest-model)、
[reasoning guide](https://developers.openai.com/api/docs/guides/reasoning)、
[Responses API reference](https://developers.openai.com/api/docs/api-reference/responses/create) を確認してください。

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

対応する `deferred_multi_release` の stage-policy 比較を作るには、次を使います。

```bash
OPENAI_API_KEY=... scripts/run_deferred_multi_release_stage_policy_gpt4mini.sh
```

この 2 つの compare output から blind な pairwise human-eval packet を組むには、次で足ります。

```bash
scripts/build_staged_release_human_eval_set.sh
```

出力先は `human_eval_sets/staged_release_pairwise_v1/` で、`manifest.json`、
`eval_items.jsonl`、`booklet.md`、`answer_sheet.csv`、`blind_key.json`、`READY.json` と、
各 item ごとの Markdown packet を `packets/` に書き出します。`READY.json` がある場合だけ
publish 完了です。再 build では最初に marker を外すため、途中で落ちた出力を review server は読みません。

`manifest.json` と `eval_items.jsonl` は reviewer-safe な公開 packet です。arm 名、provider、
model、入力 path は含みません。A/B と実 arm の対応、seed、source digest、private config は
`blind_key.json` だけに保存されます。評価中はこのファイルを reviewer に渡さないでください。
`human_eval_sets/` 以下の生成 key は gitignore 対象です。公開 packet、booklet/Markdown packet、
answer sheet、readiness marker だけを reviewer 側へ渡し、key は research 側に保持します。
builder は arm 間で user turn 列が同一であることも検査し、比較組を表さない opaque item ID を振ります。

live な qualitative playtest 用には、minimal local web app も使えます。

```bash
pip install -r playtest_requirements.txt
scripts/run_playtest_server.sh
```

この backend は crash-tolerant な session API を持っていて、`playtest_ui/dist/` があれば
build 済み UI も `http://127.0.0.1:8787` からそのまま配信します。

frontend を開発しながら使うなら、別ターミナルで:

```bash
cd playtest_ui
npm install
npm run dev
```

dev 中は `http://127.0.0.1:5173`、`npm run build` 後は `http://127.0.0.1:8787` を開いてください。
full mode の app には 2 つの workspace があります。ただし Playtest API は arm / model 設定を返すため、
この full mode 全体を researcher console として扱ってください。

`Playtest` は benchmark 用ではなく、人間が transcript と live trace と観察メモを見ながら
洗うための非盲検 UI です。

- `static` / `adaptive_flat` / `adaptive_kind_aware` の session を作成・再開できる
- protocol script の turn を seed として流し込み、その後は自由対話に切り替えられる
- transcript、conclusion state、delayed mention pressure、軽い live metrics を横で見られる
- observer note を書きながら、backend が turn ごとに session を保存する

`Blind Review` は `eval_items.jsonl` を読み、arm / provider / model / machine metric を隠したまま
pairwise 判断を収集します。

- 同じ user turn の下で response A/B を比較する
- 各 rubric に `A / B / Tie`、confidence、evidence、counterevidence を記録する
- 判断不能は `Tie` と分けて、理由つきの `Abstain` として保存する
- 判断の訂正は古い event を上書きせず、`supersedes` つきで追記する
- 全 item 完了後にだけ seal し、research artifact に設問別 raw count と digest を書き出す

review event、`unblinded_results.json`、`seal_receipt.json` は
`blind_review_sessions/<session-id>/` に保存されます。seal 後も reviewer UI は blind のままで、
receipt だけを表示します。unblinded result は後から research 側で確認します。モデル API は呼びません。

reviewer に渡す instance は、Playtest UI/API を閉じた mode で起動します。

```bash
EVAL_SETS_DIR=examples/human_eval_sets scripts/run_blind_review_server.sh
```

現時点の review mode は loopback 上の「信頼された local rater 1 人」用で、認証つき multi-rater service
ではありません。また pairwise 判断から分かるのは arm 間の相対的な好みです。「結論を言う準備が
整った正確な turn」という強い timing claim には、次の評価層で absolute readiness-turn annotation が要ります。

playtest session snapshot は `playtest_sessions/` に保存されるので、server が turn の途中で落ちても
直前の user draft を復元できます。

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
